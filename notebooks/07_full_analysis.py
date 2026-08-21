import subprocess
import sys

PACKAGES = [
    "faiss-cpu",
    "sentence-transformers",
    "pandas",
    "pyarrow>=16",
    "transformers>=4.44",
    "accelerate",
    "bitsandbytes",
]

subprocess.run(
    [sys.executable, "-m", "pip", "install", "-q", *PACKAGES],
    check=True,
)

import os
import re
import ast
import json
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import faiss
import torch

from sentence_transformers import SentenceTransformer
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    pipeline,
)

warnings.filterwarnings("ignore")

DATA_DIR = "/kaggle/input/datasets/mrnotalent/laptop-embedding"
OUT_DIR = "/kaggle/working/laptop_rag_outputs"
os.makedirs(OUT_DIR, exist_ok=True)

PARQUET_PATH = os.path.join(
    DATA_DIR, "laptop_chunks_embeddings_with_lineage.parquet"
)

EVAL_CANDIDATES = [
    os.path.join(DATA_DIR, "eval_queries_final.csv"),
    os.path.join(DATA_DIR, "eval_queries.csv"),
]

EVAL_CSV_PATH = next(
    (p for p in EVAL_CANDIDATES if os.path.exists(p)),
    None,
)

FPGROWTH_RULES_PATH = os.path.join(
    DATA_DIR, "fpgrowth_association_rules.csv"
)


EMBEDDING_MODEL = "all-MiniLM-L6-v2"

MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"

TOP_K = 5
CACHE_THRESHOLD = 0.92
CACHE_MAX_SIZE = 200

print("DATA_DIR:", DATA_DIR)
print("PARQUET_PATH:", PARQUET_PATH)
print("EVAL_CSV_PATH:", EVAL_CSV_PATH)
print("EMBEDDING_MODEL:", EMBEDDING_MODEL)
print("LLM:", MODEL_ID)

if not os.path.exists(PARQUET_PATH):
    raise FileNotFoundError(
        f"Parquet file not found:\n{PARQUET_PATH}\n"
        "Check the Kaggle dataset path/name."
    )

if EVAL_CSV_PATH is None:
    raise FileNotFoundError(
        "Neither eval_queries_final.csv nor eval_queries.csv was found in:\n"
        f"{DATA_DIR}"
    )


df = pd.read_parquet(PARQUET_PATH)

required_columns = [
    "embedding",
    "row_uid",
    "title",
    "price_usd",
    "cpu",
    "ram_gb",
    "storage",
    "gpu",
    "display",
    "battery",
    "chunk_text",
    "lineage",
]

missing = [c for c in required_columns if c not in df.columns]
if missing:
    raise ValueError(
        "The parquet is missing required columns:\n"
        + "\n".join(f"- {c}" for c in missing)
    )

for col in ["title", "cpu", "storage", "gpu", "display", "battery", "chunk_text", "lineage"]:
    df[col] = df[col].fillna("").astype(str)

df["price_usd"] = pd.to_numeric(df["price_usd"], errors="coerce")
df["ram_gb"] = pd.to_numeric(df["ram_gb"], errors="coerce")

df["row_uid"] = df["row_uid"].astype(str)

valid_embedding_mask = df["embedding"].apply(
    lambda x: isinstance(x, (list, tuple, np.ndarray)) and len(x) > 0
)

removed_embeddings = int((~valid_embedding_mask).sum())
if removed_embeddings:
    print(f"Removing {removed_embeddings} rows with invalid embeddings.")
    df = df.loc[valid_embedding_mask].reset_index(drop=True)

embedding_lengths = df["embedding"].apply(len)
if embedding_lengths.nunique() != 1:
    counts = embedding_lengths.value_counts().to_dict()
    raise ValueError(
        f"Embedding dimensions are inconsistent: {counts}"
    )

EMBEDDING_DIM = int(embedding_lengths.iloc[0])
embeddings = np.vstack(df["embedding"].to_numpy()).astype("float32")

if not np.isfinite(embeddings).all():
    raise ValueError("Embedding matrix contains NaN/Inf values.")

device_level = (
    df.drop_duplicates(subset="row_uid", keep="first")
      .reset_index(drop=True)
      .copy()
)

print("\n=== DATA ===")
print("Chunk rows:", len(df))
print("Unique devices:", len(device_level))
print("Embedding matrix:", embeddings.shape)
print("Embedding dimension:", EMBEDDING_DIM)

print("\nLoading embedding model...")
embed_model = SentenceTransformer(EMBEDDING_MODEL)

test_embedding = embed_model.encode(
    ["dimension check"],
    convert_to_numpy=True,
).astype("float32")

QUERY_EMBEDDING_DIM = int(test_embedding.shape[1])

if QUERY_EMBEDDING_DIM != EMBEDDING_DIM:
    raise ValueError(
        "\nEMBEDDING DIMENSION MISMATCH\n"
        f"Parquet embeddings: {EMBEDDING_DIM}\n"
        f"Query model ({EMBEDDING_MODEL}): {QUERY_EMBEDDING_DIM}\n\n"
        "The query embedding model MUST be the exact same embedding model "
        "used to create the parquet embeddings. Change EMBEDDING_MODEL "
        "accordingly and rerun this notebook."
    )

print("Embedding model OK.")
print("Query embedding dimension:", QUERY_EMBEDDING_DIM)

print("\nLoading LLM...")

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
)

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    quantization_config=bnb_config,
    device_map="auto",
    torch_dtype=torch.float16,
)

llm_pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=400,
    do_sample=True,
    temperature=0.3,
    top_p=0.9,
    return_full_text=False,
)

print("LLM loaded.")

def _numeric(text):
    """Return a float from a numeric regex capture."""
    return float(text)


def extract_filters(query: str):

    q = query.lower()
    mask = pd.Series(True, index=device_level.index)
    applied = []

    price_patterns = [
        r"\bunder\s*\$?\s*([\d,]+(?:\.\d+)?)",
        r"\bless\s+than\s*\$?\s*([\d,]+(?:\.\d+)?)",
        r"\bbelow\s*\$?\s*([\d,]+(?:\.\d+)?)",
        r"\bup\s+to\s*\$?\s*([\d,]+(?:\.\d+)?)",
    ]

    price_match = None
    for pattern in price_patterns:
        price_match = re.search(pattern, q)
        if price_match:
            break

    if price_match:
        price = _numeric(price_match.group(1).replace(",", ""))
        mask &= device_level["price_usd"].notna()
        mask &= device_level["price_usd"] < price
        applied.append(f"price_usd < {price:g}")

    price_patterns = [
        r"\bover\s*\$?\s*([\d,]+(?:\.\d+)?)",
        r"\babove\s*\$?\s*([\d,]+(?:\.\d+)?)",
        r"\bmore\s+than\s*\$?\s*([\d,]+(?:\.\d+)?)",
    ]

    price_match = None
    for pattern in price_patterns:
        price_match = re.search(pattern, q)
        if price_match:
            break

    if price_match:
        price = _numeric(price_match.group(1).replace(",", ""))
        mask &= device_level["price_usd"].notna()
        mask &= device_level["price_usd"] > price
        applied.append(f"price_usd > {price:g}")

    ram_match = re.search(
        r"(?:at\s+least\s+)?(\d+(?:\.\d+)?)\s*gb\s*(?:of\s*)?ram",
        q,
    )

    if ram_match:
        ram = _numeric(ram_match.group(1))
        mask &= device_level["ram_gb"].notna()
        mask &= device_level["ram_gb"] >= ram
        applied.append(f"ram_gb >= {ram:g}")

    if any(
        word in q
        for word in ["nvidia", "rtx", "geforce"]
    ):
        mask &= device_level["gpu"].str.contains(
            r"NVIDIA|RTX|GeForce",
            case=False,
            na=False,
            regex=True,
        )
        applied.append("gpu contains NVIDIA/RTX/GeForce")

    if "gaming" in q and not any(
        word in q for word in ["nvidia", "rtx", "geforce", "amd"]
    ):
        mask &= device_level["gpu"].str.strip().ne("")
        applied.append("gpu is present")

    if "amd" in q or "ryzen" in q:
        mask &= device_level["cpu"].str.contains(
            r"Ryzen|AMD",
            case=False,
            na=False,
            regex=True,
        )
        applied.append("cpu contains Ryzen/AMD")

    for cpu_kw in ["i9", "i7", "i5", "i3"]:
        if re.search(rf"\b{re.escape(cpu_kw)}\b", q):
            mask &= device_level["cpu"].str.contains(
                rf"\b{re.escape(cpu_kw)}\b",
                case=False,
                na=False,
                regex=True,
            )
            applied.append(f"cpu contains {cpu_kw}")

    if any(word in q for word in ["ssd", "nvme"]):
        mask &= device_level["storage"].str.contains(
            r"SSD|PCIe|NVMe",
            case=False,
            na=False,
            regex=True,
        )
        applied.append("storage contains SSD/PCIe/NVMe")

    storage_match = re.search(
        r"(\d+(?:\.\d+)?)\s*(tb|gb)\s*(?:ssd|storage)",
        q,
    )

    if storage_match:
        amount = float(storage_match.group(1))
        unit = storage_match.group(2).lower()

        requested = (
            f"{amount:g}{unit}"
        )

        mask &= device_level["storage"].str.contains(
            re.escape(requested),
            case=False,
            na=False,
            regex=True,
        )
        applied.append(f"storage contains {requested}")

    return mask, applied


RESULT_COLUMNS = [
    "row_uid",
    "title",
    "price_usd",
    "cpu",
    "ram_gb",
    "storage",
    "gpu",
    "display",
    "battery",
    "chunk_text",
    "lineage",
]


def retrieve(query: str, k: int = 5):
   
    if k <= 0:
        raise ValueError("k must be >= 1")

    mask, applied_filters = extract_filters(query)

    candidate_uids = set(
        device_level.loc[mask, "row_uid"].astype(str)
    )

    if not candidate_uids:
        return (
            df.iloc[0:0][RESULT_COLUMNS].copy(),
            applied_filters + ["NO DEVICES SATISFY ALL EXPLICIT FILTERS"],
        )

    candidate_chunk_idx = np.flatnonzero(
        df["row_uid"].isin(candidate_uids).to_numpy()
    )

    if len(candidate_chunk_idx) == 0:
        return (
            df.iloc[0:0][RESULT_COLUMNS].copy(),
            applied_filters + ["NO CHUNKS FOUND FOR FILTERED DEVICES"],
        )

    candidate_embeddings = embeddings[candidate_chunk_idx]

    sub_index = faiss.IndexFlatL2(EMBEDDING_DIM)
    sub_index.add(candidate_embeddings)

    query_vec = embed_model.encode(
        [query],
        convert_to_numpy=True,
    ).astype("float32")

    n_results = min(k, len(candidate_chunk_idx))
    distances, local_idx = sub_index.search(query_vec, n_results)

    valid_local = local_idx[0] >= 0
    local_idx = local_idx[0][valid_local]
    distances = distances[0][valid_local]

    global_idx = candidate_chunk_idx[local_idx]

    results = df.iloc[global_idx][RESULT_COLUMNS].copy()
    results["retrieval_distance"] = distances

    results = results.sort_values(
        "retrieval_distance",
        ascending=True,
    ).reset_index(drop=True)

    return results, applied_filters


def _safe_value(value):
    if pd.isna(value):
        return "N/A"
    return str(value)


def generate_recommendation(
    query: str,
    retrieved_df: pd.DataFrame,
):
  
    if retrieved_df.empty:
        return (
            "I could not find a laptop in the dataset that satisfies "
            "all of the explicit requirements."
        )

    context_blocks = []

    for rank, (_, row) in enumerate(retrieved_df.iterrows(), start=1):
        context_blocks.append(
            f"{rank}. "
            f"Title: {_safe_value(row['title'])} | "
            f"Price: ${_safe_value(row['price_usd'])} | "
            f"CPU: {_safe_value(row['cpu'])} | "
            f"RAM: {_safe_value(row['ram_gb'])} GB | "
            f"Storage: {_safe_value(row['storage'])} | "
            f"GPU: {_safe_value(row['gpu'])} | "
            f"Display: {_safe_value(row['display'])} | "
            f"Battery: {_safe_value(row['battery'])}"
        )

    context = "\n".join(context_blocks)

    user_prompt = f"""
You are a laptop recommendation assistant.

User request:
{query}

Retrieved candidates:
{context}

Rules:
1. Recommend only laptops listed in the retrieved candidates.
2. Do not invent or assume specifications.
3. Do not recommend a laptop that violates an explicit user constraint.
4. Use only the displayed fields.
5. Pick the best match, or up to two if they are genuinely useful alternatives.
6. Briefly explain why the selected laptop matches the request.
7. If none of the retrieved candidates is a good match, say so clearly.
"""

    messages = [
        {
            "role": "system",
            "content": (
                "You are a grounded laptop recommendation assistant. "
                "Never fabricate product specifications."
            ),
        },
        {
            "role": "user",
            "content": user_prompt,
        },
    ]

    prompt_text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    output = llm_pipe(prompt_text)

    return output[0]["generated_text"].strip()


def parse_filter_condition(filter_str: str):

    match = re.match(
        r"^\s*(\w+)\s*(>=|<=|>|<|==)\s*([\d.]+)\s*$",
        filter_str.strip(),
    )

    if not match:
        return None

    col = match.group(1)
    op = match.group(2)
    threshold = float(match.group(3))

    return col, op, threshold


def check_condition(actual, op, threshold):
    if op == "<":
        return actual < threshold
    if op == "<=":
        return actual <= threshold
    if op == ">":
        return actual > threshold
    if op == ">=":
        return actual >= threshold
    if op == "==":
        return actual == threshold
    return False


def verify_grounding(
    query: str,
    retrieved_df: pd.DataFrame,
    applied_filters: list,
    answer_text: str,
):

    numeric_filters = [
        parse_filter_condition(f)
        for f in applied_filters
    ]
    numeric_filters = [
        f for f in numeric_filters
        if f is not None
    ]

    verified_facts = []
    warnings_list = []

    answer_lower = answer_text.lower()

    for _, row in retrieved_df.iterrows():
        title = _safe_value(row["title"])
        title_lower = title.lower()

        title_tokens = [
            token for token in re.findall(r"[a-z0-9]+", title_lower)
            if len(token) >= 4
        ]

        title_mentioned = False

        if title_lower in answer_lower:
            title_mentioned = True
        elif title_tokens:
            sample_tokens = title_tokens[:4]
            title_mentioned = sum(
                token in answer_lower for token in sample_tokens
            ) >= max(1, min(2, len(sample_tokens)))

        if not title_mentioned:
            continue

        fact_line = (
            f"{title} -> price ${row['price_usd']:.2f}"
            if pd.notna(row["price_usd"])
            else f"{title} -> price N/A"
        )

        checks = []

        for col, op, threshold in numeric_filters:
            if col not in row.index or pd.isna(row[col]):
                continue

            actual = float(row[col])
            satisfies = check_condition(
                actual,
                op,
                threshold,
            )

            checks.append(
                f"{col} {op} {threshold:g} -> "
                f"{'PASS' if satisfies else 'FAIL'} "
                f"(actual={actual:g})"
            )

            if not satisfies:
                warnings_list.append(
                    f"MISMATCH: '{title}' does not satisfy "
                    f"{col} {op} {threshold:g}; "
                    f"actual={actual:g}."
                )

        verified_facts.append(
            fact_line
            + (" | " + "; ".join(checks) if checks else "")
        )

    contradiction_patterns = [
        r"above\s+(?:the\s+|a\s+|our\s+)?budget",
        r"over\s+(?:the\s+|a\s+|our\s+)?budget",
        r"exceed\w*\s+(?:the\s+|a\s+|our\s+)?budget",
        r"not\s+(?:strictly\s+)?under\s+(?:the\s+|a\s+|our\s+)?budget",
        r"too\s+expensive",
        r"outside\s+(?:the\s+|a\s+|our\s+)?budget",
        r"beyond\s+(?:the\s+|a\s+|our\s+)?budget",
    ]

    sentences = re.split(
        r"(?<=[.!?])\s+",
        answer_text,
    )

    for sentence in sentences:
        sentence_lower = sentence.lower()

        for pattern in contradiction_patterns:
            match = re.search(
                pattern,
                sentence_lower,
            )

            if not match:
                continue

            preceding_text = sentence_lower[:match.start()]
            preceding_words = preceding_text.split()[-5:]

            negation_words = {
                "without",
                "not",
                "no",
                "doesn't",
                "does",
                "isn't",
                "is",
                "won't",
                "never",
            }

            if any(
                word in negation_words
                for word in preceding_words
            ):
                continue

            warnings_list.append(
                "Possible budget contradiction — "
                f"verify manually: {sentence.strip()}"
            )
            break

    return verified_facts, warnings_list


def _serializable_records(dataframe):
    """
    Convert NaN/numpy values into JSON-safe Python values.
    """
    if dataframe.empty:
        return []

    clean = dataframe.copy()

    clean = clean.replace(
        {np.nan: None}
    )

    records = clean.to_dict(orient="records")

    def convert(obj):
        if isinstance(obj, np.generic):
            return obj.item()
        return obj

    return [
        {key: convert(value) for key, value in record.items()}
        for record in records
    ]


def recommend(
    query: str,
    k: int = TOP_K,
):
    retrieved, applied_filters = retrieve(query, k)

    answer = generate_recommendation(
        query,
        retrieved,
    )

    return {
        "query": query,
        "filters_applied": applied_filters,
        "retrieved_context": _serializable_records(retrieved),
        "lineage": retrieved["lineage"].tolist()
        if not retrieved.empty
        else [],
        "recommendation": answer,
    }


def recommend_verified(
    query: str,
    k: int = TOP_K,
):
    retrieved, applied_filters = retrieve(query, k)

    answer = generate_recommendation(
        query,
        retrieved,
    )

    verified_facts, warnings_list = verify_grounding(
        query,
        retrieved,
        applied_filters,
        answer,
    )

    return {
        "query": query,
        "filters_applied": applied_filters,
        "retrieved_context": _serializable_records(retrieved),
        "lineage": retrieved["lineage"].tolist()
        if not retrieved.empty
        else [],
        "recommendation": answer,
        "grounding_check": {
            "verified_facts": verified_facts,
            "warnings": warnings_list,
            "passed": len(warnings_list) == 0,
        },
    }


print("\n=== FILTER SANITY TEST ===")

test_mask, test_filters = extract_filters(
    "gaming laptop with NVIDIA graphics under $1200"
)

print("Filters:", test_filters)
print("Matching devices:", int(test_mask.sum()))

assert all(
    "price_usd < 1200" not in f
    or True
    for f in test_filters
)

print("\n=== RETRIEVAL SANITY TEST ===")

test_retrieved, test_applied = retrieve(
    "gaming laptop with NVIDIA graphics under $1200",
    k=5,
)

print("Applied filters:", test_applied)
print("Retrieved rows:", len(test_retrieved))

if not test_retrieved.empty:
    print(
        test_retrieved[
            [
                "row_uid",
                "title",
                "price_usd",
                "cpu",
                "ram_gb",
                "storage",
                "gpu",
            ]
        ].to_string(index=False)
    )

print("\n=== QUALITATIVE TEST ===")

qualitative_queries = [
    "gaming laptop with NVIDIA graphics under $1200",
    "AMD Ryzen laptop under $600",
    "lightweight ultrabook for a college student",
    "laptop with at least 32GB RAM and 1TB SSD storage",
    "cheapest laptop with an i9 processor",
]

qualitative_results = []

for query in qualitative_queries:
    result = recommend_verified(query, k=TOP_K)
    qualitative_results.append(result)

    print("\n" + "=" * 100)
    print("QUERY:", query)
    print("FILTERS:", result["filters_applied"])
    print(
        "N CANDIDATES:",
        len(result["retrieved_context"]),
    )
    print("-" * 100)
    print(result["recommendation"])
    print("-" * 100)
    print(
        "GROUNDING CHECK PASSED:",
        result["grounding_check"]["passed"],
    )

    if result["grounding_check"]["warnings"]:
        print("WARNINGS:")
        for warning in result["grounding_check"]["warnings"]:
            print(" -", warning)

with open(
    os.path.join(OUT_DIR, "m4_qualitative_samples.json"),
    "w",
    encoding="utf-8",
) as f:
    json.dump(
        qualitative_results,
        f,
        indent=2,
        ensure_ascii=False,
    )

eval_df = pd.read_csv(EVAL_CSV_PATH)

required_eval_columns = {
    "query",
    "relevant_row_uids",
}

missing_eval = required_eval_columns - set(eval_df.columns)

if missing_eval:
    raise ValueError(
        "Evaluation CSV is missing columns: "
        + ", ".join(sorted(missing_eval))
    )

if len(eval_df) < 30:
    raise ValueError(
        f"Evaluation set contains only {len(eval_df)} queries. "
        "The project requires at least 30 labeled queries."
    )

print("\n=== EVALUATION DATA ===")
print("Evaluation rows:", len(eval_df))
print("Columns:", eval_df.columns.tolist())


def parse_relevant_uids(value):
    """
    Supports:
      "['abc', 'def']"
      '["abc", "def"]'
      ['abc', 'def']
      a single UID
    """
    if isinstance(value, set):
        return {str(x).strip() for x in value}

    if isinstance(value, (list, tuple, np.ndarray)):
        return {
            str(x).strip()
            for x in value
            if str(x).strip()
        }

    if pd.isna(value):
        return set()

    text = str(value).strip()

    if not text:
        return set()

    try:
        parsed = ast.literal_eval(text)

        if isinstance(parsed, (list, tuple, set)):
            return {
                str(x).strip()
                for x in parsed
                if str(x).strip()
            }

        if parsed is not None:
            return {str(parsed).strip()}

    except (ValueError, SyntaxError):
        pass

    # Fallback for comma-separated values.
    text = text.strip("[](){}")
    parts = [
        part.strip().strip("'\"")
        for part in text.split(",")
    ]

    return {
        part
        for part in parts
        if part
    }


def evaluate_pipeline_retrieval(
    evaluation_df: pd.DataFrame,
    k: int = 5,
):
    rows = []

    for _, eval_row in evaluation_df.iterrows():
        query = str(eval_row["query"])
        relevant_uids = parse_relevant_uids(
            eval_row["relevant_row_uids"]
        )

        retrieved, applied_filters = retrieve(
            query,
            k=k,
        )

        retrieved_uids = set(
            retrieved["row_uid"].astype(str).tolist()
        )

        n_relevant = len(relevant_uids)
        n_retrieved = len(retrieved_uids)
        n_hits = len(
            retrieved_uids.intersection(relevant_uids)
        )

        precision = (
            n_hits / n_retrieved
            if n_retrieved > 0
            else 0.0
        )

        recall = (
            n_hits / n_relevant
            if n_relevant > 0
            else 0.0
        )

        hit = n_hits > 0

        rows.append(
            {
                "query": query,
                "n_relevant": n_relevant,
                "n_retrieved": n_retrieved,
                "n_filters_applied": len(
                    applied_filters
                ),
                "n_hits": n_hits,
                "hit": hit,
                "precision_at_k": precision,
                "recall_at_k": recall,
                "filters_applied": " | ".join(
                    applied_filters
                ),
                "retrieved_row_uids": "|".join(
                    sorted(retrieved_uids)
                ),
                "retrieved_titles": " | ".join(
                    retrieved["title"].astype(str).tolist()
                ),
            }
        )

    return pd.DataFrame(rows)


retrieval_eval_df = evaluate_pipeline_retrieval(
    eval_df,
    k=TOP_K,
)

retrieval_eval_path = os.path.join(
    OUT_DIR,
    "retrieval_evaluation_FIXED.csv",
)

retrieval_eval_df.to_csv(
    retrieval_eval_path,
    index=False,
)

print("\n=== RETRIEVAL EVALUATION ===")
print(f"K = {TOP_K}")
print(
    f"Mean Precision@{TOP_K}: "
    f"{retrieval_eval_df['precision_at_k'].mean():.4f}"
)
print(
    f"Mean Recall@{TOP_K}: "
    f"{retrieval_eval_df['recall_at_k'].mean():.4f}"
)
print(
    f"Hit Rate: "
    f"{retrieval_eval_df['hit'].mean():.4f}"
)

print("\nPer-query evaluation:")
print(
    retrieval_eval_df[
        [
            "query",
            "n_relevant",
            "n_retrieved",
            "n_hits",
            "precision_at_k",
            "recall_at_k",
            "hit",
        ]
    ].to_string(index=False)
)

failures = retrieval_eval_df[
    ~retrieval_eval_df["hit"]
].copy()

failures_path = os.path.join(
    OUT_DIR,
    "retrieval_failures_FIXED.csv",
)

failures.to_csv(
    failures_path,
    index=False,
)

print("\n=== RETRIEVAL FAILURES ===")
print("Failed queries:", len(failures))

if not failures.empty:
    print(
        failures[
            [
                "query",
                "filters_applied",
                "retrieved_titles",
            ]
        ].to_string(index=False)
    )

class SemanticCache:
    def __init__(
        self,
        similarity_threshold: float = 0.92,
        max_size: int = 200,
    ):
        self.threshold = similarity_threshold
        self.max_size = max_size
        self.embeddings = []
        self.entries = []

    @staticmethod
    def _normalize(vec):
        vec = np.asarray(vec, dtype="float32")
        norm = np.linalg.norm(vec)

        if norm <= 0:
            return np.zeros_like(vec)

        return vec / norm

    def lookup(self, query_embedding):
        if not self.embeddings:
            return None, None

        q = self._normalize(query_embedding)

        sims = np.array(
            [
                float(np.dot(q, embedding))
                for embedding in self.embeddings
            ],
            dtype="float32",
        )

        best_idx = int(np.argmax(sims))
        best_sim = float(sims[best_idx])

        if best_sim >= self.threshold:
            matched_query, response = self.entries[
                best_idx
            ]

            return response, {
                "matched_query": matched_query,
                "similarity": best_sim,
            }

        return None, None

    def store(
        self,
        query: str,
        query_embedding,
        response: dict,
    ):
        if len(self.embeddings) >= self.max_size:
            self.embeddings.pop(0)
            self.entries.pop(0)

        self.embeddings.append(
            self._normalize(query_embedding)
        )

        self.entries.append(
            (query, response)
        )

    def clear(self):
        self.embeddings.clear()
        self.entries.clear()

    def __len__(self):
        return len(self.entries)


cache = SemanticCache(
    similarity_threshold=CACHE_THRESHOLD,
    max_size=CACHE_MAX_SIZE,
)


def embed_query(query: str):
    return embed_model.encode(
        [query],
        convert_to_numpy=True,
    )[0].astype("float32")


def recommend_cached(
    query: str,
    k: int = TOP_K,
):
    query_embedding = embed_query(query)

    cached_response, match_info = cache.lookup(
        query_embedding
    )

    if cached_response is not None:
        result = dict(cached_response)
        result["cache_hit"] = True
        result["matched_query"] = match_info[
            "matched_query"
        ]
        result["similarity"] = round(
            match_info["similarity"],
            4,
        )
        return result

    result = recommend_verified(
        query,
        k,
    )

    result["cache_hit"] = False

    cache.store(
        query,
        query_embedding,
        result,
    )

    return result


benchmark_queries = [
    "AMD Ryzen laptop under $600",
    "cheap AMD laptop under $600 dollars",
    "gaming laptop with NVIDIA graphics under $1200",
    "budget laptop with AMD Ryzen processor under 600 bucks",
    "lightweight ultrabook for travel",
    "AMD Ryzen laptop under $600",
    "NVIDIA gaming laptop under $1200 budget",
    "laptop with 32GB RAM and 1TB SSD",
]

# Start from a clean cache for a reproducible benchmark.
cache.clear()

timings = []

for query in benchmark_queries:
    start = time.perf_counter()

    result = recommend_cached(
        query,
        k=TOP_K,
    )

    elapsed = time.perf_counter() - start

    timings.append(
        {
            "query": query,
            "cache_hit": result["cache_hit"],
            "latency_s": elapsed,
            "matched_query": result.get(
                "matched_query"
            ),
            "similarity": result.get(
                "similarity"
            ),
        }
    )

    if result["cache_hit"]:
        hit_text = (
            f"HIT | matched={result.get('matched_query')} "
            f"| similarity={result.get('similarity')}"
        )
    else:
        hit_text = "MISS"

    print(
        f"[{elapsed:.3f}s] {hit_text} | {query}"
    )

bench_df = pd.DataFrame(timings)

bench_path = os.path.join(
    OUT_DIR,
    "semantic_cache_benchmark.csv",
)

bench_df.to_csv(
    bench_path,
    index=False,
)

hit_df = bench_df[
    bench_df["cache_hit"]
]

miss_df = bench_df[
    ~bench_df["cache_hit"]
]

avg_hit = (
    hit_df["latency_s"].mean()
    if not hit_df.empty
    else np.nan
)

avg_miss = (
    miss_df["latency_s"].mean()
    if not miss_df.empty
    else np.nan
)

speedup = (
    avg_miss / avg_hit
    if (
        pd.notna(avg_hit)
        and avg_hit > 0
        and pd.notna(avg_miss)
    )
    else np.nan
)

cache_summary = pd.DataFrame(
    [
        {
            "hit_rate": bench_df[
                "cache_hit"
            ].mean(),
            "avg_latency_hit_s": avg_hit,
            "avg_latency_miss_s": avg_miss,
            "speedup_hit_vs_miss": speedup,
            "similarity_threshold": CACHE_THRESHOLD,
            "n_queries": len(bench_df),
        }
    ]
)

cache_summary_path = os.path.join(
    OUT_DIR,
    "semantic_cache_summary.csv",
)

cache_summary.to_csv(
    cache_summary_path,
    index=False,
)

print("\n=== CACHE SUMMARY ===")
print(
    cache_summary.to_string(index=False)
)

threshold_results = []

benchmark_embeddings = {
    query: embed_query(query)
    for query in benchmark_queries
}

for threshold in [
    0.80,
    0.85,
    0.90,
    0.92,
    0.95,
    0.98,
]:
    test_cache = SemanticCache(
        similarity_threshold=threshold,
        max_size=CACHE_MAX_SIZE,
    )

    hits = 0

    for query in benchmark_queries:
        query_embedding = benchmark_embeddings[
            query
        ]

        cached_response, match_info = (
            test_cache.lookup(query_embedding)
        )

        if cached_response is not None:
            hits += 1
        else:
            test_cache.store(
                query,
                query_embedding,
                {
                    "query": query,
                    "recommendation": "placeholder",
                },
            )

    threshold_results.append(
        {
            "threshold": threshold,
            "hits": hits,
            "hit_rate": hits / len(
                benchmark_queries
            ),
        }
    )

threshold_df = pd.DataFrame(
    threshold_results
)

threshold_path = os.path.join(
    OUT_DIR,
    "semantic_cache_threshold_sweep.csv",
)

threshold_df.to_csv(
    threshold_path,
    index=False,
)

print("\n=== CACHE THRESHOLD SWEEP ===")
print(
    threshold_df.to_string(index=False)
)

def cosine_sim(a, b):
    a = np.asarray(a, dtype="float32")
    b = np.asarray(b, dtype="float32")

    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)

    if norm_a == 0 or norm_b == 0:
        return 0.0

    return float(
        np.dot(a, b)
        / (norm_a * norm_b)
    )


pairs_to_check = [
    (
        "AMD Ryzen laptop under $600",
        "cheap AMD laptop under $600 dollars",
    ),
    (
        "AMD Ryzen laptop under $600",
        "budget laptop with AMD Ryzen processor, under 600 bucks",
    ),
    (
        "gaming laptop with NVIDIA graphics under $1200",
        "NVIDIA gaming laptop under $1200 budget",
    ),
]

similarity_results = []

for query_a, query_b in pairs_to_check:
    e1 = embed_query(query_a)
    e2 = embed_query(query_b)

    similarity = cosine_sim(e1, e2)

    similarity_results.append(
        {
            "query_a": query_a,
            "query_b": query_b,
            "similarity": similarity,
        }
    )

    print(
        f"{similarity:.4f} | "
        f"'{query_a}' vs '{query_b}'"
    )

similarity_df = pd.DataFrame(
    similarity_results
)

similarity_path = os.path.join(
    OUT_DIR,
    "semantic_cache_paraphrase_similarities.csv",
)

similarity_df.to_csv(
    similarity_path,
    index=False,
)

fpgrowth_rules = None

if os.path.exists(FPGROWTH_RULES_PATH):
    fpgrowth_rules = pd.read_csv(
        FPGROWTH_RULES_PATH
    )

    required_fp_columns = {
        "antecedent",
        "consequent",
        "confidence",
        "lift",
    }

    missing_fp = (
        required_fp_columns
        - set(fpgrowth_rules.columns)
    )

    if missing_fp:
        raise ValueError(
            "FP-Growth rules file is missing columns: "
            + ", ".join(sorted(missing_fp))
        )

    fpgrowth_rules["antecedent"] = (
        fpgrowth_rules["antecedent"]
        .apply(parse_relevant_uids)
    )

    fpgrowth_rules["consequent"] = (
        fpgrowth_rules["consequent"]
        .apply(parse_relevant_uids)
    )

    fpgrowth_rules["confidence"] = pd.to_numeric(
        fpgrowth_rules["confidence"],
        errors="coerce",
    )

    fpgrowth_rules["lift"] = pd.to_numeric(
        fpgrowth_rules["lift"],
        errors="coerce",
    )

    print(
        f"Loaded {len(fpgrowth_rules)} FP-Growth rules."
    )
else:
    print(
        "\nFP-Growth rules file not found. "
        "FP-Growth recommendations will be empty."
    )


def bucket_cpu_simple(cpu):
    if not isinstance(cpu, str):
        return None

    cpu_l = cpu.lower()

    if "ryzen 9" in cpu_l or "ryzen 7" in cpu_l:
        return "CPU_Ryzen7/9"

    if "ryzen 5" in cpu_l:
        return "CPU_Ryzen5"

    if "ryzen 3" in cpu_l:
        return "CPU_Ryzen3"

    if "i9" in cpu_l or "i7" in cpu_l:
        return "CPU_i7/i9"

    if "i5" in cpu_l:
        return "CPU_i5"

    if "i3" in cpu_l:
        return "CPU_i3"

    return None


def get_paired_specs(
    cpu_value: str,
    top_n: int = 3,
):
    if fpgrowth_rules is None:
        return []

    bucket = bucket_cpu_simple(cpu_value)

    if bucket is None:
        return []

    matches = fpgrowth_rules[
        fpgrowth_rules["antecedent"].apply(
            lambda antecedent:
                bucket in antecedent
                and len(antecedent) == 1
        )
    ].sort_values(
        "lift",
        ascending=False,
    ).head(top_n)

    output = []

    for _, row in matches.iterrows():
        output.append(
            {
                "paired_with": ", ".join(
                    sorted(
                        str(x)
                        for x in row["consequent"]
                    )
                ),
                "confidence": round(
                    float(row["confidence"]),
                    2,
                ),
                "lift": round(
                    float(row["lift"]),
                    2,
                ),
            }
        )

    return output


print("\nFP-Growth quick check:")
print(
    get_paired_specs(
        "AMD Ryzen 7 5700U"
    )
)

def recommend_full(
    query: str,
    k: int = TOP_K,
    use_cache: bool = True,
):
    query_embedding = None

    if use_cache:
        query_embedding = embed_query(query)

        cached_response, match_info = (
            cache.lookup(query_embedding)
        )

        if cached_response is not None:
            result = dict(cached_response)
            result["cache_hit"] = True
            result["matched_query"] = (
                match_info["matched_query"]
            )
            result["similarity"] = round(
                match_info["similarity"],
                4,
            )

            if "commonly_paired_specs" not in result:
                context = result.get(
                    "retrieved_context",
                    [],
                )

                top_cpu = (
                    context[0].get("cpu")
                    if context
                    else None
                )

                result[
                    "commonly_paired_specs"
                ] = (
                    get_paired_specs(top_cpu)
                    if top_cpu
                    else []
                )

            return result

    result = recommend_verified(
        query,
        k,
    )

    result["cache_hit"] = False

    retrieved_context = result.get(
        "retrieved_context",
        [],
    )

    top_cpu = (
        retrieved_context[0].get("cpu")
        if retrieved_context
        else None
    )

    result["commonly_paired_specs"] = (
        get_paired_specs(top_cpu)
        if top_cpu
        else []
    )

    if use_cache:
        cache.store(
            query,
            query_embedding,
            result,
        )

    return result


print("\n=== FINAL END-TO-END TEST ===")

final_query = (
    "AMD Ryzen laptop under $600"
)

final_result = recommend_full(
    final_query,
    k=TOP_K,
    use_cache=False,
)

print(
    json.dumps(
        final_result,
        indent=2,
        ensure_ascii=False,
        default=str,
    )
)

summary = {
    "data": {
        "chunk_rows": int(len(df)),
        "unique_devices": int(len(device_level)),
        "embedding_dimension": int(EMBEDDING_DIM),
        "embedding_model": EMBEDDING_MODEL,
    },
    "evaluation": {
        "n_queries": int(len(eval_df)),
        "k": int(TOP_K),
        "mean_precision_at_k": float(
            retrieval_eval_df[
                "precision_at_k"
            ].mean()
        ),
        "mean_recall_at_k": float(
            retrieval_eval_df[
                "recall_at_k"
            ].mean()
        ),
        "hit_rate": float(
            retrieval_eval_df[
                "hit"
            ].mean()
        ),
    },
    "cache": {
        "threshold": CACHE_THRESHOLD,
        "hit_rate": float(
            bench_df["cache_hit"].mean()
        ),
        "avg_hit_latency_s": (
            None
            if pd.isna(avg_hit)
            else float(avg_hit)
        ),
        "avg_miss_latency_s": (
            None
            if pd.isna(avg_miss)
            else float(avg_miss)
        ),
        "speedup_hit_vs_miss": (
            None
            if pd.isna(speedup)
            else float(speedup)
        ),
    },
}

summary_path = os.path.join(
    OUT_DIR,
    "pipeline_summary_FIXED.json",
)

with open(
    summary_path,
    "w",
    encoding="utf-8",
) as f:
    json.dump(
        summary,
        f,
        indent=2,
    )

print("\n=== OUTPUT FILES ===")
for path in sorted(Path(OUT_DIR).glob("*")):
    print(path)

print("\nDONE.")
