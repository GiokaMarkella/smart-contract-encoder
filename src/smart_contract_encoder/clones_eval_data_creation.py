import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from sentence_transformers.evaluation import InformationRetrievalEvaluator

from smart_contract_encoder.encoder import load_encoder


DEFAULT_CLONES_CSV = Path(__file__).resolve().parent.parent.parent / "test_with_fc_transitive_clones.csv"


def _parse_clone_list(value) -> list[int]:
    if pd.isna(value):
        return []
    if isinstance(value, (int, float)):
        value = str(int(value))
    parts = [p.strip() for p in str(value).split(";")]
    clones = []
    for p in parts:
        if not p:
            continue
        try:
            clones.append(int(p))
        except ValueError:
            continue
    return clones


def _select_query_indices(df: pd.DataFrame, n_queries: int, unique_by_func_name: bool) -> list[int]:
    candidates = df[df["clone_count"] > 0].copy()
    candidates = candidates.sort_values("clone_count", ascending=False)
    selected = []
    seen_names = set()
    for idx, row in candidates.iterrows():
        if len(selected) >= n_queries:
            break
        if unique_by_func_name:
            func_name = row.get("func_name")
            if pd.isna(func_name):
                func_name = f"__missing_name__{idx}"
            if func_name in seen_names:
                continue
            seen_names.add(func_name)
        selected.append(idx)
    return selected


def create_clone_query_dataset(
    encoder: str,
    encoder_version: str,
    field: str,
    model_to_load: str | None = None,
    clones_csv: str | os.PathLike = DEFAULT_CLONES_CSV,
    n_queries: int = 100,
    unique_by_func_name: bool = True,
):
    df = pd.read_csv(clones_csv)
    df = df.reset_index(drop=True)
    df["clones_list"] = df["clones"].apply(_parse_clone_list)
    df["clone_count"] = df["clones_list"].apply(len)

    query_indices = _select_query_indices(df, n_queries=n_queries, unique_by_func_name=unique_by_func_name)
    if not query_indices:
        raise ValueError("No queries could be selected from clones CSV")

    corpus_indices = [i for i in df.index if i not in set(query_indices)]

    corpus = {f"d{idx}": df.loc[idx, field] for idx in corpus_indices}
    queries = {f"q{idx}": df.loc[idx, field] for idx in query_indices}

    relevant_docs = {}
    dropped_queries = []
    corpus_index_set = set(corpus_indices)
    for q_idx in query_indices:
        clone_ids = set(df.loc[q_idx, "clones_list"])
        clone_ids.discard(q_idx)
        clone_ids = clone_ids.intersection(corpus_index_set)
        if not clone_ids:
            dropped_queries.append(q_idx)
            continue
        relevant_docs[f"q{q_idx}"] = {f"d{c_idx}" for c_idx in clone_ids}

    if dropped_queries:
        for q_idx in dropped_queries:
            queries.pop(f"q{q_idx}", None)
        if not queries:
            raise ValueError("All queries were dropped because no relevant clone docs were in the corpus")

    k_values = list(range(5, 110, 10)) + [1, 10, 20]
    ir_evaluator = InformationRetrievalEvaluator(
        queries=queries,
        corpus=corpus,
        relevant_docs=relevant_docs,
        show_progress_bar=True,
        mrr_at_k=k_values,
        ndcg_at_k=k_values,
        accuracy_at_k=k_values,
        precision_recall_at_k=k_values,
        map_at_k=k_values,
        corpus_chunk_size=len(df),
    )

    model_field2 = load_encoder(encoder, encoder_version, model_to_load)
    if encoder == "smartembed":
        model_field2.dataset = df

    eval_result = ir_evaluator(model_field2.model)

    metrics = ["mrr", "ndcg", "accuracy", "precision", "recall", "map"]
    results = {metric: {} for metric in metrics}
    for k in k_values:
        k_str = str(k)
        for metric in metrics:
            results[metric][k_str] = eval_result[f"cosine_{metric}@{k}"]

    output_dir = "./results"
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, f"{encoder}_{encoder_version}_{field}_clones_results.json")
    with open(output_file, "w") as file:
        json.dump(
            {
                "meta": {
                    "n_queries": len(queries),
                    "requested_queries": n_queries,
                    "unique_by_func_name": unique_by_func_name,
                    "clones_csv": str(clones_csv),
                },
                "results": results,
            },
            file,
            indent=4,
        )
    print(f"Results saved successfully to {output_file}")
