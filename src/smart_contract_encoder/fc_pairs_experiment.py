from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from smart_contract_encoder.encoder import load_encoder
from smart_contract_encoder.load_data import DATA_DIR


LINE_COMMENT_RE = re.compile(r"//.*?$", flags=re.MULTILINE)
BLOCK_COMMENT_RE = re.compile(r"/\*.*?\*/", flags=re.DOTALL)
PRAGMA_IMPORT_RE = re.compile(r"^\s*(pragma|import)\b.*?;\s*$", flags=re.MULTILINE)
CONTRACT_DECL_RE = re.compile(r"\b(contract|library|interface|abstract\s+contract)\b")
FUNCTION_START_RE = re.compile(r"\b(function|constructor|fallback|receive)\b")
WHITESPACE_RE = re.compile(r"\s+")


@dataclass
class Metrics:
    threshold: float
    accuracy: float
    precision: float
    recall: float
    f1: float
    tp: int
    fp: int
    tn: int
    fn: int


def _strip_comments(text: str) -> str:
    text = BLOCK_COMMENT_RE.sub(" ", text)
    text = LINE_COMMENT_RE.sub("", text)
    return text


def _collapse_whitespace(text: str) -> str:
    return WHITESPACE_RE.sub(" ", text).strip()


def _extract_braced_block(text: str, start_idx: int) -> str | None:
    brace_idx = text.find("{", start_idx)
    semicolon_idx = text.find(";", start_idx)
    if semicolon_idx != -1 and (brace_idx == -1 or semicolon_idx < brace_idx):
        return text[start_idx : semicolon_idx + 1]
    if brace_idx == -1:
        return None
    depth = 0
    for idx in range(brace_idx, len(text)):
        char = text[idx]
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return text[start_idx : idx + 1]
    return None


def _extract_first_function(text: str) -> str:
    match = FUNCTION_START_RE.search(text)
    if not match:
        return text.strip()
    block = _extract_braced_block(text, match.start())
    return (block or text[match.start() :]).strip()


def normalize_solidity_function(text: str) -> str:
    if not isinstance(text, str):
        return ""
    text = _strip_comments(text)
    text = PRAGMA_IMPORT_RE.sub("", text)
    if CONTRACT_DECL_RE.search(text):
        text = _extract_first_function(text)
    else:
        text = text.strip()
    return _collapse_whitespace(text)


def _run_7z_extract(archive_path: Path, output_dir: Path) -> Path:
    seven_zip = shutil.which("7z") or shutil.which("7zz")
    if not seven_zip:
        raise RuntimeError(
            "FC-pair archive extraction requires `7z`/`7zz` on PATH or a pre-extracted dataset directory."
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [seven_zip, "x", str(archive_path), f"-o{output_dir}", "-y"],
        check=True,
    )
    return output_dir


def resolve_fc_pair_root(fc_pair_input: Path, extracted_dir: Path | None = None) -> Path:
    if fc_pair_input.is_dir():
        return fc_pair_input
    if fc_pair_input.suffix.lower() != ".7z":
        raise ValueError(f"Expected an extracted FC-pair directory or .7z archive, got: {fc_pair_input}")
    target_dir = extracted_dir or fc_pair_input.with_suffix("")
    return _run_7z_extract(fc_pair_input, target_dir)


def _infer_source_id(path: Path) -> str:
    return path.stem if path.suffix else path.name


def _iter_source_files(data_dir: Path) -> Iterable[Path]:
    for path in sorted(data_dir.rglob("*")):
        if path.is_file():
            yield path


def extract_fc_pair_functions(fc_pair_root: Path) -> pd.DataFrame:
    rows: list[dict] = []
    for split in ("train", "test"):
        split_dir = fc_pair_root / f"{split}_data"
        if not split_dir.exists():
            continue
        for path in _iter_source_files(split_dir):
            source = path.read_text(encoding="utf-8", errors="ignore")
            function_id = _infer_source_id(path)
            rows.append(
                {
                    "split": split,
                    "function_id": str(function_id),
                    "source_path": str(path.relative_to(fc_pair_root)),
                    "raw_source": source,
                    "normalized_source": normalize_solidity_function(source),
                }
            )
    if not rows:
        raise ValueError(f"No source files found under {fc_pair_root}")
    df = pd.DataFrame(rows).drop_duplicates(subset=["function_id"], keep="first")
    return df.sort_values(["split", "function_id"]).reset_index(drop=True)


def _find_pair_columns(df: pd.DataFrame, allow_positional_fallback: bool = True) -> tuple[str, str, str]:
    lowered = {col.lower(): col for col in df.columns}
    first = lowered.get("fid1") or lowered.get("id1") or lowered.get("function_id_1") or lowered.get("func1")
    second = lowered.get("fid2") or lowered.get("id2") or lowered.get("function_id_2") or lowered.get("func2")
    label = lowered.get("label") or lowered.get("type") or lowered.get("clone_label")
    if first and second and label:
        return first, second, label
    if not allow_positional_fallback:
        raise ValueError("Could not infer FC-pair label columns from CSV headers")
    if len(df.columns) < 3:
        raise ValueError("FC-pair labels CSV must contain at least three columns")
    return df.columns[0], df.columns[1], df.columns[2]


def _read_fc_pair_labels_csv(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    try:
        _find_pair_columns(df, allow_positional_fallback=False)
        return df
    except ValueError:
        df = pd.read_csv(csv_path, header=None)
        if df.shape[1] < 3:
            raise ValueError(f"{csv_path} must contain at least three columns")
        df = df.iloc[:, :3].copy()
        df.columns = ["function_id_1", "function_id_2", "clone_label"]
        if not df.empty and not str(df.iloc[0]["clone_label"]).strip().isdigit():
            df = df.iloc[1:].reset_index(drop=True)
        return df


def load_fc_pair_labels(fc_pair_root: Path) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    for split in ("train", "test"):
        csv_path = fc_pair_root / f"{split}.csv"
        if not csv_path.exists():
            continue
        df = _read_fc_pair_labels_csv(csv_path)
        col1, col2, label_col = _find_pair_columns(df)
        part = df[[col1, col2, label_col]].copy()
        part.columns = ["function_id_1", "function_id_2", "clone_label"]
        part["split"] = split
        part = part.dropna(how="all", subset=["function_id_1", "function_id_2", "clone_label"]).reset_index(drop=True)
        part["function_id_1"] = part["function_id_1"].astype(str)
        part["function_id_2"] = part["function_id_2"].astype(str)
        numeric_labels = pd.to_numeric(part["clone_label"], errors="coerce")
        invalid_mask = numeric_labels.isna()
        if invalid_mask.any():
            invalid_rows = (
                part.loc[invalid_mask, ["function_id_1", "function_id_2", "clone_label"]]
                .head(5)
                .to_dict(orient="records")
            )
            raise ValueError(
                f"{csv_path} contains non-numeric or missing clone labels; "
                f"examples: {invalid_rows}"
            )
        part["clone_label"] = numeric_labels.astype(int)
        rows.append(part)
    if not rows:
        raise ValueError(f"No train.csv/test.csv found under {fc_pair_root}")
    return pd.concat(rows, ignore_index=True)


def load_merged_dataset(merged_path: Path | None = None) -> pd.DataFrame:
    merged_path = merged_path or DATA_DIR / "test_merged.pkl"
    df = pd.read_pickle(merged_path)
    required = {"func_code", "code"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Merged dataset is missing required columns: {sorted(missing)}")
    df = df.copy().reset_index(drop=True)
    df["merged_row_id"] = df.index
    df["normalized_func_code"] = df["func_code"].fillna("").map(normalize_solidity_function)
    return df


def build_match_table(fc_functions: pd.DataFrame, merged_df: pd.DataFrame) -> pd.DataFrame:
    merged_cols = ["merged_row_id", "func_code", "code", "normalized_func_code"]
    merged_lookup = merged_df[merged_cols].rename(
        columns={
            "func_code": "merged_func_code",
            "code": "decompiled_code",
            "normalized_func_code": "normalized_source",
        }
    )
    match_counts = (
        merged_lookup.groupby("normalized_source", dropna=False)
        .size()
        .rename("merged_match_count")
        .reset_index()
    )
    merged_lookup = (
        merged_lookup.sort_values("merged_row_id")
        .drop_duplicates(subset=["normalized_source"], keep="first")
        .reset_index(drop=True)
    )
    matched = fc_functions.merge(match_counts, how="left", on="normalized_source")
    matched = matched.merge(merged_lookup, how="left", on="normalized_source")
    matched["merged_match_count"] = matched["merged_match_count"].fillna(0).astype(int)
    matched["match_found"] = matched["merged_row_id"].notna()
    return matched


def build_decompiled_pairs_dataset(labels_df: pd.DataFrame, matched_functions: pd.DataFrame) -> pd.DataFrame:
    function_cols = [
        "function_id",
        "raw_source",
        "normalized_source",
        "merged_row_id",
        "merged_func_code",
        "decompiled_code",
        "match_found",
    ]
    lookup = matched_functions[function_cols]

    left = lookup.rename(
        columns={
            "function_id": "function_id_1",
            "raw_source": "raw_source_1",
            "normalized_source": "normalized_source_1",
            "merged_row_id": "merged_row_id_1",
            "merged_func_code": "matched_func_code_1",
            "decompiled_code": "decompiled_function1",
            "match_found": "match_found_1",
        }
    )
    right = lookup.rename(
        columns={
            "function_id": "function_id_2",
            "raw_source": "raw_source_2",
            "normalized_source": "normalized_source_2",
            "merged_row_id": "merged_row_id_2",
            "merged_func_code": "matched_func_code_2",
            "decompiled_code": "decompiled_function2",
            "match_found": "match_found_2",
        }
    )

    pairs = labels_df.merge(left, how="left", on="function_id_1")
    pairs = pairs.merge(right, how="left", on="function_id_2")
    pairs["both_matched"] = pairs["match_found_1"].fillna(False) & pairs["match_found_2"].fillna(False)
    pairs = pairs[pairs["both_matched"]].copy()
    pairs.reset_index(drop=True, inplace=True)
    return pairs


def _cosine_scores(emb1: np.ndarray, emb2: np.ndarray) -> np.ndarray:
    emb1 = np.asarray(emb1, dtype=float)
    emb2 = np.asarray(emb2, dtype=float)
    num = np.sum(emb1 * emb2, axis=1)
    den = np.linalg.norm(emb1, axis=1) * np.linalg.norm(emb2, axis=1)
    den = np.where(den == 0, 1e-12, den)
    return num / den


def _safe_div(num: float, den: float) -> float:
    return float(num / den) if den else 0.0


def evaluate_thresholds(
    pairs_df: pd.DataFrame,
    encoder: str,
    encoder_version: str,
    model_to_load: str | None = None,
    thresholds: Iterable[float] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if pairs_df.empty:
        raise ValueError("Matched pairs dataset is empty")
    model = load_encoder(encoder, encoder_version, model_to_load)
    if encoder == "smartembed":
        raise ValueError(
            "The SmartEmbed backend relies on the repo's func_code-indexed embedding cache and cannot score arbitrary decompiled code pairs."
        )
    emb1 = np.asarray(model.encode(pairs_df["decompiled_function1"].tolist()))
    emb2 = np.asarray(model.encode(pairs_df["decompiled_function2"].tolist()))
    scores = _cosine_scores(emb1, emb2)
    eval_df = pairs_df.copy()
    eval_df["cosine_similarity"] = scores

    if thresholds is None:
        thresholds = np.round(np.arange(0.50, 1.001, 0.05), 2)

    y_true = eval_df["clone_label"].astype(int).to_numpy()
    metrics_rows: list[Metrics] = []
    for threshold in thresholds:
        y_pred = (scores >= threshold).astype(int)
        tp = int(np.sum((y_true == 1) & (y_pred == 1)))
        fp = int(np.sum((y_true == 0) & (y_pred == 1)))
        tn = int(np.sum((y_true == 0) & (y_pred == 0)))
        fn = int(np.sum((y_true == 1) & (y_pred == 0)))
        precision = _safe_div(tp, tp + fp)
        recall = _safe_div(tp, tp + fn)
        metrics_rows.append(
            Metrics(
                threshold=float(threshold),
                accuracy=_safe_div(tp + tn, len(y_true)),
                precision=precision,
                recall=recall,
                f1=_safe_div(2 * precision * recall, precision + recall),
                tp=tp,
                fp=fp,
                tn=tn,
                fn=fn,
            )
        )

    metrics_df = pd.DataFrame([m.__dict__ for m in metrics_rows])
    return eval_df, metrics_df


def parse_thresholds(value: str | None) -> list[float] | None:
    if not value:
        return None
    return [float(item.strip()) for item in value.split(",") if item.strip()]


def print_summary(
    fc_functions: pd.DataFrame,
    labels_df: pd.DataFrame,
    matched_functions: pd.DataFrame,
    pairs_df: pd.DataFrame,
    metrics_df: pd.DataFrame | None,
) -> None:
    total_functions = len(fc_functions)
    matched_count = int(matched_functions["match_found"].sum())
    print(f"FC-pair functions parsed: {total_functions}")
    print(f"FC-pair functions matched to test_merged.pkl: {matched_count}")
    print(f"FC-pair labeled pairs: {len(labels_df)}")
    print(f"Pairs with both functions matched: {len(pairs_df)}")
    if metrics_df is not None and not metrics_df.empty:
        print("\nThreshold sweep:")
        print(metrics_df.to_string(index=False))


def run_pipeline(args: argparse.Namespace) -> None:
    fc_pair_root = resolve_fc_pair_root(args.fc_pair_input, args.extracted_dir)
    fc_functions = extract_fc_pair_functions(fc_pair_root)
    labels_df = load_fc_pair_labels(fc_pair_root)
    merged_df = load_merged_dataset(args.merged_path)
    matched_functions = build_match_table(fc_functions, merged_df)
    pairs_df = build_decompiled_pairs_dataset(labels_df, matched_functions)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    fc_functions.to_csv(args.output_dir / "fc_pair_functions.csv", index=False)
    matched_functions.to_csv(args.output_dir / "fc_pair_function_matches.csv", index=False)
    pairs_df.to_csv(args.output_dir / "fc_pair_decompiled_pairs.csv", index=False)

    metrics_df = None
    eval_pairs_df = None
    if args.evaluate:
        eval_pairs_df, metrics_df = evaluate_thresholds(
            pairs_df=pairs_df,
            encoder=args.encoder,
            encoder_version=args.encoder_version,
            model_to_load=args.model_to_load,
            thresholds=parse_thresholds(args.thresholds),
        )
        eval_pairs_df.to_csv(args.output_dir / "fc_pair_decompiled_pairs_scored.csv", index=False)
        metrics_df.to_csv(args.output_dir / "fc_pair_threshold_metrics.csv", index=False)
        with open(args.output_dir / "fc_pair_threshold_metrics.json", "w", encoding="utf-8") as handle:
            json.dump(metrics_df.to_dict(orient="records"), handle, indent=2)

    print_summary(fc_functions, labels_df, matched_functions, pairs_df, metrics_df)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Extract FC-pair functions, match them against test_merged.pkl by normalized source, "
            "build a decompiled clone dataset, and optionally evaluate similarity thresholds."
        )
    )
    parser.add_argument(
        "fc_pair_input",
        type=Path,
        help="Path to an extracted FC-pair directory or the FC-pair.7z archive.",
    )
    parser.add_argument(
        "--merged-path",
        type=Path,
        default=DATA_DIR / "test_merged.pkl",
        help="Path to test_merged.pkl. Defaults to ./data/test_merged.pkl.",
    )
    parser.add_argument(
        "--extracted-dir",
        type=Path,
        default=None,
        help="Extraction target when fc_pair_input points to FC-pair.7z.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DATA_DIR / "fc_pairs_experiment",
        help="Directory where intermediate and final CSV/JSON outputs will be written.",
    )
    parser.add_argument(
        "--evaluate",
        action="store_true",
        help="Also encode the matched decompiled pairs and sweep cosine-similarity thresholds.",
    )
    parser.add_argument(
        "--encoder",
        default="sentence_encoder",
        help="Encoder backend to use for evaluation. Examples: sentence_encoder, codebert, coderankeembed.",
    )
    parser.add_argument(
        "--encoder-version",
        default="finetuned",
        help="Encoder version to pass into the repo loader.",
    )
    parser.add_argument(
        "--model-to-load",
        default=None,
        help="Optional finetuned model path/name for encoder backends that support it.",
    )
    parser.add_argument(
        "--thresholds",
        default=None,
        help="Comma-separated cosine similarity thresholds. Default: 0.50..1.00 in 0.05 steps.",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    run_pipeline(args)


if __name__ == "__main__":
    main()
