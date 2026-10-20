#!/usr/bin/env python
"""Repeated validation/test holdout evaluation for the LLM ensemble study.

This script never calls the nine base models.  It reads their saved per-question
predictions, repeatedly selects ensemble members on a validation split, then
evaluates the fixed ensembles on the matched held-out test split.

Run ``python repeated_holdout_evaluation.py --help`` for options.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = ROOT / "RepeatedHoldoutResults"
SEEDS = (2026, 2027, 2028, 2029, 2030)
K_VALUES = (3, 5)

MODEL_FILES = {
    "gemma3:12b": "result_gemma3_12b.xlsx",
    "deepseek-r1:8b": "result_deepseek-r1_8b.xlsx",
    "qwen3:8b": "result_qwen3_8b.xlsx",
    "llama3.1:8b": "result_llama3.1_8b.xlsx",
    "mistral:7b": "result_mistral_7b.xlsx",
    "huihui_ai/internlm3-abliterated": "result_huihui_ai_internlm3_abliterated.xlsx",
    "ingu627/phi-4-mini-instruct-q4": "result_ingu627_phi-4-mini-instruct-q4.xlsx",
    "glm4:9b": "result_glm4_9b.xlsx",
    "myaniu/minicpm4-8b:Q4_K_M": "result_myaniu_minicpm4-8b_Q4_K_M.xlsx",
}


@dataclass(frozen=True)
class DatasetConfig:
    name: str
    data_file: Path
    result_dir: Path


DATASETS = (
    DatasetConfig("Tech Arena", ROOT / "科技大擂台" / "data" / "all_fix.xlsx", ROOT / "科技大擂台" / "code" / "Results" / "2"),
    DatasetConfig("C3-D", ROOT / "C3" / "data" / "c3_d_test.xlsx", ROOT / "C3" / "Results" / "d1"),
    DatasetConfig("C3-M", ROOT / "C3" / "data" / "c3_m_test.xlsx", ROOT / "C3" / "Results" / "m1"),
    DatasetConfig("RACE-M", ROOT / "RACE" / "test" / "RACE_Middle_Combined.xlsx", ROOT / "RACE" / "Results" / "m"),
    DatasetConfig("RACE-H", ROOT / "RACE" / "test" / "RACE_High_Combined.xlsx", ROOT / "RACE" / "Results" / "h"),
)


def to_answer(value: Any) -> int | None:
    if value is None or pd.isna(value):
        return None
    try:
        value = int(value)
    except (TypeError, ValueError):
        return None
    return value if value in (1, 2, 3, 4) else None


def to_confidence(value: Any) -> float:
    try:
        value = float(value)
    except (TypeError, ValueError):
        return 0.0
    return 0.0 if math.isnan(value) else min(1.0, max(0.0, value))


def normalise_id(value: Any) -> str:
    """Make Excel IDs comparable without changing their displayed source value."""
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value).strip()


def load_dataset(config: DatasetConfig) -> tuple[pd.DataFrame, list[str]]:
    if not config.data_file.exists():
        raise FileNotFoundError(f"Missing dataset file: {config.data_file}")

    base = pd.read_excel(config.data_file).copy()
    required = {"ID", "Article", "Question", "Option1", "Option2", "Option3", "Option4", "Answer"}
    missing = required.difference(base.columns)
    if missing:
        raise ValueError(f"{config.name}: dataset is missing columns: {sorted(missing)}")
    base["_key"] = base["ID"].map(normalise_id)
    if base["_key"].duplicated().any():
        raise ValueError(f"{config.name}: duplicate IDs in dataset")

    models: list[str] = []
    for model, filename in MODEL_FILES.items():
        candidate_paths = [config.result_dir / filename]
        if model == "huihui_ai/internlm3-abliterated":
            # Historical folders use both spellings. One C3 file has empty
            # predictions, so choose the existing candidate with usable output.
            candidate_paths.append(config.result_dir / "result_huihui_ai_internlm3-abliterated.xlsx")
        existing_paths = [path for path in candidate_paths if path.exists()]
        if not existing_paths:
            raise FileNotFoundError(f"{config.name}: missing saved predictions for {model}: {candidate_paths[0]}")
        candidates = [(path, pd.read_excel(path)) for path in existing_paths]
        path, pred = max(candidates, key=lambda item: item[1].get("Predicted", pd.Series(dtype=float)).notna().sum())
        required_pred = {"ID", "Predicted", "Confidence"}
        missing_pred = required_pred.difference(pred.columns)
        if missing_pred:
            raise ValueError(f"{config.name}, {model}: result file is missing {sorted(missing_pred)}")
        pred["_key"] = pred["ID"].map(normalise_id)
        if pred["_key"].duplicated().any():
            raise ValueError(f"{config.name}, {model}: duplicate prediction IDs")

        selected = pred[["_key", "Predicted", "Confidence"]].rename(
            columns={"Predicted": f"pred__{model}", "Confidence": f"conf__{model}"}
        )
        base = base.merge(selected, on="_key", how="left", validate="one_to_one")
        if base[f"pred__{model}"].isna().all():
            raise ValueError(f"{config.name}, {model}: no prediction IDs matched the dataset")
        models.append(model)

    base["actual"] = base["Answer"].map(to_answer)
    if base["actual"].isna().any():
        raise ValueError(f"{config.name}: contains invalid gold answers")
    return base, models


def accuracy(frame: pd.DataFrame, model: str) -> float:
    preds = frame[f"pred__{model}"].map(to_answer)
    return float((preds == frame["actual"]).mean())


def der(frame: pd.DataFrame, first: str, second: str) -> float:
    first_wrong = frame[f"pred__{first}"].map(to_answer) != frame["actual"]
    second_wrong = frame[f"pred__{second}"].map(to_answer) != frame["actual"]
    return float((first_wrong & second_wrong).mean())


def select_top_accuracy(validation: pd.DataFrame, models: list[str], k: int) -> list[str]:
    ranking = sorted(models, key=lambda model: (-accuracy(validation, model), model))
    return ranking[:k]


def select_des(validation: pd.DataFrame, models: list[str], k: int, seed_rule: str) -> list[str]:
    """Select a DES ensemble using validation data only.

    ``highest_accuracy`` matches the current manuscript: the best validation
    accuracy model is the first member, then the lowest mean DER candidate is
    added iteratively. ``minimum_pair`` reproduces the older notebook rule.
    """
    if k > len(models):
        raise ValueError(f"k={k} is greater than the available model count")
    ranking = sorted(models, key=lambda model: (-accuracy(validation, model), model))

    if seed_rule == "highest_accuracy":
        selected = [ranking[0]]
    elif seed_rule == "minimum_pair":
        pairs = [(der(validation, a, b), a, b) for i, a in enumerate(ranking) for b in ranking[i + 1 :]]
        _, first, second = min(pairs, key=lambda item: (item[0], item[1], item[2]))
        selected = [first, second]
    else:
        raise ValueError(f"Unknown DES seed rule: {seed_rule}")

    while len(selected) < k:
        remaining = [model for model in ranking if model not in selected]
        next_model = min(
            remaining,
            key=lambda candidate: (np.mean([der(validation, candidate, chosen) for chosen in selected]), candidate),
        )
        selected.append(next_model)
    return selected[:k]


def majority_vote(predictions: list[int | None]) -> int | None:
    counts: dict[int, int] = {}
    for prediction in predictions:
        if prediction is not None:
            counts[prediction] = counts.get(prediction, 0) + 1
    if not counts:
        return None
    maximum = max(counts.values())
    winners = [answer for answer, count in counts.items() if count == maximum]
    return winners[0] if len(winners) == 1 else None


def weighted_vote(predictions: list[int | None], confidences: list[float]) -> int | None:
    scores = {answer: 0.0 for answer in (1, 2, 3, 4)}
    for prediction, confidence in zip(predictions, confidences):
        if prediction is not None:
            scores[prediction] += confidence
    maximum = max(scores.values())
    if maximum == 0:
        return None
    winners = [answer for answer, score in scores.items() if abs(score - maximum) < 1e-12]
    return winners[0] if len(winners) == 1 else None


def build_prompt(row: pd.Series) -> str:
    valid_options = {
        index: row[f"Option{index}"]
        for index in (1, 2, 3, 4)
        if pd.notna(row[f"Option{index}"]) and str(row[f"Option{index}"]).strip()
    }
    options_text = "\n".join(f"{index}. {option}" for index, option in valid_options.items())
    valid_numbers = ", ".join(map(str, valid_options))
    return f'''Read the following article and answer the multiple-choice question.

Article:
{row["Article"]}

Question:
{row["Question"]}

Options:
{options_text}

Instruction:
1) Select the best option among ({valid_numbers}). If an option is missing/empty, do not select it.
2) Output ONLY a JSON object with:
   - "answer": an integer in [{valid_numbers}]
   - "confidence": a number in [0, 1] representing your probability the answer is correct.

Example:
{{"answer": 2, "confidence": 0.78}}'''


def parse_thinking_answer(text: str) -> int | None:
    if not text:
        return None
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if match:
        try:
            answer = to_answer(json.loads(match.group(0)).get("answer"))
            if answer is not None:
                return answer
        except (ValueError, TypeError, json.JSONDecodeError):
            pass
    labelled = re.search(r"Answer:\s*([1-4])", text, flags=re.IGNORECASE)
    if labelled:
        return int(labelled.group(1))
    numbers = re.findall(r"\b([1-4])\b", text)
    return int(numbers[-1]) if numbers else None


def cache_key(row: pd.Series, model: str, context_length: int) -> str:
    payload = json.dumps(
        {"id": normalise_id(row["ID"]), "model": model, "context_length": context_length, "prompt": build_prompt(row)},
        ensure_ascii=False,
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


class ThinkingResolver:
    def __init__(self, enabled: bool, model: str, context_length: int, cache_file: Path):
        self.enabled = enabled
        self.model = model
        self.context_length = context_length
        self.cache_file = cache_file
        self.cache: dict[str, dict[str, Any]] = {}
        if cache_file.exists():
            self.cache = json.loads(cache_file.read_text(encoding="utf-8"))

    def save(self) -> None:
        self.cache_file.parent.mkdir(parents=True, exist_ok=True)
        self.cache_file.write_text(json.dumps(self.cache, ensure_ascii=False, indent=2), encoding="utf-8")

    def resolve(self, row: pd.Series) -> tuple[int | None, bool, bool]:
        """Return answer, fallback_used, and whether a model invocation occurred."""
        key = cache_key(row, self.model, self.context_length)
        if key in self.cache:
            return self.cache[key].get("answer"), bool(self.cache[key].get("fallback")), False
        if not self.enabled:
            return None, True, False
        try:
            from ollama import chat
        except ImportError as exc:
            raise RuntimeError("CCR needs the ollama Python package. Activate the environment used for the original notebooks.") from exc
        answer = None
        for _ in range(3):
            response = chat(
                model=self.model,
                messages=[{"role": "user", "content": build_prompt(row)}],
                options={"num_ctx": self.context_length, "temperature": 0.0},
                think=True,
            )
            answer = parse_thinking_answer(response.message.content or "")
            if answer is not None:
                break
        fallback = answer is None
        self.cache[key] = {"answer": answer, "fallback": fallback}
        self.save()
        return answer, fallback, True


def evaluate_method(
    test: pd.DataFrame,
    members: list[str],
    vote: str,
    use_ccr: bool,
    fallback_model: str,
    resolver: ThinkingResolver,
) -> tuple[float, int, int]:
    correct = 0
    invocations = 0
    fallbacks = 0
    for _, row in test.iterrows():
        predictions = [to_answer(row[f"pred__{model}"]) for model in members]
        confidences = [to_confidence(row[f"conf__{model}"]) for model in members]
        answer = majority_vote(predictions) if vote == "MV" else weighted_vote(predictions, confidences)
        if answer is None:
            if use_ccr:
                answer, fallback_used, invoked = resolver.resolve(row)
                # Report logical CCR triggers, not cache misses. The cache only
                # avoids repeating identical Ollama work across repeated splits;
                # a tie still represents one Thinking Model invocation in the
                # experimental policy for this method and split.
                invocations += 1
                fallbacks += int(fallback_used)
            if answer is None:
                answer = to_answer(row[f"pred__{fallback_model}"])
        correct += int(answer == row["actual"])
    return correct / len(test), invocations, fallbacks


def make_split(frame: pd.DataFrame, seed: int, validation_fraction: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    indices = rng.permutation(len(frame))
    validation_size = int(round(len(frame) * validation_fraction))
    validation_indices = indices[:validation_size]
    test_indices = indices[validation_size:]
    return frame.iloc[validation_indices].copy(), frame.iloc[test_indices].copy()


def evaluate_dataset(config: DatasetConfig, args: argparse.Namespace) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    frame, models = load_dataset(config)
    rows: list[dict[str, Any]] = []
    selections: list[dict[str, Any]] = []
    cache_file = OUTPUT_DIR / "thinking_cache" / f"{config.name.replace(' ', '_')}.json"
    resolver = ThinkingResolver(args.enable_ccr, args.thinking_model, args.context_length, cache_file)

    for seed in args.seeds:
        validation, test = make_split(frame, seed, args.validation_fraction)
        split_dir = OUTPUT_DIR / "splits" / config.name.replace(" ", "_")
        split_dir.mkdir(parents=True, exist_ok=True)
        pd.DataFrame({"ID": validation["ID"], "split": "validation", "seed": seed}).to_csv(split_dir / f"seed_{seed}_validation.csv", index=False, encoding="utf-8-sig")
        pd.DataFrame({"ID": test["ID"], "split": "test", "seed": seed}).to_csv(split_dir / f"seed_{seed}_test.csv", index=False, encoding="utf-8-sig")

        validation_ranking = sorted(((model, accuracy(validation, model)) for model in models), key=lambda item: (-item[1], item[0]))
        for rank, (model, value) in enumerate(validation_ranking, start=1):
            selections.append({"dataset": config.name, "seed": seed, "selection_type": "validation_accuracy_ranking", "rank": rank, "model": model, "validation_accuracy": value})

        # CCRE always uses validation-selected top-2 models, independent of k.
        ccre_members = select_top_accuracy(validation, models, 2)
        for position, model in enumerate(ccre_members, start=1):
            selections.append({"dataset": config.name, "seed": seed, "selection_type": "CCRE_top2", "k": 2, "position": position, "model": model})

        for vote in ("MV", "CWV"):
            fallback_model = max(ccre_members, key=lambda model: (accuracy(validation, model), model))
            value, calls, fallbacks = evaluate_method(test, ccre_members, vote, True, fallback_model, resolver)
            rows.append({"dataset": config.name, "seed": seed, "k": 2, "method": f"CCRE-{vote}", "accuracy": value, "test_questions": len(test), "thinking_invocations": calls, "fallbacks": fallbacks, "members": " | ".join(ccre_members), "fallback_model": fallback_model})

        for k in args.k_values:
            top_members = select_top_accuracy(validation, models, k)
            des_members = select_des(validation, models, k, args.des_seed_rule)
            selection_sets = {"top_accuracy": top_members, "DES": des_members}
            for selection_type, members in selection_sets.items():
                for position, model in enumerate(members, start=1):
                    selections.append({"dataset": config.name, "seed": seed, "selection_type": selection_type, "k": k, "position": position, "model": model})

            for method, members in (("MV", top_members), ("CWV", top_members), ("MV+DES", des_members), ("CWV+DES", des_members)):
                vote = "MV" if method.startswith("MV") else "CWV"
                for with_ccr in (False, True):
                    final_method = f"{method}+CCR" if with_ccr else method
                    fallback_model = max(members, key=lambda model: (accuracy(validation, model), model))
                    value, calls, fallbacks = evaluate_method(test, members, vote, with_ccr, fallback_model, resolver)
                    rows.append({"dataset": config.name, "seed": seed, "k": k, "method": final_method, "accuracy": value, "test_questions": len(test), "thinking_invocations": calls, "fallbacks": fallbacks, "members": " | ".join(members), "fallback_model": fallback_model})

    resolver.save()
    return rows, selections


def summarise(rows: pd.DataFrame) -> pd.DataFrame:
    summary = rows.groupby(["dataset", "k", "method"], as_index=False).agg(
        mean_accuracy=("accuracy", "mean"),
        sd_accuracy=("accuracy", "std"),
        mean_invocations=("thinking_invocations", "mean"),
        sd_invocations=("thinking_invocations", "std"),
        mean_fallbacks=("fallbacks", "mean"),
        repetitions=("seed", "nunique"),
    )
    summary["accuracy_mean_sd"] = summary.apply(lambda row: f"{row['mean_accuracy'] * 100:.2f} ± {row['sd_accuracy'] * 100:.2f}", axis=1)
    return summary.sort_values(["k", "method", "dataset"]).reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Repeated 20/80 validation/test holdout evaluation using saved base-model predictions.")
    parser.add_argument("--datasets", nargs="+", default=[config.name for config in DATASETS], choices=[config.name for config in DATASETS], help="Datasets to evaluate (default: all five).")
    parser.add_argument("--seeds", nargs="+", type=int, default=list(SEEDS), help="Five or more random seeds for repeated holdouts.")
    parser.add_argument("--validation-fraction", type=float, default=0.20, help="Validation fraction; default: 0.20.")
    parser.add_argument("--k-values", nargs="+", type=int, default=list(K_VALUES), help="Ensemble sizes; default: 3 5.")
    parser.add_argument("--des-seed-rule", choices=("highest_accuracy", "minimum_pair"), default="highest_accuracy", help="DES initialisation rule; default follows the current manuscript.")
    parser.add_argument("--enable-ccr", action="store_true", help="Call Ollama only for CCR tie cases. Without it, CCR rows use the fallback model and are marked incomplete.")
    parser.add_argument("--thinking-model", default="qwen3:8b", help="Thinking Model used by CCR.")
    parser.add_argument("--context-length", type=int, default=3192, help="Thinking Model context length.")
    args = parser.parse_args()
    if not 0 < args.validation_fraction < 1:
        parser.error("--validation-fraction must be between 0 and 1")
    if len(args.seeds) < 5:
        parser.error("Use at least five seeds, as required by the revised experiment design")
    args.k_values = tuple(sorted(set(args.k_values)))

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    all_rows: list[dict[str, Any]] = []
    all_selections: list[dict[str, Any]] = []
    for config in DATASETS:
        if config.name in args.datasets:
            print(f"\n=== {config.name} ===")
            rows, selections = evaluate_dataset(config, args)
            all_rows.extend(rows)
            all_selections.extend(selections)

    results = pd.DataFrame(all_rows)
    selections = pd.DataFrame(all_selections)
    summary = summarise(results)
    results.to_csv(OUTPUT_DIR / "per_repeat_test_results.csv", index=False, encoding="utf-8-sig")
    selections.to_csv(OUTPUT_DIR / "validation_selections.csv", index=False, encoding="utf-8-sig")
    summary.to_csv(OUTPUT_DIR / "summary_mean_sd.csv", index=False, encoding="utf-8-sig")
    with pd.ExcelWriter(OUTPUT_DIR / "repeated_holdout_results.xlsx", engine="openpyxl") as writer:
        summary.to_excel(writer, sheet_name="mean_sd_summary", index=False)
        results.to_excel(writer, sheet_name="per_repeat_test_results", index=False)
        selections.to_excel(writer, sheet_name="validation_selections", index=False)
    print(f"\nSaved results to: {OUTPUT_DIR}")
    print("Base-model prediction files were read only; no base models were run.")
    if not args.enable_ccr:
        print("WARNING: CCR rows are fallback-only because --enable-ccr was not supplied. Do not use them in the paper.")


if __name__ == "__main__":
    main()
