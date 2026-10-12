import argparse
import glob
import json
import os

import pandas as pd

from criteria_pairwise import CRITERIA_PAIRWISE
from judge_pairwise_parsing import parse_judge_output

RAW_SUFFIX = "_judge_pairwise_raw"
EVAL_SUFFIX = "_judge_pairwise_evaluation"

_SWAPPED_VERDICT = {"response_a": "response_b", "response_b": "response_a"}


def _unswap_position(parsed: dict, position_swapped: bool) -> dict:
    if not position_swapped:
        return parsed
    for key, _, _ in CRITERIA_PAIRWISE:
        verdict = parsed.get(key)
        if verdict in _SWAPPED_VERDICT:
            parsed[key] = _SWAPPED_VERDICT[verdict]
    return parsed


def load_results(path: str) -> list:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def raw_path_to_eval_stem(raw_path: str, output_dir: str = None) -> str:
    directory, filename = os.path.split(raw_path)
    stem, _ = os.path.splitext(filename)
    if stem.endswith(RAW_SUFFIX):
        stem = stem[: -len(RAW_SUFFIX)] + EVAL_SUFFIX
    else:
        stem = f"{stem}{EVAL_SUFFIX}"
    return os.path.join(output_dir or directory, stem)


def extract_file(raw_path: str, output_dir: str = None):
    rows = load_results(raw_path)

    extracted = []
    invalid_counts = {}
    for row in rows: #for each question
        raw_output = row.get("judge_raw_output", "")
        parsed = parse_judge_output(raw_output) #returns the verdit, if the verdit is valid and the reason given by the judge
        parsed = _unswap_position(parsed, bool(row.get("position_swapped")))

        result = {k: v for k, v in row.items() if k != "judge_raw_output"} #copy all the original fields from the question (question_id, dataset, model_a, model_b, question, response_a, response_b, prompt_used, ...) except for judge_raw_output
        result.update(parsed) #add the extracted keys (verdict/valid/reason by criterion).
        result["judge_raw_output"] = raw_output 
        extracted.append(result)  

        for key, is_valid in parsed.items(): #check whether there were any verdicts extracted not in a valid from
            if key.endswith("_valid") and not is_valid:
                invalid_counts[key] = invalid_counts.get(key, 0) + 1

    out_dir = output_dir or os.path.dirname(raw_path)
    os.makedirs(out_dir, exist_ok=True)
    out_stem = raw_path_to_eval_stem(raw_path, output_dir)

    df = pd.DataFrame(extracted)
    df.to_json(f"{out_stem}.json", orient="records", indent=2, force_ascii=False)

    print(f"Extracted {len(extracted)} rows from {raw_path} -> {out_stem}.json")
    if invalid_counts:
        print(f"  [WARNING] Unparsed/invalid verdicts: {invalid_counts}")


def main():
    parser = argparse.ArgumentParser(description="Extract per-criterion verdicts from raw pairwise judge responses")
    parser.add_argument("--raw_paths", type=str, nargs="+", default=None,
                         help="One or more *_judge_pairwise_raw.json files produced by judge_pairwise_generate.py")
    parser.add_argument("--raw_dir", type=str, default=None,
                         help="Recursively find and extract every *_judge_pairwise_raw.json file under this directory")
    parser.add_argument("--output_dir", type=str, default=None)
    args = parser.parse_args()

    if not args.raw_paths and not args.raw_dir:
        parser.error("Provide --raw_paths and/or --raw_dir")

    raw_paths = list(args.raw_paths or [])
    if args.raw_dir:
        raw_paths.extend(sorted(glob.glob(os.path.join(args.raw_dir, "**", f"*{RAW_SUFFIX}.json"), recursive=True)))

    if not raw_paths:
        raise SystemExit("No raw pairwise judge response files found.")

    for raw_path in raw_paths:
        extract_file(raw_path, args.output_dir)


if __name__ == "__main__":
    main()
