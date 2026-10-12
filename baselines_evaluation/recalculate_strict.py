import pandas as pd
import re
import os
import glob
import argparse

def extract_strict_answer(model_output, answer_type="letters"):
    text = str(model_output).strip()
    text_lower = text.lower()

    if answer_type == "letters":
        match = re.search(r'(?:answer\s*is\s*)[\*\'\"]*([a-d])\b[\*\'\"]*', text_lower)
        if match: return match.group(1).upper(), "Regex_Conversational"

        match = re.search(r'the\s+correct\s+answer\s+is\s*:\s*([a-d])\s*$', text_lower)
        if match: return match.group(1).upper(), "Regex_The_Correct_Answer_Is_End"

        match = re.search(r'answer\s+is\s+option\s+([a-d])\b', text_lower)
        if match: return match.group(1).upper(), "Regex_Answer_Is_Option"

    return "UNKNOWN", "No_Match"

def process_single_csv(filepath):
    print(f"\n--- Processing: {os.path.basename(filepath)} ---")

    try:
        df = pd.read_csv(filepath)
    except Exception as e:
        print(f"[ERROR] Could not read {filepath}: {e}")
        return

    if 'salvaged_by_rule' not in df.columns:
        df['salvaged_by_rule'] = "Not Salvaged"

    # Auto-detect the answer type from the filename
    filepath_lower = filepath.lower()
    answer_type = "yes_no_maybe" if "pubmedqa" in filepath_lower else "letters"

    total_questions = len(df)
    salvaged_count = 0

    for index, row in df.iterrows():
        old_extracted = str(row['extracted_answer']).strip()

        # Apply the regex ONLY to rows that failed (UNKNOWN) 
        if old_extracted == "UNKNOWN":
            raw_answer = str(row['model_raw_answer'])
            expected = str(row['expected_answer']).strip()
            if answer_type == "yes_no_maybe":
                expected = expected.lower()

            new_extracted, rule_used = extract_strict_answer(raw_answer, answer_type)

            # If the answer was salvaged
            if new_extracted != "UNKNOWN":
                salvaged_count += 1
                df.at[index, 'extracted_answer'] = new_extracted
                df.at[index, 'salvaged_by_rule'] = rule_used

                is_correct = (new_extracted == expected)
                df.at[index, 'is_correct'] = int(is_correct) #finds row [index], goes to the 'is_correct' column and overwrites it with 1

    # Compute the new metrics
    final_correct = df['is_correct'].sum()
    final_unknowns = (df['extracted_answer'] == 'UNKNOWN').sum()

    new_accuracy = (final_correct / total_questions) * 100
    new_unknown_rate = (final_unknowns / total_questions) * 100

    print(f"[SALVAGE] Salvaged {salvaged_count} answers previously marked as UNKNOWN!")
    print(f"[METRICS] New Accuracy: {new_accuracy:.2f}% | New Unknown Rate: {new_unknown_rate:.2f}%")

    # Save the updated file
    new_filepath = filepath.replace(".csv", "_RECALCULATED.csv")
    df.to_csv(new_filepath, index=False)
    print(f"[SUCCESS] Saved to: {os.path.basename(new_filepath)}")


def main():
    parser = argparse.ArgumentParser(description="Salvage correct answers for the Baselines")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--csv_path", type=str, help="Path to a single results CSV")
    group.add_argument("--folder_path", type=str, help="Path to a folder containing multiple CSVs")

    args = parser.parse_args()

    print("Starting new extraction pass...")

    if args.csv_path:
        if os.path.exists(args.csv_path):
            process_single_csv(args.csv_path)
        else:
            print(f"[ERROR] File not found: {args.csv_path}")

    elif args.folder_path:
        if os.path.isdir(args.folder_path):
            csv_files = glob.glob(os.path.join(args.folder_path, "*.csv"))
            # Skip files that have already been recalculated
            csv_files = [f for f in csv_files if "_RECALCULATED" not in f]

            if not csv_files:
                print(f"[WARNING] No original CSV files found in folder: {args.folder_path}")
            else:
                for filepath in sorted(csv_files):
                    process_single_csv(filepath)
        else:
            print(f"[ERROR] Folder not found: {args.folder_path}")

if __name__ == "__main__":
    main()
