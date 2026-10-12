import pandas as pd
import argparse
import os
import glob
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

def calculate_and_print_metrics(df_subset, subset_name="Global", answer_type="letters"):
    total_questions = len(df_subset)
    if total_questions == 0: return

    correct_count = df_subset['is_correct'].sum()
    format_failure_count = (df_subset['extracted_answer'] == 'UNKNOWN').sum()

    acc = correct_count / total_questions
    format_failure_rate = format_failure_count / total_questions

    y_true = df_subset["expected_answer"].astype(str).tolist()
    y_pred = df_subset["extracted_answer"].astype(str).tolist()


    if answer_type == "yes_no_maybe":
        labels = ["yes", "no", "maybe"]
    else:
        labels = ["A", "B", "C", "D"]

    _, _, f1_macro, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, average='macro', zero_division=0
    )

    print(f"\n--- {subset_name} ---")
    print(f"Questions           : {total_questions}")
    print(f"Accuracy            : {acc * 100:.2f}%")
    print(f"Macro-F1            : {f1_macro * 100:.2f}%")
    print(f"Format Failure Rate : {format_failure_rate * 100:.2f}% (UNKNOWNs)")
    
    print("Model Prediction Distribution:")
    distribution = df_subset['extracted_answer'].value_counts(normalize=True) * 100
    for answer, percentage in distribution.items():
        print(f"  {answer}: {percentage:.2f}%")
    
    print("True Answer Distribution (Expected):")
    true_distribution = df_subset['expected_answer'].value_counts(normalize=True) * 100
    for answer, percentage in true_distribution.items():
        print(f"  {answer}: {percentage:.2f}%")

def process_single_csv(filepath):
    print("\n" + "="*70)
    print(f"CALCULATING METRICS FOR: {os.path.basename(filepath)}")
    print("="*70)

    try:
        df = pd.read_csv(filepath)
    except Exception as e:
        print(f"[ERROR] Could not read {filepath}: {e}")
        return

    filename_lower = os.path.basename(filepath).lower()
    ans_type = "yes_no_maybe" if "pubmedqa" in filename_lower else "letters"
    
    if "is_correct" not in df.columns:
        print(f"[ERROR] Missing 'is_correct' column!")
        return

    # Compute global metrics
    calculate_and_print_metrics(df, subset_name="GLOBAL METRICS", answer_type=ans_type)

    # Compute per-topic metrics (if the column exists)
    if "topic" in df.columns:
        print("-" * 50)
        print("METRICS BY TOPIC")
        print("-" * 50)
        for topic in df["topic"].unique():
            df_sub = df[df["topic"] == topic]
            calculate_and_print_metrics(df_sub, subset_name=f"Topic: {topic}", answer_type=ans_type)
    else:
        print("[WARNING] 'topic' column not found in this CSV. Skipping topic stratification.")

def main():
    parser = argparse.ArgumentParser(description="Calculate Metrics from CSVs")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--csv_path", type=str, help="Path to a single results CSV")
    group.add_argument("--folder_path", type=str, help="Path to a folder containing multiple CSVs")
    
    args = parser.parse_args()

    if args.folder_path:
        #Read the files that have already gone through extraction
        csv_files = glob.glob(os.path.join(args.folder_path, "*.csv"))
        if not csv_files:
            print(f"[WARNING] No csv files found in {args.folder_path}")
            return
        for filepath in sorted(csv_files):
            process_single_csv(filepath)

    elif args.csv_path:
        process_single_csv(args.csv_path)

if __name__ == "__main__":
    main()