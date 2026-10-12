import pandas as pd
import re
import os
import glob
import argparse

def extract_intelligent_answer(model_output, answer_type="letters"):
    text = str(model_output).strip()
    
    # 1. DeepSeek-r1
    if "</think>" in text:
        text = text.split("</think>", 1)[-1].strip()
    elif "<think>" in text and "</think>" not in text:
        return "UNKNOWN", "Unfinished_Think"
    
    text_lower = text.lower()
    
    if answer_type == "letters":
        match = re.search(r'^\s*(?:#+\s*)?\**\(?([a-d])\b\)?\**(?:\.|,|\)|:|\s*-|\s*\n|\s*$)', text_lower)
        if match: return match.group(1).upper(), "Regex_Basic_2"

        match = re.search(r'(?:#+\s*)?(?:correct\s+)?answer:\s*\**\(?([a-d])\b', text_lower)
        if match: return match.group(1).upper(), "Regex_Basic_1"

        match = re.search(r'(?:#+\s*)?letter\s*:\s*([a-d])\b', text_lower)
        if match: return match.group(1).upper(), "Regex_Letter_Header" 
        
        match = re.search(r'(?:answer\s*is\s*)[\*\'\"]*([a-d])\b[\*\'\"]*', text_lower)
        if match: return match.group(1).upper(), "Regex_Conversational"

        match = re.search(r'(?:option\s+)?([a-d])\b\s+is\s+(?:the\s+)?correct', text_lower)
        if match: return match.group(1).upper(), "Regex_Is_Correct"

        match = re.search(r'^\s*(?:#+\s*)?option\s+([a-d])\b', text_lower)
        if match: return match.group(1).upper(), "Regex_Starts_With_Option"

        match = re.search(r'^\s*([A-D])\s+[A-Z]', text)
        if match: return match.group(1).upper(), "Regex_Starts_With_Letter_Space"

        match = re.search(r'###answer:\s*option\s+([a-d])\s*(?:[\.:]|\n|$)', text_lower)
        if match: return match.group(1).upper(), "Regex_Exact_Hash_Option"

        match = re.search(r'the\s+correct\s+answer\s+is\s*:\s*([a-d])\s*$', text_lower)
        if match: return match.group(1).upper(), "Regex_The_Correct_Answer_Is_End"

        match = re.search(r'\*\*\s*answer\s*\**\s*:\s*\**\s*([a-d])\b', text_lower)
        if match: return match.group(1).upper(), "Regex_Markdown_Bold_Answer"

        match = re.search(r'answer\s+is\s+option\s+([a-d])\b', text_lower)
        if match: return match.group(1).upper(), "Regex_Answer_Is_Option"
            
    elif answer_type == "yes_no_maybe":
        match = re.search(r'(?:#+\s*)?(?:correct\s+)?answer:\s*\**(yes|no|maybe)\b', text_lower)
        if match: return match.group(1).lower(), "Regex_Basic_YN_1"

        match = re.search(r'^\s*(?:#+\s*)?\**(yes|no|maybe)\b\**(?:\.|,|!|\s*\n|\s*$)', text_lower)
        
        if match: return match.group(1).lower(), "Regex_Basic_YN_2"

        if "statement is correct" in text_lower: 
            return "yes", "Regex_Statement_Correct"
        if "statement is incorrect" in text_lower:
            return "no", "Regex_Statement_Incorrect"
        if "statement is not correct" in text_lower:
            return "no", "Regex_Statement_Not_Correct"
            
        match = re.search(r'(?:answer\s*is\s*:?\s*)[\*\'\"]*(yes|no|maybe)\b', text_lower) 
        if match: return match.group(1).lower(), "Regex_Conversational_YN"

        match = re.search(r'^\s*[\*\'\"]*(yes|no|maybe)\b', text_lower)
        if match: return match.group(1).lower(), "Regex_Starts_With_Word"
        
    return "UNKNOWN", "No_Match"

def process_single_csv(filepath):
    print(f"\n--- Extracting answers for: {os.path.basename(filepath)} ---")
    
    try:
        df = pd.read_csv(filepath)
    except Exception as e:
        print(f"[ERROR] Could not read {filepath}: {e}")
        return
    
    filepath_lower = filepath.lower()
    answer_type = "yes_no_maybe" if "pubmedqa" in filepath_lower else "letters"
    
    extracted_answers = []
    extraction_rules = []
    is_correct_list = []
    
    for index, row in df.iterrows():
        raw_answer = str(row['model_raw_answer'])
        expected = str(row['expected_answer']).strip()
        
        if answer_type == "yes_no_maybe":
            expected = expected.lower()
            
        extracted, rule_used = extract_intelligent_answer(raw_answer, answer_type)
        is_correct = 1 if (extracted == expected) else 0
        
        extracted_answers.append(extracted)
        extraction_rules.append(rule_used)
        is_correct_list.append(is_correct)

    df['extracted_answer'] = extracted_answers
    df['extraction_rule'] = extraction_rules
    df['is_correct'] = is_correct_list

    
    print("\n[EXTRACTION SUMMARY]")

    # Count how many times each regex was used
    rule_counts = df['extraction_rule'].value_counts()
    print("Rules applied:")
    for rule, count in rule_counts.items():
        print(f"  -> {rule}: {count}")
        
    # Count the UNKNOWNs
    unknowns = (df['extracted_answer'] == 'UNKNOWN').sum()
    print(f"\nTotal UNKNOWNs: {unknowns} out of {len(df)} questions.")
    print("-" * 50)
   
    new_filepath = filepath.replace(".csv", "_RECALCULATED.csv")
    df.to_csv(new_filepath, index=False)
    print(f"[SUCCESS] Extraction complete! Saved to: {os.path.basename(new_filepath)}")

def main():
    parser = argparse.ArgumentParser(description="Answer Extraction Engine (Regex Parser)")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--csv_path", type=str, help="Path to a single RAW CSV file")
    group.add_argument("--folder_path", type=str, help="Path to a folder containing RAW CSV files")
    
    args = parser.parse_args()

    if args.csv_path:
        process_single_csv(args.csv_path)

    elif args.folder_path:
        csv_files = glob.glob(os.path.join(args.folder_path, "*.csv"))
        csv_files = [f for f in csv_files if "_RECALCULATED" not in f]
        
        if not csv_files:
            print(f"[WARNING] No raw CSV files found in directory.")
        else:
            for filepath in sorted(csv_files):
                process_single_csv(filepath)

if __name__ == "__main__":
    main()