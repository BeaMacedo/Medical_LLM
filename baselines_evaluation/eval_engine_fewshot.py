import json
import torch
import argparse
import os
import pandas as pd
import re
import random
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def load_jsonl(filepath):
    data = []
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"[ERROR] File not found: {filepath}")
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    return data

def build_fewshot_prompt_base(instruction, few_shot_data, input_text): #Prompt Construction
    prompt = f"{instruction}\n\n"
    
    # Append the few-shot examples
    for shot in few_shot_data:
        prompt += f"{shot.get('input', '')}\n\nAnswer: {shot.get('output', '')}\n\n"
        
    # Append the final target question
    prompt += f"{input_text}\n\nAnswer:"
    return prompt

def extract_intelligent_answer(model_output, answer_type="letters"):
    text = model_output.strip()

    # DeepSeek-R1: Remove reasoning block
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
            
    elif answer_type == "yes_no_maybe":
        match = re.search(r'(?:#+\s*)?(?:correct\s+)?answer:\s*\**(yes|no|maybe)\b', text_lower)
        if match: return match.group(1).lower(), "Regex_Basic_YN_1"

        match = re.search(r'^\s*(?:#+\s*)?\**(yes|no|maybe)\b\**(?:\.|,|!|\s*\n|\s*$)', text_lower)
        if match: return match.group(1).lower(), "Regex_Basic_YN_2"
        
    return "UNKNOWN", "No_Match"

def main():
    set_seed(42)

    parser = argparse.ArgumentParser(description="Few-Shot LLM Evaluation Engine")
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--dataset_name", type=str, required=True)    
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--dataset_path", type=str, required=True, help="Path to the test file")
    parser.add_argument("--few_shot_path", type=str, required=True, help="Path to the 5-shot examples file")
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--answer_type", type=str, default="letters", choices=["letters", "yes_no_maybe"])
    parser.add_argument("--max_length", type=int, default=4096, help="Max token length for truncation")
    parser.add_argument("--max_new_tokens", type=int, default=40, help="Max tokens to generate")
    parser.add_argument("--is_instruct", action="store_true", help="Enable Chat Template for Instruct models")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    csv_filename = os.path.join(args.output_dir, f"fewshot_results_{args.model_name}_{args.dataset_name}.csv")
    if os.path.exists(csv_filename):
        print(f"\n[SKIP] The file {csv_filename} already exists! Skipping this dataset.")
        return

    print(f"--- Starting FEW-SHOT evaluation: {args.model_name} on {args.dataset_name} ({args.answer_type}) ---")

    test_data = load_jsonl(args.dataset_path)
    few_shot_data = load_jsonl(args.few_shot_path)
    print(f"[INFO] Loaded {len(few_shot_data)} Few-Shot examples.")

    # Load Tokenizer and Model
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    if tokenizer.pad_token is None: #it assigns the End-of-Sequence token as the pad token
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        attn_implementation="eager",
        device_map="auto"
    )
    model.eval()

    detailed_results = []
    correct_count = 0
    unknown_count = 0  
    total_questions = len(test_data)

    print(f"Evaluating {total_questions} questions...")

    with torch.no_grad(): #optimization that tells PyTorch not to compute gradients, it's only necessary during training/fine tuning models not evaluation
        for i, item in enumerate(test_data):
            instruction = item.get('instruction', '')
            input_text = item.get('input', item.get('question', '')) 
            topic_label = item.get('topic', 'General')
            expected_answer = str(item.get('output', '')).strip()
            if args.answer_type == "yes_no_maybe":
                expected_answer = expected_answer.lower()

            # Prompt Construction
            if args.is_instruct:
                is_reasoning_model = "deepseek" in args.model_name.lower() or "deepseek" in args.model_path.lower()
                
                if is_reasoning_model: #in the model card they tell to avoid using system prompt
                    messages = []
                    for j, shot in enumerate(few_shot_data):
                        if j == 0:
                            messages.append({"role": "user", "content": f"{instruction}\n\n{shot.get('input', '')}"})
                        else:
                            messages.append({"role": "user", "content": shot.get("input", "")})
                        messages.append({"role": "assistant", "content": str(shot.get("output", ""))})
                    messages.append({"role": "user", "content": input_text})
                    
                    final_prompt = tokenizer.apply_chat_template(
                        messages, tokenize=False, add_generation_prompt=True
                    )
                    if i == 0: print(f"\n[TEMPLATE INFO] DeepSeek detected. Forcing 'user role' only.\n")
                else:
                    try:
                        # 1º Try: Use role "system" 
                        messages = [{"role": "system", "content": instruction}]
                        for shot in few_shot_data:
                            messages.append({"role": "user", "content": shot.get("input", "")})
                            messages.append({"role": "assistant", "content": str(shot.get("output", ""))})
                        
                        messages.append({"role": "user", "content": input_text})
                        
                        final_prompt = tokenizer.apply_chat_template(
                            messages, tokenize=False, add_generation_prompt=True
                        )
                        if i == 0: print(f"\n[TEMPLATE INFO] Model accepted 'system role' with few-shot history.\n")
                        
                    except Exception:
                        # if the model reject
                        messages = []
                        for j, shot in enumerate(few_shot_data):
                            if j == 0:
                                messages.append({"role": "user", "content": f"{instruction}\n\n{shot.get('input', '')}"})
                            else:
                                messages.append({"role": "user", "content": shot.get("input", "")})
                            messages.append({"role": "assistant", "content": str(shot.get("output", ""))})
                        
                        messages.append({"role": "user", "content": input_text})
                        
                        final_prompt = tokenizer.apply_chat_template(
                            messages, tokenize=False, add_generation_prompt=True
                        )
                        if i == 0: print(f"\n[TEMPLATE INFO] Model rejected 'system role'. Used 'user role' for instruction.\n")
            else:
                #For models without template
                final_prompt = build_fewshot_prompt_base(instruction, few_shot_data, input_text)
            
            # Tokenization and Generation
            inputs = tokenizer(
                final_prompt, 
                return_tensors="pt",
                truncation=True,
                max_length=args.max_length
            ).to(model.device)

            input_length = inputs['input_ids'].shape[1]
            if input_length >= args.max_length:
                print(f"\n[WARNING] Prompt for question {i+1} was TRUNCATED! (Length: {input_length} / Max: {args.max_length})")
                print(f"[WARNING] The model might not have seen the full question.\n")

            outputs = model.generate(
                **inputs, 
                max_new_tokens=args.max_new_tokens, 
                pad_token_id=tokenizer.eos_token_id,
                do_sample=False  #Greedy Decoding
            )
            
            generated_tokens = outputs[0][input_length:]
            raw_model_answer = tokenizer.decode(generated_tokens, skip_special_tokens=True).strip() #numbers to letters, removes special tokens

            # Extraction and Scoring
            extracted_answer, rule_name = extract_intelligent_answer(raw_model_answer, answer_type=args.answer_type)

            is_correct = (extracted_answer == expected_answer)
            if is_correct: correct_count += 1
            if extracted_answer == "UNKNOWN": unknown_count += 1

            detailed_results.append({
                "question_id": i + 1,
                "topic": topic_label,
                "prompt": final_prompt if i == 0 else "<Prompt omitted to save space. See line 1.>",
                "expected_answer": expected_answer,
                "model_raw_answer": raw_model_answer,
                "extracted_answer": extracted_answer,
                "extraction_rule": rule_name,
                "is_correct": int(is_correct)
            })

            if (i + 1) % 50 == 0:
                print(f"Progress: {i + 1}/{total_questions} processed...")

    # Metrics
    accuracy = correct_count / total_questions
    unknown_rate = unknown_count / total_questions
    
    print(f"\n--- FINAL RESULTS ---")
    print(f"Accuracy: {correct_count}/{total_questions} ({accuracy * 100:.2f}%)")
    print(f"Instruction Failures (UNKNOWN): {unknown_count}/{total_questions} ({unknown_rate * 100:.2f}%)")

    df = pd.DataFrame(detailed_results)
    df.to_csv(csv_filename, index=False)
    print(f"Results saved to: {csv_filename}")

if __name__ == "__main__":
    main()