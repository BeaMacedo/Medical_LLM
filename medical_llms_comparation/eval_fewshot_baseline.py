import json
import torch
import argparse
import os
import pandas as pd
import random
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

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

def build_fewshot_prompt_base(instruction, few_shot_data, input_text):
    prompt = f"{instruction}\n\n"
    for shot in few_shot_data:
        prompt += f"{shot.get('input', '')}\n\nAnswer: {shot.get('output', '')}\n\n"
    prompt += f"{input_text}\n\nAnswer:"
    return prompt

def main():
    set_seed(42)

    parser = argparse.ArgumentParser(description="Few-Shot LLM Evaluation Engine for baselines (Generation Only, LoRA optional)")
    parser.add_argument("--model_name",      type=str, required=True)
    parser.add_argument("--dataset_name",    type=str, required=True)
    parser.add_argument("--model_path",      type=str, required=True,  help="Path to base model")
    parser.add_argument("--peft_model_path", type=str, default=None,   help="Path to a LoRA adapter to inject on top of the base model. Omit for baselines (no adapter)")
    parser.add_argument("--prev_adapters_paths", type=str, nargs='*', default=[], help="List of previous adapters to merge in order")
    parser.add_argument("--dataset_path",    type=str, required=True,  help="Path to the test file")
    parser.add_argument("--few_shot_path",   type=str, required=True,  help="Path to the 5-shot examples file")
    parser.add_argument("--output_dir",      type=str, required=True)
    parser.add_argument("--answer_type",     type=str, default="letters", choices=["letters", "yes_no_maybe"])
    parser.add_argument("--max_length",      type=int, default=8000,   help="Max token length for truncation")
    parser.add_argument("--max_new_tokens",  type=int, default=40,     help="Max tokens to generate")
    parser.add_argument("--is_instruct",     action="store_true",      help="Enable Chat Template for Instruct models")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    csv_filename = os.path.join(args.output_dir, f"fewshot_results_{args.model_name}_{args.dataset_name}.csv")
    if os.path.exists(csv_filename):
        print(f"\n[SKIP] {csv_filename} already exists! Skipping.")
        return

    print(f"--- Starting FEW-SHOT GENERATION: {args.model_name} on {args.dataset_name} ({args.answer_type}) ---")

    test_data = load_jsonl(args.dataset_path)
    few_shot_data = load_jsonl(args.few_shot_path)
    total_questions = len(test_data)
    print(f"[INFO] Loaded {len(few_shot_data)} few-shot examples | {total_questions} test questions")

    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("Loading base model...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        attn_implementation="eager",  
        device_map="auto"
    )

    if args.prev_adapters_paths:
        print(f"\n[INFO] Merging {len(args.prev_adapters_paths)} previous adapter(s) in strict order:")
        for i, prev_path in enumerate(args.prev_adapters_paths):
            print(f"   {i+1}. {prev_path}")
            model = PeftModel.from_pretrained(model, prev_path)
            model = model.merge_and_unload()
        print("All previous knowledge merged successfully!")

    if args.peft_model_path:
        print(f"Injecting LoRA adapter from {args.peft_model_path}...")
        model = PeftModel.from_pretrained(model, args.peft_model_path)
    else:
        print("[INFO] No --peft_model_path given — evaluating the raw base/baseline model (no adapter).")

    model.eval()

    detailed_results = []

    print(f"Generating answers for {total_questions} questions...")

    with torch.no_grad():
        for i, item in enumerate(test_data):
            instruction   = item.get('instruction', '')
            input_text    = item.get('input', item.get('question', ''))
            topic_label   = item.get('topic', 'General')
            expected_answer = str(item.get('output', '')).strip()

            if args.answer_type == "yes_no_maybe":
                expected_answer = expected_answer.lower()

            if args.is_instruct:
                is_reasoning_model = "deepseek" in args.model_name.lower() or "deepseek" in args.model_path.lower()

                if is_reasoning_model:
                    messages = []
                    for j, shot in enumerate(few_shot_data):
                        if j == 0:
                            messages.append({"role": "user", "content": f"{instruction}\n\n{shot.get('input', '')}"})
                        else:
                            messages.append({"role": "user", "content": shot.get("input", "")})
                        messages.append({"role": "assistant", "content": str(shot.get("output", ""))})
                    messages.append({"role": "user", "content": input_text})

                    final_prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                    if i == 0:
                        print(f"\n[TEMPLATE INFO] DeepSeek detected — forcing user role only.\n")

                else:
                    try:
                        messages = [{"role": "system", "content": instruction}]
                        for shot in few_shot_data:
                            messages.append({"role": "user", "content": shot.get("input", "")})
                            messages.append({"role": "assistant", "content": str(shot.get("output", ""))})
                        messages.append({"role": "user", "content": input_text})
                        final_prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                        if i == 0:
                            print(f"\n[TEMPLATE INFO] Model accepted system role with few-shot history.\n")

                    except Exception:
                        messages = []
                        for j, shot in enumerate(few_shot_data):
                            if j == 0:
                                messages.append({"role": "user", "content": f"{instruction}\n\n{shot.get('input', '')}"})
                            else:
                                messages.append({"role": "user", "content": shot.get("input", "")})
                            messages.append({"role": "assistant", "content": str(shot.get("output", ""))})
                        messages.append({"role": "user", "content": input_text})
                        final_prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                        if i == 0:
                            print(f"\n[TEMPLATE INFO] Model rejected system role — using user role fallback.\n")
            else:
                final_prompt = build_fewshot_prompt_base(instruction, few_shot_data, input_text)

            inputs = tokenizer(
                final_prompt,
                return_tensors="pt",
                truncation=True,
                max_length=args.max_length
            ).to(model.device)

            input_length = inputs['input_ids'].shape[1]
            if input_length >= args.max_length:
                print(f"\n[WARNING] Prompt for question {i+1} TRUNCATED! ({input_length}/{args.max_length} tokens)")
                print(f"[WARNING] The model may not have seen the full question.\n")

            outputs = model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                pad_token_id=tokenizer.eos_token_id,
                do_sample=False  # greedy decoding 
            )

            generated_tokens = outputs[0][input_length:]
            raw_model_answer = tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()

            detailed_results.append({
                "question_id":    i + 1,
                "topic":          topic_label,
                "prompt":         final_prompt if i == 0 else "<omitted — see question_id 1>",
                "expected_answer": expected_answer,
                "model_raw_answer": raw_model_answer
            })

            if (i + 1) % 50 == 0:
                print(f"Progress: {i + 1}/{total_questions} processed...")

    df = pd.DataFrame(detailed_results)
    df.to_csv(csv_filename, index=False)
    print(f"\nGeneration complete! Raw results saved to: {csv_filename}")

if __name__ == "__main__":
    main()
