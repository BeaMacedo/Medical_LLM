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
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line))
    return data

def build_prompt(instruction, input_text): #Prompt Construction
    prompt = f"{instruction}\n\n{input_text}\n\nAnswer:"
    return prompt

def main():
    set_seed(42)

    parser = argparse.ArgumentParser(description="LLM Evaluation Engine with LoRA")
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--model_path", type=str, required=True, help="Path to base model")
    parser.add_argument("--peft_model_path", type=str, required=True, help="Path to the LoRA adapter")
    parser.add_argument("--prev_adapters_paths", type=str, nargs='*', default=[], help="List of previous adapters to merge in order")
    parser.add_argument("--dataset_name", type=str, required=True)    
    parser.add_argument("--dataset_path", type=str, required=True, help="Path to the test file")
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--answer_type", type=str, default="letters", choices=["letters", "yes_no_maybe"])
    parser.add_argument("--max_length", type=int, default=4096, help="Max token length for truncation")
    parser.add_argument("--max_new_tokens", type=int, default=40, help="Max tokens to generate")
    parser.add_argument("--is_instruct", action="store_true", help="Enable the Chat Template for Instruct models")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    csv_filename = os.path.join(args.output_dir, f"results_{args.model_name}_{args.dataset_name}.csv")
    if os.path.exists(csv_filename):
        print(f"\n[SKIP] The file {csv_filename} already exists! Skipping this dataset.")
        return

    print(f"--- Starting GENERATION: {args.model_name} on dataset {args.dataset_name} ---")

    test_data = load_jsonl(args.dataset_path)

    tokenizer = AutoTokenizer.from_pretrained(args.model_path) 
    if tokenizer.pad_token is None: 
        tokenizer.pad_token = tokenizer.eos_token

    #Load the Model
    print("Loading Base Model...")
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

    #Inject current LoRA
    print(f"Injecting LoRA Adapter from {args.peft_model_path}...")
    model = PeftModel.from_pretrained(model, args.peft_model_path)
    model.eval()

    detailed_results = []
    total_questions = len(test_data)

    print(f"Generating answers for {total_questions} questions...")

    with torch.no_grad(): 
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
                
                if is_reasoning_model:  #in the model card they tell to avoid using system prompt
                    messages = [{"role": "user", "content": f"{instruction}\n\n{input_text}"}]
                    final_prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                    if i == 0: print(f"\n[TEMPLATE INFO] DeepSeek detected. Forcing 'user role' only.\n")
                else:
                    try:
                        messages = [
                            {"role": "system", "content": instruction},
                            {"role": "user", "content": input_text}
                        ]
                        final_prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                        if i == 0: print(f"\n[TEMPLATE INFO] Model {args.model_name} accepted 'system role'.\n")
                    except Exception:
                        messages = [{"role": "user", "content": f"{instruction}\n\n{input_text}"}]
                        final_prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                        if i == 0: print(f"\n[TEMPLATE INFO] Model {args.model_name} rejected 'system role' ('user role').\n")
            else:
                final_prompt = build_prompt(instruction, input_text)
            
            inputs = tokenizer(
                final_prompt, 
                return_tensors="pt",
                truncation=True,
                max_length=args.max_length
            ).to(model.device)

            input_length = inputs['input_ids'].shape[1] 
            if input_length >= args.max_length:
                print(f"\n[WARNING] Prompt TRUNCATED! (Length: {input_length} / Max: {args.max_length})")

            outputs = model.generate(
                **inputs, 
                max_new_tokens=args.max_new_tokens, 
                pad_token_id=tokenizer.eos_token_id,
                do_sample=False #Greedy Decoding
            )
            
            generated_tokens = outputs[0][input_length:] 
            raw_model_answer = tokenizer.decode(generated_tokens, skip_special_tokens=True).strip() 

            detailed_results.append({
                "question_id": i + 1,
                "topic": topic_label,
                "prompt": final_prompt, 
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