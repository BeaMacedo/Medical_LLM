import argparse
import os
import json
import torch
import pandas as pd
import random
import numpy as np
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def main():
    set_seed(42)

    parser = argparse.ArgumentParser(description="Run Open-Ended Evaluation")
    parser.add_argument("--model_name", type=str)
    parser.add_argument("--model_path", type=str, required=True, help="Path to base model")
    parser.add_argument("--prev_adapters_paths", type=str, nargs='*', default=[], help="List of previous adapters to merge in order")
    parser.add_argument("--peft_model_path", type=str, required=True, help="Path to the LoRA adapter")
    parser.add_argument("--dataset_name", type=str, required=True, help="Name of the dataset")
    parser.add_argument("--dataset_path", type=str, required=True, help="Path to the test dataset")
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--is_instruct", action="store_true", help="Use chat template")
    parser.add_argument("--max_length", type=int, default=8000)
    parser.add_argument("--max_new_tokens", type=int, default=1500)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Loading Base Model: {args.model_path}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.add_bos_token = False

    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        attn_implementation="eager" 
    )

    if args.prev_adapters_paths:
        print(f"\n[INFO] Merging {len(args.prev_adapters_paths)} previous adapter(s) in strict order:")
        for i, prev_path in enumerate(args.prev_adapters_paths):
            print(f"   {i+1}. {prev_path}")
            model = PeftModel.from_pretrained(model, prev_path)
            model = model.merge_and_unload()
        print("All previous knowledge merged successfully!")

    print(f"Injecting LoRA Adapter from: {args.peft_model_path}...")
    model = PeftModel.from_pretrained(model, args.peft_model_path)
    model.eval()

    data = []
    with open(args.dataset_path, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip(): continue
            data.append(json.loads(line))

    results = []

    print(f"Running inference on {len(data)} samples")
    
    for i, item in enumerate(tqdm(data)):
        system_prompt = item.get("instruction", "").strip()
        user_query = item.get("input", "").strip()
        expected_answer = item.get("output", item.get("gpt-4-answer", ""))

        # Prompt Construction
        if args.is_instruct:
            try:
                # 1st Try: Use role "system"
                messages = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_query}
                ]
                prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True) #tokenize=False so we have the actual prompt that was sent
                if i == 0:
                    print(f"\n[TEMPLATE INFO] Model {args.model_name} accepted 'system role'.\n")
            except Exception:
                # 2nd Try: If the model rejects "system"
                messages = [
                    {"role": "user", "content": f"{system_prompt}\n\n{user_query}"}
                ]
                prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                if i == 0:
                    print(f"\n[TEMPLATE INFO] Model {args.model_name} rejected 'system role'. Falling back to 'user role'.\n")
        else:
            #For models without template
            prompt = f"{system_prompt}\n\n{user_query}\n\nAnswer:\n"

        inputs = tokenizer(
            prompt, 
            return_tensors="pt",
            truncation=True, 
            max_length=args.max_length,
            add_special_tokens=True 
        ).to(model.device)
        
        input_length = inputs['input_ids'].shape[1]

        if input_length >= args.max_length:
            print(f"\n[WARNING] Prompt for question {i+1} was TRUNCATED! (Length: {input_length})")

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                do_sample=False, #Greedy Decoding
                temperature=None,
                top_p=None,
                pad_token_id=tokenizer.pad_token_id
            )
        # Extract only the generated text
        generated_tokens = outputs[0][input_length:]
        generated_text = tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()

        results.append({
            "dataset": args.dataset_name,
            "model_name": args.model_name,
            "prompt_used": prompt,
            "expected_reference": expected_answer,
            "model_generation": generated_text
        })

    output_file = os.path.join(args.output_dir, f"{args.model_name}_{args.dataset_name}_GoldenSet_OpenQA.csv")
    df_results = pd.DataFrame(results)
    df_results.to_csv(output_file, index=False, encoding='utf-8-sig')
    print(f"\nEvaluation saved to {output_file}")

if __name__ == "__main__":
    main()