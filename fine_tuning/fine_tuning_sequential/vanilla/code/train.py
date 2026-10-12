import os
import json
import argparse
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, EarlyStoppingCallback
from peft import LoraConfig, get_peft_model, PeftModel
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM
from transformers import set_seed
from transformers.trainer_utils import get_last_checkpoint

PHASE_CONFIGS = {
    1: {"eval_steps": 200},
    2: {"eval_steps": 200},
    3: {"eval_steps": 10}, 
}

def detect_system_role_support(tokenizer): #Tries to build a conversation with a system message. If the tokenizer accepts it without error, the model supports a system role.

    if tokenizer.chat_template is None:
        return False
    try:
        messages = [{"role": "system", "content": "T"}, {"role": "user", "content": "T"}]
        _ = tokenizer.apply_chat_template(messages, tokenize=False)
        return True
    except Exception:
        return False

def detect_model_mode_and_template(tokenizer, supports_system_role): #Generates a test prompt and looks for the pattern marking the start of the assistant's response, to know where the response begins. This is so the loss is computed only over the response.
    if tokenizer.chat_template is None:
        return True, "\n\nAnswer:\n"

    if supports_system_role:
        test_messages = [
            {"role": "system", "content": "Sys"},
            {"role": "user", "content": "User"},
            {"role": "assistant", "content": "Asst"}
        ]
    else:
        test_messages = [
            {"role": "user", "content": "User"},
            {"role": "assistant", "content": "Asst"}
        ]

    test_prompt = tokenizer.apply_chat_template(test_messages, tokenize=False, add_generation_prompt=False) 

    patterns = [
        "\n<start_of_turn>model\n",                           # gemma-2-9b-it
        "<|im_start|>assistant\n"                             # qwen2.5-7b-instruct
    ]

    for pattern in patterns:
        if pattern in test_prompt:
            print(f"[AUTO-DETECT] Pattern explicitly found: {repr(pattern)}")
            return False, pattern

    raise ValueError(f"Could not detect response template. Test prompt:\n{test_prompt}")

def make_prompt_builder(tokenizer, is_base_model, supports_system_role):
    def build_universal_prompt(examples):
        output_texts = []
        # Extract the full lists from the batch
        instructions = examples.get("instruction", [])
        inputs = examples.get("input", [""] * len(instructions))
        outputs = examples.get("output", [""] * len(instructions))

        # Iterate over each entry in the batch, one at a time
        for i in range(len(instructions)):
            inst = (instructions[i] or "").strip()
            inp  = (inputs[i]  or "").strip()
            out  = (outputs[i] or "").strip()

            if inp == "<noinput>": inp = ""

            if is_base_model: #for models that don't have a chat template. 
                if inp:
                    prompt = f"{inst}\n\n{inp}\n\nAnswer:\n{out}{tokenizer.eos_token}"
                else:
                    prompt = f"{inst}\n\nAnswer:\n{out}{tokenizer.eos_token}"
                output_texts.append(prompt)

            else: #for instruct models
                if supports_system_role:
                    messages = [
                        {"role": "system", "content": inst},
                        {"role": "user", "content": inp},
                        {"role": "assistant", "content": out}
                    ] if inp else [
                        {"role": "user", "content": inst},
                        {"role": "assistant", "content": out}
                    ]
                else:
                    user_content = f"{inst}\n\n{inp}" if inp else inst
                    messages = [
                        {"role": "user",  "content": user_content},
                        {"role": "assistant", "content": out}
                    ]

                text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False) 
                output_texts.append(text)

        return output_texts
    return build_universal_prompt

def main():
    set_seed(42) 
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", type=int, required=True)
    parser.add_argument("--base_model_path", type=str, required=True)
    parser.add_argument("--train_dataset_path", type=str, required=True)
    parser.add_argument("--eval_dataset_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--prev_adapter_path", type=str, default=None,
                        help="Path to previous phase adapter (Phase 2: Phase 1 adapter / Phase 3: Phase 2 adapter)")
    parser.add_argument("--prev_adapter_path_phase1", type=str, default=None,
                        help="Path to Phase 1 adapter (only required for Phase 3, cumulative merge)")
    parser.add_argument("--search_lr",   type=float, required=True,
                        help="Learning rate")
    parser.add_argument("--search_rank", type=int,   required=True,
                        help="LoRA rank")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--grad_accum", type=int, default=4)
    parser.add_argument("--epochs",     type=int)
    args = parser.parse_args()

    local_rank = int(os.environ.get("LOCAL_RANK", 0))

    world_size = int(os.environ.get("WORLD_SIZE", 1)) #total number of GPUs

    if local_rank == 0:
        print(f"\n{'='*50}\nStarting Phase {args.phase} Fine-Tuning\n{'='*50}")

    if args.phase not in PHASE_CONFIGS:
        raise ValueError(f"Invalid phase {args.phase}. Valid phases: {list(PHASE_CONFIGS.keys())}")

    default_config = PHASE_CONFIGS[args.phase]
    current_lr     = args.search_lr
    current_rank   = args.search_rank
    current_epochs = args.epochs if args.epochs is not None else 5

    current_alpha = current_rank

    tokenizer = AutoTokenizer.from_pretrained(args.base_model_path)  
    tokenizer.padding_side = 'right' 
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.add_bos_token = False 

    if local_rank == 0:
        print("\n[AUTO-DETECTION] Running diagnostics...")

    supports_system = detect_system_role_support(tokenizer)
    is_base, response_template_str = detect_model_mode_and_template(tokenizer, supports_system)
    attn_impl = "eager" 

    if local_rank == 0:
        print(f"   -> Mode: {'BASE' if is_base else 'INSTRUCT'}")
        print(f"   -> System role support: {supports_system}")
        print(f"   -> Response template: {repr(response_template_str)}") 
        print(f"   -> Attention impl: {attn_impl}")
        print(f"\nActive Configuration:")
        print(f"   -> LR: {current_lr}")
        print(f"   -> Rank: {current_rank} | Alpha: {current_alpha}")
        print(f"   -> Epochs: {current_epochs}")
        #print(f"   -> Batch size: {args.batch_size} | Effective batch: {args.batch_size * args.grad_accum}")
        print(f"   -> Batch size: {args.batch_size} | GPUs: {world_size} | Effective batch: {args.batch_size * args.grad_accum * world_size}")

    build_prompt = make_prompt_builder(tokenizer, is_base, supports_system)

    is_distributed = world_size > 1

    safe_device_map = None if is_distributed else {"": torch.cuda.current_device()}

    if local_rank == 0:
        print(f"\nLoading base model...")

    model = AutoModelForCausalLM.from_pretrained(
        args.base_model_path,
        torch_dtype=torch.bfloat16,
        attn_implementation=attn_impl,
        device_map=safe_device_map 
    )

    if torch.cuda.is_available() and local_rank == 0:
        mem_alloc = torch.cuda.memory_allocated() / (1024 ** 3) # Convert bytes to GB
        mem_res   = torch.cuda.memory_reserved()   / (1024 ** 3)
        print(f"[VRAM] After loading base model: {mem_alloc:.2f} GB allocated, {mem_res:.2f} GB reserved")

    #Adapter merging
    if args.phase == 3:
        # Phase 3: cumulative merge (Phase 1 first, then Phase 2)
        if not args.prev_adapter_path:
            raise ValueError("Phase 3 requires --prev_adapter_path (Phase 2 adapter)")
        if not args.prev_adapter_path_phase1:
            raise ValueError("Phase 3 requires --prev_adapter_path_phase1 (Phase 1 adapter)")
        if not os.path.exists(args.prev_adapter_path_phase1):
            raise FileNotFoundError(f"Phase 1 adapter not found: {args.prev_adapter_path_phase1}")
        if not os.path.exists(args.prev_adapter_path):
            raise FileNotFoundError(f"Phase 2 adapter not found: {args.prev_adapter_path}")

        if local_rank == 0:
            print(f"\n[CUMULATIVE MERGE] Merging Phase 1 + Phase 2 into base model")
            print(f"  1. Merging Phase 1 from: {args.prev_adapter_path_phase1}")
        model = PeftModel.from_pretrained(model, args.prev_adapter_path_phase1)
        model = model.merge_and_unload()

        if local_rank == 0:
            print(f"  2. Merging Phase 2 from: {args.prev_adapter_path}")
        model = PeftModel.from_pretrained(model, args.prev_adapter_path)
        model = model.merge_and_unload()

        if local_rank == 0:
            print(f"Cumulative merge complete (Phase 1 + Phase 2 now part of base model)\n")

    elif args.phase == 2:
        # Phase 2: one merge (Phase 1 only)
        if not args.prev_adapter_path:
            raise ValueError("Phase 2 requires --prev_adapter_path (Phase 1 adapter)")
        if not os.path.exists(args.prev_adapter_path):
            raise FileNotFoundError(f"Phase 1 adapter not found: {args.prev_adapter_path}")

        if local_rank == 0:
            print(f"\nMerging Phase 1 adapter from: {args.prev_adapter_path}")
        model = PeftModel.from_pretrained(model, args.prev_adapter_path)
        model = model.merge_and_unload()
        if local_rank == 0:
            print("Phase 1 knowledge merged!\n")

    # Phase 1: no merge

    # LoRA
    peft_config = LoraConfig(
        r=current_rank,
        lora_alpha=current_alpha,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"], 
        bias="none", 
        task_type="CAUSAL_LM" 
    )
    model = get_peft_model(model, peft_config)

    for name, param in model.named_parameters():
        if param.requires_grad:
            param.data = param.data.to(torch.float32)

    if local_rank == 0:
        model.print_trainable_parameters()
        if torch.cuda.is_available():
            mem_alloc = torch.cuda.memory_allocated() / (1024 ** 3)
            print(f"[VRAM] After injecting LoRA: {mem_alloc:.2f} GB")

    if local_rank == 0:
        print("\nLoading datasets...")

    train_dataset = load_dataset("json", data_files={"train": args.train_dataset_path})["train"]
    eval_dataset  = load_dataset("json", data_files={"validation": args.eval_dataset_path})["validation"]

    if local_rank == 0:
        print(f"[DATASETS] Train: {len(train_dataset)}, Eval: {len(eval_dataset)}")

    response_template_ids = tokenizer.encode(response_template_str, add_special_tokens=False) 

    if local_rank == 0:
        print(f"[DEBUG] Response template IDs: {response_template_ids}")

    collator = DataCollatorForCompletionOnlyLM(
        response_template=response_template_ids,
        tokenizer=tokenizer
    ) #Ignores everything before the response_template, so the loss is computed only over the assistant's response


    if local_rank == 0:
        print("\n[SANITY CHECK] Validating response template...")
        if is_base:
            dummy = f"Test instruction\n\nTest input\n\nAnswer:\nTest answer{tokenizer.eos_token}"
        elif supports_system:
            dummy = tokenizer.apply_chat_template(
                [{"role": "system", "content": "S"}, {"role": "user", "content": "X"}, {"role": "assistant", "content": "Y"}],
                tokenize=False, add_generation_prompt=False
            )
        else:
            dummy = tokenizer.apply_chat_template(
                [{"role": "user", "content": "X"}, {"role": "assistant", "content": "Y"}],
                tokenize=False, add_generation_prompt=False
            )

        dummy_ids = tokenizer.encode(dummy, add_special_tokens=False)
        ref = list(response_template_ids)
        seq = list(dummy_ids)
        found = any(seq[i:i+len(ref)] == ref for i in range(len(seq) - len(ref) + 1))
        print(f"   -> Template found in test sequence: {found}")
        if not found:
            print(f"[ERROR] Sanity check failed! Template: {repr(response_template_str)}")
            print(f"Test prompt:\n{dummy}")
        assert found, "Response template not found in generated sequence!"
        print("   -> Validation passed!")

    # Training arguments
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=current_epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        per_device_eval_batch_size=4, 
        learning_rate=current_lr,
        optim="adamw_torch",
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        logging_steps=10,
        eval_on_start=True,
        eval_strategy="steps",
        save_strategy="steps",
        eval_steps=default_config["eval_steps"],
        save_steps=default_config["eval_steps"],
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        bf16=True,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        ddp_find_unused_parameters=False, 
        report_to="none"
    )

    trainer = SFTTrainer(
        model=model,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        max_seq_length= 2048,
        tokenizer=tokenizer,
        args=training_args,
        formatting_func=build_prompt, 
        data_collator=collator,
        packing=False,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
    )

    if local_rank == 0:
        print("\n" + "="*70)
        print("First formatted training example:")
        print("="*70)
        sample_text = build_prompt(train_dataset[:1])[0]
        print(sample_text)
        print("="*70 + "\n")

    training_succeeded = False
    try:
        last_checkpoint = get_last_checkpoint(args.output_dir) if os.path.isdir(args.output_dir) else None 
        trainer.train(resume_from_checkpoint=last_checkpoint)
        training_succeeded = True
    except Exception as e:
        if local_rank == 0:
            print(f"\n[ERROR] Training failed: {e}")
        raise
    finally: 
        if local_rank == 0 and trainer.state.log_history:
            log_path = os.path.join(args.output_dir, "training_logs.json")
            with open(log_path, "w") as f:
                json.dump(trainer.state.log_history, f, indent=4)
            print(f"\nLogs saved to: {log_path}")

    # Save adapter
    if local_rank == 0 and training_succeeded:
        final_save_path = os.path.join(args.output_dir, f"phase_{args.phase}_final_adapter")
        trainer.model.save_pretrained(final_save_path)
        tokenizer.save_pretrained(final_save_path)
        print(f"\n{'='*50}")
        print(f"Training complete!")
        print(f"Adapter saved to: {final_save_path}")
        print(f"{'='*50}\n")

if __name__ == "__main__":
    main()
