#http://localhost:7860

import argparse
import gc

import gradio as gr
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_REGISTRY = {
    "Qwen2.5-7B-Instruct | Mixed_Baseline 1-Fase (r16, lr1.1247e-4)": {
        "base_model": "./models/Qwen2.5-7B-Instruct",
        "adapters": [
            "./fine_tuning/fine_tuning_1phase/Qwen2.5-7B-Instruct/checkpoints/"
            "Mixed_Baseline/r16/lr_1e4/lr1.1247e-4_rank16/phase_1_final_adapter",
        ],
    },
}

DEFAULT_SYSTEM_INSTRUCTION = (
    "You are a clinical decision support assistant. Answer the medical "
    "question accurately and concisely, based on current medical knowledge."
)

EXAMPLE_PROMPTS = [
    "What are the clinical indications, composition, and side effects of Ramcor 1.25 Capsule?",
    "What is the mechanism of action of Ramipril?",
    "Which drugs are indicated for treatment of panic disorder?",
    "I'm considering adding Sertraline to a regimen that includes Tolbutamide. Are there any contraindications?"
]

MAX_LENGTH = 8000  
MAX_NEW_TOKENS = 1500 

#stores the model/tokeniser currently loaded into memory
STATE = {
    "model": None,
    "tokenizer": None,
    "model_key": None,
    "supports_system_role": False,
}


def detect_system_role_support(tokenizer): #checks if the model supports the role system
    if tokenizer.chat_template is None:
        return False
    try:
        messages = [{"role": "system", "content": "T"}, {"role": "user", "content": "T"}]
        tokenizer.apply_chat_template(messages, tokenize=False)
        return True
    except Exception:
        return False


def unload_current_model(): #Clear the STATE to free up VRAM before loading a new model
    if STATE["model"] is not None:
        del STATE["model"]
        STATE["model"] = None
    if STATE["tokenizer"] is not None:
        del STATE["tokenizer"]
        STATE["tokenizer"] = None
    STATE["model_key"] = None
    STATE["supports_system_role"] = False
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

def load_model(model_key, progress=gr.Progress()): #Loads the base model and merges the adapter/s for the selected input, in order, into memory.
    if not model_key: #if doesn't select the model
        return "Select a template from the drop-down menu before uploading."

    config = MODEL_REGISTRY[model_key]
    base_model_path = config["base_model"]
    adapter_paths = config["adapters"]

    progress(0.0, desc="Releasing previous model (if any)...")
    unload_current_model()  # deletes the previous model/tokenizer from STATE

    progress(0.1, desc=f"Loading tokenizer from {base_model_path}...")
    tokenizer = AutoTokenizer.from_pretrained(base_model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.add_bos_token = False

    progress(0.25, desc=f"Loading base model {base_model_path}...")  
    #Loads the base model:
    model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        device_map="auto",
        dtype=torch.bfloat16,
        attn_implementation="eager",
    )
    #For each adapter in the list (supports sequential fine-tuning, in order)
    n_adapters = len(adapter_paths)
    for i, adapter_path in enumerate(adapter_paths):
        frac = 0.25 + 0.65 * ((i + 1) / max(n_adapters, 1))  
        progress(frac, desc=f"Applying and merging adapter {i + 1}/{n_adapters}: {adapter_path}")
        model = PeftModel.from_pretrained(model, adapter_path)
        model = model.merge_and_unload()

    model.eval()

    progress(0.95, desc="Detecting 'system role' support in the chat template...")
    supports_system_role = detect_system_role_support(tokenizer)  # tests whether the tokenizer's chat template accepts role: "system"

    #Stores everything in STATE and returns a status string for the textbox
    STATE["model"] = model
    STATE["tokenizer"] = tokenizer
    STATE["model_key"] = model_key
    STATE["supports_system_role"] = supports_system_role

    progress(1.0, desc="Done.")
    status = (
        f"Model loaded: {model_key}\n"
        f"Base: {base_model_path}\n"
        f"Adapters merged: {n_adapters}\n"
        f"Supports 'system role': {supports_system_role}"
    )
    return status


def _message_text(content): #Formats the content of a chatbot message
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(
            item.get("text", "") if isinstance(item, dict) else str(item)
            for item in content
            if not (isinstance(item, dict) and item.get("type") == "file")
        )
    return str(content)


def build_prompt(tokenizer, supports_system_role, system_instruction, history, user_message):  #Builds the chat prompt 
    #Normalizes the history
    history = [{"role": m["role"], "content": _message_text(m["content"])} for m in history]

    messages = []
    if supports_system_role:  #If the tokenizer supports "system role": builds [system] + history + [current user]
        messages.append({"role": "system", "content": system_instruction})
        messages.extend(history)
        messages.append({"role": "user", "content": user_message})
    else:
        if not history:  #If not supported: there is no separate system message the system instruction is prepended to the content of the first message (either the first one in the history, or the current user message if the history is still empty).
            first_user_content = f"{system_instruction}\n\n{user_message}"
            messages.append({"role": "user", "content": first_user_content})
        else:
            first_message = dict(history[0])
            first_message["content"] = f"{system_instruction}\n\n{first_message['content']}"
            messages.append(first_message)
            messages.extend(history[1:])
            messages.append({"role": "user", "content": user_message})

    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True) 


def chat_fn(user_message, history, system_instruction):
    if STATE["model"] is None or STATE["tokenizer"] is None:
        return "No template has been loaded. Select a template from the drop-down menu and click 'Load Template' before starting a conversation."

    model = STATE["model"]
    tokenizer = STATE["tokenizer"]
    supports_system_role = STATE["supports_system_role"]

    prompt = build_prompt(tokenizer, supports_system_role, system_instruction, history, user_message)

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=MAX_LENGTH,
        add_special_tokens=True,
    ).to(model.device)
    input_length = inputs["input_ids"].shape[1]
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
            temperature=None,
            top_p=None,
            pad_token_id=tokenizer.pad_token_id,
        )

    generated_tokens = outputs[0][input_length:]

    response_text = tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()
    return response_text


CUSTOM_CSS = """
html, body, .gradio-container { background: #ffffff !important; height: 100vh !important; }
.gradio-container { max-width: 900px !important; margin: auto !important; display: flex !important; flex-direction: column !important; }
footer { display: none !important; }
#chat-column { flex: 1 !important; display: flex !important; flex-direction: column !important; min-height: 0 !important; }
#chat-column > div { flex: 1 !important; min-height: 0 !important; }
"""

FORCE_LIGHT_JS = """
function refresh() {
    const url = new URL(window.location);
    if (url.searchParams.get('__theme') !== 'light') {
        url.searchParams.set('__theme', 'light');
        window.location.href = url.href;
    }
}
"""


def build_interface(): 
    with gr.Blocks(title="AI assistant", css=CUSTOM_CSS, fill_height=True) as demo:
        with gr.Accordion("Definitions", open=False):  #collapsible menu, the one for model selection
            with gr.Row():
                model_dropdown = gr.Dropdown(  #model_dropdown lists the keys of MODEL_REGISTRY
                    choices=list(MODEL_REGISTRY.keys()),
                    label="Model / Pipeline",
                    value=None,
                )
                load_button = gr.Button("Load Model", variant="primary") 
            load_status = gr.Textbox(label="Status of the loaded model", interactive=False, lines=3)  

            load_button.click(fn=load_model, inputs=[model_dropdown], outputs=[load_status])  #Connects the button click to the load_model function

            system_instruction_box = gr.Textbox(  #editable textbox with the system instructioN
                label="System instruction",
                value=DEFAULT_SYSTEM_INSTRUCTION,
                lines=2,
            )

        with gr.Column(elem_id="chat-column"):
            gr.ChatInterface(  #ChatInterface keeps the conversation history as internal component state (type "messages", list of dicts {"role", "content"}) and automatically passes it as the second argument to the fn function on each call
                fn=lambda message, history, system_instruction: chat_fn(message, history, system_instruction),
                additional_inputs=[system_instruction_box],
                examples=[[prompt] for prompt in EXAMPLE_PROMPTS],
                fill_height=True,
            )

        demo.load(fn=None, js=FORCE_LIGHT_JS)  #forces the light theme in the browser

    return demo


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--server_port", type=int, default=7860)
    args = parser.parse_args()

    demo = build_interface()
    demo.queue().launch(
        server_name="0.0.0.0",
        server_port=args.server_port,
        share=False,  
    )


if __name__ == "__main__": 
    main()
