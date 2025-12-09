import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import json
import sys
import os
import re

# --- Configuration ---
MODEL_ID = "mistralai/Mistral-7B-Instruct-v0.3"
OUTPUT_DIR = "generated_data" 

# Target Intent and Output File
TARGET_INTENT = "adjust_offer_positioning" 
FINAL_OUTPUT_FILE = os.path.join(OUTPUT_DIR, "intent4.json")

# Generation parameters (Targeting 75 samples)
TOTAL_SAMPLES_TO_GENERATE = 75 
BATCH_SIZE = 10
RETRY_LIMIT = 5
TOKENS_PER_SAMPLE = 80 

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",  
    bnb_4bit_compute_dtype=torch.bfloat16,
)

print("Loading Model and Tokenizer...")
try:
    if 'model' not in locals() or 'tokenizer' not in locals():
        tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID,
            quantization_config=bnb_config,
            device_map="auto"
        )
        model.eval()
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id 
except Exception as e:
    print(f"FATAL ERROR during model loading: {e}")
    sys.exit(1)

# ---------------- EXAMPLES (Competitive/Objection) ---------------- #

# Using the high-quality samples from the assessment
FEW_SHOT_EXAMPLES = [
    {"text": "I've noticed that your product has a higher monthly cost compared to competitor 'Z', is there any room for negotiation?", "label": TARGET_INTENT},
    {"text": "Your offering seems to lack real-time data analysis, while 'Competitor A' provides this feature. Is there a possibility to add it?", "label": TARGET_INTENT},
    {"text": "Could we discuss the possibility of waiving the setup fee for our company's initial implementation?", "label": TARGET_INTENT},
    {"text": "I've observed that 'Solution B' offers a longer trial period. Can we extend the trial period for our evaluation?", "label": TARGET_INTENT},
    {"text": "The monthly cost per user for your product seems high. Are there any discounts available for larger teams?", "label": TARGET_INTENT},
]

# ---------------- PROMPTS ---------------- #

SYSTEM_INSTRUCTION = (
    "You are an expert data generator for text classification. The samples must be realistic, standalone customer utterances (not agent responses) from a B2B presales conversation. "
    "Output ONLY a JSON array. No explanations, no markdown, no filler text. The output MUST be a list of objects with two keys: 'text' (the user's utterance) and 'label' (the intent)."
)

TARGETED_TASK = (
    "Generate {num_to_generate} unique, single-turn customer utterances that strictly express the intent '{intent_name}'. "
    "These utterances must be **objections, questions comparing the product to competitors, requests for discounts, or statements of perceived feature gaps**. Output JSON array only."
)

def build_full_prompt(examples, task_instruction, intent_name, num_to_generate):
    ex_str = ""
    for ex in examples:
        ex_str += f"<EXAMPLE>\n{json.dumps(ex)}\n</EXAMPLE>\n"

    return (
        f"{SYSTEM_INSTRUCTION}\n\n"
        f"--- FEW-SHOT EXAMPLES ---\n{ex_str}\n"
        f"--- FINAL REQUEST ---\n"
        f"{task_instruction.format(intent_name=intent_name, num_to_generate=num_to_generate)}"
    )

# ---------------- CLEAN JSON (Robust Extraction) ---------------- #
def extract_json(raw_text):
    response_start_tag = "[/INST]"
    start_index_search = raw_text.rfind(response_start_tag)
    if start_index_search == -1:
        start_index_search = 0
    else:
        start_index_search += len(response_start_tag)
    
    clean_text = raw_text[start_index_search:].strip()
    match = re.search(r'\[\s*\{.*\}\s*\]', clean_text, flags=re.DOTALL)
    if match:
        json_str = match.group(0).strip()
        json_str = json_str.replace("```json", "").replace("```", "").strip()
        return json_str
    return None

# ---------------- GENERATION ---------------- #

def generate_data(intent_name, total_samples, examples, task_instruction, tokens_per_sample):
    data = []
    attempts = 0

    print(f"\n--- Starting generation for {intent_name} ({total_samples} samples) ---")

    while len(data) < total_samples and attempts < RETRY_LIMIT * (total_samples / BATCH_SIZE):
        attempts += 1
        num = min(BATCH_SIZE, total_samples - len(data))
        max_tokens_batch = tokens_per_sample * num * 2

        prompt = build_full_prompt(examples, task_instruction, intent_name, num)
        messages = [{"role": "user", "content": prompt}]

        try:
            encoded = tokenizer.apply_chat_template(messages, return_tensors="pt", tokenize=True, add_generation_prompt=True)
            input_ids = encoded.to(model.device)
            pad_token_id = tokenizer.eos_token_id 

            with torch.no_grad():
                out = model.generate(input_ids=input_ids, max_new_tokens=max_tokens_batch, do_sample=True, temperature=0.8, top_p=0.9, pad_token_id=pad_token_id)

            decoded = tokenizer.decode(out[0][input_ids.shape[-1]:], skip_special_tokens=True)
            json_str = extract_json(decoded)

            if not json_str: continue
            
            parsed = json.loads(json_str)

            if not isinstance(parsed, list) or len(parsed) != num or not all('text' in item and 'label' in item for item in parsed): continue

            data.extend(parsed)
            print(f"✅ Batch: {len(parsed)}. Total {len(data)}/{total_samples}.")

        except json.JSONDecodeError: pass
        except Exception: pass

    return data

# ---------------- MAIN ---------------- #

if __name__ == "__main__":
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("=======================================================")
    print(f"GENERATING SETFIT DATA FOR INTENT: {TARGET_INTENT}")
    print("=======================================================")

    generated_data = generate_data(
        TARGET_INTENT,
        TOTAL_SAMPLES_TO_GENERATE,
        FEW_SHOT_EXAMPLES,
        TARGETED_TASK,
        TOKENS_PER_SAMPLE
    )

    if generated_data:
        # Save the result to intent4.json
        with open(FINAL_OUTPUT_FILE, "w") as f:
            json.dump(generated_data, f, indent=2)

        print("\n=== GENERATION COMPLETE ===")
        print(f"Intent: {TARGET_INTENT}")
        print(f"Total Generated: {len(generated_data)}/{TOTAL_SAMPLES_TO_GENERATE}")
        print(f"Saved to: {FINAL_OUTPUT_FILE}")
    else:
        print("\n⚠️ Generation failed or returned empty data.")