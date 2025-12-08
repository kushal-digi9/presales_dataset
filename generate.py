import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import json
import sys
import os
import re

# --- Configuration ---
MODEL_ID = "mistralai/Mistral-7B-Instruct-v0.3"
OUTPUT_DIR = "input/"

# The intent and output file name must reflect the new target
UNIFIED_INTENT = "adjust_offer_positioning" 
FINAL_OUTPUT_FILE = os.path.join(OUTPUT_DIR, f"{UNIFIED_INTENT}.json")

# --- Sample Counts based on 60:40 Ratio ---
TOTAL_SAMPLES_NEEDED = 100
SINGLE_TURN_SAMPLES = int(TOTAL_SAMPLES_NEEDED * 0.6) # 60 samples
MULTI_TURN_SAMPLES = TOTAL_SAMPLES_NEEDED - SINGLE_TURN_SAMPLES # 40 samples

BATCH_SIZE = 5
RETRY_LIMIT = 5
TOKENS_PER_SAMPLE_SINGLE = 300 
TOKENS_PER_SAMPLE_MULTI = 700 

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
)

print("Loading Model and Tokenizer...")
try:
    # Check if model is already loaded (for interactive environments)
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
    print(f"Model {MODEL_ID} loaded successfully.")
except Exception as e:
    print(f"FATAL ERROR during model loading: {e}")
    sys.exit(1)

# ---------------- EXAMPLES ---------------- #

# --- Single Turn Examples (adjust_offer_positioning) ---
SINGLE_TURN_EXAMPLES = [
    {
        "intent": UNIFIED_INTENT,
        "dialogue": [
            {"role": "user", "content": "Your Pro tier is too expensive; we can get similar features from Competitor X for 15% less."},
            {"role": "agent", "content": "I understand the need to be cost-effective. However, the Pro tier’s value comes from the included 24/7 dedicated support, which translates directly into less downtime and an estimated 20% faster deployment for your team, minimizing setup costs in the long run. Isn't rapid deployment a top priority?"}
        ],
        "notes": "Adjusts positioning from price vs. features to value vs. cost (downtime/deployment speed)."
    },
    {
        "intent": UNIFIED_INTENT,
        "dialogue": [
            {"role": "user", "content": "We need a one-month trial, not just the 14-day free period you offer."},
            {"role": "agent", "content": "While we strictly maintain the 14-day period to ensure fair resource allocation, we can adjust the positioning by offering a dedicated Solutions Engineer for three days within that period to guarantee you hit your key integration milestones faster, effectively doubling the value of the trial. Would access to a dedicated engineer help your evaluation?"}
        ],
        "notes": "Adjusts a term request (length of trial) by repositioning the offer with enhanced service/support (speed of integration)."
    }
]

# --- Multi-Turn Examples (adjust_offer_positioning) ---
MULTI_TURN_EXAMPLES = [
    {
        "intent": UNIFIED_INTENT,
        "dialogue": [
            {"role": "user", "content": "We need a 20% discount on the annual plan to meet our budget targets."},
            {"role": "agent", "content": "I appreciate you sharing your budget constraint. While a 20% price reduction is outside our standard policy for the annual plan, we can reposition the offer: what if we reduced the initial cost by 5% and included 5 hours of premium consulting services (worth $1,500) for free to ensure immediate ROI?"},
            {"role": "user", "content": "The consulting hours are helpful, but the budget reduction is the non-negotiable part for my finance team."},
            {"role": "agent", "content": "Understood. The finance team needs the numbers to work. If we kept the list price but **reduced the per-unit cost** by shifting your organization to a consumption-based metering system (often reducing overall cost by 10-15% for growing startups), would that meet the finance requirement?"},
            {"role": "user", "content": "A consumption model might work, yes, but we'd need a projection first."},
            {"role": "agent", "content": "Absolutely. I can generate a projection model right now. Based on your current usage estimates, the consumption model would position the offer to achieve your 15% saving goal by month three. Shall I send that breakdown over?"}
        ],
        "notes": "Multi-turn flow demonstrating sustained negotiation, shifting from direct discount to value-add (consulting) and then to a structural cost adjustment (consumption model)."
    }
]

# ---------------- PROMPTS ---------------- #

SYSTEM_INSTRUCTION = (
    "You are a highly professional Presales Agent for a **Startup Suite** software company (API-first, cloud-native, focused on scale). "
    "Output ONLY a JSON array. No explanations."
)

SINGLE_TURN_TASK = (
    "Generate {num_to_generate} new 2-turn dialogues for the intent '{intent_name}'. "
    "The Agent's response MUST adjust the offer positioning by **reframing a customer's objection** (price, term, feature) into a discussion about **unique Startup Suite value** (e.g., speed, scalability, API maturity, total cost of ownership). Output JSON array only."
)

MULTI_TURN_TASK = (
    "Generate {num_to_generate} new multi-turn dialogues for the intent '{intent_name}'. "
    "Each dialogue must have at least 6 turns (User, Agent, U, A, U, A). "
    "The Agent MUST use conversational memory to handle a sustained negotiation or objection over multiple turns, **adjusting the offer's positioning multiple times** (e.g., first repositioning price to service, then shifting to a structural change like a different pricing model). Output JSON array only."
)

def build_full_prompt(examples, task_instruction, intent_name, num_to_generate):
    ex_str = ""
    for i, ex in enumerate(examples):
        ex_str += f"<EXAMPLE {i+1}>\n{json.dumps(ex)}\n</EXAMPLE>\n"

    return (
        f"{SYSTEM_INSTRUCTION}\n\n"
        f"--- FEW-SHOT EXAMPLES ---\n{ex_str}\n"
        f"--- FINAL REQUEST ---\n"
        f"{task_instruction.format(intent_name=intent_name, num_to_generate=num_to_generate)}"
    )

# ---------------- CLEAN JSON (Robust Extraction) ---------------- #

def extract_json(raw_text):
    """
    Extracts the JSON array from the LLM's raw output, robustly searching for [ ... ].
    """
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
    
    # Fallback
    start_bracket = clean_text.find('[')
    end_bracket = clean_text.rfind(']')
    
    if start_bracket != -1 and end_bracket != -1 and start_bracket < end_bracket:
        json_str = clean_text[start_bracket : end_bracket + 1].strip()
        return json_str
        
    return None

# ---------------- GENERATION ---------------- #

def generate_data(intent_name, total_samples, examples, task_instruction, tokens_per_sample):
    data = []
    attempts = 0

    print(f"\n--- Starting generation for {intent_name} (Format: {task_instruction.split()[4]}) ({total_samples} samples) ---")

    while len(data) < total_samples and attempts < RETRY_LIMIT * (total_samples / BATCH_SIZE):
        attempts += 1
        num = min(BATCH_SIZE, total_samples - len(data))
        
        max_tokens_batch = tokens_per_sample * num

        prompt = build_full_prompt(examples, task_instruction, intent_name, num)
        messages = [{"role": "user", "content": prompt}]

        try:
            encoded = tokenizer.apply_chat_template(
                messages,
                return_tensors="pt",
                tokenize=True,
                add_generation_prompt=True
            )
            
            input_ids = encoded.to(model.device)
            pad_token_id = tokenizer.eos_token_id 

            with torch.no_grad():
                out = model.generate(
                    input_ids=input_ids,
                    max_new_tokens=max_tokens_batch, 
                    do_sample=True,
                    temperature=0.8,
                    top_p=0.9,
                    pad_token_id=pad_token_id
                )

            decoded = tokenizer.decode(out[0][input_ids.shape[-1]:], skip_special_tokens=True)
            
            json_str = extract_json(decoded)

            if not json_str:
                print("❌ No valid JSON structure found in output. Retrying.")
                continue
            
            parsed = json.loads(json_str)

            if not isinstance(parsed, list) or len(parsed) != num:
                print(f"❌ JSON array size mismatch (Expected {num}, Got {len(parsed)}). Retrying.")
                continue

            data.extend(parsed)
            print(f"✅ Batch: {len(parsed)}. Total {len(data)}/{total_samples}.")

        except json.JSONDecodeError as e:
            print(f"❌ JSON Decode Error: Expecting value: {e}. Retrying.")
        except Exception as e:
            print(f"❌ Unexpected Error during generation: {e}. Retrying.")

    return data

# ---------------- MAIN ---------------- #

if __name__ == "__main__":
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    all_data = []

    # --- Phase 1: Single-Turn Generation (60%) ---
    single = generate_data(
        UNIFIED_INTENT,
        SINGLE_TURN_SAMPLES,
        SINGLE_TURN_EXAMPLES,
        SINGLE_TURN_TASK,
        TOKENS_PER_SAMPLE_SINGLE
    )
    all_data.extend(single)

    # --- Phase 2: Multi-Turn Generation (40%) ---
    multi = generate_data(
        UNIFIED_INTENT,
        MULTI_TURN_SAMPLES,
        MULTI_TURN_EXAMPLES,
        MULTI_TURN_TASK,
        TOKENS_PER_SAMPLE_MULTI
    )
    all_data.extend(multi)

    if all_data:
        with open(FINAL_OUTPUT_FILE, "w") as f:
            json.dump(all_data, f, indent=2)

        print("\n=== COMPLETE ===")
        print(f"Intent: {UNIFIED_INTENT}")
        print(f"Total Generated: {len(all_data)}/{TOTAL_SAMPLES_NEEDED}")
        print(f"Single-Turn: {len(single)} | Multi-Turn: {len(multi)}")
        print(f"Saved to: {FINAL_OUTPUT_FILE}")
    else:
        print("\n⚠️ No samples generated.")