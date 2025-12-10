import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import json
import sys
import os
import re

# ==============================================================================
#                             GLOBAL CONFIGURATION
# ==============================================================================

MODEL_ID = "mistralai/Mistral-7B-Instruct-v0.3"
OUTPUT_DIR = "input"

# Target Sample Count
TOTAL_SAMPLES_TO_GENERATE = 300  # Total samples for the single intent
BATCH_SIZE = 5
RETRY_LIMIT = 5
INTENT_TO_RUN = "handle_scheduling" # Only run this intent

# Quantization Config (for running on limited GPU resources)
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",  
    bnb_4bit_compute_dtype=torch.bfloat16,
)

# ==============================================================================
#                             INTENT EXAMPLES AND STRUCTURES
# ==============================================================================

# --- Intent 1: handle_scheduling (Booking/Admin) ---
SCHEDULING_MULTI_EX = [
    {"intent": "handle_scheduling", "dialogue": [
        {"role": "user", "content": "I need to set up a technical call with your API lead, but I'm only available after 3 PM Pacific Time."},
        {"role": "agent", "content": "Understood. Our API lead is available late next Wednesday. Would 4:30 PM PT work for a 60-minute technical review?"},
        {"role": "user", "content": "That time works, but can we keep it to 45 minutes?"},
        {"role": "agent", "content": "Absolutely. Confirmed: Wednesday at 4:30 PM PT for a 45-minute call. I'll send the invite, noting the concise agenda."}
    ]},
    {"intent": "handle_scheduling", "dialogue": [
        {"role": "user", "content": "Can we schedule a 30-minute demo on VoizPanda's sentiment analysis capabilities for next Tuesday at 11 AM EST?"},
        {"role": "agent", "content": "I see that 11 AM EST on Tuesday is currently free. I've tentatively booked the 'Sentiment Deep Dive' for you; the calendar invite will follow shortly."},
        {"role": "user", "content": "Is that invite going to my work email or the generic contact one?"},
        {"role": "agent", "content": "I'll send it to your work email, confirming the title as: VoizPanda - Sentiment Deep Dive. Does that work, or should I loop in your CTO as well?"},
        {"role": "user", "content": "Please loop in the CTO, their email is [cto_email]."},
        {"role": "agent", "content": "Done. The invite for Tuesday at 11 AM EST now includes the CTO. Please accept the invitation to confirm your attendance."}
    ]},
]

# --- Centralized Intent Configuration (Simplified for the single run) ---
INTENT_CONFIGS = {
    "handle_scheduling": {
        "multi_ex": SCHEDULING_MULTI_EX,
        "tokens_per_sample_multi": 750, 
    },
}

# ==============================================================================
#                             PROMPTS AND CORE FUNCTIONS
# ==============================================================================

# **UPDATED SYSTEM INSTRUCTION WITH VOIZPANDA PERSONA AND STRICT JSON RULE**
SYSTEM_INSTRUCTION = (
    "You are a highly professional Presales Agent for **VoizPanda**, a specialized CRM company that excels at **AI-based voice calling features** (e.g., predictive dialing, real-time sentiment analysis, auto-summarization). "
    "Your responses MUST be natural conversational statements suitable for Text-to-Speech (TTS) generation. "
    "**ABSOLUTELY CRITICAL RULE: Output ONLY a single, valid JSON array. DO NOT INCLUDE ANY TEXT, WARNINGS, REASONING, OR MARKDOWN FENCES (like ```json) OUTSIDE OF THE ARRAY.**"
)

MULTI_TURN_TASK = (
    "Generate {num_to_generate} new multi-turn dialogues for the unified intent '{intent_name}'. "
    "Each dialogue must have a **minimum of 6 turns** (User, Agent, U, A, U, A). "
    "The Agent MUST use conversational memory to clarify details and, within the turns, **guide the conversation toward the successful booking of a meeting, demo, or follow-up call** before finalizing the dialogue. "
    "**STRICTLY ENFORCE:** Output only the JSON array structure provided in the examples."
)

# --- Standard Utility Functions ---
def build_full_prompt(examples, task_instruction, intent_name, num_to_generate):
    """Formats the system instruction, few-shot examples, and the final request."""
    ex_str = ""
    for i, ex in enumerate(examples):
        ex_str += f"<EXAMPLE {i+1}>\n{json.dumps(ex)}\n</EXAMPLE>\n"

    return (
        f"{SYSTEM_INSTRUCTION}\n\n"
        f"--- FEW-SHOT EXAMPLES ---\n{ex_str}\n"
        f"--- FINAL REQUEST ---\n"
        f"{task_instruction.format(intent_name=intent_name, num_to_generate=num_to_generate)}"
    )

def extract_json(raw_text):
    """Extracts the JSON array from the LLM's raw output robustly."""
    response_start_tag = "[/INST]"
    start_index_search = raw_text.rfind(response_start_tag)
    start_index = start_index_search + len(response_start_tag) if start_index_search != -1 else 0
    
    clean_text = raw_text[start_index:].strip()
    
    match = re.search(r'\[\s*\{.*\}\s*\]', clean_text, flags=re.DOTALL)
    
    if match:
        json_str = match.group(0).strip().replace("```json", "").replace("```", "").strip()
        return json_str
    
    # Fallback to simple bracket search
    start_bracket = clean_text.find('[')
    end_bracket = clean_text.rfind(']')
    
    if start_bracket != -1 and end_bracket != -1 and start_bracket < end_bracket:
        return clean_text[start_bracket : end_bracket + 1].strip()
        
    return None

def generate_data(intent_name, total_samples, examples, task_instruction, tokens_per_sample):
    """Generates a batch of synthetic data for a single intent type."""
    data = []
    attempts = 0
    
    filtered_examples = examples 

    print(f"--- Starting generation for {intent_name} (Format: Multi-Turn Only) ({total_samples} samples) ---")

    while len(data) < total_samples and attempts < RETRY_LIMIT * (total_samples / BATCH_SIZE):
        attempts += 1
        num = min(BATCH_SIZE, total_samples - len(data))
        max_tokens_batch = tokens_per_sample * num

        prompt = build_full_prompt(filtered_examples, task_instruction, intent_name, num)
        messages = [{"role": "user", "content": prompt}]

        try:
            # Tokenize and encode
            encoded = tokenizer.apply_chat_template(
                messages, return_tensors="pt", tokenize=True, add_generation_prompt=True
            )
            input_ids = encoded.to(model.device)
            pad_token_id = tokenizer.eos_token_id 

            # Generate
            with torch.no_grad():
                out = model.generate(
                    input_ids=input_ids,
                    max_new_tokens=max_tokens_batch, 
                    do_sample=True,
                    # NEW CHANGE: Lowered temperature for better JSON adherence
                    temperature=0.7, # Was 0.8
                    top_p=0.9,
                    pad_token_id=pad_token_id
                )

            decoded = tokenizer.decode(out[0][input_ids.shape[-1]:], skip_special_tokens=True)
            json_str = extract_json(decoded)

            if not json_str:
                print(f"❌ Attempt {attempts}: Failed to extract valid JSON.")
                continue
            
            parsed = json.loads(json_str)

            if not isinstance(parsed, list) or len(parsed) != num:
                print(f"❌ Attempt {attempts}: Parsed {len(parsed)} samples, expected {num}.")
                continue

            data.extend(parsed)
            print(f"✅ Batch: {len(parsed)}. Total {len(data)}/{total_samples}. (Attempt {attempts})")

        except json.JSONDecodeError:
            print(f"❌ Attempt {attempts}: JSON Decode Error.")
            continue
        except Exception as e:
            print(f"❌ Attempt {attempts}: General Error - {e}")
            continue

    return data

# ==============================================================================
#                             MAIN EXECUTION LOOP
# ==============================================================================

if __name__ == "__main__":
    
    # --- Load Model and Tokenizer ---
    print("Loading Model and Tokenizer...")
    try:
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

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    total_generated_count = 0
    
    print("\n==================== STARTING BATCH GENERATION (handle_scheduling @ 300 Multi-Turn Samples) ====================\n")
    
    # --- Processing the single target intent ---
    intent_name = INTENT_TO_RUN
    config = INTENT_CONFIGS[intent_name]
    current_output_file = os.path.join(OUTPUT_DIR, f"{intent_name}.json")
    
    # Checkpoint: Skip if the output file already exists and has the target count
    if os.path.exists(current_output_file):
        with open(current_output_file, "r") as f:
             existing_data = json.load(f)
        if len(existing_data) >= TOTAL_SAMPLES_TO_GENERATE:
             total_generated_count += len(existing_data)
             print(f"⏩ Skipping {intent_name}: Output file already exists with {len(existing_data)} samples.")
             print("Please delete the existing file if you wish to re-run the generation.")
        else:
             # If file exists but has fewer than the target, we allow re-run.
             # Delete and re-run to ensure a fresh batch.
             print(f"⚠️ Existing file has {len(existing_data)} samples (less than target {TOTAL_SAMPLES_TO_GENERATE}). Deleting and restarting generation.")
             os.remove(current_output_file)
             
             multi_data = generate_data(
                intent_name,
                TOTAL_SAMPLES_TO_GENERATE,
                config["multi_ex"], 
                MULTI_TURN_TASK,
                config["tokens_per_sample_multi"]
            )
             
             # Save Checkpoint
             if multi_data:
                with open(current_output_file, "w") as f:
                    json.dump(multi_data, f, indent=2)

                print(f"\n✅ Intent {intent_name} complete. Saved {len(multi_data)} samples to {current_output_file}")
                total_generated_count += len(multi_data)
             else:
                print(f"\n⚠️ Intent {intent_name} failed to generate any samples.")
        
    else:
        # --- Phase: Multi-Turn Generation Only ---
        multi_data = generate_data(
            intent_name,
            TOTAL_SAMPLES_TO_GENERATE,
            config["multi_ex"], # Use only multi-turn examples
            MULTI_TURN_TASK,
            config["tokens_per_sample_multi"]
        )
        
        all_intent_data = multi_data
        
        # --- Save Checkpoint ---
        if all_intent_data:
            with open(current_output_file, "w") as f:
                json.dump(all_intent_data, f, indent=2)

            print(f"\n✅ Intent {intent_name} complete. Saved {len(all_intent_data)} samples to {current_output_file}")
            total_generated_count += len(all_intent_data)
        else:
            print(f"\n⚠️ Intent {intent_name} failed to generate any samples.")
            
    print("\n\n#################### BATCH GENERATION FINAL REPORT ####################")
    print(f"Total Intents Processed: 1")
    print(f"Grand Total Samples Generated (Target: {TOTAL_SAMPLES_TO_GENERATE}): {total_generated_count}")
    print(f"File saved in the '{OUTPUT_DIR}/' directory: {INTENT_TO_RUN}.json")
    print("#####################################################################")