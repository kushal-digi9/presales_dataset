import json
import os
import glob

# --- Configuration ---
INPUT_DIR = "input"
OUTPUT_DIR = "data"
SYSTEM_PROMPT = (
    "You are a helpful and enthusiastic Presales Agent for a B2B SaaS product. "
    "Your goal is to qualify leads, answer product-related questions accurately, "
    "and gently guide the conversation towards scheduling a detailed demo."
)

# --- Llama 3 Chat Template Constants ---
# NOTE: Using the standard Llama 3.1/3.2 tokens for alignment
BOS = "<|begin_of_text|>"
EOS = "<|eot|>"
EOT = "<|end_of_text|>" # This token is typically the same as EOS but used here for clarity
SYSTEM_HEADER = "<|start_header_id|>system<|end_header_id|>\n\n"
USER_HEADER = "<|start_header_id|>user<|end_header_id|>\n\n"
ASSISTANT_HEADER = "<|start_header_id|>assistant<|end_header_id|>\n\n"

def format_conversation_for_llama3(conversation_data, system_prompt):
    """
    Applies the Llama 3 chat template to a single conversation.

    Args:
        conversation_data (list): A list of message objects.
        system_prompt (str): The initial system instruction.

    Returns:
        str: The fully templated conversation string, or None if the conversation is empty.
    """
    if not conversation_data:
        return None

    # Start the full sequence with the BOS token
    full_text = BOS

    # 1. Add the System Prompt
    full_text += f"{SYSTEM_HEADER}{system_prompt}{EOS}"

    # 2. Iterate through all subsequent turns
    for message in conversation_data:
        role = message.get("role")
        content = message.get("content", "").strip()

        if role == "user":
            # Add the USER_HEADER and content
            if content:
                full_text += f"{USER_HEADER}{content}{EOS}"
        elif role == "agent": # NOTE: Assumes agent role is synonymous with assistant
            # Add the ASSISTANT_HEADER and content
            if content:
                full_text += f"{ASSISTANT_HEADER}{content}{EOS}"
        # If the role is neither user nor agent/assistant, it's ignored.

    # 3. End the entire sequence with the EOT token
    full_text += EOT
    return full_text

def transform_json_to_jsonl(input_path, output_path, system_prompt):
    """
    Reads a raw JSON file containing conversations, formats them, and writes to a JSONL file.
    """
    try:
        with open(input_path, 'r', encoding='utf-8') as f:
            raw_data = json.load(f)
    except FileNotFoundError:
        print(f"Error: Input file '{input_path}' not found. Skipping.")
        return
    except json.JSONDecodeError:
        print(f"Error: Failed to parse '{input_path}'. Skipping.")
        return

    # Assuming raw_data is a list of conversation objects
    if not isinstance(raw_data, list):
        print(f"Warning: Expected a list in {input_path}, found {type(raw_data)}. Skipping.")
        return

    processed_count = 0
    with open(output_path, 'w', encoding='utf-8') as outfile:
        # Assuming each object in raw_data contains the 'dialogue' key
        for conversation_obj in raw_data:
            conversation = conversation_obj.get("dialogue", [])

            templated_text = format_conversation_for_llama3(conversation, system_prompt)

            if templated_text:
                # Write the templated text as a JSON object with a 'text' key on a single line
                jsonl_line = json.dumps({"text": templated_text}, ensure_ascii=False)
                outfile.write(jsonl_line + '\n')
                processed_count += 1

    print(f"✅ Processed {processed_count} conversations from '{os.path.basename(input_path)}'.")
    print(f"Output saved to '{output_path}'.")

if __name__ == "__main__":
    print("--- Starting Batch JSON to JSONL Conversion ---")
    print(f"Targeting all files in '{INPUT_DIR}' for Llama 3 SFT format.")

    # 1. Ensure the output directory exists
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 2. Find all relevant JSON files in the input directory
    json_files = glob.glob(os.path.join(INPUT_DIR, "*.json"))

    if not json_files:
        print(f"❌ No JSON files found in the '{INPUT_DIR}' directory. Exiting.")
    else:
        # 3. Process each file
        for input_path in json_files:
            # Determine output filename: e.g., input/file.json -> data/file.jsonl
            base_filename = os.path.basename(input_path)
            output_filename = base_filename.replace(".json", ".jsonl")
            output_path = os.path.join(OUTPUT_DIR, output_filename)

            transform_json_to_jsonl(input_path, output_path, SYSTEM_PROMPT)

        print("\n--- Batch Conversion Complete 🎉 ---")
        print("All intent JSON files are now in the Llama 3.2 SFT JSONL format.")