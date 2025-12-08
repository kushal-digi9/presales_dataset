import json

# --- Configuration ---
INPUT_FILE = "reframe_to_value.json"
OUTPUT_FILE = "data/reframe_to_value.jsonl"
SYSTEM_PROMPT = (
    "You are a helpful and enthusiastic Presales Agent for a B2B SaaS product. "
    "Your goal is to qualify leads, answer product-related questions accurately, "
    "and gently guide the conversation towards scheduling a detailed demo."
)

# --- Llama 3 Chat Template Constants ---
BOS = "<|begin_of_text|>"
EOS = "<|eot|>"
EOT = "<|end_of_text|>"
SYSTEM_HEADER = "<|start_header_id|>system<|end_header_id|>\n\n"
USER_HEADER = "<|start_header_id|>user<|end_header_id|>\n\n"
ASSISTANT_HEADER = "<|start_header_id|>assistant<|end_header_id|>\n\n"

def format_conversation_for_llama3(conversation_data, system_prompt):
    """
    Applies the Llama 3 chat template to a single conversation.

    Args:
        conversation_data (list): A list of message objects, e.g.,
            [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]
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
            full_text += f"{USER_HEADER}{content}{EOS}"
        elif role == "assistant":
            full_text += f"{ASSISTANT_HEADER}{content}{EOS}"
        # Ignoring other roles (like 'system' if it appeared mid-chat) for simplicity

    # 3. End the entire sequence with the EOT token
    full_text += EOT
    return full_text

def transform_json_to_jsonl(input_file, output_file, system_prompt):
    """
    Reads a raw JSON file containing conversations, formats them, and writes to a JSONL file.
    """
    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            raw_data = json.load(f)
    except FileNotFoundError:
        print(f"Error: Input file '{input_file}' not found.")
        return
    except json.JSONDecodeError:
        print(f"Error: Failed to parse '{input_file}'. Ensure it is valid JSON.")
        return

    # Normalize loaded JSON to a list of conversation objects
    # Common input shapes:
    # - top-level list of conversations
    # - top-level dict with a wrapper key like 'conversations', 'items', 'data', or 'messages'
    # - a single conversation dict (possibly containing 'messages' or 'conversation')
    if isinstance(raw_data, dict):
        # If it's a wrapper dict containing a list, extract it
        for key in ("conversations", "items", "data", "messages"):
            if key in raw_data and isinstance(raw_data[key], list):
                raw_data = raw_data[key]
                break
        else:
            # If the dict itself contains a 'messages' or 'conversation' list, wrap that
            if "messages" in raw_data and isinstance(raw_data["messages"], list):
                # keep outer structure so downstream can use .get('messages')
                raw_data = [raw_data]
            elif "conversation" in raw_data and isinstance(raw_data["conversation"], list):
                raw_data = [raw_data]
            else:
                # Treat the dict as a single conversation object
                raw_data = [raw_data]

    elif not isinstance(raw_data, list):
        # Unknown top-level type (e.g., string/number) -> wrap to avoid crashes
        raw_data = [raw_data]

    processed_count = 0
    with open(output_file, 'w', encoding='utf-8') as outfile:
        # Assuming raw_data is a list of conversation objects
        for conversation_obj in raw_data:
            # Assuming the conversation history is stored under a key like 'messages' or 'conversation'
            # Adjust this key if your raw JSON structure is different
            conversation = conversation_obj.get("dialogue") or conversation_obj.get("messages") or conversation_obj.get("conversation", [])

            templated_text = format_conversation_for_llama3(conversation, system_prompt)

            if templated_text:
                # Write the templated text as a JSON object with a 'text' key on a single line
                jsonl_line = json.dumps({"text": templated_text}, ensure_ascii=False)
                outfile.write(jsonl_line + '\n')
                processed_count += 1

    print(f"\nTransformation complete.")
    print(f"Processed {processed_count} conversations.")
    print(f"Output saved to '{output_file}'. This is ready for Llama 3.2 fine-tuning.")
    print("\n--- Example of the Llama 3.2 Format (1 line in JSONL) ---")
    if processed_count > 0:
        print(jsonl_line[:200] + "...") # Print first 200 chars of the last line

if __name__ == "__main__":
    transform_json_to_jsonl(INPUT_FILE, OUTPUT_FILE, SYSTEM_PROMPT)