"""Retained SFT serializer and original sequence-building statement prefix.

The serializer is copied unchanged. The original nested tokenize_function is
lifted to module scope with its name and sequence statements unchanged, and an
explicit result return replaces tensor conversion/label masking. Only records
with messages and answer_messages are supported by the release replay CLI.
"""
def convert_to_sft_format(data_entry: dict) -> str:
    """Convert eval.jsonl format to training format

    Supports new format from make_sft_dataset.py with label_mode variations:
    - messages: contains system and user prompts
    - answer_messages: contains assistant response (format varies by label_mode)
      * full: Analogy + Code + Grid
      * analogy: Analogy + Grid
      * code: Code + Grid
      * base: Grid only
    """
    if "messages" in data_entry:
        # New format from make_sft_dataset.py (supports all label_modes)
        messages = data_entry["messages"]

        # Extract system and user messages as input
        input_parts = []
        output_text = ""

        for msg in messages:
            if msg["role"] == "system":
                input_parts.append(f"System: {msg['content']}")
            elif msg["role"] == "user":
                input_parts.append(f"Human: {msg['content']}")
            elif msg["role"] == "assistant":
                # Found assistant response in messages array
                output_text = msg['content']

        # Prioritize answer_messages field (for ground_truth files with label_mode)
        # This contains the correct label format based on label_mode setting
        if 'answer_messages' in data_entry and len(data_entry['answer_messages']) > 0:
            output_text = data_entry['answer_messages'][0]['content']

        if not output_text:
            print(f"Warning: No output text found for problem_id {data_entry.get('problem_id', 'unknown')}")
            return None

        # Format as training example
        input_text = "\n\n".join(input_parts) + "\n\nAssistant:"

        # Combine input and output for training
        return input_text + " " + output_text
    
    elif "source" in data_entry and "examples" in data_entry:
        # This is the old format from final_gifarc_output_100.jsonl
        # We need to create a proper input-output pair
        
        # Check if examples are empty
        if not data_entry.get('examples'):
            print(f"Warning: Empty examples for problem_id {data_entry.get('problem_id', 'unknown')}")
            return None
        
        # Extract analogy from seeds
        analogy = extract_analogy_from_source(data_entry['seeds'][-1])
        
        # Extract functions
        source_parts = data_entry['source'].split('\n\n')
        libraries = ""
        main_function = ""
        generate_input = ""

        # Extract libraries (everything before def main)
        if 'def main(' in data_entry['source']:
            libraries = data_entry['source'].split('def main(')[0].strip()

        # Find specific functions in parts
        for part in source_parts:
            if 'def main(' in part:
                main_function = part
            elif 'def generate_input(' in part:
                generate_input = part

        # Validate extraction
        if not libraries:
            print(f"Warning: libraries is empty for problem_id {data_entry.get('problem_id', 'unknown')}")
        if not main_function:
            print(f"Warning: main_function is empty for problem_id {data_entry.get('problem_id', 'unknown')}")

        
        # Get test output from examples
        test_output_grid = "[[]]"
        if data_entry['examples'] and len(data_entry['examples']) > 0:
            test_example = data_entry['examples'][-1]
            if len(test_example) >= 2:
                test_output_grid = str(test_example[1])
        
        # Use the common function to create input prompt
        input_text = create_input_prompt(data_entry['examples'])
        
        # Create the expected output in our 3-step format
        output_text = f"""# THINKING ANALOGY
{analogy}

# SOLUTION CODE
{libraries}

{main_function}

# RESULT TEST OUTPUT GRID
{test_output_grid}"""
        
        # Return the complete training example
        return input_text + " " + output_text
    
    else:
        raise ValueError("Unknown data format")


def tokenize_function(examples):
    # First, we need to handle the truncation properly
    processed_texts = []
    for idx, text in enumerate(examples["text"]):
        # Find where "Assistant:" is
        assistant_token = "Assistant:"
        assistant_idx = text.find(assistant_token)

        if assistant_idx != -1:
            # Split into input and output
            input_part = text[:assistant_idx + len(assistant_token)]
            output_part = text[assistant_idx + len(assistant_token):]

            # Tokenize to check lengths
            input_tokens = tokenizer(input_part, add_special_tokens=True)["input_ids"]
            output_tokens = tokenizer(output_part, add_special_tokens=False)["input_ids"]

            # If total is too long, we need to reduce examples
            if len(input_tokens) + len(output_tokens) > max_length:
                # Try to reduce examples by removing from the beginning
                if "original_data" in examples and idx < len(examples["original_data"]):
                    original_data = examples["original_data"][idx]
                    if isinstance(original_data, dict) and "examples" in original_data:
                        example_list = original_data["examples"]

                        # Keep removing examples from the beginning until it fits
                        for num_examples_to_keep in range(len(example_list) - 1, 0, -1):
                            # Create new input with fewer examples
                            reduced_examples = example_list[-num_examples_to_keep:]
                            reduced_input = create_input_prompt(reduced_examples)
                            reduced_input_tokens = tokenizer(reduced_input + "Assistant:", add_special_tokens=True)["input_ids"]

                            # Check if it fits now
                            if len(reduced_input_tokens) + len(output_tokens) <= max_length:
                                input_part = reduced_input + "Assistant:"
                                text = input_part + " " + output_part
                                if idx == 0:  # Debug print for first example
                                    print(f"[DEBUG] Reduced examples from {len(example_list)} to {num_examples_to_keep} to fit in {max_length} tokens")
                                    print(f"[DEBUG] Input tokens: {len(reduced_input_tokens)}, Output tokens: {len(output_tokens)}, Total: {len(reduced_input_tokens) + len(output_tokens)}")
                                    print(f"[DEBUG] Output preserved: {'YES' if len(output_tokens) == len(tokenizer(output_part, add_special_tokens=False)['input_ids']) else 'NO (truncated)'}")
                                break
                        else:
                            # Even with 1 example it's too long, fallback to truncation
                            print(f"[WARNING] Even with 1 example, still too long. Using truncation.")
                            # Truncate output to fit - leave some buffer for special tokens
                            max_output_length = max_length - len(reduced_input_tokens) - 10
                            output_part = tokenizer.decode(output_tokens[:max_output_length], skip_special_tokens=True)
                            text = reduced_input + "Assistant: " + output_part
                    else:
                        # No examples data available, fallback to original truncation
                        # Reserve space for output by limiting input
                        max_input_length = int(max_length * 0.7)  # Use 70% for input
                        if len(input_tokens) > max_input_length:
                            # Truncate input instead of output
                            input_part = tokenizer.decode(input_tokens[:max_input_length], skip_special_tokens=True)
                            # Make sure it still ends with "Assistant:"
                            if not input_part.endswith("Assistant:"):
                                input_part = input_part + "Assistant:"
                        text = input_part + " " + output_part
                else:
                    # No original_data available, use simple truncation
                    # Reserve space for output by limiting input
                    max_input_length = int(max_length * 0.7)  # Use 70% for input
                    if len(input_tokens) > max_input_length:
                        # Truncate input instead of output
                        input_part = tokenizer.decode(input_tokens[:max_input_length], skip_special_tokens=True)
                        # Make sure it still ends with "Assistant:"
                        if not input_part.endswith("Assistant:"):
                            input_part = input_part + "Assistant:"
                    text = input_part + " " + output_part
            else:
                text = text

            processed_texts.append(text)
        else:
            processed_texts.append(text)

    # Now tokenize the processed texts - work with lists internally
    all_input_ids = []
    all_attention_masks = []

    for text in processed_texts:
        # Tokenize without return_tensors to get lists
        tokens = tokenizer(text, add_special_tokens=True)
        input_ids = tokens["input_ids"]

        if len(input_ids) > max_length:
            # Need to truncate - find Assistant: position
            assistant_idx = text.find("Assistant:")
            if assistant_idx != -1:
                input_part = text[:assistant_idx + len("Assistant:")]
                output_part = text[assistant_idx + len("Assistant:"):]

                # Tokenize parts separately (as lists)
                input_tokens = tokenizer(input_part, add_special_tokens=True)["input_ids"]
                output_tokens = tokenizer(output_part, add_special_tokens=False)["input_ids"]

                # If output is too long, truncate it
                max_output_tokens = max_length - len(input_tokens) - 1  # -1 for EOS token
                if len(output_tokens) > max_output_tokens:
                    output_tokens = output_tokens[:max_output_tokens]

                # Reconstruct full token sequence
                full_tokens = input_tokens + output_tokens

                # Pad or truncate to max_length
                if len(full_tokens) < max_length:
                    padding_length = max_length - len(full_tokens)
                    full_tokens = full_tokens + [tokenizer.pad_token_id] * padding_length
                else:
                    full_tokens = full_tokens[:max_length]

                all_input_ids.append(full_tokens)
            else:
                # Fallback: truncate from the end
                truncated_tokens = input_ids[:max_length]
                # Pad if needed
                if len(truncated_tokens) < max_length:
                    padding_length = max_length - len(truncated_tokens)
                    truncated_tokens = truncated_tokens + [tokenizer.pad_token_id] * padding_length
                all_input_ids.append(truncated_tokens)
        else:
            # Pad to max_length
            padding_length = max_length - len(input_ids)
            padded_tokens = input_ids + [tokenizer.pad_token_id] * padding_length
            all_input_ids.append(padded_tokens)

    # Create attention masks
    for input_ids in all_input_ids:
        attention_mask = [1 if token_id != tokenizer.pad_token_id else 0 for token_id in input_ids]
        all_attention_masks.append(attention_mask)
    return {'processed_texts': processed_texts, 'input_ids': all_input_ids}
