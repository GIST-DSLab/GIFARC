import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    BitsAndBytesConfig
)
from datasets import Dataset
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
    TaskType,
    PeftModel
)
import json
from typing import Dict, List
import argparse
import os
import warnings
from tqdm import tqdm
from datetime import datetime
import wandb
import re
# Removed arc_evaluation import - evaluation is now handled in sft_eval.py

# Suppress warnings
warnings.filterwarnings("ignore", message=".*tokenizer.*deprecated.*")
warnings.filterwarnings("ignore", message=".*gradient checkpointing.*")
warnings.filterwarnings("ignore", message=".*past_key_value.*")

# Model configuration
MAX_LENGTH = 2048

# LoRA configuration
LORA_CONFIG = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=[
        "q_proj",
        "k_proj", 
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ],
    lora_dropout=0.15,
    bias="none",
    task_type=TaskType.CAUSAL_LM,
)

def extract_analogy_from_source(source_code: str) -> str:
    """Extract analogical description from source code comments"""
    analogy = source_code.split('\nDescription:')[1]
    return analogy

# Removed create_evaluation_prompt as the format is now included in the user prompt

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

def create_input_prompt(examples: list) -> str:
    """Create input prompt for training
    
    Args:
        examples: List of (input, output) tuples
    """
    # Create system prompt
    system_prompt = "You are a world-class puzzle solver with exceptional pattern recognition skills and expertise in Python programming. Your task is to analyze puzzles and provide Python solutions."
    
    # Create user prompt with examples
    user_prompt = "Here are some examples of input-output transformations:\n\n"
    
    # Add all examples except the last one as training examples
    if examples and len(examples) > 1:
        for i, (input_grid, output_grid) in enumerate(examples[:-1]):
            user_prompt += f"Example {i+1}:\n"
            user_prompt += f"Input: {str(input_grid)}\n"
            user_prompt += f"Output: {str(output_grid)}\n\n"
    
    # Add the test input
    if examples:
        test_input = examples[-1][0]
        user_prompt += f"Test Input: {str(test_input)}\n\n"
        user_prompt += "Analyze the patterns in these examples and provide a solution for the test input. "
        user_prompt += "You MUST structure your response using EXACTLY the following format with the specific section headers:\n\n"
        user_prompt += "# THINKING ANALOGY\n"
        user_prompt += "[Describe the pattern inherent in the input-output relationship using analogies, metaphors, or real-world comparisons]\n\n"
        user_prompt += "# SOLUTION CODE\n"
        user_prompt += "[Translate the pattern from Step 1's analogical thinking into code]\n\n"
        user_prompt += "# RESULT TEST OUTPUT GRID\n"
        user_prompt += "[The result of applying Step 2's transformation code to the test input]"
    
    # Format as input prompt
    input_prompt = f"System: {system_prompt}\n\nHuman: {user_prompt}\n\nAssistant:"
    
    return input_prompt

def create_input_only_format(data_entry: dict) -> str:
    """Create input-only version for evaluation (used in sft_eval.py)"""
    return create_input_prompt(data_entry['examples'])

def prepare_arc_dataset(data_path: str, tokenizer, max_length: int = MAX_LENGTH, augment_examples: bool = False, max_augment_factor: int = 10) -> Dataset:
    """Prepare ARC dataset for training
    
    Args:
        data_path: Path to JSONL data file
        tokenizer: Tokenizer to use
        max_length: Maximum sequence length
        augment_examples: If True, create multiple samples by using different example combinations when truncation is needed
        max_augment_factor: Maximum number of augmented samples per original sample
    """
    formatted_data = []
    
    # Read JSONL format file and convert to SFT format
    with open(data_path, 'r', encoding='utf-8') as f:
        for idx, line in enumerate(f):
            try:
                data = json.loads(line.strip())
                
                # Convert to new SFT format
                if "source" in data and "examples" in data:
                    if augment_examples and len(data["examples"]) > 2:
                        # Check if augmentation is needed
                        full_sft_text = convert_to_sft_format(data)
                        if full_sft_text is None:
                            raise ValueError(f"Failed to convert data to SFT format for problem_id {data.get('problem_id', idx)}")
                        full_tokens = tokenizer(full_sft_text, add_special_tokens=True)["input_ids"]
                        
                        if len(full_tokens) > max_length:
                            # Need truncation - let's augment!
                            num_examples = len(data["examples"])
                            
                            # Find max examples that fit while preserving the assistant response
                            max_examples_fit = 1
                            
                            # First, check how much space the output needs
                            assistant_idx = full_sft_text.find("Assistant:")
                            if assistant_idx != -1:
                                output_part = full_sft_text[assistant_idx + len("Assistant:"):]
                                output_tokens = tokenizer(output_part, add_special_tokens=False)["input_ids"]
                                # Reserve space for output + some buffer
                                reserved_tokens = len(output_tokens) + 50
                                available_tokens = max_length - reserved_tokens
                            else:
                                # If can't find Assistant:, reserve 40% for output
                                reserved_tokens = int(max_length * 0.4)
                                available_tokens = int(max_length * 0.6)  # Use 60% for input if can't find Assistant:
                            
                            for n in range(2, num_examples + 1):
                                # Try with last n examples (maintaining order)
                                test_examples = data["examples"][-n:]
                                test_input = create_input_prompt(test_examples) + "Assistant:"
                                test_input_tokens = tokenizer(test_input, add_special_tokens=True)["input_ids"]
                                
                                if len(test_input_tokens) <= available_tokens:
                                    max_examples_fit = n
                                else:
                                    break
                            
                            if max_examples_fit < num_examples:
                                # Generate augmented samples
                                import itertools
                                
                                # Calculate total possible combinations
                                # Choose (max_examples_fit - 1) training examples from (num_examples - 1)
                                # Then choose 1 as test from remaining
                                num_train_select = max_examples_fit - 1
                                total_combinations = 0
                                
                                augmented_count = 0
                                
                                # For each possible test input position
                                for test_idx in range(num_examples):
                                    # Get training examples (all except the chosen test)
                                    train_indices = [i for i in range(num_examples) if i != test_idx]
                                    
                                    # Select combinations of training examples
                                    for train_combo in itertools.combinations(train_indices, num_train_select):
                                        if augmented_count >= max_augment_factor:
                                            break
                                        
                                        # Build the example list
                                        selected_examples = []
                                        for i in train_combo:
                                            selected_examples.append(data["examples"][i])
                                        # Add test input at the end
                                        selected_examples.append(data["examples"][test_idx])
                                        
                                        # Create augmented data
                                        augmented_data = {**data, "examples": selected_examples}
                                        sft_text = convert_to_sft_format(augmented_data)
                                        
                                        formatted_data.append({
                                            "text": sft_text,
                                            "problem_id": f"{data['problem_id']}_aug{augmented_count}",
                                            "original_data": augmented_data
                                        })
                                        augmented_count += 1
                                    
                                    if augmented_count >= max_augment_factor:
                                        break
                                
                                if idx == 0:  # Debug print
                                    print(f"[AUGMENTATION] Original examples: {num_examples}, Max fit: {max_examples_fit}")
                                    print(f"[AUGMENTATION] Created {augmented_count} augmented samples from 1 original")
                                    print(f"[AUGMENTATION] Output tokens reserved: {reserved_tokens}, Available for input: {available_tokens}")
                            else:
                                # All examples fit, no augmentation
                                sft_text = convert_to_sft_format(data)

                                if sft_text is None:

                                    continue  # Skip this entry
                                formatted_data.append({
                                    "text": sft_text,
                                    "problem_id": data['problem_id'],
                                    "original_data": data
                                })
                        else:
                            # No truncation needed
                            sft_text = convert_to_sft_format(data)

                            if sft_text is None:

                                continue  # Skip this entry
                            formatted_data.append({
                                "text": sft_text,
                                "problem_id": data['problem_id'],
                                "original_data": data
                            })
                    else:
                        # No augmentation requested or too few examples
                        sft_text = convert_to_sft_format(data)

                        if sft_text is None:

                            continue  # Skip this entry
                        formatted_data.append({
                            "text": sft_text,
                            "problem_id": data['problem_id'],
                            "original_data": data
                        })
                elif "messages" in data:
                    # Use the same format conversion for consistency
                    sft_text = convert_to_sft_format(data)

                    if sft_text is None:

                        continue  # Skip this entry
                    
                    formatted_data.append({
                        "text": sft_text,
                        "idx": idx,
                        "original_data": data
                    })
            except Exception as e:
                print(f"Error processing line {idx}: {e}")
                continue
    
    print(f"[DATASET] Total samples after augmentation: {len(formatted_data)}")
    dataset = Dataset.from_list(formatted_data)
    
    # Tokenization function
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
        
        # Convert lists to tensors at the end
        outputs = {
            "input_ids": torch.tensor(all_input_ids),
            "attention_mask": torch.tensor(all_attention_masks)
        }
        outputs["labels"] = outputs["input_ids"].clone()
        
        # Mask padding tokens
        outputs["labels"][outputs["labels"] == tokenizer.pad_token_id] = -100
        
        # Mask input tokens (only train on assistant response)
        for idx, text in enumerate(processed_texts):
            # 더 정확한 마스킹을 위해
            full_tokens = outputs["input_ids"][idx]
            assistant_position = None

            assistant_tokens = tokenizer.encode("Assistant:", add_special_tokens=False)
            assistant_len = len(assistant_tokens)  # 실제 토큰 개수 확인
            
            # Debug print for first example
            if idx == 0:
                print(f"\n[DEBUG] Processing example {idx}")
                print(f"Assistant tokens: {assistant_tokens} (length: {assistant_len})")
                print(f"First 20 tokens: {full_tokens[:20].tolist()}")
                
            # "Assistant:" 토큰의 정확한 위치 찾기
            for i in range(len(full_tokens) - assistant_len + 1):
                candidate_tokens = full_tokens[i:i+assistant_len].tolist()
                decoded_text = tokenizer.decode(candidate_tokens)
                
                if idx == 0 and i < 10:  # Debug first few positions
                    print(f"Position {i}: tokens={candidate_tokens}, decoded='{decoded_text}'")
                
                if decoded_text.strip() == "Assistant:":
                    assistant_position = i + assistant_len
                    if idx == 0:
                        print(f"Found 'Assistant:' at position {i}, will mask up to position {assistant_position}")
                    break
            
            if assistant_position:
                outputs["labels"][idx][:assistant_position] = -100
            else:
                print(f"WARNING: Could not find 'Assistant:' in example {idx}")
                # Try to find it in the original text for debugging
                full_text = tokenizer.decode(full_tokens, skip_special_tokens=True)
                if "Assistant:" in full_text:
                    print(f"  'Assistant:' exists in decoded text at char position: {full_text.find('Assistant:')}")
                else:
                    print(f"  'Assistant:' NOT FOUND in decoded text")
        
        # Keep idx and original_data for evaluation
        if "problem_id" in examples:
            outputs["problem_id"] = examples["problem_id"]
        if "idx" in examples:
            outputs["idx"] = examples["idx"]
        if "original_data" in examples:
            outputs["original_data"] = examples["original_data"]
        elif "messages" in examples:
            outputs["messages"] = examples["messages"]
        
        return outputs
    
    tokenized_dataset = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=["text"]
    )
    
    # Validation: Check that Assistant responses are preserved
    print("\n[VALIDATION] Checking tokenized dataset...")
    for i in range(min(5, len(tokenized_dataset))):
        sample = tokenized_dataset[i]
        input_ids = sample["input_ids"]
        labels = sample["labels"]
        
        # Decode the full text
        if isinstance(input_ids, torch.Tensor):
            input_ids = input_ids.tolist()
        full_text = tokenizer.decode(input_ids, skip_special_tokens=True)
        
        # Find where training starts (first non-masked label)
        # Convert to tensor if it's a list
        if isinstance(labels, list):
            labels_tensor = torch.tensor(labels)
        else:
            labels_tensor = labels
            
        non_masked_positions = (labels_tensor != -100).nonzero(as_tuple=True)[0]
        if len(non_masked_positions) > 0:
            first_trained_pos = non_masked_positions[0].item()
            if isinstance(input_ids, torch.Tensor):
                decode_ids = input_ids[first_trained_pos:].tolist()
            elif isinstance(input_ids, list):
                decode_ids = input_ids[first_trained_pos:]
            else:
                decode_ids = input_ids[first_trained_pos:]
            trained_text = tokenizer.decode(decode_ids, skip_special_tokens=True)
            
            print(f"\n[VALIDATION] Sample {i}:")
            print(f"  - Full text length: {len(full_text)} chars")
            print(f"  - Has 'Assistant:': {'Yes' if 'Assistant:' in full_text else 'No'}")
            print(f"  - Training starts at position: {first_trained_pos}")
            print(f"  - Trained text preview: {trained_text[:100]}...")
            
            # Check if the trained text contains expected patterns
            has_thinking = "# THINKING ANALOGY" in trained_text
            has_solution = "# SOLUTION CODE" in trained_text
            has_result = "# RESULT TEST OUTPUT GRID" in trained_text
            
            print(f"  - Has THINKING section: {has_thinking}")
            print(f"  - Has SOLUTION section: {has_solution}")
            print(f"  - Has RESULT section: {has_result}")
            
            if not all([has_thinking, has_solution, has_result]):
                print(f"  - WARNING: Missing expected sections!")
    
    return tokenized_dataset

def print_trainable_parameters(model):
    """Print number of trainable parameters"""
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params:,} || "
        f"all params: {all_param:,} || "
        f"trainable%: {100 * trainable_params / all_param:.2f}"
    )

# Custom data collator that removes non-tensor fields for training
class CustomDataCollatorForLanguageModeling(DataCollatorForLanguageModeling):
    def __call__(self, features):
        # Store labels before processing to preserve our custom masking
        saved_labels = []
        for feature in features:
            if "labels" in feature:
                # Convert to tensor if it's a list
                if isinstance(feature["labels"], list):
                    saved_labels.append(torch.tensor(feature["labels"]))
                else:
                    saved_labels.append(feature["labels"])
                
            # Remove non-tensor fields
            if "idx" in feature:
                del feature["idx"]
            if "messages" in feature:
                del feature["messages"]
            if "original_data" in feature:
                del feature["original_data"]
            if "problem_id" in feature:
                del feature["problem_id"]
        
        # Call parent collator
        batch = super().__call__(features)
        
        # Restore our custom labels if we saved them
        if saved_labels:
            batch["labels"] = torch.stack(saved_labels)
        
        return batch

# Custom trainer with W&B integration
class CustomTrainer(Trainer):
    def __init__(self, use_wandb=False, tokenizer=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.use_wandb = use_wandb
        self.tokenizer = tokenizer
        # Removed eval_logged_examples - evaluation is handled in sft_eval.py
        self.train_examples_buffer = []  # Buffer for training examples
        
    def compute_loss(self, model, inputs, return_outputs=False, **_):
        """
        Custom loss computation with logging
        """
        labels = inputs.pop("labels")
        
        # Save original data for logging
        original_data = inputs.pop("original_data") if "original_data" in inputs else None
        idx = inputs.pop("idx") if "idx" in inputs else None
        # Remove token_type_ids if present (Mistral doesn't use them)
        if "token_type_ids" in inputs:
            inputs.pop("token_type_ids")
        
        # Debug: Check masking during training
        if self.state.global_step % 10 == 0:  # Log every 10 steps
            # Check labels masking
            batch_size = labels.shape[0] if labels.dim() > 1 else 1
            for i in range(min(batch_size, 2)):  # Check first 2 examples in batch
                sample_labels = labels[i] if labels.dim() > 1 else labels
                non_masked = (sample_labels != -100).sum().item()
                total = sample_labels.numel()
                masked = total - non_masked
                
                print(f"\n[Step {self.state.global_step}] Sample {i} masking check:")
                print(f"  - Total tokens: {total}")
                print(f"  - Masked tokens (-100): {masked} ({masked/total*100:.1f}%)")
                print(f"  - Non-masked tokens: {non_masked} ({non_masked/total*100:.1f}%)")
                
                # Find first non-masked position
                non_masked_positions = (sample_labels != -100).nonzero(as_tuple=True)[0]
                if len(non_masked_positions) > 0:
                    first_non_masked = non_masked_positions[0].item()
                    print(f"  - First non-masked position: {first_non_masked}")
                    
                    # Decode around that position to verify it's after "Assistant:"
                    input_ids = inputs["input_ids"][i] if inputs["input_ids"].dim() > 1 else inputs["input_ids"]
                    # Convert CUDA tensor to CPU for decoding
                    before_text = self.tokenizer.decode(input_ids[:first_non_masked].cpu().tolist())
                    if "Assistant:" in before_text:
                        print(f"  - ✓ Correctly masked: non-masked tokens start after 'Assistant:'")
                    else:
                        print(f"  - ✗ WARNING: Non-masked tokens found before 'Assistant:'")
                        print(f"  - Text before first non-masked: ...{before_text[-100:]}")
        
        outputs = model(**inputs)
        logits = outputs.get('logits')
        
        # output_text = self.tokenizer.decode(model.generate(**inputs)[0][inputs['input_ids'].shape[-1]:].cpu().tolist(), skip_special_tokens=True)
        # self.tokenizer.decode(labels[labels != -100].contiguous().cpu().tolist(), skip_special_tokens=True)
        
        # Shift for next token prediction
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        
        # Compute loss
        loss_fct = torch.nn.CrossEntropyLoss(ignore_index=-100)
        loss = loss_fct(
            shift_logits.view(-1, shift_logits.size(-1)), 
            shift_labels.view(-1)
        )
        
        # Collect training examples for logging
        if self.use_wandb and wandb.run is not None:
            # Save this example to buffer
            self.train_examples_buffer.append({
                'inputs': {k: v.clone() if torch.is_tensor(v) else v for k, v in inputs.items()},
                'outputs': outputs,
                'labels': labels.clone() if torch.is_tensor(labels) else labels,
                'original_data': original_data,
                'idx': idx
            })
            
            # Log when we have 10 examples or at logging step
            if len(self.train_examples_buffer) >= 10 or self.state.global_step % 20 == 0:
                self._log_training_examples()
                self.train_examples_buffer = []
        
        return (loss, outputs) if return_outputs else loss
    
    def log(self, logs: Dict[str, float], start_time: float = None) -> None:
        """
        Override log method to add custom W&B logging
        """
        # Call parent log method
        if start_time is not None:
            super().log(logs, start_time)
        else:
            super().log(logs)
        
        # Additional W&B logging if enabled
        if self.use_wandb and wandb.run is not None:
            # Custom metrics
            custom_logs = {}
            
            # Add learning rate if available
            if "learning_rate" in logs:
                custom_logs["training/learning_rate"] = logs["learning_rate"]
            
            # Add gradient norm if available
            if "grad_norm" in logs:
                custom_logs["training/gradient_norm"] = logs["grad_norm"]
                
            # Log to W&B
            if custom_logs:
                wandb.log(custom_logs, step=self.state.global_step)
    
    def _log_training_examples(self):
        """Log multiple training examples to W&B"""
        if self.tokenizer is None or not self.train_examples_buffer:
            return
            
        try:
            # Create a table for training examples
            train_table = wandb.Table(columns=["Step", "Example_Index", "Input", "Predicted", "Label", "Ground Truth"])
            
            # Process up to 10 examples from buffer
            for i, example in enumerate(self.train_examples_buffer[:10]):
                inputs = example['inputs']
                outputs = example['outputs']
                labels = example['labels']
                original_data = example['original_data']
                idx = example['idx']
                
                # Take only the first example in the batch
                input_ids = inputs["input_ids"][0] if len(inputs["input_ids"].shape) > 1 else inputs["input_ids"]
                logits = outputs.logits[0] if len(outputs.logits.shape) > 2 else outputs.logits
                label = labels[0] if len(labels.shape) > 1 else labels
                
                # Generate prediction from logits
                predicted_ids = torch.argmax(logits, dim=-1)
                
                # Decode texts
                input_text = self.tokenizer.decode(input_ids, skip_special_tokens=True)
                predicted_text = self.tokenizer.decode(predicted_ids, skip_special_tokens=True)
                label_text = self.tokenizer.decode(label[label != -100], skip_special_tokens=True)
                
                # Get ground truth if available
                ground_truth = ""
                if original_data and len(original_data) > 0:
                    data = original_data[0] if isinstance(original_data, list) else original_data
                    if isinstance(data, dict) and "examples" in data and data["examples"]:
                        last_example = data["examples"][-1]
                        if len(last_example) >= 2:
                            ground_truth = str(last_example[1])  # Full ground truth
                
                # Add to table
                train_table.add_data(
                    self.state.global_step,
                    i,
                    input_text,  # Full input text
                    predicted_text,  # Full predicted text
                    label_text,  # Full label text
                    ground_truth  # Full ground truth
                )
            
            # Log the table
            wandb.log({
                "train/examples": train_table,
                "train/num_examples_logged": len(self.train_examples_buffer[:10]),
                "train/logging_step": self.state.global_step
            }, step=self.state.global_step)
            
        except Exception as e:
            # Log any errors but don't stop training
            print(f"Error logging training examples: {e}")
    
    # Removed _save_evaluation_outputs method - evaluation is now handled in sft_eval.py

def main():
    parser = argparse.ArgumentParser(description="ARC Model Training")
    parser.add_argument("--model_name", type=str, required=True, help="Model name")
    parser.add_argument("--ground_truth_data_path", type=str, required=True, help="Training data file")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory")
    parser.add_argument("--num_epochs", type=int, default=4, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size per device")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8, help="Gradient accumulation steps")
    parser.add_argument("--learning_rate", type=float, default=2e-4, help="Learning rate")
    parser.add_argument("--max_length", type=int, default=MAX_LENGTH, help="Maximum sequence length")
    parser.add_argument("--use_8bit", action="store_true", help="Use 8-bit quantization")
    parser.add_argument("--use_4bit", action="store_true", help="Use 4-bit quantization")
    
    # W&B arguments
    parser.add_argument("--use_wandb", action="store_true", help="Use Weights & Biases for logging")
    parser.add_argument("--wandb_project", type=str, default="arc-sft", help="W&B project name")
    parser.add_argument("--wandb_run_name", type=str, default=None, help="W&B run name")
    parser.add_argument("--wandb_tags", nargs="+", default=[], help="W&B tags")
    
    # Evaluation is now handled by sft_eval.py
    parser.add_argument("--max_new_tokens", type=int, default=8192, help="Maximum new tokens for generation")
    
    # Data augmentation arguments
    parser.add_argument("--augment_examples", action="store_true", help="Augment dataset by creating multiple samples with different example combinations when truncation is needed")
    parser.add_argument("--max_augment_factor", type=int, default=10, help="Maximum number of augmented samples per original sample")
    # Removed evaluation_mode and save_eval_during_training - use sft_eval.py for evaluation
    parser.add_argument("--merge_model", action="store_true", help="Merge LoRA weights and save full model after training")

    # Random seed for reproducibility
    parser.add_argument("--seed", type=int, default=777, help="Random seed for reproducibility (default: 42)")

    args = parser.parse_args()

    # Set random seed for reproducibility
    import random
    import numpy as np
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    print(f"Random seed set to: {args.seed}")

    # Add seed to output_dir if not already present
    if f"seed{args.seed}" not in args.output_dir and f"seed-{args.seed}" not in args.output_dir:
        # Append seed to output directory name
        args.output_dir = f"{args.output_dir}-seed{args.seed}"
        print(f"Output directory updated to include seed: {args.output_dir}")
    
    # Initialize W&B if requested
    if args.use_wandb:
        # Set up W&B config
        wandb_config = {
            "model": args.model_name,
            "learning_rate": args.learning_rate,
            "batch_size": args.batch_size,
            "gradient_accumulation_steps": args.gradient_accumulation_steps,
            "num_epochs": args.num_epochs,
            "max_length": args.max_length,
            "seed": args.seed,
            "lora_r": LORA_CONFIG.r,
            "lora_alpha": LORA_CONFIG.lora_alpha,
            "lora_dropout": LORA_CONFIG.lora_dropout,
            "target_modules": LORA_CONFIG.target_modules,
            "use_4bit": args.use_4bit,
            "use_8bit": args.use_8bit,
            "augment_examples": args.augment_examples,
            "max_augment_factor": args.max_augment_factor,
        }

        # Initialize W&B with seed in run name
        run_name = args.wandb_run_name or f"arc-sft-seed{args.seed}-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
        wandb.init(
            project=args.wandb_project,
            name=run_name,
            config=wandb_config,
            tags=args.wandb_tags or ["arc", "sft", "lora"],
        )
    
    # Load tokenizer
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    
    # Add padding token if not present
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Check GPU availability
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load model with proper configuration
    print("Loading model for training...")
    
    if args.use_4bit:
        print("Using 4-bit quantization")
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16
        )
        model = AutoModelForCausalLM.from_pretrained(
            args.model_name,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
        )
    elif args.use_8bit:
        print("Using 8-bit quantization")
        model = AutoModelForCausalLM.from_pretrained(
            args.model_name,
            load_in_8bit=True,
            device_map="auto",
            trust_remote_code=True,
        )
    else:
        # Load normally without quantization
        model = AutoModelForCausalLM.from_pretrained(
            args.model_name,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
        )
        # Manually move to device for non-quantized models
        model = model.to(device)
    
    # Prepare model for LoRA
    if args.use_8bit or args.use_4bit:
        model = prepare_model_for_kbit_training(model)
    
    model = get_peft_model(model, LORA_CONFIG)
    
    # Enable gradient checkpointing for memory efficiency
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()
    
    if hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()
    
    print_trainable_parameters(model)
    
    # Log model info to W&B
    if args.use_wandb:
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in model.parameters())
        wandb.log({
            "model/trainable_params": trainable_params,
            "model/total_params": total_params,
            "model/trainable_percentage": 100 * trainable_params / total_params
        })
    
    # Load datasets
    print(f"Loading datasets from {args.ground_truth_data_path}")
    train_dataset = prepare_arc_dataset(
        args.ground_truth_data_path, 
        tokenizer, 
        args.max_length,
        augment_examples=args.augment_examples,
        max_augment_factor=args.max_augment_factor
    )
    # Don't augment evaluation dataset, but still handle truncation properly
    
    print(f"Train dataset size: {len(train_dataset)}")
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.num_epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        warmup_steps=100,
        logging_steps=10,
        save_steps=500,
        save_total_limit=3,
        learning_rate=args.learning_rate,
        weight_decay=0.01,
        fp16=False,
        bf16=True if device.type == "cuda" else False,
        max_grad_norm=1.0,
        logging_dir="./logs",
        report_to=["wandb"] if args.use_wandb else [],
        load_best_model_at_end=False,
        remove_unused_columns=False,
        dataloader_num_workers=0,
        save_safetensors=True,
        run_name=args.wandb_run_name if args.use_wandb else None,
    )
    
    # Data collator
    data_collator = CustomDataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
    )
    
    # Initialize trainer with W&B support
    trainer = CustomTrainer(
        use_wandb=args.use_wandb,
        tokenizer=tokenizer,
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=data_collator,
    )
    
    # Start training
    print("Starting training...")
    try:
        trainer.train()
        save_dir = args.output_dir
        os.makedirs(save_dir, exist_ok=True)
        print(f"Saving model to {save_dir}...")
        trainer.save_model(save_dir)
        tokenizer.save_pretrained(save_dir)
        
        print("Training completed!")
        
        # Optional: merge and save the full model (controlled by argument)
        if hasattr(args, 'merge_model') and args.merge_model:
            print("Merging LoRA weights...")
            merged_model = model.merge_and_unload()

            merged_output_dir = save_dir + "_merged"
            os.makedirs(merged_output_dir, exist_ok=True)
            merged_model.save_pretrained(merged_output_dir)
            tokenizer.save_pretrained(merged_output_dir)
            print(f"Merged model saved to {merged_output_dir}")
        
        # Evaluation is now handled by sft_eval.py after training
        
        # Finish W&B run
        if args.use_wandb:
            wandb.finish()
            
    except Exception as e:
        print(f"Training error: {e}")
        if args.use_wandb:
            wandb.finish()
        raise

if __name__ == "__main__":
    main()