import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig
)
from peft import PeftModel
import json
import argparse
import os
import warnings
from tqdm import tqdm
import time
import re
from sft_train import create_input_only_format
from arc_evaluation import evaluate_arc_output, parse_grid_from_text, extract_test_output_grid

# Suppress warnings
warnings.filterwarnings("ignore", message=".*tokenizer.*deprecated.*")
warnings.filterwarnings("ignore", message=".*gradient checkpointing.*")
warnings.filterwarnings("ignore", message=".*past_key_value.*")

# Model configuration
# MODEL_NAME = "openai/gpt-oss-20b" # "nvidia/Mistral-NeMo-Minitron-8B-Base"


def load_model_and_tokenizer(model_name: str, model_path: str, use_4bit: bool = False, use_8bit: bool = False):
    """Load model for evaluation"""
    print(f"Loading model from {model_path} for evaluation...")
    
    # Load tokenizer (use model_name for compatibility; LoRA checkpoints share base tokenizer)
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_path)
    except (AttributeError, Exception):
        print(f"Tokenizer loading from checkpoint failed, falling back to base model: {model_name}")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Check if it's a LoRA adapter or merged model
    adapter_config_path = os.path.join(model_path, "adapter_config.json")
    
    if os.path.exists(adapter_config_path):
        # Load as LoRA model
        print("Loading LoRA adapter...")
        if use_4bit:
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16
            )
            base_model = AutoModelForCausalLM.from_pretrained(
                model_name,
                quantization_config=bnb_config,
                device_map="auto",
                trust_remote_code=True,
            )
        elif use_8bit:
            base_model = AutoModelForCausalLM.from_pretrained(
                model_name,
                load_in_8bit=True,
                device_map="auto",
                trust_remote_code=True,
            )
        else:
            base_model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.bfloat16,
                device_map="auto",
                trust_remote_code=True,
            )
        model = PeftModel.from_pretrained(base_model, model_path)
    else:
        # Load as merged model
        print("Loading merged model...")
        if use_4bit:
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16
            )
            model = AutoModelForCausalLM.from_pretrained(
                model_path,
                quantization_config=bnb_config,
                device_map="auto",
                trust_remote_code=True,
            )
        elif use_8bit:
            model = AutoModelForCausalLM.from_pretrained(
                model_path,
                load_in_8bit=True,
                device_map="auto",
                trust_remote_code=True,
            )
        else:
            model = AutoModelForCausalLM.from_pretrained(
                model_path,
                torch_dtype=torch.bfloat16,
                device_map="auto",
                trust_remote_code=True,
            )
    
    model.eval()
    return model, tokenizer

def generate_evaluation_outputs(model_name, model, tokenizer, input_data_path: str, ground_truth_data_path: str, output_path: str, max_length: int, max_new_tokens: int = 512, evaluation_mode: str = "transduction", model_path: str = "", temperature: float = 0.1):
    """Generate evaluation outputs with prompt template
    
    Args:
        model: The model to evaluate
        tokenizer: The tokenizer
        data_path: Path to evaluation data
        output_path: Path to save results
        max_new_tokens: Maximum new tokens to generate
        evaluation_mode: None, 'induction', 'transduction', or 'both'
        model_path: Path to the model (used to determine if it's a base model)
        temperature: Temperature for generation (default: 0.1)
    """
    model.eval()
    device = next(model.parameters()).device
    
    # Load existing results to check completed (problem_id, test_index) pairs
    completed_tasks = set()
    if os.path.exists(output_path):
        print(f"Found existing output file: {output_path}")
        with open(output_path, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    result = json.loads(line.strip())
                    if 'problem_id' in result and 'test_index' in result:
                        completed_tasks.add((result['problem_id'], result['test_index']))
                except:
                    pass
        print(f"Found {len(completed_tasks)} already completed tasks")
    
    # Open output file in append mode
    output_file = open(output_path, 'a', encoding='utf-8')
    
    eval_outputs = []
    total_time = 0
    skipped_count = 0
    ground_truth_datasets = []
    
    with open(ground_truth_data_path, 'r', encoding='utf-8') as gt_file:
        for line in gt_file:
            try:
                data = json.loads(line.strip())
                ground_truth_datasets.append([data])
            except json.JSONDecodeError:
                print(f"Skipping invalid JSON line in ground truth data: {line.strip()}")
    try:
        # Count total lines for progress bar
        with open(input_data_path, 'r', encoding='utf-8') as f:
            total_lines = sum(1 for _ in f)
        
        with open(input_data_path, 'r', encoding='utf-8') as f:
            for idx, line in enumerate(tqdm(f, total=total_lines, desc="Evaluating", unit="sample")):
                try:
                    data = json.loads(line.strip())
                
                    # Check if this problem was already completed
                    problem_id = data.get("problem_id", None)
                    test_index = data.get("test_index", None)
                    
                    target_ground_truth_dataset = None
                    for ground_truth_dataset in ground_truth_datasets:
                        if ground_truth_dataset[0]["problem_id"] == problem_id and ground_truth_dataset[0]["test_index"] == test_index:
                            target_ground_truth_dataset = ground_truth_dataset[0]
                            break
                            
                    if problem_id and test_index is not None and (problem_id, test_index) in completed_tasks:
                        skipped_count += 1
                        continue
                    
                    if "source" in data and "examples" in data:
                        # Create input-only prompt with template
                        input_prompt = create_input_only_format(data)
                        
                        # Tokenize input
                        inputs = tokenizer(
                            input_prompt,
                            return_tensors="pt",
                            truncation=True,
                            max_length=max_length, #6144,
                            padding=False
                        )
                        
                        # Remove token_type_ids if present (not used by this model)
                        if "token_type_ids" in inputs:
                            del inputs["token_type_ids"]
                        
                        # Move to device
                        inputs = {k: v.to(device) for k, v in inputs.items()}
                        
                        # Generate output
                        start_time = time.time()
                        with torch.no_grad():
                            outputs = model.generate(
                                **inputs,
                                max_new_tokens=max_new_tokens, #50000, #2048,
                                temperature=temperature,
                                do_sample=True if temperature > 0 else False,
                                pad_token_id=tokenizer.eos_token_id,
                                eos_token_id=tokenizer.eos_token_id,
                                # repetition_penalty=1.2, 
                                # no_repeat_ngram_size=3,
                            )
                        generation_time = time.time() - start_time
                        total_time += generation_time
                        
                        # Decode output
                        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
                        
                        # Remove input prompt from output
                        if generated_text.startswith(input_prompt):
                            generated_output = generated_text[len(input_prompt):].strip()
                        else:
                            generated_output = generated_text
                        
                        # Use ground_truth_output from data if available
                        ground_truth = ""
                        if "ground_truth_output" in data:
                            ground_truth = str(data["ground_truth_output"])
                        elif "examples" in data and data["examples"]:
                            last_example = data["examples"][-1]
                            if len(last_example) >= 2:
                                ground_truth = str(last_example[1])
                        
                        result_dict = {
                            "idx": idx,
                            "problem_id": data.get("problem_id", None),
                            "test_index": data.get("test_index", None),
                            "input_prompt": input_prompt,
                            "generated_output": generated_output,
                            "ground_truth_response": ground_truth,
                            "generation_time": generation_time,
                            "original_data": data
                        }
                        
                        # Perform evaluation if requested
                        if evaluation_mode and "examples" in data and data["examples"]:
                            try:
                                # Get test input and output
                                test_input = data["examples"][-1][0]
                                
                                # Always use ground_truth_output from data if available
                                if "ground_truth_output" in data:
                                    test_output = data["ground_truth_output"]
                                else:
                                    test_output = data["examples"][-1][1] if len(data["examples"][-1]) > 1 else None
                                
                                if test_output:
                                    # Check if this is transduction format (no sections expected)
                                    # Transduction format: directly outputs grid without headers
                                    user_content = data.get("source", "")
                                    is_transduction_format = (
                                        "Directly provide the output grids" in user_content or
                                        (evaluation_mode == "transduction")  # If explicitly transduction mode
                                    )
                                    
                                    if is_transduction_format:
                                        # For transduction, evaluate directly without checking sections
                                        missing_sections = []
                                    else:
                                        # Check if generated output contains the required sections
                                        missing_sections = []
                                        if "# SOLUTION CODE" not in generated_output:
                                            missing_sections.append("SOLUTION CODE")
                                        if "# RESULT TEST OUTPUT GRID" not in generated_output:
                                            missing_sections.append("RESULT TEST OUTPUT GRID")
                                    
                                    if missing_sections:
                                        # Set appropriate results to -1 based on missing sections
                                        if "SOLUTION CODE" in missing_sections:
                                            result_dict["induction_result"] = -1
                                            # Don't set induction_output (omit it)
                                        if "RESULT TEST OUTPUT GRID" in missing_sections:
                                            result_dict["transduction_result"] = -1
                                            # Don't set transduction_output (omit it)
                                        result_dict["evaluation_error"] = f"Missing sections: {', '.join(missing_sections)}"
                                        result_dict["ground_truth_output"] = test_output
                                    else:
                                        device = next(model.parameters()).device
                                        eval_results = evaluate_arc_output(
                                            llm_output=generated_output,
                                            test_input=test_input,
                                            test_output=test_output,
                                            model=model if evaluation_mode == "both" else None,
                                            tokenizer=tokenizer if evaluation_mode == "both" else None,
                                            device=device if evaluation_mode == "both" else None,
                                            evaluation_mode=evaluation_mode
                                        )
                                        # Add evaluation results in compatible format
                                        if "induction" in eval_results:
                                            ind_res = eval_results["induction"]
                                            result_dict["induction_output"] = ind_res.get("predicted_grid", None)
                                            result_dict["induction_result"] = ind_res.get("result", 0)
                                        
                                        # Add ground truth output
                                        result_dict["ground_truth_output"] = test_output
                                        
                                        if "transduction" in eval_results:
                                            trans_res = eval_results["transduction"]
                                            result_dict["transduction_output"] = trans_res.get("predicted_grid", None)
                                            result_dict["transduction_result"] = trans_res.get("result", 0)
                                        
                                        if "negative_log_probability_correct" in eval_results:
                                            result_dict["negative_log_probability_correct"] = eval_results["negative_log_probability_correct"]
                                        
                                        # Add nlp_debug if present
                                        if "nlp_debug" in eval_results:
                                            result_dict["nlp_prompt_until_grid"] = eval_results["nlp_debug"]["nlp_prompt_until_grid"]
                                            result_dict["nlp_target_grid"] = eval_results["nlp_debug"]["nlp_target_grid"]
                                        
                                        # Keep full evaluation for debugging
                                        result_dict["evaluation_details"] = eval_results
                            except Exception as e:
                                result_dict["evaluation_error"] = str(e)
                                tqdm.write(f"Evaluation error for sample {idx}: {e}")
                        
                        # Write result immediately to file
                        output_file.write(json.dumps(result_dict, ensure_ascii=False) + '\n')
                        output_file.flush()  # Ensure it's written to disk
                        eval_outputs.append(result_dict)
                        
                        # Update progress bar with time info
                        tqdm.write(f"Sample {idx+1}: {generation_time:.2f}s")
                        
                        # Print first few for monitoring
                        if idx < 3:
                            tqdm.write(f"\n=== Example {idx+1} ===")
                            tqdm.write(f"Input prompt: {input_prompt[:200]}...")
                            tqdm.write(f"Generated output: {generated_output[:300]}...")
                            tqdm.write(f"Generation time: {generation_time:.2f}s")
                            tqdm.write("="*50)
                    elif "messages" in data:
                        # Fallback for old chat format
                        prompt = ""
                        ground_truth_response = ""
                        
                        for msg in data["messages"]:
                            if msg["role"] == "system":
                                prompt += f"System: {msg['content']}\n\n"
                            elif msg["role"] == "user":
                                prompt += f"Human: {msg['content']}\n\n"
                        
                        for msg in target_ground_truth_dataset["answer_messages"]:
                            if msg["role"] == "assistant":
                                ground_truth_response = msg['content']
                        
                        # Add format guide only for base models
                        is_base_model = model_path == model_name
                        
                        if is_base_model:
                            format_guide = """\n\nCRITICAL INSTRUCTIONS - YOU MUST FOLLOW EXACTLY:

1. Start your response with: # RESULT TEST OUTPUT GRID
2. Next line must be: The output grid for the test input grid is:
3. Next line must be: ```
4. Then write the grid using ONLY these color names: Black, Blue, Red, Green, Yellow, Gray, Pink, Orange, Purple, Brown
5. End with: ```

DO NOT write explanations, steps, or analysis. ONLY output the grid.

CORRECT EXAMPLE:
# RESULT TEST OUTPUT GRID
The output grid for the test input grid is:

```
Purple Purple Orange Purple Purple
Purple Orange Orange Orange Purple
Purple Purple Purple Purple Purple
Orange Orange Orange Orange Orange
```

WRONG EXAMPLE (DO NOT DO THIS):
1. Analyze the patterns...
2. Identify the rules...

START YOUR RESPONSE NOW WITH:
# RESULT TEST OUTPUT GRID

"""
                            prompt += format_guide + "Assistant: "
                        else:
                            prompt += "Assistant: "
                    else:
                        prompt += "Assistant: "
                    
                    # Generate response
                    inputs = tokenizer(
                        prompt, 
                        return_tensors="pt", 
                        truncation=True, 
                        max_length=max_length #6144
                    )
                    
                    # Remove token_type_ids if present (not used by this model)
                    if "token_type_ids" in inputs:
                        del inputs["token_type_ids"]
                    
                    # Move to device
                    inputs = {k: v.to(device) for k, v in inputs.items()}
                    
                    start_time = time.time()
                    with torch.no_grad():
                        outputs = model.generate(
                            **inputs,
                            max_new_tokens=max_new_tokens,#2048,
                            temperature=temperature,
                            do_sample=True if temperature > 0 else False,
                            pad_token_id=tokenizer.eos_token_id,
                            eos_token_id=tokenizer.eos_token_id,
                            # repetition_penalty=1.2, 
                            # no_repeat_ngram_size=3,
                        )
                    generation_time = time.time() - start_time
                    total_time += generation_time
                    
                    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
                    generated_output = generated_text.split("Assistant: ")[-1].strip()
                    
                    result_dict = {
                        "idx": idx,
                        "problem_id": problem_id,
                        "test_index": test_index,
                        "input_prompt": prompt,
                        "generated_output": generated_output,
                        "ground_truth_response": ground_truth_response,
                        "generation_time": generation_time,
                    }
                    
                    # Add ground_truth_output to result if available
                    if "ground_truth_output" in target_ground_truth_dataset:
                        result_dict["ground_truth_output"] = target_ground_truth_dataset["ground_truth_output"]
                    
                    # Perform evaluation if requested (for messages format)
                    if evaluation_mode:
                        try:
                            # Extract test grids from user message
                            test_input = None
                            test_output = None
                            
                            # If ground_truth_output is provided in data, use it
                            if "ground_truth_output" in target_ground_truth_dataset:
                                test_output = target_ground_truth_dataset.get("ground_truth_output", "")
                            
                            # Find user message content
                            user_content = ""
                            for msg in data["messages"]:
                                if msg["role"] == "user":
                                    user_content = msg["content"]
                                    break
                            
                            if user_content:
                                # Extract test input from "Here is the input grid for the test example:"
                                test_input_match = re.search(r'Here is the input grid for the test example:\s*\nInput:\s*\n((?:[^\n]+\n)*)', user_content)
                                if not test_input_match:
                                    # Fallback to other patterns
                                    test_input_match = re.search(r'Test Input:\s*\n([^\n]+(?:\n[^\n]+)*?)(?:\n\n|$)', user_content)
                                if test_input_match:
                                    test_input_text = test_input_match.group(1)
                                    test_input = parse_grid_from_text(test_input_text)
                                
                                # Skip trying to extract ground truth from examples
                                # We already have it from data["ground_truth_output"] above
                                
                            if test_input:
                                # Check if this is transduction format (no sections expected)
                                # Transduction format: directly outputs grid without headers
                                is_transduction_format = (
                                    "Directly provide the output grids" in user_content or
                                    "direct solutions" in data.get("messages", [{}])[0].get("content", "")
                                )
                                
                                if is_transduction_format:
                                    # For transduction, evaluate directly without checking sections
                                    missing_sections = []
                                else:
                                    # Check if generated output contains the required sections
                                    missing_sections = []
                                    if "# SOLUTION CODE" not in generated_output:
                                        missing_sections.append("SOLUTION CODE")
                                    if "# RESULT TEST OUTPUT GRID" not in generated_output:
                                        missing_sections.append("RESULT TEST OUTPUT GRID")
                                
                                if missing_sections:
                                    # Set appropriate results to -1 based on missing sections
                                    if "SOLUTION CODE" in missing_sections:
                                        result_dict["induction_result"] = -1
                                        # Don't set induction_output (omit it)
                                    if "RESULT TEST OUTPUT GRID" in missing_sections:
                                        result_dict["transduction_result"] = -1
                                        # Don't set transduction_output (omit it)
                                    result_dict["evaluation_error"] = f"Missing sections: {', '.join(missing_sections)}"
                                    result_dict["ground_truth_output"] = test_output
                                else:
                                    # Only try to extract from ground_truth_response if we don't have ground_truth_output
                                    if not test_output:
                                        test_output_parsed = extract_test_output_grid(ground_truth_response)
                                        if test_output_parsed:
                                            test_output = test_output_parsed
                                        else:
                                            test_output = test_input  # Fallback
                                    
                                    # Even without perfect test_output, we can still evaluate
                                    device = next(model.parameters()).device
                                    eval_results = evaluate_arc_output(
                                        llm_output=generated_output,
                                        test_input=test_input,
                                        test_output=test_output,
                                        model=model,
                                        tokenizer=tokenizer,
                                        device=device,
                                        evaluation_mode=evaluation_mode
                                    )
                                    
                                    # Add evaluation results
                                    if "induction" in eval_results:
                                        ind_res = eval_results["induction"]
                                        result_dict["induction_output"] = ind_res.get("predicted_grid", None)
                                        result_dict["induction_result"] = ind_res.get("result", 0)
                                    else:
                                        result_dict["induction_output"] = None
                                        result_dict["induction_result"] = 0
                                    
                                    # Add ground truth output if available
                                    if test_output:
                                        result_dict["ground_truth_output"] = test_output
                                    
                                    if "transduction" in eval_results:
                                        trans_res = eval_results["transduction"]
                                        result_dict["transduction_output"] = trans_res.get("predicted_grid", None)
                                        result_dict["transduction_result"] = trans_res.get("result", 0)
                                    else:
                                        result_dict["transduction_output"] = None
                                        result_dict["transduction_result"] = 0
                                    
                                    if "negative_log_probability_correct" in eval_results:
                                        result_dict["negative_log_probability_correct"] = eval_results["negative_log_probability_correct"]
                                    
                                    # Add nlp_debug if present
                                    if "nlp_debug" in eval_results:
                                        result_dict["nlp_prompt_until_grid"] = eval_results["nlp_debug"]["nlp_prompt_until_grid"]
                                        result_dict["nlp_target_grid"] = eval_results["nlp_debug"]["nlp_target_grid"]
                                    
                                    # Keep full evaluation for debugging
                                    result_dict["evaluation_details"] = eval_results
                            else:
                                # Could not parse test input
                                result_dict["induction_output"] = None
                                result_dict["induction_result"] = 0
                                result_dict["transduction_output"] = None
                                result_dict["transduction_result"] = 0
                                result_dict["evaluation_error"] = "Could not parse test input from messages"
                                
                                # Still try to calculate negative log probability even without test input
                                if evaluation_mode == "both":
                                    try:
                                        device = next(model.parameters()).device
                                        # Try to extract expected grid from ground_truth_response
                                        expected_grid = extract_test_output_grid(ground_truth_response)
                                        if expected_grid:
                                            # Use actual expected grid
                                            test_output_for_nlp = expected_grid
                                        else:
                                            # Fallback to dummy
                                            test_output_for_nlp = [[0,0,0],[0,0,0]]
                                        
                                        dummy_test = [[0,0,0],[0,0,0]]  # Still dummy for test input
                                        eval_results = evaluate_arc_output(
                                            llm_output=generated_output,
                                            test_input=dummy_test,
                                            test_output=test_output_for_nlp,
                                            model=model,
                                            tokenizer=tokenizer,
                                            device=device,
                                            evaluation_mode="both"
                                        )
                                        if "negative_log_probability_correct" in eval_results:
                                            result_dict["negative_log_probability_correct"] = eval_results["negative_log_probability_correct"]
                                    except Exception as e:
                                        print(f"Warning: Failed to calculate negative_log_probability for sample {idx}: {e}")
                                        pass  # Continue without negative_log_probability
                        except Exception as e:
                            result_dict["evaluation_error"] = str(e)
                            result_dict["induction_output"] = None
                            result_dict["induction_result"] = 0
                            result_dict["transduction_output"] = None
                            result_dict["transduction_result"] = 0
                    
                    # Write result immediately to file
                    output_file.write(json.dumps(result_dict, ensure_ascii=False) + '\n')
                    output_file.flush()  # Ensure it's written to disk
                    eval_outputs.append(result_dict)
                    
                    # Update progress bar with time info
                    tqdm.write(f"Sample {idx+1}: {generation_time:.2f}s")
                        
                except Exception as e:
                    print(f"Error processing evaluation line {idx}: {e}")
                    continue
    
    finally:
        # Always close output file
        output_file.close()
        
        print(f"\nEvaluation completed! Outputs saved to {output_path}")
        print(f"Generated {len(eval_outputs)} new evaluation outputs")
        if skipped_count > 0:
            print(f"Skipped {skipped_count} already completed tasks")
        print(f"Total generation time: {total_time:.2f}s")
        print(f"Average time per sample: {total_time/len(eval_outputs):.2f}s" if eval_outputs else "N/A")
    
    # Print evaluation summary if available
    if evaluation_mode and eval_outputs:
        induction_correct = sum(1 for r in eval_outputs if r.get("induction_result", 0) == 1)
        transduction_correct = sum(1 for r in eval_outputs if r.get("transduction_result", 0) == 1)
        
        print(f"\nEvaluation Results:")
        if evaluation_mode in ["induction", "both"]:
            print(f"Induction accuracy: {induction_correct}/{len(eval_outputs)} ({100*induction_correct/len(eval_outputs):.1f}%)")
        if evaluation_mode in ["transduction", "both"]:
            print(f"Transduction accuracy: {transduction_correct}/{len(eval_outputs)} ({100*transduction_correct/len(eval_outputs):.1f}%)")
    
    return eval_outputs

def main():
    parser = argparse.ArgumentParser(description="ARC Model Evaluation")
    parser.add_argument("--model_name", type=str, default="nvidia/Mistral-NeMo-Minitron-8B-Base", help="Model name")
    parser.add_argument("--model_path", type=str, required=True, help="Path to trained model")
    parser.add_argument("--input_data_path", type=str, default="input_final_gifarc_output_100.jsonl", help="Evaluation data file")
    parser.add_argument("--ground_truth_data_path", type=str, default="ground_truth_final_gifarc_output_100.jsonl", help="Evaluation data file")
    parser.add_argument("--output_path", type=str, default="evaluation_outputs.jsonl", help="Output file for results")
    parser.add_argument("--max_new_tokens", type=int, default=1536, help="Maximum new tokens for generation")
    parser.add_argument("--max_length", type=int, default=6144, help="Maximum new tokens for generation")
    parser.add_argument("--use_4bit", action="store_true", help="Use 4-bit quantization")
    parser.add_argument("--use_8bit", action="store_true", help="Use 8-bit quantization")
    parser.add_argument("--evaluation_mode", type=str, choices=["induction", "transduction", "both"], default="transduction", help="Evaluation mode (default: transduction)")
    parser.add_argument("--temperature", type=float, default=0.1, help="Temperature for generation (default: 0.1)")
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
    print(f"Evaluation random seed set to: {args.seed}")
    
    # Load model and tokenizer
    model, tokenizer = load_model_and_tokenizer(
        args.model_name,
        args.model_path,
        use_4bit=args.use_4bit,
        use_8bit=args.use_8bit
    )
    
    # Generate evaluation outputs
    generate_evaluation_outputs(
        model_name=args.model_name,
        model=model,
        tokenizer=tokenizer,
        input_data_path=args.input_data_path,
        ground_truth_data_path=args.ground_truth_data_path,
        output_path=args.output_path,
        max_length=args.max_length,
        max_new_tokens=args.max_new_tokens,
        evaluation_mode=args.evaluation_mode,
        model_path=args.model_path,
        temperature=args.temperature
    )

if __name__ == "__main__":
    main()