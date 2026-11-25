import numpy as np
import re
import sys
import os
import signal
import traceback
from contextlib import contextmanager
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
import torch
from transformers import AutoTokenizer

# Resolve the shared support library from the repository layout.
sys.path.append(str(Path(__file__).resolve().parents[2] / "src"))

# Color mappings
COLOR_MAPPING = {
    0: "Black",
    1: "Blue", 
    2: "Red",
    3: "Green",
    4: "Yellow",
    5: "Gray",
    6: "Pink",
    7: "Orange",
    8: "Purple",
    9: "Brown"
}

REVERSED_MAPPING = {v: k for k, v in COLOR_MAPPING.items()}

# Handle color variations
COLOR_REPLACEMENTS = {
    "Grey": "Gray",
    "Teal": "Purple",
    "Maroon": "Brown",
}

class TimeoutException(Exception):
    pass

@contextmanager
def time_limit(seconds):
    """Context manager for setting time limits"""
    def signal_handler(signum, frame):
        raise TimeoutException("Code execution timed out")
    
    # Set the signal handler and alarm
    signal.signal(signal.SIGALRM, signal_handler)
    signal.alarm(seconds)
    try:
        yield
    finally:
        # Disable the alarm
        signal.alarm(0)

def parse_grid_from_text(text: str) -> Optional[List[List[int]]]:
    """Parse grid from LLM output text"""
    # Find the grid pattern - lines with color names
    lines = text.strip().split('\n')
    grid_lines = []
    
    for line in lines:
        # Check if line contains color names
        if any(color in line for color in COLOR_MAPPING.values()):
            grid_lines.append(line)
    
    if not grid_lines:
        return None
    
    grid = []
    for line in grid_lines:
        row = []
        # Split by spaces and process each token
        tokens = line.split()
        
        for token in tokens:
            # Clean token (remove punctuation)
            token = token.strip('.,;:')
            
            # Handle color replacements
            if token in COLOR_REPLACEMENTS:
                token = COLOR_REPLACEMENTS[token]
            
            # Convert to number
            if token in REVERSED_MAPPING:
                row.append(REVERSED_MAPPING[token])
        
        if row:  # Only add non-empty rows
            grid.append(row)
    
    return grid if grid else None

def extract_solution_code(llm_output: str) -> Optional[str]:
    """Extract solution code from LLM output"""
    # Look for code block after "# SOLUTION CODE"
    solution_marker = "# SOLUTION CODE"
    
    if solution_marker not in llm_output:
        return None
    
    # Get text after solution marker
    after_marker = llm_output.split(solution_marker, 1)[1]
    
    # Find code block (```python ... ```)
    code_pattern = r'```python\n(.*?)\n```'
    matches = re.findall(code_pattern, after_marker, re.DOTALL)
    
    if matches:
        return matches[0]
    
    # Alternative: Find transform function
    transform_pattern = r'def transform\(.*?\):(.*?)(?=\n(?:def|class|#|$))'
    matches = re.findall(transform_pattern, after_marker, re.DOTALL)
    
    if matches:
        return f"def transform(input_grid):{matches[0]}"
    
    return None

def extract_test_output_grid(llm_output: str) -> Optional[List[List[int]]]:
    """Extract test output grid from LLM output"""
    # Look for grid after "# RESULT TEST OUTPUT GRID" (induction format)
    grid_marker = "# RESULT TEST OUTPUT GRID"

    # Also support transduction format: "The output grid for the test input grid is:"
    transduction_marker = "The output grid for the test input grid is:"

    if grid_marker in llm_output:
        after_marker = llm_output.split(grid_marker, 1)[1]
    elif transduction_marker in llm_output:
        after_marker = llm_output.split(transduction_marker, 1)[1]
    else:
        return None

    # Check if grid is in code block
    if "```" in after_marker:
        # Extract content between first ``` and second ```
        parts = after_marker.split("```")
        if len(parts) >= 2:
            grid_text = parts[1]
            # Remove language identifier if present (e.g., "python")
            lines = grid_text.strip().split('\n')
            if lines and not any(color in lines[0] for color in COLOR_MAPPING.values()):
                grid_text = '\n'.join(lines[1:])
        else:
            grid_text = after_marker
    else:
        # Look for next section marker or end
        end_markers = ["# RESULT TEST OUTPUT GRID NAME", "# RESULT TEST OUTPUT GRID DESCRIPTION", "#"]

        grid_text = after_marker
        for marker in end_markers:
            if marker in grid_text:
                grid_text = grid_text.split(marker)[0]
                break

    return parse_grid_from_text(grid_text)

def execute_solution_code(code: str, test_input: List[List[int]], timeout_seconds: int = 180) -> Union[List[List[int]], int]:
    """Execute solution code with timeout"""
    try:
        with time_limit(timeout_seconds):
            # Create namespace with seeds.common
            namespace = {}
            
            # Import seeds.common
            exec("from seeds.common import *", namespace)
            exec("import numpy as np", namespace)
            
            # Convert input to numpy array
            input_array = np.array(test_input)
            
            # Execute the code
            exec(code, namespace)
            
            # Call transform function
            if 'transform' in namespace:
                result = namespace['transform'](input_array)
                
                # Convert result to list
                if isinstance(result, np.ndarray):
                    return result.tolist()
                else:
                    return result
            else:
                print("Warning: transform function not found in code")
                return -1
                
    except TimeoutException:
        print(f"Code execution timed out after {timeout_seconds} seconds")
        return -1
    except Exception as e:
        print(f"Error executing code: {str(e)}")
        traceback.print_exc()
        return -1

def compare_grids(predicted: List[List[int]], actual: List[List[int]]) -> float:
    """Compare two grids and return accuracy"""
    if not predicted or not actual:
        return 0.0

    # Check dimensions
    if len(predicted) != len(actual):
        return 0.0

    total_cells = 0
    correct_cells = 0

    for i in range(len(actual)):
        if len(predicted[i]) != len(actual[i]):
            # Row length mismatch
            return 0.0

        for j in range(len(actual[i])):
            total_cells += 1
            if predicted[i][j] == actual[i][j]:
                correct_cells += 1

    return correct_cells / total_cells if total_cells > 0 else 0.0

def compare_grids_foreground(predicted: List[List[int]], actual: List[List[int]], input_grid: List[List[int]]) -> Optional[float]:
    """Compare only foreground cells (where actual != input_grid).

    For cells within the overlapping region of input and actual grids,
    foreground = cells where actual[i][j] != input[i][j].
    For cells outside the input grid bounds, they are always foreground.
    """
    if not actual:
        return None

    # Count foreground cells in actual grid
    def count_fg():
        total = 0
        for i in range(len(actual)):
            for j in range(len(actual[i])):
                in_val = input_grid[i][j] if (input_grid and i < len(input_grid) and j < len(input_grid[i])) else -1
                if actual[i][j] != in_val:
                    total += 1
        return total

    # If predicted grid is missing or has wrong dimensions, all foreground cells are wrong
    if not predicted or len(predicted) != len(actual):
        fg_total = count_fg()
        return 0.0 if fg_total > 0 else None

    foreground_total = 0
    foreground_correct = 0

    for i in range(len(actual)):
        if len(predicted[i]) != len(actual[i]):
            return 0.0  # Row length mismatch in prediction

        for j in range(len(actual[i])):
            # Compare element-by-element; out-of-bounds input = foreground
            in_val = input_grid[i][j] if (input_grid and i < len(input_grid) and j < len(input_grid[i])) else -1

            if actual[i][j] != in_val:
                foreground_total += 1
                if predicted[i][j] == actual[i][j]:
                    foreground_correct += 1

    return foreground_correct / foreground_total if foreground_total > 0 else None

def evaluate_induction(llm_output: str, test_input: List[List[int]], test_output: List[List[int]]) -> Dict:
    """Evaluate using induction method (execute code)"""
    # Extract solution code
    code = extract_solution_code(llm_output)
    
    if not code:
        return {
            "predicted_grid": None,
            "result": 0,  # Failed to extract code
            "error": "Could not extract solution code",
            "accuracy": 0.0
        }
    
    # Execute code
    predicted = execute_solution_code(code, test_input)
    
    if predicted == -1:
        return {
            "predicted_grid": None,
            "result": -1,  # Timeout or execution error
            "error": "Code execution failed or timed out",
            "accuracy": 0.0
        }
    
    # Compare grids
    accuracy = compare_grids(predicted, test_output)
    
    return {
        "predicted_grid": predicted,
        "result": 1 if accuracy == 1.0 else 0,  # 1 if perfect match, 0 otherwise
        "accuracy": accuracy
    }

def parse_bare_grid(text: str) -> Optional[List[List[int]]]:
    """Accept a standalone grid without treating prose that names colors as a grid."""
    lines = text.strip().splitlines()
    if len(lines) >= 3 and lines[0].strip() in ("```", "```text") and lines[-1].strip() == "```":
        lines = lines[1:-1]
    rows = [line.split() for line in lines]
    if not rows or not rows[0] or any(len(row) != len(rows[0]) for row in rows):
        return None
    rows = [[COLOR_REPLACEMENTS.get(token, token) for token in row] for row in rows]
    if any(token not in REVERSED_MAPPING for row in rows for token in row):
        return None
    return [[REVERSED_MAPPING[token] for token in row] for row in rows]


def evaluate_transduction(llm_output: str, test_output: List[List[int]], test_input: List[List[int]] = None) -> Dict:
    """Evaluate using transduction method (direct grid parsing)"""
    # Extract predicted grid
    predicted = extract_test_output_grid(llm_output) or parse_bare_grid(llm_output)

    if not predicted:
        result = {
            "predicted_grid": None,
            "result": 0,  # Failed to extract grid
            "error": "Could not extract test output grid",
            "accuracy": 0.0
        }
        if test_input is not None:
            result["foreground_accuracy"] = None
        return result

    # Compare grids
    accuracy = compare_grids(predicted, test_output)

    result = {
        "predicted_grid": predicted,
        "result": 1 if accuracy == 1.0 else 0,  # 1 if perfect match, 0 otherwise
        "accuracy": accuracy
    }

    if test_input is not None:
        result["foreground_accuracy"] = compare_grids_foreground(predicted, test_output, test_input)

    return result

def calculate_conditional_probability(model, tokenizer, prompt: str, target_text: str, device) -> float:
    """Calculate -log P(target_text | prompt) which is the cross-entropy"""
    # Prepare full text
    full_text = prompt + target_text
    
    # Tokenize
    prompt_tokens = tokenizer(prompt, return_tensors="pt", return_token_type_ids=False).to(device)
    full_tokens = tokenizer(full_text, return_tensors="pt", return_token_type_ids=False).to(device)
    
    # Get model outputs
    model.eval()
    with torch.no_grad():
        outputs = model(**full_tokens)
        logits = outputs.logits
    
    # Calculate cross-entropy for target tokens only
    prompt_length = prompt_tokens['input_ids'].shape[1]
    target_length = full_tokens['input_ids'].shape[1] - prompt_length
    
    if target_length <= 0:
        return float('inf')
    
    # Shift logits and labels for next token prediction
    shift_logits = logits[..., prompt_length-1:-1, :].contiguous()
    shift_labels = full_tokens['input_ids'][..., prompt_length:].contiguous()
    
    # Calculate cross-entropy
    loss_fct = torch.nn.CrossEntropyLoss(reduction='sum')
    loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
    
    # Return negative log probability (cross-entropy)
    return loss.item() 

def evaluate_arc_output(
    llm_output: str,
    test_input: List[List[int]], 
    test_output: List[List[int]],
    model=None,
    tokenizer=None,
    device=None,
    evaluation_mode="transduction"  # "induction", "transduction", or "both"
) -> Dict:
    """
    Comprehensive ARC evaluation
    
    Args:
        llm_output: Full LLM response
        test_input: Input grid as list of lists
        test_output: Expected output grid as list of lists
        model: Model for probability calculation (optional)
        tokenizer: Tokenizer for probability calculation (optional)
        device: Device for model (optional)
        evaluation_mode: Which evaluation methods to use
        
    Returns:
        Dictionary with evaluation results
    """
    results = {}
    
    # Induction evaluation
    if evaluation_mode in ["induction", "both"]:
        results["induction"] = evaluate_induction(llm_output, test_input, test_output)
    
    # Transduction evaluation
    if evaluation_mode in ["transduction", "both"]:
        results["transduction"] = evaluate_transduction(llm_output, test_output, test_input)
    
    # Calculate conditional probability if model provided
    if model and tokenizer and device and test_output is not None:
        # Find where the actual grid generation should start
        # The prompt should end after "The output grid for the test input grid is:\n```\n"
        grid_start_marker = "The output grid for the test input grid is:"
        
        if grid_start_marker in llm_output:
            # Split at the marker and include it plus the code block start
            prompt_parts = llm_output.split(grid_start_marker)
            prompt = prompt_parts[0] + grid_start_marker + "\n```\n"
            
            # Convert actual test output grid to string format
            target_lines = []
            for row in test_output:
                row_colors = [COLOR_MAPPING.get(cell, str(cell)) for cell in row]
                target_lines.append(" ".join(row_colors))
            target = "\n".join(target_lines)
            
            try:
                cross_entropy = calculate_conditional_probability(
                    model, tokenizer, prompt, target, device
                )
                results["negative_log_probability_correct"] = cross_entropy
                results["perplexity"] = np.exp(cross_entropy)
                
                # Store full prompt and target for debugging
                results["nlp_debug"] = {
                    "nlp_prompt_until_grid": prompt,
                    "nlp_target_grid": target,
                    "prompt_length": len(prompt),
                    "target_length": len(target)
                }
            except Exception as e:
                print(f"Error calculating probability: {e}")
                results["negative_log_probability_correct"] = None
                results["perplexity"] = None
        else:
            # Fallback to old method if new marker not found
            grid_marker = "# RESULT TEST OUTPUT GRID"
            if grid_marker in llm_output:
                prompt = llm_output.split(grid_marker)[0] + grid_marker + "\n"
                
                # Convert actual test output grid to string format
                target_lines = []
                for row in test_output:
                    row_colors = [COLOR_MAPPING.get(cell, str(cell)) for cell in row]
                    target_lines.append(" ".join(row_colors))
                target = "\n".join(target_lines)
                
                try:
                    cross_entropy = calculate_conditional_probability(
                        model, tokenizer, prompt, target, device
                    )
                    results["negative_log_probability_correct"] = cross_entropy
                    results["perplexity"] = np.exp(cross_entropy)
                    
                    # Store full prompt and target for debugging
                    results["nlp_debug"] = {
                        "nlp_prompt_until_grid": prompt,
                        "nlp_target_grid": target,
                        "prompt_length": len(prompt),
                        "target_length": len(target),
                        "fallback": True
                    }
                except Exception as e:
                    print(f"Error calculating probability: {e}")
                    results["negative_log_probability_correct"] = None
                    results["perplexity"] = None
    
    return results