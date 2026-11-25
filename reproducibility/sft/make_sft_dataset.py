import os
import re
import io
import csv
import json
import random
import argparse
import numpy as np
import tiktoken
import traceback
from tqdm import tqdm
from pathlib import Path

# Try to import arc library - make it optional
try:
    from datasets import Dataset
    from arc.types import ArcProblem, ArcIOPair
    from arc import train_problems, validation_problems
    HAS_ARC_LIB = True
except ImportError:
    HAS_ARC_LIB = False
    print("Warning: arc library not found. Only external data files will be supported.")

# Try to import prompt module - make it optional
try:
    from prompt import get_common_lib_from_file
    HAS_PROMPT_LIB = True
except ImportError:
    HAS_PROMPT_LIB = False
    print("Warning: prompt module not found. Common library will not be available.")


ANALOGY_PROCESS_CODE = """ # THINKING ANALOGY
[Let's think step by step about the analogical pattern in the given task, and describe what kind of analogical, metaphorical, or figurative pattern it is]

{analogy}

# SOLUTION CODE
[Write the final solution code for the given task based on the analogical approach identified in the thinking analogy step]

{solution_code}

# RESULT TEST OUTPUT GRID
[Write the final output grid for the given task based on the solution code and the test input grid]

{test_output_grid}
"""

VERSION = "0.3"
EXTRA_NEWLINE = "\n"

COLOR_MAPPING = {
    0: "Black",
    1: "Blue",
    2: "Red",
    3: "Green",
    4: "Yellow",
    5: "Grey",
    6: "Pink",
    7: "Orange",
    8: "Teal",
    9: "Maroon"
}

COLOR_REPLACEMENTS = {
    "Grey": "Gray",
    "Teal": "Purple",
    "Maroon": "Brown",
}

# Fix Color Mapping
for k, v in COLOR_MAPPING.items():
    if v in COLOR_REPLACEMENTS:
        COLOR_MAPPING[k] = COLOR_REPLACEMENTS[v]


DEFAULT_SYSTEM_PROMPT = "You are a world-class puzzle solver with exceptional pattern recognition skills and expertise in Python programming. Your task is to analyze puzzles and provide Python solutions."
DEFAULT_SYSTEM_PROMPT_TRANSDUCTION = "You are a world-class puzzle solver with exceptional pattern recognition skills. Your task is to analyze puzzles, spot patterns, and provide direct solutions."


def log_skipped_file(path, reason, flag, tb_str="", filename="skipped_files.csv"):
    write_header = not os.path.exists(filename) or os.stat(filename).st_size == 0
    with open(filename, "a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["path", "reason", "flag", "traceback"])
        if write_header:
            writer.writeheader()
        writer.writerow({
            "path": path,
            "reason": reason,
            "flag": flag,
            "traceback": tb_str.replace("\n", "\\n")
        })


def color_deterministic(problem_source_code, old_color, new_color):
    """Map a hard coded color to a deterministic some other color in source code, keeping cases same"""
    upper_template = f"(((?<=[^a-zA-Z])|^)({old_color.upper()})(?=[^a-zA-Z]|$))"
    capitalized_template = (
        f"(((?<=[^a-zA-Z])|^)({old_color.lower().capitalize()})(?=[^a-zA-Z]|$))"
    )
    lower_template = f"(((?<=[^a-zA-Z])|^)({old_color.lower()})(?=[^a-zA-Z]|$))"

    upper_regex = re.compile(upper_template)
    capitalized_regex = re.compile(capitalized_template)
    lower_regex = re.compile(lower_template)

    replace_upper = re.sub(
        upper_regex, lambda x: new_color.upper(), problem_source_code
    )
    replace_capitalized = re.sub(
        capitalized_regex,
        lambda x: new_color.lower().capitalize(),
        replace_upper,
    )
    replace_lower = re.sub(
        lower_regex,
        lambda x: new_color.lower(),
        replace_capitalized,
    )

    return replace_lower


def convert_color_name(text, mapping):
    for old_color, new_color in mapping.items():
        text = color_deterministic(text, old_color, new_color)
    return text


class IOPair:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        assert isinstance(self.x, np.ndarray)
        assert isinstance(self.y, np.ndarray)
        assert len(self.x.shape) == 2
        assert len(self.y.shape) == 2


class Problem:
    def __init__(self, filename=None, code=None, seed_id=None, train_pairs=None, test_pairs=None, analogy=None):
        self.filename = filename
        self.seed_id = seed_id
        self.analogy = analogy
        self.code = code if code else "def transform(grid): pass"
        self.train_pairs = train_pairs if train_pairs else []
        self.test_pairs = test_pairs if test_pairs else []
        
        # Only assert for non-ARC modes
        if code and code != "def transform(grid): pass":
            assert self.train_pairs, "Train pairs are not provided"
            assert self.test_pairs, "Test pairs are not provided"
            assert isinstance(self.train_pairs, list)
            assert isinstance(self.test_pairs, list)
            assert all(isinstance(pair, IOPair) for pair in self.train_pairs)
            assert all(isinstance(pair, IOPair) for pair in self.test_pairs)


def grid_to_input(grid, transpose: bool):
    if transpose:
        transformed_grid = grid.T
    else:
        transformed_grid = grid
    return "\n".join(" ".join(COLOR_MAPPING[c] for c in row) for row in transformed_grid) + EXTRA_NEWLINE


def make_problem_input_str(problem: Problem, transpose: bool):
    prompt = "Given input-output grid pairs as reference examples, carefully observe the patterns to predict the output grid for new test input. Each pair follows the same transformation rule. Grids are 2D arrays represented as strings, with cells (colors) separated by spaces and rows by newlines."
    prompt += "\nHere are the input and output grids for the reference examples:\n"
    
    for i, pair in enumerate(problem.train_pairs):
        prompt += f"Example {i+1}\n"
        prompt += f"Input:\n{grid_to_input(pair.x, transpose)}\nOutput:\n{grid_to_input(pair.y, transpose)}\n\n" 
    
    prompt += "Here is the input grid for the test example:\n"
    # Only use the first test pair
    prompt += "Input:\n" + grid_to_input(problem.test_pairs[0].x, transpose)
    return prompt


def make_input_prompt_induction(problem: Problem, transpose: bool, common_lib_path: str = None):
    common_lib_prefix = ""
    
    if common_lib_path and HAS_PROMPT_LIB:
        try:
            common_lib = get_common_lib_from_file(common_lib_path)
            # For now, we'll keep the common lib prefix empty as in original
            common_lib_prefix = ""
        except:
            pass
    
    question = common_lib_prefix + make_problem_input_str(problem, transpose=transpose)
    question += "\nWrite a Python function `transform` that can convert any given input grid to its corresponding output grid based on the pattern observed in the reference examples."
    # question += "\n\nAssistant:"
    return question


def make_input_prompt_transduction(problem: Problem, transpose: bool):
    """Create transduction prompt - directly ask for the output grid without code"""
    prompt = make_problem_input_str(problem, transpose=transpose)
    prompt += "\n\nDirectly provide the output grids corresponding to the given test input grids, based on the patterns observed in the reference examples."
    return prompt


def make_input_prompt_for_analogy_generation(problem: Problem, transpose: bool):
    """
    Create prompt for Stage 1: Generate analogy only (for Berman-style refinement)
    This is used to train a model that generates analogies given a task.
    """
    question = make_problem_input_str(problem, transpose=transpose)
    question += "\n\nAnalyze the pattern in the reference examples and describe the analogical, metaphorical, or figurative pattern you observe. Focus on identifying the core transformation rule."
    return question


def make_input_prompt_for_grid_generation(problem: Problem, analogy: str, transpose: bool):
    """
    Create prompt for Stage 2: Generate grid given analogy (for Berman-style refinement)
    This is used to train a model that generates output grids given a task + analogy.

    Args:
        problem: Problem object containing task information
        analogy: The analogy text to include in the prompt
        transpose: Whether to transpose grids
    """
    question = make_problem_input_str(problem, transpose=transpose)
    question += f"\n\nGiven Analogy:\n{analogy}\n\n"
    question += "Based on the above analogy and the test input, directly provide the output grid."
    return question




def generate_label_based_on_mode(label_mode, analogy, solution_code, output_grid):
    """
    Generate label based on the specified mode for ablation study.

    Args:
        label_mode: One of 'full', 'analogy', 'code', 'base'
        analogy: Analogy text (required for 'full' and 'analogy' modes)
        solution_code: Solution code text
        output_grid: Output grid text

    Returns:
        Generated label string

    Raises:
        ValueError: If analogy is required but not provided
    """
    if label_mode == "full":
        # Full Components: Analogy + Code + Grid
        if not analogy:
            raise ValueError("Analogy is required for 'full' label_mode but not provided")
        return ANALOGY_PROCESS_CODE.format(
            analogy=analogy,
            solution_code=solution_code,
            test_output_grid=output_grid
        )

    elif label_mode == "analogy":
        # With Analogy: Analogy + Grid (no code)
        if not analogy:
            raise ValueError("Analogy is required for 'analogy' label_mode but not provided")
        return f"""# THINKING ANALOGY
[Let's think step by step about the analogical pattern in the given task, and describe what kind of analogical, metaphorical, or figurative pattern it is]

{analogy}

# RESULT TEST OUTPUT GRID
[Write the final output grid for the given task based on the analogical approach identified in the thinking analogy step]

{output_grid}
"""

    elif label_mode == "code":
        # With Code: Code + Grid (no analogy)
        return f"""# SOLUTION CODE
[Write the final solution code for the given task]

{solution_code}

# RESULT TEST OUTPUT GRID
[Write the final output grid for the given task based on the solution code and the test input grid]

{output_grid}
"""

    elif label_mode == "base":
        # Base Component: Grid only
        return f"""# RESULT TEST OUTPUT GRID
[Write the final output grid for the given task]

{output_grid}
"""

    elif label_mode == "refining-stage1-analogy":
        # Berman-style Stage 1: Analogy only (NO GRID!)
        if not analogy:
            raise ValueError("Analogy is required for 'refining-stage1-analogy' label_mode but not provided")
        return f"""# THINKING ANALOGY
[Let's think step by step about the analogical pattern in the given task, and describe what kind of analogical, metaphorical, or figurative pattern it is]

{analogy}
"""

    elif label_mode == "refining-stage2-grid":
        # Berman-style Stage 2: Grid only (given analogy in input)
        return f"""# RESULT TEST OUTPUT GRID
[Write the final output grid for the given task based on the given analogy and the test input grid]

{output_grid}
"""

    else:
        raise ValueError(f"Unknown label_mode: {label_mode}")


def convert_chat_format(question, answer, system_prompt=None, problem_id=None, test_index=None, ground_truth_output=None):
    if system_prompt is None:
        system_prompt = DEFAULT_SYSTEM_PROMPT
    messages = {
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": question},
        ]
    }
    if answer:
        messages["answer_messages"] = [{"role": "assistant", "content": answer}]

    # Add problem_id if provided
    if problem_id is not None:
        messages["problem_id"] = problem_id

    # Add test_index if provided
    if test_index is not None:
        messages["test_index"] = test_index

    # Add ground_truth_output if provided
    if ground_truth_output is not None:
        messages["ground_truth_output"] = ground_truth_output

    return messages


def parse_data_based_on_mode(data, mode, data_key='train'):
    """Parse data based on the mode and return IOPairs"""
    all_pairs = []
    
    if mode == "arc":
        # Standard ARC format with train/test split
        # For ARC mode, we DON'T add test to all_pairs to avoid data leakage
        # Test cases are handled separately later
        if "train" == data_key:
            for example in data["train"]:
                input_grid = np.array(example["input"])
                output_grid = np.array(example["output"])
                all_pairs.append(IOPair(input_grid, output_grid))
        elif "test" == data_key:
            # If only test is available, we can still create pairs
            for example in data["test"]:
                input_grid = np.array(example["input"])
                output_grid = np.array(example["output"])
                all_pairs.append(IOPair(input_grid, output_grid))
    
    elif mode == "gifarc":
        # GIFARC format with examples field (also handles rebuttal format)
        # For GIFARC, we need to handle train/test split properly
        # The last example should be reserved for test, others for train
        if "examples" in data:
            examples = data["examples"]
            # Use all but last example for training
            for example in examples[:-1]:
                input_grid = np.array(example[0])
                output_grid = np.array(example[1])
                all_pairs.append(IOPair(input_grid, output_grid))
    
    else:
        # Try to auto-detect format
        if "examples" in data:
            # GIFARC/BARC style - reserve last for test
            examples = data["examples"]
            for example in examples[:-1]:
                input_grid = np.array(example[0])
                output_grid = np.array(example[1])
                all_pairs.append(IOPair(input_grid, output_grid))
        elif "train" in data:
            # Check if it's ARC
            if len(data["train"]) > 0:
                for example in data["train"]:
                    input_grid = np.array(example["input"])
                    output_grid = np.array(example["output"])
                    all_pairs.append(IOPair(input_grid, output_grid))
    
    return all_pairs


def main():
    parser = argparse.ArgumentParser(description="Create SFT dataset for ARC-like puzzles")
    parser.add_argument("--mode", type=str, default="auto", 
                       choices=["arc", "barc", "gifarc", "auto"],
                       help="Data format mode (default: auto)")
    parser.add_argument("--input_path", type=str, required=True,
                       help="Path to input data file (JSONL format)")
    parser.add_argument("--output_dir", type=str, default="data",
                       help="Output directory for generated datasets (default: data/)")
    parser.add_argument("--max_data", type=int, default=None,
                       help="Maximum number of samples to randomly sample without replacement (default: use all data)")
    parser.add_argument("--model", type=str, default="gpt-4",
                       help="Model name for tokenizer (default: gpt-4)")
    parser.add_argument("--max_tokens", type=int, default=10000,
                       help="Maximum tokens per sample (default: 10000)")
    parser.add_argument("--num_pairs", type=int, default=None,
                       help="Number of example pairs to use (default: all available)")
    parser.add_argument("--transpose", action="store_true",
                       help="Transpose grids before processing")
    parser.add_argument("--common_lib_path", type=str, default="seeds/common.py",
                       help="Path to common library file")
    parser.add_argument("--skip_analogy", action="store_true",
                       help="Skip analogy section in output")
    parser.add_argument("--seed", type=int, default=777,
                       help="Random seed for reproducibility (default: 777)")
    parser.add_argument("--eval_type", type=str, default="induction",
                       choices=["induction", "transduction"],
                       help="Evaluation type: induction (default) or transduction")
    parser.add_argument("--label_mode", type=str, default="full",
                       choices=["full", "analogy", "code", "base", "all", "refining-stage1-analogy", "refining-stage2-grid"],
                       help="Label mode: full (Analogy+Code+Grid), analogy (Analogy+Grid), code (Code+Grid), base (Grid only), all (generate all 4 versions), refining-stage1-analogy (Analogy only for Berman Stage 1), refining-stage2-grid (Grid only for Berman Stage 2)")
    parser.add_argument("--dataset_type", type=str, required=True,
                       help="Dataset type identifier (e.g., 'sft', 'base', 'train', 'eval')")

    args = parser.parse_args()

    # Set random seed for reproducibility
    random.seed(args.seed)
    np.random.seed(args.seed)
    print(f"Random seed set to: {args.seed}")
    
    # Validate input path
    if not os.path.exists(args.input_path):
        raise FileNotFoundError(f"Input file not found: {args.input_path}")
    
    # Create output directory with seed subdirectory
    seed_output_dir = os.path.join(args.output_dir, f"seed{args.seed}")
    os.makedirs(seed_output_dir, exist_ok=True)
    print(f"Output directory: {seed_output_dir}")
    
    # Read input data
    print(f"Reading data from {args.input_path}...")
    with open(args.input_path, "r", encoding="utf-8") as f:
        lines = f.readlines()
    
    # Sample data if needed (without replacement)
    # Sample extra to account for skipped samples during processing
    if args.max_data and args.max_data < len(lines):
        sample_size = min(args.max_data * 2, len(lines))  # Sample 2x to account for skipping
        print(f"Sampling {sample_size} samples from {len(lines)} total samples (will trim to {args.max_data} after processing)")
        lines = random.sample(lines, sample_size)
    
    loaded_data = []
    for line in lines:
        try:
            loaded_data.append(json.loads(line))
        except json.JSONDecodeError:
            print(f"Warning: Skipping invalid JSON line")
            continue
    
    print(f"Loaded {len(loaded_data)} samples")
    
    # Process problems with tracking of original data
    problems = []
    problem_data_mapping = []  # Track original data for each problem
    skipped_count = 0
    
    for idx, d in enumerate(tqdm(loaded_data, desc="Processing samples")):
        try:
            # Determine how many pairs to use
            if args.mode == "gifarc" or args.mode == "barc" or (args.mode == "auto" and "examples" in d):
                # For GIFARC mode, the test pair comes from the last example
                # which was NOT added to all_pairs
                # Parse data based on mode
                all_pairs = parse_data_based_on_mode(d, args.mode, args.num_pairs)
                
                # Determine minimum pairs needed
                min_pairs = args.num_pairs if args.num_pairs else 2  # Need at least 2 (1 train, 1 test)
                
                if len(all_pairs) < min_pairs:
                    skipped_count += 1
                    continue
                examples = d["examples"]
                last_example = examples[-1]
                test_input = np.array(last_example[0])
                test_output = np.array(last_example[1])
                test_all_pairs = [IOPair(test_input, test_output)]
                
                # For train pairs, use what's in all_pairs
                if args.num_pairs and args.num_pairs <= len(all_pairs) + 1:
                    # Use first num_pairs-1 from all_pairs
                    train_pairs = all_pairs[:args.num_pairs-1]
                else:
                    # Use all pairs from all_pairs
                    train_pairs = all_pairs
            
            # For ARC mode, skip code requirements
            elif args.mode == "arc":
                train_all_pairs = parse_data_based_on_mode(d, args.mode, data_key='train')
                test_all_pairs = parse_data_based_on_mode(d, args.mode, data_key='test')
                # Process each test case separately
                if "test" in d and len(d["test"]) > 0:
                    # For ARC mode, train_pairs come from the train field in original data
                    # all_pairs already contains the train examples
                    arc_train_pairs = train_all_pairs
                    
                    # Create a separate problem for each test case
                    # for test_idx, test_case in enumerate(d["test"]):
                    #     if "output" not in test_case:
                    #         continue  # Skip test cases without output
                        
                    #     # Create a copy of the original data for this test case
                    
                    # Create problem without code
                    for test_pair_index in range(len(test_all_pairs)):
                        d_copy = d.copy()
                        d_copy['visited_flag'] = False
                        d_copy["selected_test_index"] = test_pair_index
                        problem = Problem(
                            code="def transform(grid): pass",  # Dummy code
                            train_pairs=arc_train_pairs,
                            test_pairs=[test_all_pairs[test_pair_index]],
                            analogy=None
                        )
                        problems.append(problem)
                        problem_data_mapping.append(d_copy)  # Store original data with test index
                else:
                    # No test cases, skip
                    skipped_count += 1
                continue
            
            # Extract code
            code = d.get("source", "")
            if not code:
                skipped_count += 1
                continue
            
            # Basic code validation
            if "def generate_input" in code and "def main(" in code:
                code = code.split("def generate_input")[0].strip()
                code = code.replace("def main(", "def transform(")
            elif "def transform(" not in code:
                # If no transform function, skip
                skipped_count += 1
                continue
            
            # Extract analogy if available
            analogy = None
            if not args.skip_analogy:
                if "seeds" in d and isinstance(d["seeds"], list) and len(d["seeds"]) > 0:
                    last_seed = d["seeds"][-1]
                    if "Description:" in last_seed:
                        analogy = last_seed.split('Description:')[-1].strip()
                elif "analogy" in d:
                    analogy = d["analogy"]
            
            # Create problem
            for test_pair_index in range(len(test_all_pairs)):
                problem = Problem(
                    code=code,
                    train_pairs=train_pairs,
                    test_pairs=[test_all_pairs[test_pair_index]],
                    analogy=analogy
                )
                problems.append(problem)
                problem_data_mapping.append(d)  # Store original data
            
        except Exception as e:
            print(f"Error processing sample {idx}: {e}")
            skipped_count += 1
            continue
    
    print(f"Successfully processed {len(problems)} problems, skipped {skipped_count}")

    # Determine which label modes to process based on eval_type
    if args.eval_type == "induction":
        # For induction mode (dataset_type=sft), process requested label modes
        if args.label_mode == "all":
            label_modes_to_process = ["full", "analogy", "code", "base"]
        else:
            label_modes_to_process = [args.label_mode]
        print(f"Generating induction dataset with label_modes={label_modes_to_process}")

    elif args.eval_type == "transduction":
        # For transduction mode (dataset_type=base), only use label_mode "base"
        label_modes_to_process = ["base"]
        print(f"Generating transduction dataset with label_mode=base (Grid only)")

    # Process each label mode
    for current_label_mode in label_modes_to_process:
        print(f"\n{'='*60}")
        print(f"Generating dataset for label_mode: {current_label_mode}")
        print(f"{'='*60}")

        # Generate datasets
        input_dataset = []
        ground_truth_dataset = []

        for problem_idx, problem in enumerate(tqdm(problems, desc=f"Generating {current_label_mode} datasets")):
            try:
                # Choose prompt format based on label_mode for refinement modes
                if current_label_mode == "refining-stage1-analogy":
                    # Stage 1: Task → Analogy only
                    question = make_input_prompt_for_analogy_generation(problem, args.transpose)
                    system_prompt = DEFAULT_SYSTEM_PROMPT
                elif current_label_mode == "refining-stage2-grid":
                    # Stage 2: Task + Analogy → Grid only
                    if not problem.analogy:
                        # Skip problems without analogy for Stage 2
                        continue
                    question = make_input_prompt_for_grid_generation(problem, problem.analogy, args.transpose)
                    system_prompt = DEFAULT_SYSTEM_PROMPT
                elif args.mode == "arc" and args.eval_type == "transduction":
                    question = make_input_prompt_transduction(problem, args.transpose)
                    system_prompt = DEFAULT_SYSTEM_PROMPT_TRANSDUCTION
                else:
                    # Default to induction format for non-ARC modes or induction/unified eval_type
                    question = make_input_prompt_induction(problem, args.transpose, args.common_lib_path)
                    system_prompt = DEFAULT_SYSTEM_PROMPT
                
                # For ARC mode, adjust answer based on eval_type
                if args.mode == "arc":
                    answer_output_grid = f"The output grid for the test input grid is:\n\n```\n{grid_to_input(problem.test_pairs[0].y, False).strip()}\n```"
                    if args.eval_type == "transduction":
                        # For transduction, only output grid
                        answer = answer_output_grid
                    else:
                        # For induction/unified, include RESULT TEST OUTPUT GRID header
                        answer = f"# RESULT TEST OUTPUT GRID\n[Write the final output grid for the given task based on the solution code and the test input grid]\n\n{answer_output_grid}"
                else:
                    # For GIFARC/BARC modes, use label_mode to generate answer
                    answer_solution_code = f"""Let's solve this task using Python code with the common library functions.
We first define a common library that contains the functions that you can use to solve the task.
Here is the common library function signature and docstring that you can use to solve the problem (skipping the implementation for brevity):
```python
{problem.code}
```
Using the above common library, solve the given task. Name the main function transform. The transform function should take the input grid and return the output grid.
"""

                    answer_output_grid = f"The output grid for the test input grid is:\n\n```\n{grid_to_input(problem.test_pairs[0].y, False).strip()}\n```"

                    # Generate answer based on current label_mode
                    answer = generate_label_based_on_mode(
                        current_label_mode,
                        problem.analogy,
                        answer_solution_code,
                        answer_output_grid
                    )
                
                answer = convert_color_name(answer, COLOR_REPLACEMENTS)
                question = convert_color_name(question, COLOR_REPLACEMENTS)

                # Get problem_id from original data
                problem_id = problem_data_mapping[problem_idx].get("problem_id", None)

                # Get ground truth output based on label_mode
                if current_label_mode == "refining-stage1-analogy":
                    # For Stage 1, ground truth is the analogy text
                    ground_truth_output = problem.analogy
                else:
                    # For other modes, ground truth is the output grid
                    ground_truth_output = problem.test_pairs[0].y.tolist()

                # Get test index from the stored data

                actual_test_index = problem_data_mapping[problem_idx].get("selected_test_index", 0)
                
                dataset = convert_chat_format(question, answer, system_prompt, problem_id, test_index=actual_test_index, ground_truth_output=ground_truth_output)
                eval_input_dict = {
                    "messages": dataset["messages"],
                    "problem_id": problem_id,
                    "test_index": actual_test_index,
                }
                input_dataset.append(eval_input_dict)
                
                ground_truth_dict = {
                    "messages": dataset["messages"],
                    "answer_messages": dataset["answer_messages"] if "answer_messages" in dataset else [],
                    "problem_id": problem_id,
                    "test_index": actual_test_index,
                    "ground_truth_output": ground_truth_output,
                }
                ground_truth_dataset.append(ground_truth_dict)
                
            except Exception as e:
                print(f"Error generating data for problem: {e}")
                continue
        
        # Filter by token length
        try:
            tokenizer = tiktoken.encoding_for_model(args.model)
        except:
            print(f"Warning: Could not find tokenizer for {args.model}, using cl100k_base")
            tokenizer = tiktoken.get_encoding("cl100k_base")
        
        def filter_by_token_limit(data, limit):
            filtered = []
            dropped = []
            for item in data:
                if "messages" in item:
                    total_tokens = sum(len(tokenizer.encode(m["content"])) for m in item["messages"])
                else:
                    prompt_tokens = len(tokenizer.encode(item.get("prompt", "")))
                    completion_tokens = len(tokenizer.encode(item.get("completion", "")))
                    total_tokens = prompt_tokens + completion_tokens
                
                if total_tokens < limit:
                    filtered.append(item)
                else:
                    dropped.append(item)
            return filtered, dropped
        
        input_dataset, input_dataset_dropped = filter_by_token_limit(input_dataset, args.max_tokens)
        ground_truth_dataset, ground_truth_dataset_dropped = filter_by_token_limit(ground_truth_dataset, args.max_tokens)
        
        print(f"Filtered out {len(input_dataset_dropped)} samples due to token limit")
        print(f"Filtered out {len(ground_truth_dataset_dropped)} samples due to token limit")

        # Final random sampling to get exactly max_data samples if specified
        if args.max_data and len(input_dataset) > args.max_data:
            print(f"Randomly sampling {args.max_data} samples from {len(input_dataset)} generated samples (without replacement)")
            # Create indices and sample them without replacement
            sampled_indices = random.sample(range(len(input_dataset)), args.max_data)
            input_dataset = [input_dataset[i] for i in sampled_indices]
            ground_truth_dataset = [ground_truth_dataset[i] for i in sampled_indices]

        # Generate output filenames
        base_name = Path(args.input_path).stem

        input_dataset_filename = f"[{args.dataset_type}]{base_name}_{current_label_mode}_seed{args.seed}_{len(input_dataset)}.jsonl"
        ground_truth_dataset_filename = f"[{args.dataset_type}]ground_truth_{base_name}_{current_label_mode}_seed{args.seed}_{len(ground_truth_dataset)}.jsonl"
        
        # Write to file in seed subdirectory
        input_dataset_path = os.path.join(seed_output_dir, input_dataset_filename)
        ground_truth_dataset_path = os.path.join(seed_output_dir, ground_truth_dataset_filename)
        
        
        with open(input_dataset_path, "w", encoding="utf-8") as f:
            for item in input_dataset:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
        
        print(f"\nDataset generation complete!")
        print(f"Saved {len(input_dataset)} samples to: {input_dataset_path}")
        
        # Write summary
        # input_summary = {
        #     "input_file": args.input_path,
        #     "mode": args.mode,
        #     "total_input_samples": len(loaded_data),
        #     "processed_problems": len(problems),
        #     "skipped_samples": skipped_count,
        #     "generated_samples": len(input_dataset),
        #     "dropped_token_limit": len(input_dataset_dropped),
        #     "max_tokens": args.max_tokens,
        #     "num_pairs": args.num_pairs,
        #     "transpose": args.transpose
        # }
        
        # input_summary_path = os.path.join(args.output_dir, f"[{args.dataset_type}]{base_name}_{current_label_mode}_seed{args.seed}_summary.json")
        # with open(input_summary_path, "w", encoding="utf-8") as f:
        #     json.dump(input_summary, f, indent=2)

        # print(f"Saved summary to: {input_summary_path}")

        with open(ground_truth_dataset_path, "w", encoding="utf-8") as f:
            for item in ground_truth_dataset:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")

        print(f"\nDataset generation complete!")
        print(f"Saved {len(ground_truth_dataset)} samples to: {ground_truth_dataset_path}")

        # Write ground truth summary
        # ground_truth_summary = {
        #     "input_file": args.input_path,
        #     "mode": args.mode,
        #     "total_input_samples": len(loaded_data),
        #     "processed_problems": len(problems),
        #     "skipped_samples": skipped_count,
        #     "generated_samples": len(ground_truth_dataset),
        #     "dropped_token_limit": len(ground_truth_dataset_dropped),
        #     "max_tokens": args.max_tokens,
        #     "num_pairs": args.num_pairs,
        #     "transpose": args.transpose
        # }

        # ground_truth_summary_path = os.path.join(args.output_dir, f"[{args.dataset_type}]ground_truth_{base_name}_{current_label_mode}_seed{args.seed}_summary.json")
        # with open(ground_truth_summary_path, "w", encoding="utf-8") as f:
        #     json.dump(ground_truth_summary, f, indent=2)

        # print(f"Saved ground truth summary to: {ground_truth_summary_path}")


if __name__ == "__main__":
    main()