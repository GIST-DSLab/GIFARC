from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

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
from arc_evaluation import COLOR_MAPPING, evaluate_arc_output, extract_test_output_grid, parse_grid_from_text

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

BASE_FORMAT_GUIDE = """\n\nCRITICAL INSTRUCTIONS - YOU MUST FOLLOW EXACTLY:

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


@dataclass
class EvaluationExample:
    key: tuple[str | int, int]
    prompt: str
    ground_truth_response: str
    test_input: list[list[int]] | None
    test_output: list[list[int]] | None
    original_data: dict | None = None


def result_key(record, location):
    if not isinstance(record, dict):
        raise ValueError(f"{location}: expected a JSON object")
    problem_id = record.get("problem_id")
    default_index = 0 if "source" in record and "examples" in record else None
    test_index = record.get("test_index", default_index)
    if not ((isinstance(problem_id, str) and problem_id.strip()) or type(problem_id) is int):
        raise ValueError(f"{location}: missing or invalid problem_id")
    if type(test_index) is not int or test_index < 0:
        raise ValueError(f"{location}: missing or invalid test_index for {problem_id!r}")
    return problem_id, test_index


def read_index(path):
    records = {}
    with open(path, encoding="utf-8-sig") as file:
        for number, line in enumerate(file, start=1):
            location = f"{path}:{number}"
            try:
                record = json.loads(line)
            except ValueError as error:
                raise ValueError(f"{location}: invalid JSON") from error
            key = result_key(record, location)
            if key in records:
                raise ValueError(f"{location}: duplicate key {key!r}")
            records[key] = record
    return records


def message_texts(record, field, role, key):
    messages = record.get(field, [])
    if not isinstance(messages, list):
        raise ValueError(f"{key!r}: {field} must be a list")
    for message in messages:
        if not (isinstance(message, dict) and isinstance(message.get("role"), str)
                and isinstance(message.get("content"), str)):
            raise ValueError(f"{key!r}: invalid {field} entry")
        if message["role"] == role:
            yield message["content"]


def is_arc_grid(value):
    return (isinstance(value, list) and bool(value) and isinstance(value[0], list) and bool(value[0])
            and all(isinstance(row, list) and len(row) == len(value[0])
                    and all(type(cell) is int and cell in COLOR_MAPPING for cell in row) for row in value))


def prepare_example(key, source, target, guide, mode):
    raw = "source" in source and "examples" in source
    response = next(reversed(list(message_texts(target, "answer_messages", "assistant", key))), "")
    target_output = target.get("ground_truth_output")
    if target_output is None and response:
        target_output = extract_test_output_grid(response)

    if raw:
        pairs = source["examples"]
        if not (isinstance(pairs, list) and pairs and isinstance(pairs[-1], list) and pairs[-1]):
            raise ValueError(f"{key!r}: missing test example")
        test_input = pairs[-1][0]
        if target_output is None:
            target_pairs = target.get("examples")
            if isinstance(target_pairs, list) and target_pairs and isinstance(target_pairs[-1], list):
                target_output = target_pairs[-1][1] if len(target_pairs[-1]) > 1 else None
        if target_output is None:
            target_output = source.get("ground_truth_output", pairs[-1][1] if len(pairs[-1]) > 1 else None)
        prompt = create_input_only_format(source)
        response = response or (str(target_output) if target_output is not None else "")
    else:
        users = list(message_texts(source, "messages", "user", key))
        if not users:
            raise ValueError(f"{key!r}: missing user message")
        labels = {"system": "System", "user": "Human"}
        prompt = "".join(f"{labels[m['role']]}: {m['content']}\n\n"
                         for m in source["messages"] if m["role"] in labels)
        prompt += guide + "Assistant: "
        match = re.search(r"Here is the input grid for the test example:\s*\nInput:\s*\n((?:[^\n]+\n)*)", users[0])
        if not match:
            match = re.search(r"Test Input:\s*\n([^\n]+(?:\n[^\n]+)*?)(?:\n\n|$)", users[0])
        test_input = parse_grid_from_text(match.group(1)) if match else None

    if mode and not is_arc_grid(target_output):
        raise ValueError(f"{key!r}: missing or invalid ground truth output grid")
    if not is_arc_grid(test_input):
        if mode in ("induction", "both"):
            raise ValueError(f"{key!r}: missing or invalid test input grid for {mode}")
        test_input = None
    return EvaluationExample(key, prompt, response, test_input, target_output, source if raw else None)


def prepare_evaluation(input_path, target_path, output_path, mode, model_name, model_path):
    if mode not in (None, "", "transduction", "induction", "both"):
        raise ValueError(f"Unknown evaluation mode {mode!r}")
    output = Path(output_path)
    for source in map(Path, (input_path, target_path)):
        if output.resolve() == source.resolve() or (output.exists() and source.exists() and output.samefile(source)):
            raise ValueError("Output path must differ from the evaluation input and ground truth files")
    inputs = read_index(input_path)
    targets = read_index(target_path)
    if not inputs:
        raise ValueError(f"{input_path}: no evaluation inputs")
    missing = inputs.keys() - targets.keys()
    if missing:
        samples = [key for key in inputs if key in missing][:5]
        raise ValueError(f"Missing ground truth for {len(missing)} input key(s): {samples!r}")
    guide = BASE_FORMAT_GUIDE if model_path == model_name else ""
    return {key: prepare_example(key, row, targets[key], guide, mode) for key, row in inputs.items()}


def scoring_modes(mode):
    if mode == "both":
        return ("induction", "transduction")
    return (mode,) if mode else ()


@contextmanager
def locked_output(path):
    with open(path, "a+b") as file:
        try:
            if os.name == "nt":
                import msvcrt

                file.seek(0)
                msvcrt.locking(file.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl

                fcntl.flock(file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as error:
            raise ValueError(f"{path}: could not lock evaluation output") from error
        yield file


def load_completed(file, path, examples, mode, settings):
    completed = {}
    file.seek(0)
    number = 0
    while True:
        offset = file.tell()
        line = file.readline()
        if not line:
            break
        number += 1
        location = f"{path}:{number}"
        terminated = line.endswith(b"\n")
        try:
            result = json.loads(line)
        except ValueError as error:
            if terminated:
                raise ValueError(f"{location}: invalid JSON in existing output") from error
            file.truncate(offset)
            print(f"{location}: removed a truncated final result")
            break
        if isinstance(result, dict) and result.get("test_index") is None:
            original = result.get("original_data")
            if isinstance(original, dict) and "source" in original and "examples" in original:
                raise ValueError(f"{location}: legacy raw result lacks a test_index; use a new output path")
        key = result_key(result, location)
        if key not in examples:
            raise ValueError(f"{location}: unexpected result key {key!r}")
        if key in completed:
            raise ValueError(f"{location}: duplicate result key {key!r}")
        example = examples[key]
        if result.get("input_prompt", example.prompt) != example.prompt:
            raise ValueError(f"{location}: result prompt differs for {key!r}")
        if result.get("ground_truth_output", example.test_output) != example.test_output:
            raise ValueError(f"{location}: result ground truth differs for {key!r}")
        if result.get("evaluation_settings", settings) != settings:
            raise ValueError(f"{location}: result settings differ for {key!r}")
        if result.get("evaluation_mode", mode) != mode:
            raise ValueError(f"{location}: result evaluation mode differs for {key!r}")
        if not isinstance(result.get("generated_output"), str) or result.get("evaluation_error"):
            raise ValueError(f"{location}: incomplete evaluation result for {key!r}")
        for score_mode in scoring_modes(mode):
            score = result.get(f"{score_mode}_result")
            details = result.get("evaluation_details")
            if type(score) is not int or score not in (-1, 0, 1) or (isinstance(details, dict) and score_mode not in details):
                raise ValueError(f"{location}: incomplete {score_mode} result for {key!r}")
        completed[key] = result
        if not terminated:
            file.write(b"\n")
            break
    file.seek(0, os.SEEK_END)
    return completed


def evaluate_example(example, index, model, tokenizer, device, max_length, max_new_tokens, mode, temperature):
    inputs = tokenizer(example.prompt, return_tensors="pt", truncation=True, max_length=max_length)
    inputs.pop("token_type_ids", None)
    inputs = {name: value.to(device) for name, value in inputs.items()}
    start = time.monotonic()
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=temperature > 0,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    elapsed = time.monotonic() - start
    decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
    if example.original_data is not None:
        generated = decoded.removeprefix(example.prompt).strip()
    else:
        generated = decoded.rsplit("Assistant: ", 1)[-1].strip()
    result = {
        "idx": index,
        "problem_id": example.key[0],
        "test_index": example.key[1],
        "input_prompt": example.prompt,
        "generated_output": generated,
        "ground_truth_response": example.ground_truth_response,
        "generation_time": elapsed,
        "evaluation_mode": mode,
    }
    if example.original_data is not None:
        result["original_data"] = example.original_data
    if example.test_output is not None:
        result["ground_truth_output"] = example.test_output
    if mode:
        include_probability = example.original_data is None or mode == "both"
        scored = evaluate_arc_output(
            llm_output=generated,
            test_input=example.test_input,
            test_output=example.test_output,
            model=model if include_probability else None,
            tokenizer=tokenizer if include_probability else None,
            device=device if include_probability else None,
            evaluation_mode=mode,
        )
        for score_mode in scoring_modes(mode):
            score = scored[score_mode]
            result[f"{score_mode}_output"] = score["predicted_grid"]
            result[f"{score_mode}_result"] = score["result"]
        if "negative_log_probability_correct" in scored:
            result["negative_log_probability_correct"] = scored["negative_log_probability_correct"]
        if "nlp_debug" in scored:
            result["nlp_prompt_until_grid"] = scored["nlp_debug"]["nlp_prompt_until_grid"]
            result["nlp_target_grid"] = scored["nlp_debug"]["nlp_target_grid"]
        result["evaluation_details"] = scored
    return result


def run_evaluation(examples, output_path, load_model, mode, settings):
    new_results = []
    with locked_output(output_path) as file:
        completed = load_completed(file, output_path, examples, mode, settings)
        pending = [(index, example) for index, (key, example) in enumerate(examples.items()) if key not in completed]
        if pending:
            model, tokenizer = load_model()
            model.eval()
            device = next(model.parameters()).device
            for index, example in tqdm(pending, total=len(pending), desc="Evaluating", unit="sample"):
                try:
                    result = evaluate_example(
                        example, index, model, tokenizer, device, settings["max_length"],
                        settings["max_new_tokens"], mode, settings["temperature"],
                    )
                    result["evaluation_settings"] = settings
                    line = json.dumps(result, ensure_ascii=False) + "\n"
                    file.write(line.encode("utf-8"))
                    file.flush()
                except Exception as error:
                    raise RuntimeError(f"Evaluation failed for {example.key!r}: {error}") from error
                completed[example.key] = result
                new_results.append(result)
                tqdm.write(f"Sample {index + 1}: {result['generation_time']:.2f}s")
        if completed.keys() != examples.keys():
            raise RuntimeError(f"Evaluation incomplete: {len(completed)}/{len(examples)} results")
    total = len(examples)
    print(f"Evaluation completed! {total}/{total} results saved to {output_path}")
    print(f"Generated {len(new_results)} new evaluation outputs")
    if total > len(new_results):
        print(f"Skipped {total - len(new_results)} already completed tasks")
    for score_mode in scoring_modes(mode):
        correct = sum(row[f"{score_mode}_result"] == 1 for row in completed.values())
        print(f"{score_mode.capitalize()} accuracy: {correct}/{total} ({100 * correct / total:.1f}%)")
    return new_results


def generate_evaluation_outputs(
    model_name, model, tokenizer, input_data_path: str, ground_truth_data_path: str, output_path: str,
    max_length: int, max_new_tokens: int = 512, evaluation_mode: str = "transduction",
    model_path: str = "", temperature: float = 0.1,
):
    """Evaluate validated inputs and return newly generated rows."""
    examples = prepare_evaluation(input_data_path, ground_truth_data_path, output_path, evaluation_mode,
                                  model_name, model_path)
    settings = {
        "model_name": model_name, "model_path": model_path, "max_length": max_length,
        "max_new_tokens": max_new_tokens, "temperature": temperature,
    }
    return run_evaluation(examples, output_path, lambda: (model, tokenizer), evaluation_mode, settings)


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

    examples = prepare_evaluation(
        args.input_data_path, args.ground_truth_data_path, args.output_path, args.evaluation_mode,
        args.model_name, args.model_path,
    )

    def load_model():
        import random
        import numpy as np

        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)
        print(f"Evaluation random seed set to: {args.seed}")
        return load_model_and_tokenizer(args.model_name, args.model_path, args.use_4bit, args.use_8bit)

    settings = {
        name: getattr(args, name) for name in
        ("model_name", "model_path", "max_length", "max_new_tokens", "temperature", "seed", "use_4bit", "use_8bit")
    }
    run_evaluation(examples, args.output_path, load_model, args.evaluation_mode, settings)


if __name__ == "__main__":
    main()
