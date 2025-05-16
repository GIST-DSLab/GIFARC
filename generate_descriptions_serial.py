import os
import re
import random
from tqdm import tqdm
from utils import get_description_from_lines, get_concepts_from_lines

from llm import *
import ast
from utils_gif import *

# add seeds/ to the python path
from seeds.common import *

def extract_concepts_and_descriptions(content):
    all_lines = content.split("\n")

    all_concepts = []
    all_descriptions = []

    last_concept_line = None
    # find the line containing "BEST SOLUTION"
    for i, line in enumerate(all_lines):
        if "# concepts" in line:
            last_concept_line = i
            lines = all_lines[last_concept_line:]
            # Extract the concepts, which come as a comment after the line containing "# concepts:"
            concepts = get_concepts_from_lines(lines)
            all_concepts.append(concepts)
            # Extract the descriptions, which come as a comment after the line containing "# description:"
            description = get_description_from_lines(lines)
            all_descriptions.append(description)

    return all_concepts, all_descriptions


def make_self_instruct_prompt(seeds_contents, rng_seed, num_descriptions=None, use_concepts=True, num_generations=5):
    # make a random generator
    rng = random.Random(rng_seed)

    # Sort the seeds so that the order is consistent
    seeds_contents = list(sorted(seeds_contents, key=lambda x: x[0]))
    rng.shuffle(seeds_contents)
    if num_descriptions is not None:
        seeds_contents = seeds_contents[:num_descriptions]

    # get the content of the seeds
    seed_content = []
    for _ , content in seeds_contents:
        assert "# ============= remove below this point for prompting =============" in content
        content = content.split("# ============= remove below this point for prompting =============")[0].strip()
        seed_content.append(content)

    # extract the concepts and descriptions from the seeds
    concepts_and_descriptions_in_seeds = []
    for content in seed_content:
        concepts, description = extract_concepts_and_descriptions(content)

        # only one concept and description per seed, so we take the first element
        concepts = concepts[0]
        description = description[0]

        # remove "color change" from the concepts, because it is problematic and easily misinterpreted
        concepts = [c for c in concepts if "color change" not in c]
        # deduplicate and randomly permute
        concepts = list(sorted(set(concepts)))
        rng.shuffle(concepts)
        concept_list = ", ".join(concepts)
        
        concepts_and_descriptions_in_seeds.append((concept_list, description))

    if use_concepts:
        examples = "\n\n".join([f"Example puzzle concepts and description:\n```python\n# concepts:\n# {concept_list}\n\n# description:\n# {description}\n```" for concept_list, description in concepts_and_descriptions_in_seeds])
    else:
        examples = "\n\n".join([f"Example puzzle description:\n```python\n# description:\n# {description}\n```" for concept_list, description in concepts_and_descriptions_in_seeds])

    # read the prompt template from prompts/description_prompt.md
    with open("prompts/description_prompt.md") as f:
        prompt_template = f.read()
    
    prompt = prompt_template.format(examples=examples, num_generations=num_generations)
    # print(prompt)
    return prompt

def make_self_instruct_prompt_with_gif(seeds_contents, rng_seed, num_descriptions=None, use_concepts=True, num_generations=5, gif_result=None, intergrated=False):
    # make a random generator
    rng = random.Random(rng_seed)

    # Sort the seeds so that the order is consistent
    seeds_contents = list(sorted(seeds_contents, key=lambda x: x[0]))
    rng.shuffle(seeds_contents)
    if num_descriptions is not None:
        seeds_contents = seeds_contents[:num_descriptions]

    # get the content of the seeds
    seed_content = []
    for _ , content in seeds_contents:
        assert "# ============= remove below this point for prompting =============" in content
        content = content.split("# ============= remove below this point for prompting =============")[0].strip()
        seed_content.append(content)

    # extract the concepts and descriptions from the seeds
    concepts_and_descriptions_in_seeds = []
    for content in seed_content:
        concepts, description = extract_concepts_and_descriptions(content)

        # only one concept and description per seed, so we take the first element
        concepts = concepts[0]
        description = description[0]

        # remove "color change" from the concepts, because it is problematic and easily misinterpreted
        concepts = [c for c in concepts if "color change" not in c]
        # deduplicate and randomly permute
        concepts = list(sorted(set(concepts)))
        rng.shuffle(concepts)
        concept_list = ", ".join(concepts)
        
        concepts_and_descriptions_in_seeds.append((concept_list, description))

    if use_concepts:
        examples = "\n\n".join([f"Example puzzle concepts and description:\n```python\n# concepts:\n# {concept_list}\n\n# description:\n# {description}\n```" for concept_list, description in concepts_and_descriptions_in_seeds])
    else:
        examples = "\n\n".join([f"Example puzzle description:\n```python\n# description:\n# {description}\n```" for concept_list, description in concepts_and_descriptions_in_seeds])

    if intergrated:
        with open("prompts/description_prompt_with_gif_intergrated.md") as f:
            prompt_template = f.read()

    else:
        with open("prompts/description_prompt_with_gif.md") as f:
            prompt_template = f.read()

    if intergrated:
        prompt = prompt_template.format(
            examples=examples,
            scenario=gif_result.get('scenario', ''),
            objects=gif_result.get('objects', []),
            composite_objects=gif_result.get('composite_objects', []),
            static_patterns=gif_result.get('static_patterns', []),
            dynamic_patterns=gif_result.get('dynamic_patterns', []),
            interactions=gif_result.get('interactions', []),
            core_principles=gif_result.get('core_principles', []),
            fundamental_principle=gif_result.get('fundamental_principle', ''),
            similar_situations=gif_result.get('similar_situations', [])
        )
    else:
        prompt = prompt_template.format(examples=examples, visual_elements=gif_result['visual_elements'], 
                                        static_patterns=gif_result['static_patterns'], dynamic_patterns=gif_result['dynamic_patterns'], 
                                        core_principles=gif_result['core_principles'])
    # print(prompt)
    return prompt

import csv, os
import json, logging
from datetime import datetime
from pathlib import Path
# ── 1) CSV 로거 설정 ───────────────────────────────────────────────────────────
LOG_DIR  = Path("error_logging")
LOG_DIR.mkdir(exist_ok=True)
LOG_NAME = 'error_log_geometry_o4_mini_base_o3_mini.csv'
LOG_FILE = LOG_DIR / LOG_NAME

CSV_HEADER = [
    "time",            # 1. 에러 발생 시간
    "error_type",      # 2. 에러 종류
    "server_response", # 3. 상대 서버 응답(본문 일부 or 전문)
    "status_code",     # 4. HTTP/SDK 응답 코드
    "gif_name",        # 5‑a. 요청 데이터: 파일명
    "model",           # 5‑b. 요청 데이터: 모델
    "temperature",     # 5‑c. 요청 파라미터
    "max_tokens"       # 5‑d. 요청 파라미터
]

def append_row_to_csv(row: dict) -> None:
    """CSV 파일이 없으면 헤더부터 쓰고, 있으면 행만 append."""
    write_header = not LOG_FILE.exists()
    with LOG_FILE.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_HEADER)
        if write_header:
            writer.writeheader()
        writer.writerow(row)

def log_llm_error_csv(exc: Exception,
                      server_resp: str | None,
                      status_code: int | str | None,
                      req_meta: dict) -> None:
    row = {
        "time": datetime.now().isoformat(timespec="seconds"),
        "error_type": type(exc).__name__,
        "server_response": server_resp,
        "status_code": status_code,
        **req_meta     # gif_name, model, temperature, max_tokens …
    }
    append_row_to_csv(row)

def main():
    import argparse
    parser = argparse.ArgumentParser(description = "problem generator")

    parser.add_argument("--num_descriptions", "-d", type=int, default=None, help="how many descriptions to show in the prompt, if not all of them")
    parser.add_argument("--batch_size", "-b", type=int, default=10, help="how many batches of descriptions to generate")
    parser.add_argument("--num_generations", "-n", type=int, default=5, help="how many generations to generate in the prompt")
    parser.add_argument("--temperature", "-t", type=float, default=0.7)
    parser.add_argument("--model", "-m", type=str, default="gpt-4-turbo", help="which model to use", 
                        choices=[m.value for model_list in LLMClient.AVAILABLE_MODELS.values() for m in model_list])
    parser.add_argument("--sample_parallel", "-sp", type=int, default=1, help="how many parallel workers to use for sampling")
    parser.add_argument("--max_tokens", type=int, default=2048, help="max number of tokens for generation")
    parser.add_argument("--rng_offset", type=int, default=0, help="offset to rng_seed_offset")
    parser.add_argument("--use_concepts", "-uc", action="store_false", help="make the prompts not use concepts", default=True)
    parser.add_argument("--batch_request", "-br", action="store_true", help="generate a batch request, cheaper and high throughput but bad latency")
    parser.add_argument("--outdir", type=str, default=None, help="output directory for the descriptions")
    
    arguments = parser.parse_args()

    # convert model into enum
    for provider, model in [(provider, model) for provider, model_list in LLMClient.AVAILABLE_MODELS.items() for model in model_list]:
        if model.value == arguments.model:
            # should break on the correct values of model and provider, so we can use those variables later
            break

    # get all files in seeds directory
    # get current directory path
    current_file_dir = os.path.dirname(os.path.realpath(__file__))
    seeds = os.listdir(os.path.join(current_file_dir, "seeds"))
    # filter files with .py extension and 8 hex value characters in the file name
    pattern = r"[0-9a-f]{8}(_[a-zA-Z]+)?\.py"
    # get all files and its content
    seeds = [seed for seed in seeds if re.match(pattern, seed)]
    seeds_contents = []
    for seed in seeds:
        with open(os.path.join(current_file_dir, "seeds", seed)) as f:
            seeds_contents.append((seed, f.read()))

    # print all files
    print(f"Using the following {len(seeds)} seeds:", ", ".join(seeds).replace(".py", ""))
    # derive a offset from rng_seed_offset by hashing it if the rng_seed_orig is not 0
    from hashlib import md5
    if arguments.rng_offset != 0:
        rng_offset_str = md5(str(arguments.rng_offset).encode()).hexdigest()[:7]
        # to integer
        rng_offset = int(rng_offset_str, 16)
    else:
        rng_offset = 0

    batch_size = arguments.batch_size
    prompts = [ make_self_instruct_prompt(seeds_contents=seeds_contents, 
                                                    rng_seed=str(rng_seed) + str(rng_offset), 
                                                    num_descriptions=arguments.num_descriptions,
                                                    use_concepts=arguments.use_concepts,
                                                    num_generations=arguments.num_generations)
               for rng_seed in tqdm(range(batch_size)) ]
    
    client = LLMClient(provider=provider, cache_dir=f"{current_file_dir}/cache")

    if arguments.batch_request:

        model_name = arguments.model.replace("/", "_")
        job_name = f"batch_requests_self_instruct_descriptions_fewshot_{arguments.num_descriptions}_{model_name}_temp{arguments.temperature:.2f}_maxtokens{arguments.max_tokens}_rng{arguments.rng_offset}"
        if arguments.use_concepts:
            job_name += "_used_concepts"

        samples = client.batch_request(job_name, prompts, model, temperature=arguments.temperature, max_tokens=arguments.max_tokens, top_p=1, num_samples=1, 
                                       blocking=True)
        # filter out None samples, and take the first (of 1) sample
        samples = [ sample[0] for sample in samples if sample is not None ]

    elif arguments.sample_parallel == 1:
        samples = []
        for prompt in tqdm(prompts):
            try:
                sample = client.generate(prompt, num_samples=1, max_tokens=arguments.max_tokens, temperature=arguments.temperature, model=model)[0]
                samples.append(sample)        
            except Exception as e:
                print("no samples, prompt was too big")
    else:
        list_of_lists_of_samples = client.generate_parallel(prompts, num_samples=1, max_tokens=arguments.max_tokens, num_workers=arguments.sample_parallel, model=model, temperature=arguments.temperature)
        # flatten the list
        samples = [sample for sublist in list_of_lists_of_samples for sample in sublist]

    concepts_descriptions = []
    for sample in samples:
        print(f"sample: {sample}")
        parsed_concepts_lst, parsed_description_lst = extract_concepts_and_descriptions(sample)
        for parsed_concepts, parsed_description in zip(parsed_concepts_lst, parsed_description_lst):
            if parsed_concepts != [] and parsed_description != []:
                parsed_concepts = ", ".join(parsed_concepts)
                concepts_descriptions.append((parsed_concepts, parsed_description))

    model_name = arguments.model.replace("/", "_")
    # write the codes to jsonl file
    file_name_base = f"self_instruct_descriptions_fewshot_{arguments.num_descriptions}_{model_name}_temp{arguments.temperature:.2f}_maxtokens{arguments.max_tokens}_rng{arguments.rng_offset}"
    if arguments.use_concepts:
        file_name_base += "_used_concepts"
    file_name_json = file_name_base + ".jsonl"
    if arguments.outdir is not None: # join with the base path
        file_name_json = os.path.join(arguments.outdir, os.path.basename(file_name_json))
    print(f"Writing to jsonl {file_name_json}")
    with open(file_name_json, "w") as f:
        # jsonl, one json per line
        import json
        for concepts, description in concepts_descriptions:
            f.write(json.dumps({"concepts": concepts,
                                "description": description,
                                }) + "\n")
    print(f"{len(concepts_descriptions)} descriptions written to {file_name_json}")
    client.show_token_usage()
    client.show_global_token_usage()

def main2():
    import argparse
    parser = argparse.ArgumentParser(description = "problem generator")

    parser.add_argument("--num_descriptions", "-d", type=int, default=None, help="how many descriptions to show in the prompt, if not all of them")
    parser.add_argument("--batch_size", "-b", type=int, default=10, help="how many batches of descriptions to generate")
    parser.add_argument("--num_generations", "-n", type=int, default=5, help="how many generations to generate in the prompt")
    parser.add_argument("--temperature", "-t", type=float, default=0.7)
    parser.add_argument("--model", "-m", type=str, default="gpt-4-turbo", help="which model to use", 
                        choices=[m.value for model_list in LLMClient.AVAILABLE_MODELS.values() for m in model_list])
    parser.add_argument("--sample_parallel", "-sp", type=int, default=1, help="how many parallel workers to use for sampling")
    parser.add_argument("--max_tokens", type=int, default=2048, help="max number of tokens for generation")
    parser.add_argument("--rng_offset", type=int, default=0, help="offset to rng_seed_offset")
    parser.add_argument("--use_concepts", "-uc", action="store_false", help="make the prompts not use concepts", default=True)
    parser.add_argument("--batch_request", "-br", action="store_true", help="generate a batch request, cheaper and high throughput but bad latency")
    parser.add_argument("--outdir", type=str, default=None, help="output directory for the descriptions")
    parser.add_argument("-tar", "--target", dest="target", default="nature", type=str)
    parser.add_argument("-sam", "--samples", dest="samples", default=1, type=int)
    parser.add_argument("-i", "--intergrated", action="store_true", help="use intergrated prompt", default=False)
    
    arguments = parser.parse_args()
    TARGET = arguments.target 
    DATA_DIR = f"./data/{TARGET}/"
    MAX_SIZE = 20 * 1024 * 1024 
    MAX_SAMPLES = arguments.samples
    
    LOG_NAME = 'error_log_{arguments.model}_{TARGET}.csv'
    LOG_FILE = LOG_DIR / LOG_NAME

    video_files = [
        os.path.join(DATA_DIR, f)
        for f in os.listdir(DATA_DIR)
        if f.endswith(".gif") or f.endswith(".webm")
    ]

    # Function to check if the result file already exists
    def result_exists(video_path, target):
        base_name = os.path.splitext(os.path.basename(video_path))[0]
        result_path = f"./results/{target}/{base_name}.json"
        return os.path.exists(result_path)

    # Get samples without replacement, avoiding files with existing results
    sampled_gif_paths = []
    available_files = video_files.copy()  # Create a copy to avoid modifying the original list
    count = 0

    while count < min(len(video_files), MAX_SAMPLES) and available_files:

        selected_file = None
        with open(f"test_{TARGET}.txt", "r") as f:
            selected_file = f.readline().split(",")
            sampled_gif_paths = [os.path.join(DATA_DIR, file.strip())+'.gif' for file in selected_file]
        break

    print(f"Successfully selected {len(sampled_gif_paths)} files for processing")


    # convert model into enum
    for provider, model in [(provider, model) for provider, model_list in LLMClient.AVAILABLE_MODELS.items() for model in model_list]:
        if model.value == arguments.model:
            # should break on the correct values of model and provider, so we can use those variables later
            break
    
    gif_mode_name = "o4-mini" if arguments.model != "o4-mini" else arguments.model
    for gif_provider, gif_model in [(provider, model) for provider, model_list in LLMClient.AVAILABLE_MODELS.items() for model in model_list]:
        if gif_model.value == gif_mode_name:
            # should break on the correct values of model and provider, so we can use those variables later
            break

    # get all files in seeds directory
    # get current directory path
    current_file_dir = os.path.dirname(os.path.realpath(__file__))
    seeds = os.listdir(os.path.join(current_file_dir, "seeds"))
    # filter files with .py extension and 8 hex value characters in the file name
    pattern = r"[0-9a-f]{8}(_[a-zA-Z]+)?\.py"
    # get all files and its content
    seeds = [seed for seed in seeds if re.match(pattern, seed)]
    seeds_contents = []
    for seed in seeds:
        with open(os.path.join(current_file_dir, "seeds", seed)) as f:
            seeds_contents.append((seed, f.read()))

    # print all files
    print(f"Using the following {len(seeds)} seeds:", ", ".join(seeds).replace(".py", ""))
    # derive a offset from rng_seed_offset by hashing it if the rng_seed_orig is not 0
    from hashlib import md5
    if arguments.rng_offset != 0:
        rng_offset_str = md5(str(arguments.rng_offset).encode()).hexdigest()[:7]
        # to integer
        rng_offset = int(rng_offset_str, 16)
    else:
        rng_offset = 0
    
    for gif_idx, gif_path in enumerate(tqdm(sampled_gif_paths,desc="Processing GIFs"),1):
        print(f"[{gif_idx}/{len(sampled_gif_paths)}] Processing {os.path.basename(gif_path)}")

        total_history = []

        # Check the size of the gif
        check_path, compress_flag = check_file_size(gif_path, MAX_SIZE)
        # Directly encode gif to base64
        base64_encoded = direct_encode_gif_to_base64(check_path)

        if arguments.intergrated:
            with open("prompts/gif_intergrated.md", encoding="utf-8") as f:
                gif_prompt = f.read()
            with open("prompts/system_prompt_gif_intergrated.md", encoding="utf-8") as f: 
                gif_system_prompt = f.read()
        else:
            with open("prompts/gif.md", encoding="utf-8") as f:
                gif_prompt = f.read()
            with open("prompts/system_prompt_gif.md", encoding="utf-8") as f:
                gif_system_prompt = f.read()

        image_block = {"type": "image_url", "image_url": {"url": f"data:image/gif;base64,{base64_encoded}"}}
        message_user = [{"type": "text", "text": gif_prompt}, image_block]
        message_system = [{"type": "text", "text": gif_system_prompt}]

        client = LLMClient(provider=gif_provider, cache_dir=f"{current_file_dir}/cache", system_content=message_system)

        total_history.append(message_user)
        total_history.append(message_system)
        req_meta = {
            "gif_name": os.path.basename(gif_path),
            "model":    gif_model.value,
            "temperature": arguments.temperature,
            "max_tokens":  arguments.max_tokens
        }
        try:
            response = client.send_request(message_user, num_samples=1, max_tokens=arguments.max_tokens, temperature=arguments.temperature, model=gif_model,  top_p=1)
            # list_of_lists_of_samples = client.generate_parallel(prompts, num_samples=1, max_tokens=arguments.max_tokens, num_workers=arguments.sample_parallel, model=model, temperature=arguments.temperature, top_p=1)
            # flatten the li,st
            # samples = [sample for sublist in list_of_lists_of_samples for sample in sublist]
        except Exception as e:
            resp_obj    = getattr(e, "response", None)
            server_resp = None
            status_code = None

            if resp_obj is not None:
                try:
                    server_resp = resp_obj.text if hasattr(resp_obj, "text") else str(resp_obj)
                    status_code = getattr(resp_obj, "status_code", None)
                except Exception:
                    server_resp = str(resp_obj)
            else:
                status_code = getattr(e, "status_code", None)
                server_resp = str(e)

            # CSV에 기록
            log_llm_error_csv(e, server_resp, status_code, req_meta)

            continue   # 다음 GIF 로 넘어가거나 필요하면 raise
        
        total_history.append(response.choices[0].message.content)

        client.update_usage(gif_model.value, response.usage)
        import json
        try:
            gif_result = json.loads(response.choices[0].message.content)
        except json.JSONDecodeError as e:
            filtered_response = re.sub(r'^```json\n|\n```$', '', response.choices[0].message.content)
            gif_result = json.loads(filtered_response)

        batch_size = arguments.batch_size
        prompts = [ make_self_instruct_prompt_with_gif(seeds_contents=seeds_contents, 
                                                        rng_seed=str(rng_seed) + str(rng_offset), 
                                                        num_descriptions=arguments.num_descriptions,
                                                        use_concepts=arguments.use_concepts,
                                                        num_generations=arguments.num_generations,
                                                        gif_result=gif_result,
                                                        intergrated=arguments.intergrated)
                for rng_seed in tqdm(range(batch_size)) ]
        
        client = LLMClient(provider=provider, cache_dir=f"{current_file_dir}/cache")

        if arguments.batch_request:

            model_name = arguments.model.replace("/", "_")
            job_name = f"batch_requests_self_instruct_descriptions_fewshot_{arguments.num_descriptions}_{model_name}_temp{arguments.temperature:.2f}_maxtokens{arguments.max_tokens}_rng{arguments.rng_offset}"
            if arguments.use_concepts:
                job_name += "_used_concepts"

            samples = client.batch_request(job_name, prompts, model, temperature=arguments.temperature, max_tokens=arguments.max_tokens, top_p=1, num_samples=1, 
                                        blocking=True)
            # filter out None samples, and take the first (of 1) sample
            samples = [ sample[0] for sample in samples if sample is not None ]

        elif arguments.sample_parallel == 1:
            samples = []
            for prompt in tqdm(prompts):
                try:
                    sample = client.generate(prompt, num_samples=1, max_tokens=arguments.max_tokens, temperature=arguments.temperature, model=model)[0]
                    samples.append(sample)        
                except Exception as e:
                    print("no samples, prompt was too big")
        else:
            list_of_lists_of_samples = client.generate_parallel(prompts, num_samples=1, max_tokens=arguments.max_tokens, num_workers=arguments.sample_parallel, model=model, temperature=arguments.temperature)
            # flatten the list
            samples = [sample for sublist in list_of_lists_of_samples for sample in sublist]

        concepts_descriptions = []
        for sample in samples:
            print(f"sample: {sample}")
            parsed_concepts_lst, parsed_description_lst = extract_concepts_and_descriptions(sample)
            for parsed_concepts, parsed_description in zip(parsed_concepts_lst, parsed_description_lst):
                if parsed_concepts != [] and parsed_description != []:
                    parsed_concepts = ", ".join(parsed_concepts)
                    concepts_descriptions.append((parsed_concepts, parsed_description))

        model_name = arguments.model.replace("/", "_")
        # write the codes to jsonl file
        target_file_name = os.path.splitext(os.path.basename(gif_path))[0]
        file_name_base = f"{TARGET}_{target_file_name}"
        file_name_json = file_name_base + ".jsonl"
        if arguments.outdir is not None: # join with the base path
            file_name_json = os.path.join(arguments.outdir, os.path.basename(file_name_json))
        print(f"Writing to jsonl {file_name_json}")
        with open(file_name_json, "w") as f:
            # jsonl, one json per line
            import json
            for concepts, description in concepts_descriptions:
                if arguments.intergrated:
                    f.write(json.dumps({
                    "concepts": concepts,
                    "description": description,
                    "scenario": gif_result.get('scenario', ""),
                    "objects": gif_result.get('objects', []),
                    "composite_objects": gif_result.get('composite_objects', []),
                    "static_patterns": gif_result.get('static_patterns', []),
                    "dynamic_patterns": gif_result.get('dynamic_patterns', []),
                    "interactions": gif_result.get('interactions', []),
                    "core_principles": gif_result.get('core_principles', []),
                    "fundamental_principle": gif_result.get('fundamental_principle', ""),
                    "similar_situations": gif_result.get('similar_situations', []),
                    "gif_path": gif_path,
                    "intergrated": arguments.intergrated
                }) + "\n")
                else:
                    f.write(json.dumps({"concepts": concepts,
                                        "description": description,
                                        "visual_elements": gif_result['visual_elements'], 
                                        "static_patterns": gif_result['static_patterns'],
                                        "dynamic_patterns": gif_result['dynamic_patterns'], 
                                        "core_principles": gif_result['core_principles'],
                                        "gif_path": gif_path,
                                        "intergrated": arguments.intergrated
                                        }) + "\n")
        print(f"{len(concepts_descriptions)} descriptions written to {file_name_json}")
        client.show_token_usage()
        client.show_global_token_usage()
    


if __name__ == "__main__":
    # main()
    main2()
