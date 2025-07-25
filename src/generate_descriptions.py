import os, re, uuid, csv, json
from itertools import islice
from typing import Iterable, Any
from tqdm import tqdm
from datetime import datetime
from pathlib import Path

from utility.llm import *
from utility.utils_gif import *
from seeds.common import *
from utility.prompt_utils import *

from GIFARC_utils.arg_parser import parse_cli_args
from GIFARC_utils.data_collector import process_data_list_loader
from GIFARC_utils.simple_rag import get_seeds_idx_ordered_content_from_files, get_rng_offeset
from GIFARC_utils.prompt_template import prompt_template_for_step_1_desc
from GIFARC_utils.PromptHistory import HistoryManager
from GIFARC_utils.generate_metadata_desc import generate_metadata_csv_of_step_descriptions
from datetime import datetime, timezone


# ── 1) CSV logger settings ───────────────────────────────────────────────────────────
LOG_DIR  = Path("error_logging")
LOG_DIR.mkdir(exist_ok=True)
LOG_NAME = 'error_log_geometry_o4_mini_base_o3_mini.csv'
LOG_FILE = LOG_DIR / LOG_NAME

CSV_HEADER = [
    "time",            # Error occurance time
    "error_type",      
    "server_response", 
    "status_code",     
    "gif_name",        # filename of gif
    "model",           # used LLM model
    "temperature",     # LLM temperature parameter
    "max_tokens"       # LLM maximum token boundary
]

def append_row_to_csv(row: dict) -> None:
    # util function that writes row in log csv file
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

# ── end of 1) CSV logger settings ───────────────────────────────────────────────────────────

# ── 2) jsonl file io functions ───────────────────────────────────────────────────────────
def parse_step_1_result(
    output_dir, 
    file_name_json, 
    intergrated, 
    gif_path, 
    concepts_descriptions, 
    gif_result):
    
    if output_dir is not None: # join with the base path
        file_name_json = os.path.join(output_dir, os.path.basename(file_name_json))
    print(f"Writing to jsonl {file_name_json}")
    with open(file_name_json, "w") as f:
        # jsonl, one json per line
        import json
        for concepts, description in concepts_descriptions:
            if intergrated:
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
                "intergrated": intergrated
            }) + "\n")
            else:
                f.write(json.dumps({"concepts": concepts,
                                    "description": description,
                                    "visual_elements": gif_result['visual_elements'], 
                                    "static_patterns": gif_result['static_patterns'],
                                    "dynamic_patterns": gif_result['dynamic_patterns'], 
                                    "core_principles": gif_result['core_principles'],
                                    "gif_path": gif_path,
                                    "intergrated": intergrated
                                    }) + "\n")
    print(f"{len(concepts_descriptions)} descriptions written to {file_name_json}")

# ── end of 2) jsonl file io functions ───────────────────────────────────────────────────────────


def batched(it: Iterable[Any], size: int):
    """A simple batched generator that works for legacy python version < 3.12 """
    it = iter(it)
    while (chunk := list(islice(it, size))):
        yield chunk

def main():
    arguments = parse_cli_args()
    TARGET = arguments.target 
    DATA_DIR = f"./data/GIF/"
    MAX_SIZE = 10 * 1024 * 1024 
    AVAILABLE_DATA_FORMATS = [".gif", ".webm"]
    MAX_SAMPLES =  -1 #  arguments.samples
    METADATA_CSV_PATH = f'./results/metadata/step_descriptions_metadata.csv'
    SELECTOR_FILE = f"./GIFARC_data_batch/uuid_batchs/batch_{TARGET}.txt"
    print(SELECTOR_FILE," will load as batch")
    LOG_NAME = 'error_log_{arguments.model}_{TARGET}.csv'
    LOG_FILE = LOG_DIR / LOG_NAME
    ENCODING='ISO-8859-1'
    SPLITOR=','
    BATCH_SIZE   = 300 
    MAX_WORKERS  = 300
    batch_pbar = tqdm(desc="Batch progress", total=(len(data_path_list) + BATCH_SIZE - 1) // BATCH_SIZE)
    results = []
    failed_results = []
    
    # 0. setting up data, RAG, model.
     
    # dada loading
    data_path_list, missing_path_list = process_data_list_loader(SELECTOR_FILE, MAX_SAMPLES, DATA_DIR, AVAILABLE_DATA_FORMATS, SPLITOR=SPLITOR, ENCODING=ENCODING)
    if len(missing_path_list) > 0:
        raise Exception("missing_path_list exist")

    # setting RAG
    current_file_dir = os.path.dirname(os.path.realpath(__file__))
    seeds, seeds_contents = get_seeds_idx_ordered_content_from_files(current_file_dir)
    rng_offset = get_rng_offeset(arguments.rng_offset, seeds)
    
    # model setting in accordance with the argument
    for provider, model in [(provider, model) for provider, model_list in LLMClient.AVAILABLE_MODELS.items() for model in model_list]:
        if model.value == arguments.model:
            break
    
    gif_mode_name = "o1" if arguments.model != "o4-mini" else arguments.model
    for gif_provider, gif_model in [(provider, model) for provider, model_list in LLMClient.AVAILABLE_MODELS.items() for model in model_list]:
        if gif_model.value == gif_mode_name:
            break

    
    # 1. Generate prompts with gif files
    prompt_manager = HistoryManager()
    system_prompt_manager = HistoryManager()
    
    for gif_idx, gif_paths in enumerate(tqdm(data_path_list, desc="Processing GIFs"),1):
        gif_path = str(gif_paths)
        print(f"[{gif_idx}/{len(data_path_list)}] Processing {gif_path}")
        message_user, image_block, system_prompt = prompt_template_for_step_1_desc(gif_path, arguments.intergrated)
       
        prompt_manager.add_message_direct(gif_path, message_user)
        system_prompt_manager.add_message_direct(gif_path, system_prompt) # idx상 1번
        
        
    # run LLM prompt batch by batch
    for batch in batched(data_path_list, BATCH_SIZE):
        num_samples = 1
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            futures = [
                executor.submit(
                    LLMClient(provider=gif_provider, 
                        cache_dir=f"{current_file_dir}/cache", 
                        system_content=system_prompt_manager.get_last_prompt(str(gif_path))
                    ).generate_sub_laber,
                    gif_path,
                    prompt_manager.get_last_prompt(str(gif_path)), 
                    num_samples, model=gif_model, 
                    temperature=arguments.temperature, 
                    max_tokens=arguments.max_tokens, top_p=1) 
                for gif_path in batch
            ]

            for future in tqdm(as_completed(futures), total=len(futures), desc="Generating samples"):  
                result = future.result()
                if result['response'] == []:
                    failed_results.append(result)
                    print(gif_path," was failed")
                else:
                    results.append(result)     
        batch_pbar.update(1)


    gif_results = {}
    for response_idx, response in enumerate(tqdm(results, desc="Processing GIFs"),1):
        gif_path = str(response['id'])
        content = response['response'][0]
        try:
            gif_result = json.loads(content)
        except json.JSONDecodeError as e:
            filtered_response = re.sub(r'^```json\n|\n```$', '', content)
            gif_result = json.loads(filtered_response)
        finally:
            gif_results[gif_path] = gif_result
        message = make_self_instruct_prompt_with_gif(seeds_contents=seeds_contents, 
                                                    rng_seed=str(response_idx) + str(rng_offset), 
                                                    num_descriptions=arguments.num_descriptions,
                                                    use_concepts=arguments.use_concepts,
                                                    num_generations=arguments.num_generations,
                                                    gif_result=gif_result,
                                                    intergrated=arguments.intergrated)
        prompt_manager.add_message_direct(gif_path, message)

    batch_pbar = tqdm(
        total=(len(data_path_list) + BATCH_SIZE - 1) // BATCH_SIZE,
        desc="Batch progress"
    )
    
    final_results = {}
    failed_final_results = {}
    for batch in batched(results, BATCH_SIZE):
        num_samples = 1
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            futures = [
                executor.submit(LLMClient(provider=gif_provider, cache_dir=f"{current_file_dir}/cache").generate_sub_laber,
                                str(elem['id']),
                                prompt_manager.get_last_prompt(str(elem['id'])), 
                                num_samples, model=gif_model, 
                                temperature=arguments.temperature, 
                                max_tokens=arguments.max_tokens, top_p=1) 
                for elem in batch]
            for future in tqdm(as_completed(futures), total=len(futures), desc="Generating samples"):  
                final_result = future.result()
                if final_result['response'] == []:
                    failed_final_results['id'] = final_result['response']
                    print(gif_path," was failed")
                else:
                    final_results[final_result['id']] = final_result['response']
        batch_pbar.update(1)
        print('donw')
        
    concepts_descriptions = []
    
    for raw_data_path in data_path_list:
        data_name = os.path.splitext(os.path.basename(raw_data_path))[0]
        data_path = str(raw_data_path)
        step_id = str(uuid.uuid4())
        try:
            sample = final_results[str(data_path)][0]
            if sample == "":
                raise ValueError("Requested Error")
            
            parsed_concepts_lst, parsed_description_lst = extract_concepts_and_descriptions(sample)
            for parsed_concepts, parsed_description in zip(parsed_concepts_lst, parsed_description_lst):
                if parsed_concepts != [] and parsed_description != []:
                    parsed_concepts = ", ".join(parsed_concepts)
                    concepts_descriptions.append((parsed_concepts, parsed_description))
            
            # write the codes to jsonl file
            parse_step_1_result(
                arguments.outdir,
                step_id+'.jsonl',
                arguments.intergrated, 
                data_path, 
                concepts_descriptions, 
                gif_results[data_path]
            )
            sample_records = [{
                "id": step_id,
                "step_name": "description",
                "prev_step_id": "GIF",
                "gif_id": str(data_name),
                "gen_model": "o1 o3-mini",
                "result_code": 1,
                "result_path": os.path.join(arguments.outdir, step_id+".jsonl"),
                "error_message": '',
                "createAt": datetime.now(timezone.utc),   # ← 현재 UTC 시각
                # "token_usage": {"prompt": 142, "completion": 23},
            }]
            generate_metadata_csv_of_step_descriptions( sample_records, output_csv=METADATA_CSV_PATH )
        except KeyError as e:
            print("Key Error occured : ", e)
            sample_records = [{
                "id": step_id,
                "step_name": "description",
                "prev_step_id": "GIF",
                "gif_id": str(data_name),
                "gen_model": "o1 o3-mini",
                "result_code": 0,
                "result_path": "",
                "error_message": e,
                "createAt": datetime.now(timezone.utc),   # ← 현재 UTC 시각
                # "token_usage": {"prompt": 142, "completion": 23},
            }]
            generate_metadata_csv_of_step_descriptions( sample_records, output_csv=METADATA_CSV_PATH )
        except ValueError as e:
            print(e)
            sample_records = [{
                "id": step_id,
                "step_name": "description",
                "prev_step_id": "GIF",
                "gif_id": str(data_name),
                "gen_model": "o1 o3-mini",
                "result_code": 0,
                "result_path": "",
                "error_message": e,
                "createAt": datetime.now(timezone.utc),   # ← 현재 UTC 시각
                # "token_usage": {"prompt": 142, "completion": 23},
            }]
            generate_metadata_csv_of_step_descriptions( sample_records, output_csv=METADATA_CSV_PATH )

if __name__ == "__main__":
    # main()
    main()
