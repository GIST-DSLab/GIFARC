
import os
import re
import random
from tqdm import tqdm
import uuid
import ast
# add seeds/ to the python path
from seeds.common import *

import csv, os
import json, logging
from datetime import datetime
from pathlib import Path

import json    

import os, re, uuid, csv, json
from itertools import islice
from typing import Iterable, List, Dict, Any

from tqdm import tqdm
from datetime import datetime
from pathlib import Path

from utility.llm import *
from utility.utils_gif import *
from seeds.common import *
from utility.prompt_utils import *
from utility.utils  import ensure_dir

from GIFARC_utils.arg_parser import parse_cli_args
from GIFARC_utils.data_collector import process_data_list_loader
from GIFARC_utils.simple_rag import get_seeds_idx_ordered_content_from_files, get_rng_offeset
from GIFARC_utils.prompt_template import prompt_template_for_step_1_desc
from GIFARC_utils.PromptHistory import HistoryManager
from GIFARC_utils.generate_metadata_desc import generate_metadata_csv_of_step_descriptions
from datetime import datetime, timezone


# ── 1) CSV logger settup ───────────────────────────────────────────────────────────
LOG_DIR  = Path("./loggings/error_desc") 
LOG_NAME = 'error_log_geometry_o4_mini_base_o3_mini.csv'
LOG_FILE = LOG_DIR / LOG_NAME

CSV_HEADER = [
    "time",            # 1. Error Raised time
    "error_type",      # 2. Error type
    "server_response", 
    "status_code",     # 4. HTTP/SDK response code
    "gif_name",        # 5‑a. Req data: file name
    "model",           # 5‑b. Req data: model name
    "temperature",     # 5‑c. Req param
    "max_tokens"       # 5‑d. Req param
]


if not LOG_FILE.exists():
    with open(LOG_FILE, "w", newline="") as f:
        csv.writer(f).writerow(CSV_HEADER)

def log_error(exc: Exception, gif_name: str = "N/A"):
    with open(LOG_FILE, "a", newline="") as f:
        csv.writer(f).writerow([
            datetime.now().isoformat(timespec="seconds"),
            type(exc).__name__,
            (str(exc) or "")[:200],     # Cut off the long text message
            gif_name
        ])


def append_row_to_csv(row: dict) -> None:
    """CSV header check and write, also write data"""
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
def strip_code_fence(s: str) -> str:
    if not s:
        return ""
    s = s.strip()
    if s.startswith("```"):
        return re.sub(r"^```(?:json)?\s*|\s*```$", "", s).strip()
    return s

def safe_json_loads(text: str, default=None):
    try:
        return json.loads(text)
    except Exception:
        return default

def safe_first_response(result: dict) -> str | None:
    try:
        return first_response_text(result)
    except Exception:
        return None

def log_and_mark_failed(exc: Exception, gif_path: str, failed_box: list | None):
    log_llm_error_csv(    # ← CSV loggind
        exc=exc,
        server_resp=str(exc)[:200],
        status_code=None,
        req_meta={"gif_name": gif_path},
    )

    if isinstance(failed_box, list):
        failed_box.append({"id": gif_path, "error": str(exc)})

def parse_step_1_result(
    output_dir, 
    file_name_json, 
    intergrated, 
    gif_path, 
    concepts_descriptions, 
    gif_result):
    # folder_path = "/path/to/your/folder"

    if output_dir is not None: # join with the base path
        ensure_dir(output_dir)
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

def first_response_text(result: dict) -> str | None:
    """
    Safely return the first response text from the result dictionary.
    - If the 'response' field is missing or is not a list/tuple, returns None.
    - If the first element is None or consists only of whitespace, returns None.
    """
    resp = result.get("response")
    if not resp:                    # None, [], '' is False
        return None
    if isinstance(resp, (list, tuple)):
        first = resp[0] if resp else None
    else:                           # Some time is could be single string
        first = resp
    if first is None:
        return None
    first = str(first).strip()
    return first or None            # empty space None



def batched(it: Iterable[Any], size: int):
    """Python <3.12 batched generator"""
    it = iter(it)
    while (chunk := list(islice(it, size))):
        yield chunk

def main():
    arguments = parse_cli_args()
    DATA_DIR = arguments.data_dir
    AVAILABLE_DATA_FORMATS = [arguments.avaliable_data_formats]
    MAX_SAMPLES =  arguments.samples
    METADATA_CSV_PATH = arguments.metadata_csv_path
    # LOG_NAME = f'error_log_{arguments.model}_{TARGET}.csv'
    ENCODING= arguments.encoding
    SPLITOR= arguments.splitor
    BATCH_SIZE   = arguments.batch_size   # parallel send gif count
    MAX_WORKERS  = arguments.batch_size
    SELECTOR_FILE = arguments.batch_list_path


    data_path_list, missing_path_list = process_data_list_loader(SELECTOR_FILE, MAX_SAMPLES, DATA_DIR, AVAILABLE_DATA_FORMATS, SPLITOR=SPLITOR, ENCODING=ENCODING)
    if len(missing_path_list) > 0:
        raise Exception("missing_path_list exist")
    # Setting RAG
    current_file_dir = os.path.dirname(os.path.realpath(__file__))
    seeds, seeds_contents = get_seeds_idx_ordered_content_from_files(current_file_dir)
    rng_offset = get_rng_offeset(arguments.rng_offset, seeds)
    
    # Model setting progress
    for provider, model in [(provider, model) for provider, model_list in LLMClient.AVAILABLE_MODELS.items() for model in model_list]:
        if model.value == arguments.model:
            # should break on the correct values of model and provider, so we can use those variables later
            break
    
    gif_mode_name = "o4-mini" if arguments.model != "o4-mini" else arguments.model
    for gif_provider, gif_model in [(provider, model) for provider, model_list in LLMClient.AVAILABLE_MODELS.items() for model in model_list]:
        if gif_model.value == gif_mode_name:
            # should break on the correct values of model and provider, so we can use those variables later
            break
    print(gif_mode_name, "models: prod", gif_provider)
    # build pre - prompt 
    prompt_manager = HistoryManager()
    system_prompt_manager = HistoryManager()
    
    for gif_idx, gif_paths in enumerate(tqdm(data_path_list, desc="Processing GIFs"),1):
        gif_path = str(gif_paths)
        print(f"[{gif_idx}/{len(data_path_list)}] Processing {gif_path}")
        message_user, image_block, system_prompt = prompt_template_for_step_1_desc(gif_path, arguments.intergrated, arguments.prompts_path)
       
        prompt_manager.add_message_direct(gif_path, message_user)
        system_prompt_manager.add_message_direct(gif_path, system_prompt) # idx: 1
        
    # print(system_prompt_manager.get_all_history())
        # print(prompt_manager.get_history(gif_path)[0])
        # print(message_system)
    batch_pbar = tqdm(
        total=(len(data_path_list) + BATCH_SIZE - 1) // BATCH_SIZE,
        desc="Batch progress"
    )
    results = []
    failed_results = []
    for batch in batched(data_path_list, BATCH_SIZE):
        num_samples = 1
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            futures = [
                executor.submit(LLMClient(provider=gif_provider, cache_dir=f"{current_file_dir}/cache", system_content=system_prompt_manager.get_last_prompt(str(gif_path))).generate_sub_laber,
                                gif_path,
                                prompt_manager.get_last_prompt(str(gif_path)), 
                                num_samples, model=gif_model, 
                                temperature=arguments.temperature, 
                                max_tokens=arguments.max_tokens, top_p=1) 
                for gif_path in batch]
            for future in tqdm(as_completed(futures), total=len(futures), desc="Generating samples"):
                try:
                    result = future.result()
                except Exception as e:
                    # When failed to call LLM SDK/HTTP 
                    log_and_mark_failed(e, gif_path="UNKNOWN", failed_box=failed_results)
                    continue

                gif_id = str(result.get("id", "UNKNOWN"))
                try:
                    text = safe_first_response(result)
                    if not text:
                        raise ValueError("empty response")

                    results.append(result)

                except Exception as e:
                    print(gif_id, "-> response invalid, skipped")
                    log_and_mark_failed(e, gif_id, failed_box=failed_results)
                    continue  
        batch_pbar.update(1)
    gif_results = {}
    for response_idx, response in enumerate(tqdm(results, desc="Processing GIFs"),1):
        gif_path = str(response.get('id', 'UNKNOWN'))
        try:
            content = safe_first_response(response)
            if not content:
                raise ValueError("Response is empty so that cannot convert in  to JSON.")

            filtered = strip_code_fence(content)
            gif_result = safe_json_loads(filtered, default=None)
            if gif_result is None:
                raise ValueError("JSON parsing failed")

            gif_results[gif_path] = gif_result

        except Exception as e:
            log_and_mark_failed(e, gif_path, failed_box=failed_results)
            gif_results[gif_path] = {}   # At least make as empty dict check None after
            continue
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
                try:
                    final_result = future.result()
                    text = safe_first_response(final_result)
                    if not text:
                        raise ValueError("empty final response")

                    final_results[final_result['id']] = final_result['response']

                except Exception as e:
                    log_and_mark_failed(e, gif_path=str(final_result.get('id', 'UNKNOWN')), failed_box=failed_final_results)
                    continue
        batch_pbar.update(1)
        print('donw')
        
    concepts_descriptions = []

    # print(final_results)
    for raw_data_path in data_path_list:
        data_name = os.path.splitext(os.path.basename(raw_data_path))[0]
        data_path = str(raw_data_path)
        step_id = str(uuid.uuid4())
        try:
            # Data Name
            sample = final_results[str(data_path)][0]
            if sample == "":
                raise ValueError("Requested Error")
            # print(f"sample: {sample}")
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
                "gen_model": "o4-mini",
                "result_code": 1,
                "result_path": os.path.join(arguments.outdir, step_id+".jsonl"),
                "error_message": '',
                "createAt": datetime.now(timezone.utc),   # ← Curr UTC
                # "token_usage": {"prompt": 142, "completion": 23},
            }]
            generate_metadata_csv_of_step_descriptions( sample_records, output_csv=METADATA_CSV_PATH )
            concepts_descriptions=[]
        except KeyError as e:
            print("test", e)
            sample_records = [{
                "id": step_id,
                "step_name": "description",
                "prev_step_id": "GIF",
                "gif_id": str(data_name),
                "gen_model": "o4-mini",
                "result_code": 0,
                "result_path": "",
                "error_message": str(e),
                "createAt": datetime.now(timezone.utc),   # ← Curr UTC
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
                "gen_model": "o4-mini",
                "result_code": 0,
                "result_path": "",
                "error_message": str(e),
                "createAt": datetime.now(timezone.utc),   # ← Curr UTC
                # "token_usage": {"prompt": 142, "completion": 23},
            }]
            generate_metadata_csv_of_step_descriptions( sample_records, output_csv=METADATA_CSV_PATH )
        except Exception as e:
            sample_records = [{
                "id": step_id,
                "step_name": "description",
                "prev_step_id": "GIF",
                "gif_id": str(data_name),
                "gen_model": "o4-mini",
                "result_code": 0,
                "result_path": "",
                "error_message": str(e),
                "createAt": datetime.now(timezone.utc),   # ← Curr UTC
                # "token_usage": {"prompt": 142, "completion": 23},
            }]
            generate_metadata_csv_of_step_descriptions( sample_records, output_csv=METADATA_CSV_PATH )

if __name__ == "__main__":
    # main()
    main()
