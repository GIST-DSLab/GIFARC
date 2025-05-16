import os
import re
import random
import csv
from pathlib import Path
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Any

from tqdm import tqdm

# project‑local imports (existing modules)
from utils import get_description_from_lines, get_concepts_from_lines
from utils_gif import (
    check_file_size,
    direct_encode_gif_to_base64,
)
from llm import LLMClient
from seeds.common import *  # noqa

# ──────────────────────────────────────────────────────────────
# 1.  공통 로딩 & 프롬프트 빌더
# ──────────────────────────────────────────────────────────────

PROJECT_DIR = Path(__file__).resolve().parent
CACHE_DIR = PROJECT_DIR / "cache"
SEED_DIR = PROJECT_DIR / "seeds"
PROMPT_DIR = PROJECT_DIR / "prompts"
RESULT_DIR = PROJECT_DIR / "results"
LOG_DIR = PROJECT_DIR / "error_logging"
LOG_DIR.mkdir(exist_ok=True)

CSV_HEADER = [
    "time",
    "error_type",
    "server_response",
    "status_code",
    "gif_name",
    "model",
    "temperature",
    "max_tokens",
]

csv_lock = None  # will be set to threading.Lock() after import


def append_row_to_csv(row: Dict[str, Any], log_file: Path):
    """Thread‑safe CSV append."""
    import threading

    global csv_lock
    if csv_lock is None:
        csv_lock = threading.Lock()

    write_header = not log_file.exists()
    with csv_lock:
        with log_file.open("a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=CSV_HEADER)
            if write_header:
                writer.writeheader()
            writer.writerow(row)


def log_llm_error_csv(exc: Exception, server_resp: str | None, status_code: int | str | None, req_meta: Dict[str, Any], log_file: Path):
    row = {
        "time": datetime.now().isoformat(timespec="seconds"),
        "error_type": type(exc).__name__,
        "server_response": server_resp,
        "status_code": status_code,
        **req_meta,
    }
    append_row_to_csv(row, log_file=log_file)


# ----------------------------- seed utils -----------------------------

def load_seeds(pattern: str = r"[0-9a-f]{8}(_[a-zA-Z]+)?\.py") -> List[tuple[str, str]]:
    seed_files = [f for f in os.listdir(SEED_DIR) if re.match(pattern, f)]
    contents = []
    for fname in seed_files:
        with open(SEED_DIR / fname, "r", encoding="utf-8") as f:
            contents.append((fname, f.read()))
    return contents


# We reuse the original prompt builders.
from prompt_utils import (
    make_self_instruct_prompt,
    make_self_instruct_prompt_with_gif,
    extract_concepts_and_descriptions,
)  # noqa: E402, import after definition

# ----------------------------- gif utils -----------------------------

def process_gif_via_llm(gif_path: Path, gif_model, gif_provider, args) -> Dict[str, Any]:
    """Send GIF to LLM and return parsed JSON result."""
    # encode
    MAX_SIZE = 20 * 1024 * 1024
    check_path, _ = check_file_size(str(gif_path), MAX_SIZE)
    base64_encoded = direct_encode_gif_to_base64(check_path)

    # load prompt template
    prompt_tpl = (
        PROMPT_DIR / ("gif_intergrated.md" if args.intergrated else "gif.md")
    ).read_text(encoding="utf-8")
    sys_tpl = (
        PROMPT_DIR / ("system_prompt_gif_intergrated.md" if args.intergrated else "system_prompt_gif.md")
    ).read_text(encoding="utf-8")

    image_block = {
        "type": "image_url",
        "image_url": {"url": f"data:image/gif;base64,{base64_encoded}"},
    }

    message_user = [{"type": "text", "text": prompt_tpl}, image_block]
    message_system = [{"type": "text", "text": sys_tpl}]

    client = LLMClient(provider=gif_provider, cache_dir=str(CACHE_DIR), system_content=message_system)

    req_meta = {
        "gif_name": gif_path.name,
        "model": gif_model.value,
        "temperature": args.temperature,
        "max_tokens": args.max_tokens,
    }

    try:
        resp = client.send_request(
            message_user,
            num_samples=1,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            model=gif_model,
            top_p=1,
        )
    except Exception as e:
        server_resp, status_code = None, None
        resp_obj = getattr(e, "response", None)
        if resp_obj is not None:
            server_resp = getattr(resp_obj, "text", str(resp_obj))
            status_code = getattr(resp_obj, "status_code", None)
        log_llm_error_csv(e, server_resp, status_code, req_meta, LOG_DIR / "gif_errors.csv")
        raise

    # parse JSON safely
    import json, re

    txt = resp.choices[0].message.content
    try:
        return json.loads(txt)
    except json.JSONDecodeError:
        filtered = re.sub(r"^```json\n|\n```$", "", txt)
        return json.loads(filtered)


# ----------------------------- core pipeline -----------------------------

def build_seed_prompt(seeds_contents, rng_seed, args):
    return make_self_instruct_prompt(
        seeds_contents=seeds_contents,
        rng_seed=str(rng_seed),
        num_descriptions=args.num_descriptions,
        use_concepts=args.use_concepts,
        num_generations=args.num_generations,
    )


def build_gif_prompt(seed_prompt: str, gif_json: Dict[str, Any], rng_seed, seeds_contents, args):
    """Wrap existing helper to embed GIF context."""
    return make_self_instruct_prompt_with_gif(
        seeds_contents=seeds_contents,
        rng_seed=str(rng_seed),
        num_descriptions=args.num_descriptions,
        use_concepts=args.use_concepts,
        num_generations=args.num_generations,
        gif_result=gif_json,
        intergrated=args.intergrated,
    )


# LLM call for seed or gif prompt

def call_llm(prompt: str, model, provider, args):
    client = LLMClient(provider=provider, cache_dir=str(CACHE_DIR))
    return client.generate(prompt, num_samples=1, max_tokens=args.max_tokens, temperature=args.temperature, model=model)[0]


# writing result

def write_result(task: Dict[str, Any], samples: List[str], gif_json: Dict[str, Any] | None, args):
    concepts_descriptions = []
    for sample in samples:
        parsed_concepts_lst, parsed_description_lst = extract_concepts_and_descriptions(sample)
        for c, d in zip(parsed_concepts_lst, parsed_description_lst):
            if c and d:
                concepts_descriptions.append((", ".join(c), d))

    if not concepts_descriptions:
        return None

    RESULT_DIR.mkdir(exist_ok=True)
    if task["mode"] == "seed":
        fname = RESULT_DIR / f"seed_only_{task['seed_rng']}.jsonl"
    else:
        base = Path(task["gif_path"]).stem
        fname = RESULT_DIR / f"gif_{base}.jsonl"

    with fname.open("w", encoding="utf-8") as f:
        import json

        for concepts, desc in concepts_descriptions:
            data = {"concepts": concepts, "description": desc}
            if gif_json:
                data.update(gif_json)
                data["gif_path"] = task.get("gif_path")
            f.write(json.dumps(data) + "\n")
    return str(fname)


# ----------------------------- wrapper per task -----------------------------

def run_pipeline(task: Dict[str, Any], idx, seeds_contents, args, seed_model, seed_provider, gif_model, gif_provider):
    """Return dict(result or error)"""
    print(f"Running task: {idx} is started")
    try:
        seed_prompt = build_seed_prompt(seeds_contents, task["seed_rng"], args)

        if task["mode"] == "seed":
            final_prompt = seed_prompt
            gif_json = None
        else:
            gif_json = process_gif_via_llm(Path(task["gif_path"]), gif_model, gif_provider, args)
            final_prompt = build_gif_prompt(seed_prompt, gif_json, task["seed_rng"], seeds_contents, args)

        sample = call_llm(final_prompt, seed_model, seed_provider, args)
        out_file = write_result(task, [sample], gif_json, args)
        return {"ok": True, "file": out_file}
    except Exception as e:
        return {"ok": False, "error": str(e), "task": task}


# ----------------------------- CLI entry -----------------------------

def cli():
    import argparse, hashlib, json, threading

    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["seed", "gif", "both"], default="both")
    p.add_argument("--parallel", type=int, default=4, help="동시 요청 수")
    p.add_argument("--use_concepts", "-uc", action="store_false", help="make the prompts not use concepts", default=True)
    p.add_argument("--gif_dir", default="data/geometry", help="GIF 샘플 폴더")
    p.add_argument("--samples", type=int, default=4, help="GIF 샘플 개수")
    p.add_argument("--num_descriptions", type=int, default=None)
    p.add_argument("--batch_size", type=int, default=10)
    p.add_argument("--num_generations", type=int, default=5)
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--model", default="o3-mini")
    p.add_argument("--max_tokens", type=int, default=2048)
    p.add_argument("--intergrated", action="store_true")
    args = p.parse_args()

    # convert models
    for provider, model in [
        (pr, m)
        for pr, lst in LLMClient.AVAILABLE_MODELS.items()
        for m in lst
    ]:
        if model.value == args.model:
            seed_provider, seed_model = provider, model
            break
    # GIF 분석용 모델(o4-mini default)
    gif_mode_name = "o4-mini"
    for provider, model in [
        (pr, m)
        for pr, lst in LLMClient.AVAILABLE_MODELS.items()
        for m in lst
    ]:
        if model.value == gif_mode_name:
            gif_provider, gif_model = provider, model
            break

    seeds_contents = load_seeds()

    # task list 구성
    tasks: List[Dict[str, Any]] = []
    rng = random.Random(42)

    if args.mode in ("seed", "both"):
        for i in range(args.samples):
            tasks.append({"mode": "seed", "seed_rng": rng.randint(0, 1 << 30)})

    if args.mode in ("gif", "both"):
        gifs = [f for f in Path(args.gif_dir).iterdir() if f.suffix in {".gif", ".webm"}]   
        rng.shuffle(gifs)
        for gif_path in gifs[: args.samples]:
            tasks.append({
                "mode": "gif",
                "gif_path": str(gif_path),
                "seed_rng": rng.randint(0, 1 << 30),
            })

    # 병렬 실행
    results = []
    with ThreadPoolExecutor(max_workers=args.parallel) as ex:
        fut_map = {
            ex.submit(
                run_pipeline,
                idx,
                t,
                seeds_contents,
                args,
                seed_model,
                seed_provider,
                gif_model,
                gif_provider,
            ): t
            for t,idx in enumerate(tasks)
        }
        for fut in tqdm(as_completed(fut_map), total=len(fut_map), desc="Tasks"):
            results.append(fut.result())

    # 요약 출력
    success = [r for r in results if r["ok"]]
    fail = [r for r in results if not r["ok"]]

    print(f"✅ {len(success)} succeeded, ❌ {len(fail)} failed")
    for r in fail:
        print(r["task"], "→", r["error"])


if __name__ == "__main__":
    cli()
