
from llm import *
import argparse

def parse_cli_args():
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
    return arguments

def parse_cli_args_step_3():
    parser = argparse.ArgumentParser(description = "problem generator")

    parser.add_argument("--jsonl", type=str, default=None, help="jsonl file descriptions to use in prompts")
    parser.add_argument("--num_seeds", "-s", type=int, default=1, help="how many seeds to show in the prompt, if more than 1")
    parser.add_argument("--temperature", "-t", type=float, default=0.7)
    parser.add_argument("--num_samples", "-n", type=int, default=1, help="how many samples to generate")
    parser.add_argument("--prompt_model", "-pm", type=str, default="gpt-4-turbo", help="which model to use for problem generation", 
                        choices=[m.value for model_list in LLMClient.AVAILABLE_MODELS.values() for m in model_list])
    parser.add_argument("--embedding_model", "-em", type=str, default="text-embedding-ada-002", help="which model to use for embedding",
                        choices=[m.value for model_list in LLMClient.AVAILABLE_MODELS.values() for m in model_list])
    parser.add_argument("--sample_parallel", "-sp", type=int, default=1, help="how many parallel workers to use for sampling")
    parser.add_argument("--max_tokens", type=int, default=2048, help="max number of tokens for generation")
    parser.add_argument("--brief_common", "-bc", action="store_false", help="whether to not include common functions that are called in the seed code", default=True)
    parser.add_argument("--nohtml", action="store_true", help="don't generate html", default=False)
    parser.add_argument("--use_concept_embeddings", "-uc", action="store_true", help="use concept embeddings in addition to description embeddings", default=False)
    parser.add_argument("--ignore_cache_samples", "-ics", action="store_true", help="ignore cache for samples", default=False)
    parser.add_argument("--suggest_function", "-sf", action="store_true", help="suggest a function to use in the prompt", default=False)
    parser.add_argument("--batch_request", "-br", action="store_true", help="use batch request API", default=False)
    parser.add_argument("--outdir", default=None, help="output directory for the code")
    parser.add_argument("--test", default=None, help="output directory for the code")

    arguments = parser.parse_args()
    return arguments
    