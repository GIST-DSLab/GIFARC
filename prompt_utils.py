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