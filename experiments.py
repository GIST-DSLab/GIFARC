import os
import json
import csv
import argparse
import re
import ast
import traceback
from datetime import datetime
from openai import AzureOpenAI, OpenAI
import backoff
import numpy as np
from prompt import get_common_lib_from_file
import ast
import pickle
import tqdm
from arc import train_problems, validation_problems, ArcProblem
from execution import multi_execute_transformation
from seeds.common import *
from enum import Enum
import glob
import seeds.common
import copy
import random

random.seed(777)

API_MODE = 'azure'
API_KEY =  os.getenv("OPENAI_API_KEY")
API_BASE = os.getenv("AZURE_OPENAI_API_BASE")

TRANSPOSE = False

MULTI_EXECUTE = True

TEST_LIMIT = 13

FEW_SHOT_COUNT = 15

LOG_FILE = "error_log.csv"

if API_MODE == 'azure':
    openai = AzureOpenAI(api_key="API_KEY", azure_endpoint=API_BASE, api_version = "2024-12-01-preview")
else:
    openai = OpenAI(api_key=API_KEY, base_url=API_BASE)

DEFAULT_SYSTEM_PROMPT = """
###############################
###  ROLE
###############################
You are an expert Python solver for ARC (Abstraction & Reasoning Corpus) puzzles.  
Your task is always to produce the **correct output grid** as a NumPy 2-D array of
integers 0-9.

###############################
###  RULES
###############################
1. **INPUT** : a single NumPy 2-D array (`ndarray`) of integers 0-9.  
2. **OUTPUT**: a NumPy 2-D array (`ndarray`) of the shape required by the puzzle.  
   *No lists, nested lists, or other types—**always a NumPy array***.  
3. Do **not** print intermediate debug info (keep diagnostics in comments).

###############################
###  LIB
###############################
{common_lib}
No external libraries other than **NumPy** are allowed.

"""

DEFAULT_HEAD_PROMPT = """
────────────────────────────────────────────
  TOP-LEVEL INSTRUCTIONS  –  READ FIRST
────────────────────────────────────────────
• The section between <# Few-Shot Examples> and <# Task> is an example block that demonstrates how to solve the task using analogies. 
  └─ *Do NOT execute or return their outputs.*  
• The lines that follow # Task present the task you must solve, consisting of Training Examples and Test Examples.
"""

DEFAULT_EVAL_HEAD_PROMPT = """
────────────────────────────────────────────
  TOP-LEVEL INSTRUCTIONS  –  READ FIRST
────────────────────────────────────────────
You will receive two short texts:

• the ground-truth analogy, which faithfully describes the transformation pattern of an ARC task
• the generated analogy, which attempts to describe the same pattern through a comparison, metaphor, or other figurative device

Your job is to evaluate how well the generated analogy aligns with the ground-truth analogy.
Focus on whether the key concepts, processes, and purposes match.
"""

ANALOGY_PROCESS_CODE = """
# Please solve {uid} task thoroughly and structure your response with the following sections:

# THINKING ANALOGY
[Let's think the analogical pattern about {uid} task step by step]
Let's think step by step about the analogical pattern in the {uid} task, and describe what kind of analogical, metaphorical, or figurative pattern it is

# THINKING PROCESS
[Let's think the {uid} task step by step by leveraging thinking analogy and task analogy]
Let's think the {uid} task leveraging the analogical, metaphorical, or figurative patterns observed in the 15 examples to uncover and apply the underlying analogy

# SOLUTION CODE
[Write the final solution code for the {uid} task based on the analogical approach identified in the thinking process]
"""

PROCESS_CODE = """
# Please solve {uid} task thoroughly and structure your response with the following sections:

# THINKING PROCESS
[Let's think the {uid} task step by step]

# SOLUTION CODE
[Write the final solution code for the {uid} task based on the analogical approach identified in the thinking process]
"""

EVALUATION_GROUND_TRUTH_ANALOGY_AND_GENERATED_ANALOGY = """
You will be given a ground-truth analogy from the Abstraction and Reasoning Corpus (ARC) along with an generated analogy sentence that expresses the pattern of that task using a comparison, metaphor, or figurative expression.

Evaluate how well the given generated analogy aligns with the ground-truth analogy that include actual transformation pattern of the task.
Score the alignment on a scale from 0 to 1:

A score close to 1 means the generated analogy aligns very well with the ground-truth analogy.

A score close to 0 means the generated analogy and the ground-truth analogy do not align meaningfully.

Your response should include only the numerical score between 0 and 1.

[Evaluation criteria]
- Does the generated analogy accurately reflect the core concepts/processes/purposes of the ground-truth analogy?

The following [Example] is a two-shot example for this.

===============================================

[Example]

*Ground-truth analogy*: The task shows a Tetris-like grid system where objects fall from above, and when a line is completely filled with objects, that line disappears. After line elimination, all objects above the removed line descend downward by the number of lines that have been eliminated. This pattern is analogous to a gravitational system where suspended objects naturally fall when their support is removed. The figurative pattern resembles how books on a bookshelf would react if a shelf were suddenly removed - all books on higher shelves would drop down precisely to fill the newly created empty space.

----------------------

*Generated analogy*: The task demonstrates a filtering system where elements are selectively removed based on their color values. When specific color combinations are detected, the system applies a masking algorithm that preserves only certain elements while eliminating others according to predefined rules. This process is like a digital image filter that recognizes and removes particular pixel values while maintaining others. This is analogous to how a sieve works with different sized particles - larger particles remain on top while smaller ones pass through, creating a gradual stratification of materials. The figurative pattern resembles how a photo editing software selectively removes red-eye while preserving the rest of the image intact.

{
    "analysis": "Both analogies involve removing elements, but they diverge in the trigger and the subsequent behavior. The ground-truth analogy centers on whole-row elimination followed by a gravity-like downward shift of everything above, emphasizing positional dynamics. The generated analogy instead frames the task as color-based filtering with no mention of rows, gravity, or cascading descent. While they share the general notion of selective removal, the key structural and dynamic aspects (line completion, gravitational drop) are missing in the generated version, so alignment is weak.",
    "answer": "0.15"
}

===============================================

Now I'll provide you with a ground-truth analogy and a generated analogy for you to analyze and evaluate whether they align.

*Ground-truth analogy*: {ground_truth_analogy}

----------------------

*Generated analogy*: {generated_analogy}

Please respond in the following JSON format:
{
    "analysis": "Analysis of whether ground-truth analogy and a generated analogy align according to the above criteria",
    "answer": "Analyze ground-truth analogy and a generated analogy, and assign a score from 0 to 1 indicating how well they align — with 1 meaning perfect alignment and 0 meaning no alignment. Score value is float between 0 and 1"
}

IMPORTANT: In the "answer" field, include only the numeric value (e.g., 0.35) without any additional text, quotes, or explanations.
"""


EVALUATION_TASK_AND_ANALOGY = """
You will be given a task from the Abstraction and Reasoning Corpus (ARC) along with an analogy sentence that expresses the pattern of that task using a comparison, metaphor, or figurative expression.

Evaluate how well the given analogy aligns with the actual transformation pattern of the task.
Score the alignment on a scale from 0 to 1:

A score close to 1 means the analogy aligns very well with the task's transformation pattern.

A score close to 0 means the analogy and the task do not align meaningfully.

Your response should include only the numerical score between 0 and 1.

[Evaluation criteria]
- Does the analogy accurately reflect the core concepts/processes/purposes of the task?

The following [Example] is a two-shot example for this.

===============================================

[Example-Alignment]
Task: 
Input-1:
[[0,0,1,1],
[0,0,0,0],
[0,4,0,0],
[3,4,0,0]]
Output-1:
[[0,0,0,0],
[0,0,0,0],
[0,0,0,0],
[0,4,0,0]]

Input-2:
[[3,3,3,0],
[0,0,0,0],
[5,5,0,0],
[5,5,2,0]]
Output-2:
[[0,0,0,0],
[3,3,3,0],
[5,5,0,0],
[5,5,2,0]]

Input-3:
[[0,0,1,1],
[0,0,0,0],
[4,4,0,0],
[4,4,0,0]]
Output-3:
[[0,0,0,0],
[0,0,0,0],
[0,0,0,0],
[4,4,0,0]]

analogy: "The task shows a Tetris-like grid system where objects fall from above, and when a line is completely filled with objects, that line disappears. After line elimination, all objects above the removed line descend downward by the number of lines that have been eliminated. This pattern is analogous to a gravitational system where suspended objects naturally fall when their support is removed. The figurative pattern resembles how books on a bookshelf would react if a shelf were suddenly removed - all books on higher shelves would drop down precisely to fill the newly created empty space."

{
    "analysis": "This task involves objects falling from above like in Tetris, and when a line is completely filled, it disappears, causing the objects above to descend by the number of lines that disappeared. Therefore, the provided analogy accurately reflects the core concepts/processes/purposes of the task's pattern.",
    "answer": "1.0"
}

===============================================

[Example-Misalignment]

Task: 
Input-1:
[[0,0,1,1],
[0,0,0,0],
[0,4,0,0],
[3,4,0,0]]
Output-1:
[[0,0,0,0],
[0,0,0,0],
[0,0,0,0],
[0,4,0,0]]

Input-2:
[[3,3,3,0],
[0,0,0,0],
[5,5,0,0],
[5,5,2,0]]
Output-2:
[[0,0,0,0],
[3,3,3,0],
[5,5,0,0],
[5,5,2,0]]

Input-3:
[[0,0,1,1],
[0,0,0,0],
[4,4,0,0],
[4,4,0,0]]
Output-3:
[[0,0,0,0],
[0,0,0,0],
[0,0,0,0],
[4,4,0,0]]

analogy: "The task demonstrates a filtering system where elements are selectively removed based on their color values. When specific color combinations are detected, the system applies a masking algorithm that preserves only certain elements while eliminating others according to predefined rules. This process is like a digital image filter that recognizes and removes particular pixel values while maintaining others. This is analogous to how a sieve works with different sized particles - larger particles remain on top while smaller ones pass through, creating a gradual stratification of materials. The figurative pattern resembles how a photo editing software selectively removes red-eye while preserving the rest of the image intact."

{
    "analysis": "This analogy incorrectly describes the task's pattern. The task shows a clear pattern where completely filled rows disappear and objects above fall down to fill the empty space - like Tetris. However, the provided analogy describes a filtering system based on color values and selective removal based on predefined rules, which doesn't match what's happening. There's no evidence of filtering by color or selective preservation in the examples; instead, entire rows are eliminated when filled. The sieve analogy also misrepresents the process, as there's no gradual stratification - just complete row removal and downward movement.",
    "answer": "0.0"
}

===============================================

Now I'll provide you with a task and an analogy for you to analyze and evaluate whether they align.

{uid} Task
{task}

analogy: {analogy}

Please respond in the following JSON format:
{
    "analysis": "Analysis of whether the task and analogy align according to the above criteria",
    "answer": "Analyze the task and the analogy, and assign a score from 0 to 1 indicating how well they align — with 1 meaning perfect alignment and 0 meaning no alignment."
}

IMPORTANT: In the "answer" field, include ONLY the number (0, 1) without any additional text, quotes, or explanations.
"""

EVALUATION_TASK_AND_ANALOGY_MULTIPLE_CHOICE = """
First, briefly summarize each option and compare its pros and cons in one to two sentences each. List your summaries and comparisons before the final JSON.

===============================================

The following shows a task from the Abstraction and Reasoning Corpus (ARC) and related information. Analyze it and choose the most appropriate answer.

{uid} Task
{task}

Which analogy (comparison/metaphor) best explains the pattern in the above task?

[Option 1]
{analogy_1}

[Option 2]
{analogy_2}

[Option 3]
{analogy_3}

[Option 4]
{analogy_4}

[Option 5]
{analogy_5}

===============================================

Only output the JSON object, with no extra text:

===============================================
```json
{
    "reasoning": {
    "option_summaries": [
        "Summary and pros/cons of option 1.",
        "Summary and pros/cons of option 2.",
        "Summary and pros/cons of option 3.",
        "Summary and pros/cons of option 4.",
        "Summary and pros/cons of option 5."
    ]
    },
    "answer": "number only, from 1 to 5, without any additional text"
    "explanation": "Write explanation of why you chose this option",
}
```
===============================================

"""

DEFAULT_COT = "\n\nLet's solve the problem step by step.\n"

ERROR_LOG_FILE = "experiments_error.csv"

class GridComparisonResult(Enum):
    EQUAL = 0
    SHAPE_MISMATCH = 1
    CONTENT_MISMATCH = 2
    TYPE_MISMATCH = 3
    ERROR = 4
    NON_2D_ARRAY = 5

class IOPair:
    x: np.ndarray
    y: np.ndarray
    def __init__(self, x, y):
        self.x = x
        self.y = y
        # check type
        assert isinstance(self.x, np.ndarray)
        assert isinstance(self.y, np.ndarray)
        # check shape
        assert len(self.x.shape) == 2
        assert len(self.y.shape) == 2

class Problem:
    # typing hint for the members
    filename: str
    seed_id: str
    code: str
    train_pairs: list
    test_pairs: list

    def __init__(self, filename=None, code=None, seed_id=None, train_pairs=None, test_pairs=None, mode=None):
        self.filename = filename
        self.seed_id = None
        self.mode=mode
        if filename:
            self.seed_id = filename.split(".")[0]
            if "_" in self.seed_id:
                self.seed_id= self.seed_id.split("_")[0]
        if seed_id:
            self.seed_id = seed_id
        if self.seed_id:
            pattern = r"[0-9a-f]{8}"
            assert re.match(pattern, self.seed_id)
            self.load_arc_problem(self.seed_id, mode=self.mode)

        self.code = code
        if train_pairs:
            self.train_pairs = train_pairs
        if test_pairs:
            self.test_pairs = test_pairs

        assert self.code, "Code is not provided"
        assert self.train_pairs, "Train pairs are not provided"
        assert self.test_pairs, "Test pairs are not provided"
        # check type
        assert isinstance(self.train_pairs, list)
        assert isinstance(self.test_pairs, list)
        assert all(isinstance(pair, IOPair) for pair in self.train_pairs)
        assert all(isinstance(pair, IOPair) for pair in self.test_pairs)


    def load_arc_problem(self, seed_id, mode=None):
        # using train_problems
        arc_problem = None
        for problem in train_problems + validation_problems:
            if problem.uid == seed_id:
                arc_problem = problem
                break
        assert arc_problem is not None
        self.train_pairs = []
        for pair in arc_problem.train_pairs:
            self.train_pairs.append(IOPair(pair.x.T, pair.y.T))
        self.test_pairs = []
        for pair in arc_problem.test_pairs:
            self.test_pairs.append(IOPair(pair.x.T, pair.y.T))

# @backoff.on_exception(backoff.expo, (Exception), max_tries=11, jitter=None)
def chat_with_retry(**kwargs):
    response = openai.chat.completions.create(**kwargs)
    return response

def send_embedding_request(input, model):
    response = openai.embeddings.create(
        model=model,
        input=input,
        encoding_format="float"
    )
    return response.data[0].embedding

def parse_thinking_analogy(text):
    match = re.search(
        r'#\s*THINKING\s+ANALOGY\s*\n(.*?)(?=\n#\s*(THINKING\s+PROCESS|SOLUTION\s+CODE)|\Z)',
        text,
        re.DOTALL | re.IGNORECASE
    )
    if match:
        thinking_analogy = match.group(1).strip()
    else:
        thinking_analogy = None
    return thinking_analogy

def parse_arc_solution(text):
    # THINKING PROCESS 추출
    thinking_match = re.search(r'# THINKING PROCESS\n(.*?)\n# SOLUTION CODE', text, re.DOTALL)
    thinking_process = thinking_match.group(1).strip() if thinking_match else None

    # SOLUTION CODE 추출 (더 유연한 패턴 허용)
    solution_match = re.search(r'# SOLUTION CODE\s*```(?:python)?\n(.*?)```', text, re.DOTALL)
    solution_code = solution_match.group(1).strip() if solution_match else None

    return thinking_process, solution_code

def grid_to_input(grid):
    return "\n".join("|".join(str(c) for c in row) for row in grid)

def make_problem_input_str(problem: Problem, uid=None):
    prompt = f"<{uid} task>\n"
    prompt += "<Training Examples>\n"
    prompt += "The following is a puzzle from the ARC dataset. Given training examples of input and output grids, predict the output grid for the test inputs.\n"
    prompt += "Here are the input and output grids for the training examples:\n"
    for pair in problem.train_pairs:
        prompt += f"Input:\n{grid_to_input(pair.x)}\nOutput:\n{grid_to_input(pair.y)}\n\n" 
    prompt += "<Test Examples>\n"
    prompt += "Here are the input grids for the test example:\n"
    prompt += "Input:\n" + "\n".join(grid_to_input(pair.x) for pair in problem.test_pairs)
    prompt += "\nGiven training examples of input and output grids, predict the output grid for the test inputs."
    prompt += "\n</task>"
    return prompt


def make_input_prompt(problem: Problem, common_lib: str, experiment_id: int = 0, test_mode: bool = False, uid=None):
    assert experiment_id in [1,2,3,4], "experiment_id should be 1, 2, 3 or 4"

    common_lib_prefix = f"""
We first define a common library that contains the functions that you can use to solve the Puzzle.
Here is the common library function signature and docstring that you can use to solve the problem (skipping the implementation for brevity):
```python
{common_lib}
```
"""
    # question = '\n\n# Task\n' + common_lib_prefix + make_problem_input_str(problem)
    question = '\n\n# Task\n' + make_problem_input_str(problem, uid) 

    # few_shot_prompt = f"\n# Few-Shot Examples\n<additional_information>\nThis information provides helpful context to solve the given ARC task.\n\n"

    few_shot_prompt = DEFAULT_HEAD_PROMPT + f"\n\n# Few-Shot Examples\n<additional_information>\nThis information provides helpful context to solve the given ARC task.\n\n"
    # few_shot_prompt =  f"\n\n# Few-Shot Examples\n<additional_information>\nThis information provides helpful context to solve the given ARC task.\n\n"
    data = []
    examples = []

    # if experiment_id == 1:
    #     with open("experiment_results/data/improving_analogy_and_solution.jsonl", "r", encoding="utf-8") as f:
    #         for i, line in enumerate(f,1):
    #             if i > FEW_SHOT_COUNT:
    #                 break
    #             data.append(json.loads(line))
    #             grid = ""
    #             for pair in problem.train_pairs:
    #                 grid += f"Input:\n{grid_to_input(pair.x)}\nOutput:\n{grid_to_input(pair.y)}\n\n" 
    #             examples.append(f'''<example-{i}>\n<grid>\n{grid}\n</grid>\n</example-{i}>''')
    if experiment_id == 1:
        with open("experiment_results/data/flat_analogy_and_solution.jsonl", "r", encoding="utf-8") as f:
            for i, line in enumerate(f,1):
                if i > FEW_SHOT_COUNT:  
                    break
                data.append(json.loads(line))
                grid = ""
                for pair in problem.train_pairs:
                    grid += f"Input:\n{grid_to_input(pair.x)}\nOutput:\n{grid_to_input(pair.y)}\n\n" 
                examples.append(f'''<example-{i}>\n<grid>\n{grid}\n</grid>\n<solution>\n{data[i-1]['solution']}\n</solution>\n</example-{i}>\n\n''')
    elif experiment_id == 2:
        with open("experiment_results/data/flat_analogy_and_solution.jsonl", "r", encoding="utf-8") as f:
            for i, line in enumerate(f,1):
                if i > FEW_SHOT_COUNT:  
                    break
                data.append(json.loads(line))
                grid = ""
                for pair in problem.train_pairs:
                    grid += f"Input:\n{grid_to_input(pair.x)}\nOutput:\n{grid_to_input(pair.y)}\n\n" 
                examples.append(f'''<example-{i}>\n<grid>\n{grid}\n</grid>\n<analogy>\n{data[i-1]['analogy']}\n</analogy>\n<solution>\n{data[i-1]['solution']}\n</solution>\n</example-{i}>\n\n''')
    elif experiment_id == 3:
        with open("experiment_results/data/improving_analogy_and_solution.jsonl", "r", encoding="utf-8") as f:
            for i, line in enumerate(f,1):
                if i > FEW_SHOT_COUNT:  
                    break
                data.append(json.loads(line))
                grid = ""
                for pair in problem.train_pairs:
                    grid += f"Input:\n{grid_to_input(pair.x)}\nOutput:\n{grid_to_input(pair.y)}\n\n" 
                examples.append(f'''<example-{i}>\n<grid>\n{grid}\n</grid>\n<analogy>\n{data[i-1]['analogy']}\n</analogy>\n<solution>\n{data[i-1]['solution']}\n</solution>\n</example-{i}>\n\n''')

    few_shot_prompt += ('\n').join(examples)
    # question += few_shot_prompt + '\n</additional_information>' #+ temp
    few_shot_prompt += '\n</additional_information>' + question + PROCESS_CODE.format(uid=uid) if experiment_id == 1 else question + ANALOGY_PROCESS_CODE.format(uid=uid)
    question = few_shot_prompt
    return question

def convert_chat_format(question, answer, method='induction', test_mode=False, uid=None, common_lib=None):
    global DEFAULT_SYSTEM_PROMPT

    if method == 'induction':
        with open("prompts/system_prompt_induction_solver_analogy.md", encoding="utf-8") as f: 
                system_prompt = f.read()
    elif method == 'induction_simple':
        with open("prompts/system_prompt_induction_solver_analogy_simple.md", encoding="utf-8") as f:
                system_prompt = f.read()
    elif method == 'transduction':
        with open("prompts/system_prompt_transduction_solver.md", encoding="utf-8") as f: 
                system_prompt = f.read()
    
    # if test_mode:
    #     with open("prompts/system_prompt_induction_solver_analogy_simple_xml.md", encoding="utf-8") as f:
    #             system_prompt = f.read()

    messages =  {
        "messages": [
            {"role": "system", "content": DEFAULT_SYSTEM_PROMPT + system_prompt if not test_mode else DEFAULT_SYSTEM_PROMPT.format(common_lib=common_lib) + DEFAULT_COT},
            {"role": "user", "content": question if not test_mode else question + DEFAULT_COT},
        ]
    }

    
    if answer:
        messages["messages"].append({"role": "assistant", "content": answer})
    return messages

def log_error(step, file_name, error, tb):
    print(f"[{step} ERROR] File: {file_name}")
    print(f"Error Message: {error}")
    print(f"Traceback:\n{tb}")

    os.makedirs(os.path.dirname(ERROR_LOG_FILE), exist_ok=True) if os.path.dirname(ERROR_LOG_FILE) else None
    file_exists = os.path.isfile(ERROR_LOG_FILE)

    with open(ERROR_LOG_FILE, mode='a', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        if not file_exists:
            writer.writerow(["step", "file_name", "error", "traceback"])
        writer.writerow([step, file_name, error, tb])

def extract_json_from_code_block(text):
    # JSON code block 패턴 (```json ... ``` 또는 ``` ... ```)
    json_pattern = r'```(?:json)?\s*([\s\S]*?)\s*```'
    matches = re.findall(json_pattern, text)
    
    if matches:
        return matches[0]  # 첫 번째 코드 블록 반환
    return None

def parse_json(text, file_name=None, method=None, mode=None):
    # 1차 시도: 직접 JSON 파싱
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        # 2차 시도: code block에서 JSON 추출
        if mode != 'other':
            json_text = extract_json_from_code_block(text)
            if json_text:
                try:
                    return json.loads(json_text)
                except json.JSONDecodeError:
                    # code block에서 추출한 JSON도 파싱 실패 시 다른 방법 진행
                    text = json_text  # 이후 과정은 추출된 텍스트로 계속 진행
        
        # 3차 시도: 백슬래시 문제 해결
        try:
            # 작은 따옴표 앞의 백슬래시 제거
            safe_str = text.replace("\\'", "'")
            # 백슬래시 이스케이프
            safe_str = safe_str.replace("\\", "\\\\")
            # 다시 따옴표 앞의 이중 백슬래시를 단일 백슬래시로 복원
            safe_str = safe_str.replace("\\\\'", "\\'")
            safe_str = safe_str.replace('\\\\"', '\\"')
            
            return json.loads(safe_str)
        except json.JSONDecodeError:
            # 4차 시도: ast.literal_eval로 Python dict 파싱
            try:
                return ast.literal_eval(text)
            except (SyntaxError, ValueError):
                # 5차 시도: 더 간단한 방법 - 백슬래시와 따옴표 정리
                try:
                    simple_safe_str = re.sub(r'\\\'|\\\"|\\|\'', '', text)
                    return json.loads(simple_safe_str)
                except json.JSONDecodeError as e:
                    if file_name:
                        print(f"JSON 파싱 실패 ({file_name}): {e}")
                        print(f"처리된 문자열: {simple_safe_str[:100]}...")
                    return None
                except Exception as e:
                    if file_name:
                        tb = traceback.format_exc()
                        log_error("parse_json", file_name, str(e), tb)
                    raise e

def parse_description_and_code(text: str):
    """
    Returns (description, code) tuple where
    - description: "# description:" 로 시작해서 그 뒤에 나오는 모든 주석 라인들
    - code: description 블록과 def generate_input 함수 전체를 제거한 나머지 코드 (import, def main, 기타 함수 포함)
    """
    # 1) description 블록 뽑기
    desc_match = re.search(
        r'(^#\s*description:[\s\S]*?)(?=^([^#]|\Z))',
        text, flags=re.MULTILINE
    )
    description = desc_match.group(1).rstrip() if desc_match else ""

    # 2) 전체에서 description 블록 제거
    without_desc = re.sub(
        r'^#\s*description:[\s\S]*?(?=^([^#]|\Z))',
        '',
        text, flags=re.MULTILINE
    )

    # 3) def generate_input(...) ~ 다음 def 또는 파일 끝까지 제거
    cleaned_code = re.sub(
        r'^def\s+generate_input\s*\([\s\S]*?)(?=^def\s|\Z)',
        '',
        without_desc, flags=re.MULTILINE
    )

    # 4) 앞뒤 공백 정리
    return description.strip(), cleaned_code.strip()

def check_content(text):
    if '# concepts:' in text:
        return False
    
    if 'def generate_input' in text:
        return False
    
    if 'def main' not in text:
        return False

    return True

def get_description_from_lines(lines):
    description = []
    for i, line in enumerate(lines):
        if "# description:" in line:
            while i+1 < len(lines) and lines[i+1].startswith("# "):
                description.append(lines[i+1][2:])
                i += 1
            description = " ".join(description)
            break
    if description == []:
        for i, line in enumerate(lines):
            if "description:" in line.lower():
                description.append(lines[i+1][2:])
                i += 1
                description = " ".join(description)
    return description

def extract_descriptions(content: str):

    lines = content.splitlines()
    all_descriptions = []

    for idx, line in enumerate(lines):
        low = line.lstrip().lower()

        if low.startswith("# description") or low.startswith("description"):
            # 이 지점부터 끝까지 넘겨서 설명 추출
            desc = get_description_from_lines(lines[idx:])
            if desc:
                all_descriptions.append(desc)

    return all_descriptions

def remove_concepts_and_descriptions(content: str) -> str:
    """
    content에서
      - '# concepts:' 블록 (다음 # 주석 라인들 포함)
      - '# description:' 블록 (다음 # 주석 라인들 포함)
    을 모두 제거하고 남은 텍스트를 반환합니다.
    """
    lines = content.splitlines()
    result = []
    skip = False

    for i, line in enumerate(lines):
        low = line.lstrip().lower()

        # concepts 블록 시작
        if low.startswith("# concepts"):
            skip = True
            continue

        # description 블록 시작
        if low.startswith("# description"):
            skip = True
            continue

        # 블록 내부(주석 라인)는 skip
        if skip and line.lstrip().startswith("# "):
            continue

        # 주석이 아닌 다른 라인이 나오면 skip 해제
        if skip:
            skip = False

        # skip이 꺼져 있을 때만 결과에 추가
        if not skip:
            result.append(line)

    return "\n".join(result)

def get_gifarc_examples(mode):
    gifarc_information = {}
    grid_list = []
    analogy_list = []
    solution_list = []
    file_name_list = []
    gifarc_data = []
    target_folder = 'experiment_results/data'
    target_file_path = os.path.join(target_folder, 'random_few_shot_gifarc_info.jsonl')

    os.makedirs(target_folder, exist_ok=True)
    try:
        if mode != 'curated':
            if not os.path.exists(target_file_path):
                with open('few_shot_example_ids.txt', 'r', encoding='utf-8') as f:
                    lines = f.read()
                gifarc_file_name_lists = lines.split(',')
                gifarc_file_name_lists = [file_name.strip() + '.jsonl' for file_name in gifarc_file_name_lists if file_name.strip()]

                with open('postprocessing_dataset/gifarc_integrated.jsonl', 'r', encoding='utf-8') as f:
                    lines = f.readlines()
                    for line in lines:
                        line_data = json.loads(line)
                        if line_data['file_name'] in gifarc_file_name_lists:
                            grid_list.append(line_data['examples'][:4])
                            analogy_text = extract_descriptions(line_data['seeds'][-1])[0]
                            solution_text = remove_concepts_and_descriptions(line_data['source'].split("def generate_input")[0])
                            if check_content(('\n').join([analogy_text, solution_text])):
                                analogy_list.append(analogy_text)
                                solution_list.append(solution_text)
                                file_name_list.append(line_data['file_name'])
                            else:
                                raise ValueError(f"Invalid content in file: {line_data['file_name']}")
                with open(target_file_path, 'w', encoding='utf-8') as f:
                    for i in range(len(grid_list)):
                        gifarc_information['grid'] = grid_list[i]
                        gifarc_information['analogy'] = analogy_list[i]
                        gifarc_information['solution'] = solution_list[i]
                        gifarc_information['file_name'] = file_name_list[i]
                        f.write(json.dumps(gifarc_information, ensure_ascii=False) + '\n')
                        gifarc_data.append({
                            'grid': gifarc_information['grid'],
                            'analogy': gifarc_information['analogy'],
                            'solution': gifarc_information['solution'],
                            'file_name': gifarc_information['file_name']
                        })
            else:
                with open(target_file_path, 'r', encoding='utf-8') as f:
                    lines = f.readlines()
                    for line in lines:
                        line_data = json.loads(line)
                        grid = line_data['grid']
                        analogy = line_data['analogy']
                        solution = line_data['solution']
                        file_name = line_data['file_name']
                        
                        gifarc_data.append({
                            'grid': grid,
                            'analogy': analogy,
                            'solution': solution,
                            'file_name': file_name
                        })
        else:
            with open('postprocessing_dataset/curated_set_1_gifarc_integrated.jsonl', 'r', encoding='utf-8') as f:
                lines = f.readlines()
                for line in lines:
                    line_data = json.loads(line)
                    grid_list.append(line_data['examples'][:4])
                    analogy_text = extract_descriptions(line_data['seeds'][-1])[0]
                    solution_text = remove_concepts_and_descriptions(line_data['source'].split("def generate_input")[0])
                    file_name = line_data['file_name']
                    if check_content(('\n').join([analogy_text, solution_text])):
                        analogy_list.append(analogy_text)
                        solution_list.append(solution_text)
                        file_name_list.append(file_name)
            
            with open(target_file_path, 'w', encoding='utf-8') as f:
                for i in range(len(grid_list)):
                    gifarc_information['grid'] = grid_list[i]
                    gifarc_information['analogy'] = analogy_list[i]
                    gifarc_information['solution'] = solution_list[i]
                    gifarc_information['file_name'] = file_name_list[i]
                    f.write(json.dumps(gifarc_information, ensure_ascii=False) + '\n')
                    gifarc_data.append({
                        'grid': gifarc_information['grid'],
                        'analogy': gifarc_information['analogy'],
                        'solution': gifarc_information['solution'],
                        'file_name': gifarc_information['file_name']
                    })
    except Exception as e:
        tb = traceback.format_exc()
        log_error("get_gifarc_example", "few_shot_example_ids.txt", str(e), tb)
        raise e
    
    return gifarc_data

def get_barc_examples(mode):
    gifarc_data = get_gifarc_examples(mode)
    barc_data = []
    target_folder = 'experiment_results/data'
    target_file_path = os.path.join(target_folder, 'few_shot_barc_info.jsonl')

    if not os.path.exists(target_file_path):
        with open('few_shot_gifarc_info.jsonl', 'r', encoding='utf-8') as f:
            lines = f.readlines()
            for line in lines:
                line_data = json.loads(line)
                grid = line_data['grid']
                analogy = line_data['analogy']
                solution = line_data['solution']
                
                gifarc_data.append({
                    'grid': grid,
                    'analogy': analogy,
                    'solution': solution
                })

        with open('postprocessing_dataset/barc_integrated.jsonl', 'r', encoding='utf-8') as f:
            lines = f.readlines()
            for i, line in enumerate(lines[:20],1):
                line_data = json.loads(line)
                grid = line_data['train'][:3]
                grid.extend(line_data['test'][0])

                analogy_text = extract_descriptions(line_data['seeds'][-1])[0]
                solution_text = remove_concepts_and_descriptions(line_data['source'].split("def generate_input")[0])
                removed_text = ('\n').join([analogy_text, solution_text])
                
                if not check_content(removed_text):
                    raise ValueError(f"Invalid content in file: {line_data['file_name']}")
                
                barc_data.append({
                    'grid': grid,
                    'analogy': analogy,
                    'solution': solution
                })
        
        with open(target_file_path, 'w', encoding='utf-8') as f:
            for i in range(len(barc_data)):
                f.write(json.dumps(barc_data[i], ensure_ascii=False) + '\n')
    else:
        with open(target_file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            for line in lines:
                line_data = json.loads(line)
                grid = line_data['grid']
                analogy = line_data['analogy']
                solution = line_data['solution']
                
                barc_data.append({
                    'grid': grid,
                    'analogy': analogy,
                    'solution': solution
                })
    
    return barc_data

def evaluation_origin_and_flat(file_name, original_text, flat_text, model_name, max_tokens, mode, target_setting_file_name):
    target_setting_folder = 'experiment_results/settings'
    if target_setting_file_name is None:
        target_setting_file_name = 'comparing_origin_and_flat' if mode == 'curated' else 'previous_comparing_origin_and_flat'
    target_jsonl_path = os.path.join(target_setting_folder, f'{target_setting_file_name}.jsonl')
    target_csv_path = os.path.join(target_setting_folder, f'{target_setting_file_name}.csv')
    os.makedirs(target_setting_folder, exist_ok=True)

    system_prompt = '''
You are an expert in analyzing the use of figurative language (analogies, metaphors, and other figurative expressions) in texts that contain both Task Explanation and Solution sections. Your analysis should consider how figurative language is used in both sections, both separately and collectively.

For each text, assess the degree to which figurative language is used according to this scale:
1. No usage - The text does not use figurative language
2. Light usage - The text uses figurative language sparingly or rarely
3. Moderate usage - The text regularly uses figurative language
4. Heavy usage - The text frequently or extensively uses figurative language

Then, evaluate the difference between the two texts using this scale:
1. No difference - There is no discernible difference in figurative language use between the two texts
2. Subtle difference - There are some differences in figurative language use, but they are not prominent
3. Clear difference - There are noticeable differences in figurative language use between the texts
4. Pronounced difference - There are very clear and significant differences in figurative language use

When analyzing, pay attention to:
- How figurative language is used in the Task Explanation section (which presents the problem)
- How figurative language is used in the Solution section (which presents the answer)
- How the overall text uses figurative language to connect concepts and facilitate understanding

Your response should be in the following JSON format:

{
    \"degree_of_A_task_explanation\": \"<integer between 1-4, where 1=No usage, 2=Light usage, 3=Moderate usage, 4=Heavy usage>\",
    \"degree_of_A_solutio\n": \"<integer between 1-4, where 1=No usage, 2=Light usage, 3=Moderate usage, 4=Heavy usage>\",
    \"degree_of_A_overall\": \"<integer between 1-4, where 1=No usage, 2=Light usage, 3=Moderate usage, 4=Heavy usage>\",
    \"degree_of_B_task_explanation\": \"<integer between 1-4, where 1=No usage, 2=Light usage, 3=Moderate usage, 4=Heavy usage>\",
    \"degree_of_B_solution\": \"<integer between 1-4, where 1=No usage, 2=Light usage, 3=Moderate usage, 4=Heavy usage>\",
    \"degree_of_B_overall\": \"<integer between 1-4, where 1=No usage, 2=Light usage, 3=Moderate usage, 4=Heavy usage>\",
    \"result\": \"<A or B - choose only one letter>\",
    \"result_reason\": \"<explanation text>\",
    \"degree_of_difference\": \"<integer between 1-4, where 1=No difference, 2=Subtle difference, 3=Clear difference, 4=Pronounced difference>\",
    \"degree_of_difference_reason\": \"<explanation text>\"
}

IMPORTANT: 
1. For all numeric degree ratings, use ONLY the integer values (1, 2, 3, or 4) WITHOUT quotation marks.
2. For text fields ("result", "result_reason", "degree_of_difference_reason"), ALWAYS enclose values in double quotation marks.
3. Ensure your JSON is properly formatted and can be parsed by a standard JSON parser.
4. For "result", use ONLY "A" or "B" (with the quotation marks).
5. AVOID using any backslashes (\) or escape characters in your text fields. Use alternative expressions where possible.
6. AVOID using single quotes (') within your text to prevent JSON parsing issues.
'''

    user_prompt = '''
Please analyze the differences in figurative language (analogies, metaphors, and other figurative expressions) between texts [A] and [B] below. Both texts have a structure consisting of a Task Explanation section followed by a Solution section.

For each text, assess how much figurative language is used in:
1. The Task Explanation section
2. The Solution section
3. The text as a whole

Use this scale for your assessment:
1. No usage - The section does not use figurative language
2. Light usage - The section uses figurative language sparingly or rarely
3. Moderate usage - The section regularly uses figurative language
4. Heavy usage - The section frequently or extensively uses figurative language

Then determine which text uses figurative expressions more frequently or effectively overall, and what is the degree of difference between them according to the evaluation scale (1-4). Provide your judgment with specific examples from both the Task Explanation and Solution sections.

[A]
{original_text}
[B]
{flat_text}
'''

    question = user_prompt.format(
        original_text=original_text,
        flat_text=flat_text
    )

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": question},
    ]

    response = chat_with_retry(
        model=model_name,
        messages=messages,
        max_completion_tokens=max_tokens,
        top_p=1.0,
        temperature=0.0,
        n=1
        )

    result_text = response.choices[0].message.content

    try:
        result_json = parse_json(result_text, file_name)
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON: {e}")
        print(f"Response content: {result_text}")
        raise e
    result_json['A-original'] = original_text
    result_json['B-flat'] = flat_text
    result_json['file_name'] = file_name
    with open(target_jsonl_path, 'a', encoding='utf-8') as f_jsonl:
        f_jsonl.write(json.dumps(result_json, ensure_ascii=False) + '\n')

    csv_fields = ["file_name", "result", "result_reason", "degree_of_difference", "degree_of_difference_reason", "A-original", "B-flat", "degree_of_A_overall", "degree_of_A_task_explanation", "degree_of_A_solution", "degree_of_B_overall", "degree_of_B_task_explanation", "degree_of_B_solution"]
    write_header = not os.path.exists(target_csv_path) 

    with open(target_csv_path, 'a', newline='', encoding='utf-8') as f_csv:
        writer = csv.DictWriter(f_csv, fieldnames=csv_fields)
        if write_header:
            writer.writeheader()
        writer.writerow({field: result_json.get(field, "") for field in csv_fields})

    
    return result_json

def imporving_analogy_and_solution(model_name, max_tokens, source_file_name):
    data_folder = 'experiment_results/data'
    target_file_name = 'improving_analogy_and_solution'

    source_file_path = os.path.join(data_folder, f'{source_file_name}.jsonl')
    target_data_file_path = os.path.join(data_folder, f'{target_file_name}.jsonl')

    system_prompt = '''
You are a specialized AI assistant that improves consistency between task explanations and solution code by enhancing metaphorical and figurative language.

Your primary function is to analyze technical task explanations and solution code, identify metaphorical expressions in the task explanation, and then apply those metaphors consistently to the solution code (primarily through variable names and comments) while preserving all functionality.

# Figurative Language Usage Scale
1. No usage - The text does not use figurative language
2. Light usage - The text uses figurative language sparingly or rarely
3. Moderate usage - The text regularly uses figurative language
4. Heavy usage - The text frequently or extensively uses figurative language

# Core Responsibilities:
1. Extract figurative language from task explanations
2. Apply these metaphors to solution code (variable names, comments)
3. Ensure appropriate metaphorical intensity (aim for at least MODERATE USAGE - level 3 or higher)
4. Maintain technical precision and code functionality
5. Create consistent metaphorical frameworks throughout the code

# Process Guidelines:

## Analysis Stage:
- Identify all metaphors, similes, and figurative expressions in the task explanation
- Determine which technical concepts each metaphor describes
- Assess the general metaphorical "tone" (scientific, natural, mechanical, etc.)

## Application Stage:
- Rename variables to incorporate metaphors while maintaining clarity
- Enhance comments with figurative language that explains the code's purpose
- Ensure the figurative language usage reaches at least MODERATE USAGE (level 3) or higher
- Maintain consistent metaphorical themes throughout

## Constraints:
- Never change the functional code logic
- Preserve all numerical values and computational steps
- Only modify variable names and comments
- Do not introduce metaphors not derived from the task explanation
- Keep technical clarity as the highest priority

# Input:
You will receive a task explanation and solution code in the following format:
[Task Explanation]
{task explanation text}

[Solution]
{solution code}

# Output:
You must respond with a JSON object in the following strict format:
{
  \"task_explanation\": \"<original task explanation - do not modify>\",
  \"solution\": \"<improved solution with metaphorically enhanced variable names and comments>\"
}

Remember to preserve all code functionality while enhancing the metaphorical consistency between the task explanation and solution code. Aim to achieve at least MODERATE USAGE (level 3) or higher of figurative language in your solution, where figurative expressions are regularly used throughout variable names and comments with a consistent metaphorical framework.
'''
    user_prompt = '''
This is a task to improve the consistency of figurative expressions between [Task Explanation] and [Solution] code:

[Task Explanation]
{task_explanation}

[Solution]
{solution}

[Instructions]
Please modify the Solution code to be consistent with the figurative expressions in the Task Explanation. Follow these principles:

1. Analysis Stage:
   - First, extract the key figurative expressions used in the Task Explanation.
   - Identify which technical concepts or operations each metaphor describes.

2. Application Scope:
   - Variable names: Maintain technical clarity while appropriately adding metaphorical elements. (e.g., 'width' → 'pulse_width')
   - Comments: Preserve technical explanations while naturally integrating metaphors from the Task Explanation.
   - Function names: Modify only when necessary, and maintain names that clearly indicate functionality.

3. Metaphor Intensity Control:
   - Avoid excessive poetic expressions or verbose metaphors.
   - Keep figurative expressions at a level that enhances code understanding.
   - Make the code slightly more metaphorically rich than the Task Explanation, but don't compromise the code's professionalism.

4. Maintain Consistency:
   - Don't introduce new metaphorical systems not present in the Task Explanation.
   - Use consistent metaphors for the same concept.
   - Maintain a consistent metaphorical style throughout the code.

5. Preserve Functionality:
   - Never alter the functional operation of the code.
   - Preserve algorithm logic and numerical calculations as they are.
   - Modify only variable names and comments.

Please explain the following along with your modified code:
1. The main figurative expressions you found in the Task Explanation
2. How you applied each metaphor to the code (including examples of variable/comment changes)
3. How you maintained overall metaphorical consistency
'''
    
    # TODO 필터링 된 데이터에서 analogy와 solution을 강화시키기
    with open(source_file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        for line in lines:
            target = json.loads(line)
            grid = target['grid']
            analogy = target['analogy']
            solution = target['solution']
            file_name = target['file_name']

            examples = []
            for i in range(len(grid)):
                examples.append(f"Example {i+1}: {grid[i]}")
            
            question = user_prompt.format(
                task_explanation=target['analogy'],
                solution=target['solution'],
                examples="\n".join(examples)
            )

            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": question},
            ]

            response = chat_with_retry(
                model=model_name,
                messages=messages,
                max_completion_tokens=max_tokens,
                top_p=1.0,
                n=1
                )

            result_text = response.choices[0].message.content

            try:
                result_json = parse_json(result_text, file_name)
            except json.JSONDecodeError as e:
                print(f"Error decoding JSON: {e}")
                print(f"Response content: {result_text}")
                raise e
            
            imporved_task_explanation = result_json['task_explanation']
            improved_solution = result_json['solution']

            improved_target = {}
            improved_target['analogy'] = imporved_task_explanation
            improved_target['solution'] = improved_solution
            improved_target['file_name'] = file_name
            improved_target['grid'] = grid
            
            with open(target_data_file_path, 'a', encoding='utf-8') as f_jsonl:
                f_jsonl.write(json.dumps(improved_target, ensure_ascii=False) + '\n')
            
            make_flat_information(gifarc_data=improved_target, barc_data=None, model_name=model_name, max_tokens=max_tokens, mode='curated', target_setting_file_name=target_file_name)


    return 

def filtering_analogy_and_solution_to_imporving(model_name, max_tokens):
    base_data_folder = 'experiment_results/data'
    base_setting_folder = 'experiment_results/settings'

    curated_set_source_file_name = 'few_shot_gifarc_info'
    previouse_set_source_file_name = 'previous_few_shot_gifarc_info'

    curated_jsonl_file_name = 'comparing_origin_and_flat'
    previous_jsonl_file_name = 'previous_comparing_origin_and_flat'

    target_file_name = 'filtering_analogy_and_solution_to_imporving'

    curated_set_source_file_path = os.path.join(base_data_folder, f'{curated_set_source_file_name}.jsonl')
    previous_set_source_file_path = os.path.join(base_data_folder, f'{previouse_set_source_file_name}.jsonl')

    curated_evaluation_file_path = os.path.join(base_setting_folder, f'{curated_jsonl_file_name}.jsonl')
    previous_evaluation_file_path = os.path.join(base_setting_folder, f'{previous_jsonl_file_name}.jsonl')

    target_data_jsonl_path = os.path.join(base_data_folder, f'{target_file_name}.jsonl')
    target_setting_jsonl_path = os.path.join(base_setting_folder, f'{target_file_name}.jsonl')
    target_setting_csv_path = os.path.join(base_setting_folder, f'{target_file_name}.csv')

    total_source_data = []
    total_evaluation_data = []
    with open(curated_set_source_file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        for line in lines:
            line_data = json.loads(line)
            total_source_data.append(line_data)
    
    with open(previous_set_source_file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        for line in lines:
            line_data = json.loads(line)
            total_source_data.append(line_data)

    curated_filtered_file_list = []
    with open(curated_evaluation_file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        for line in lines:
            line_data = json.loads(line)
            if line_data['degree_of_A_task_explanation'] >= 3:
                curated_filtered_file_list.append(line_data['file_name'])
                total_evaluation_data.append(line_data)
    
    previous_filtered_file_list = []
    with open(previous_evaluation_file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        for line in lines:
            line_data = json.loads(line)
            if line_data['degree_of_A_task_explanation'] >= 3:
                previous_filtered_file_list.append(line_data['file_name'])
                total_evaluation_data.append(line_data)
    
    for i, target in enumerate(total_source_data):
        file_name = target['file_name']

        if file_name in curated_filtered_file_list or file_name in previous_filtered_file_list:
            with open(target_data_jsonl_path, 'a', encoding='utf-8') as f_jsonl:
                f_jsonl.write(json.dumps(target, ensure_ascii=False) + '\n')

    for i, target in enumerate(total_evaluation_data):
        with open(target_setting_jsonl_path, 'a', encoding='utf-8') as f_jsonl:
            f_jsonl.write(json.dumps(target, ensure_ascii=False) + '\n')

    csv_fields = ["file_name", "result", "result_reason", "degree_of_difference", "degree_of_difference_reason", "A-original", "B-flat", "degree_of_A_overall", "degree_of_A_task_explanation", "degree_of_A_solution", "degree_of_B_overall", "degree_of_B_task_explanation", "degree_of_B_solution"]
    write_header = not os.path.exists(target_setting_csv_path) 

    for i, target in enumerate(total_evaluation_data):
        with open(target_setting_csv_path, 'a', newline='', encoding='utf-8') as f_csv:
            writer = csv.DictWriter(f_csv, fieldnames=csv_fields)
            if write_header:
                writer.writeheader()
            writer.writerow({field: target.get(field, "") for field in csv_fields})



    
    return

def make_flat_information(gifarc_data=None, barc_data=None, model_name='o3-mini',max_tokens=40000, mode='curated', target_setting_file_name=None):
    if target_setting_file_name is None:
        target_setting_file_name = 'comparing_origin_and_flat' if mode == 'curated' else 'previous_comparing_origin_and_flat'
    if gifarc_data is None:
        gifarc_data = get_gifarc_examples(mode)
    
    if barc_data is None:
        barc_data = get_barc_examples(mode)

    examples = []
    user_prompt = '''
    You need to transform the Target Content by removing analogical language, following the style of the examples I provide.

    [Target Content]
    <Task Explanation>
    {task_explanation}
    <Solution>
    {solution}

    [Style Examples]
    {examples}

    Based on the examples above, please transform the Target Content by:
    1. Identifying and removing all analogies, metaphors, and other figurative expressions
    2. Replacing them with direct, literal, and clear explanations
    3. Maintaining the original meaning while making the language more straightforward
    4. Preserving the overall structure

    The examples demonstrate the style I'm looking for - direct, literal, and free from analogical language. Analyze these examples to understand the desired style, then apply the same transformation approach to the Target Content.
    '''
    system_prompt = '''
    You are a specialized assistant that transforms text by removing analogical language. Your expertise lies in identifying and converting figurative expressions, metaphors, and analogies into direct, literal language while maintaining the original meaning and structure of the content.

    When given examples of the style, you carefully analyze them to understand the desired level of directness and clarity. You then apply this understanding to transform target content by:

    1. Precisely identifying all instances of analogical language, including metaphors, similes, and other figurative expressions
    2. Converting these instances to clear, straightforward explanations that convey the same information
    3. Ensuring the transformed content preserves the original meaning and technical accuracy
    4. Maintaining the overall structure, flow, and organization of the original content

    You focus exclusively on transforming analogical language into literal descriptions without changing the core information, terminology, or document structure. Your responses should only contain the transformed text unless additional explanation is specifically requested.

    You excel at understanding both the linguistic elements that need transformation and the subject matter context to ensure accurate conversions from figurative to literal language.

    You should return following JSON format:
    {
        \"task_explanation\": \"<Transformed Task Explanation>\",
        \"solution\": \"<Transformed Solution>\",
    }
    '''

    for i, target in enumerate(barc_data):
        analogy = target['analogy']
        solution = target['solution']

        examples.append(f'''(Example-{i})\n<Task Explanation>\n{analogy}\n\n<Solution>\n{solution}''')
    
    # api 호출하고 barc와 같이 analogy와 solution을 flat하게 만들기
    gifarc_data = gifarc_data if type(gifarc_data) == list else [gifarc_data]
    for i, target in enumerate(gifarc_data):
        analogy = target['analogy']
        solution = target['solution']
        question = user_prompt.format(
            task_explanation=target['analogy'],
            solution=target['solution'],
            examples="\n".join(examples)
        )

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": question},
        ]

        response = chat_with_retry(
        model=model_name,
        messages=messages,
        max_completion_tokens=max_tokens,
        top_p=1.0,
        n=1
        )

        original_text = target['analogy'] + target['solution']
        result = response.choices[0].message.content
        flat_result = json.loads(result)
        flat_text = flat_result['task_explanation'] + flat_result['solution']
        file_name = gifarc_data[i]['file_name']
        
        flat_json = {}
        flat_json['analogy'] = flat_result['task_explanation']
        flat_json['solution'] = flat_result['solution']
        flat_json['file_name'] = file_name
        flat_json['grid'] = target['grid']
        

        # TODO 나중에 이 부분 분리시켜야 함
        evaluation_origin_and_flat(file_name, original_text, flat_text, model_name, max_tokens, mode, target_setting_file_name)

        with open(f"experiment_results/data/flat_analogy_and_solution.jsonl", 'a', encoding='utf-8') as f_jsonl:
            f_jsonl.write(json.dumps(flat_json, ensure_ascii=False) + '\n')

def load_arcproblems(pkl_path):
    with open(pkl_path, "rb") as f:
        return pickle.load(f)

def write_jsonl(path, data_list):
    with open(path, "a", encoding="utf-8") as f:
        for entry in data_list:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

def solve_problem(arc_problem, model_name, common_lib, max_tokens, method=None, experiment_id=0, test_mode=False, temperature=0.0, flag=None, human_mode=False, human_result='unknown'):
    uid = arc_problem.uid
    question = make_input_prompt(arc_problem, common_lib=common_lib, experiment_id=experiment_id, uid=uid)
    messages = convert_chat_format(question, None, method=method, test_mode=test_mode, uid=uid, common_lib=common_lib)['messages']

    if flag=="gifarc":
        result_path  = f"experiment_results/gifarc_results/{args.model_name}_exp_{args.exp_id}_{args.temperature}"
    elif flag=='selected':
        if human_mode:
            result_path  = f"experiment_results/selected_results/[{human_result}]"
        else:
            result_path  = f"experiment_results/selected_results/{args.model_name}_exp_{args.exp_id}_{args.temperature}"
    else:
        result_path  = f"experiment_results/results/{args.model_name}_exp_{args.exp_id}_{args.temperature}"

    os.makedirs(result_path, exist_ok=True)
    with open(f"{result_path}/[{uid}]user_prompt_{model_name}_exp_{experiment_id}.txt", "w", encoding="utf-8") as f:
        f.write(messages[1]['content'])
    with open(f"{result_path}/[{uid}]system_prompt_{model_name}_exp_{experiment_id}.txt", "w", encoding="utf-8") as f:
        f.write(messages[0]['content'])

    if model_name == "o3-mini":
        response = chat_with_retry(
            model=model_name,
            messages=messages,
            max_completion_tokens=40000,
            top_p=1.0,
            n=1
        )
    else:
        response = chat_with_retry(
            model=model_name,
            messages=messages,
            max_completion_tokens=max_tokens,
            temperature=temperature,
            top_p=1.0,
            n=1
        )

    answers = [c.message.content for c in response.choices]

    with open(f"{result_path}/[{uid}]result_{model_name}_exp_{experiment_id}.txt", "w", encoding="utf-8") as f:
        f.write(answers[0])

    if test_mode:
        thinking_process, solution_code = parse_arc_solution(answers[0])

        if experiment_id == 2 or experiment_id == 3:
            thinking_analogy = parse_thinking_analogy(answers[0])
            return {
                "uid": uid,
                "messages": messages,
                "responses": answers,
                "thinking_process": thinking_process,
                "solution_code": solution_code,
                "thinking_analogy": thinking_analogy,
            }
        else:
            return {
                "uid": uid,
                "messages": messages,
                "responses": answers,
                "thinking_process": thinking_process,
                "solution_code": solution_code,
            }
    
    if method:
        result = parse_json(answers[0], method=method)
        return {
            "uid": uid,
            "messages": messages,
            "responses": answers,
            "result": result,
        }
    else:
        return {
            "uid": uid,
            "messages": messages,
            "responses": answers,
        }

def run(eval: str, method: str, model_name: str, split: str = None, temperature: float = 0.0,
        pickle_path: str = None, problems=None, max_tokens: int = 16384, experiment_id: int = 0, test_mode: bool = False):

    experiment_results_folder = 'experiment_results/results'
    os.makedirs(experiment_results_folder, exist_ok=True)

    assert experiment_id in [1, 2, 3, 4], "experiment_id must be 1, 2, 3 or 4"

    filename_parts = [eval, method, model_name]
    if eval == "arc":
        filename_parts.append(split)

    if test_mode:
        filename_parts.insert(0, "test")
        test_count = 0
        
    saving_file_name = "_".join(filename_parts+[f'exp_{experiment_id}']) + ".jsonl"
    result_path  = f"experiment_results/results/{args.model_name}_exp_{args.exp_id}_{args.temperature}"
    os.makedirs(result_path, exist_ok=True)
    saving_file = f"{result_path}/{saving_file_name}"

    solved_uids = set()
    if os.path.exists(saving_file):
        with open(saving_file, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    entry = json.loads(line)
                    solved_uids.add(entry.get("uid"))
                except json.JSONDecodeError:
                    continue
    print(f"➡️   {len(solved_uids)} problems already solved, will skip those.")
    if problems is None:
        problems = load_arcproblems(pickle_path)
    
    if eval != "arc":
        problems = [p for p in problems if p.uid not in solved_uids]
    
    print(f"Should solve thn number of problems {len(problems)}")
    print(f"problems: {problems}")


    print(f"📥 Loaded {len(problems)} problems")
    print(f"💾 Saving to {saving_file}")
    print(f"⚙️  Max tokens: {max_tokens}")

    common_lib, _ = get_common_lib_from_file("seeds/common.py")
    all_results = []

    for arc_problem in tqdm.tqdm(problems):
        try:
            if test_mode:
                if test_count == TEST_LIMIT:
                    break
                test_count += 1
            result = solve_problem(arc_problem, model_name, common_lib, max_tokens, method=method, experiment_id=experiment_id, test_mode=test_mode, temperature=temperature)
            all_results.append(result)
            write_jsonl(saving_file, [result])
        except Exception as e:
            ts = datetime.now().isoformat()
            uid = getattr(arc_problem, "uid", "unknown")
            stack = traceback.format_exc()

            # CSV에 로깅
            with open(LOG_FILE, "a", newline="", encoding="utf-8") as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow([
                    ts, uid, eval, method, model_name, split or "",
                    str(e), stack
                ])

            print(f"❌ [UID:{uid}] Error logged to {LOG_FILE}: {e}")
            continue
    print(f"✅ Saved {len(all_results)} entries to {saving_file}")

def run_experiment(args):
    if args.eval == "arc":
        from arc import train_problems, validation_problems
        problems = train_problems if args.split == "train" else validation_problems
        run(args.eval, args.method, args.model_name, split=args.split, temperature=args.temperature,
            problems=problems, max_tokens=args.max_tokens, experiment_id=args.exp_id, test_mode=args.test_mode)
    else:
        pickle_path = f"postprocessing_dataset/{args.eval}_arcproblems.pkl"
        run(args.eval, args.method, args.model_name, split=args.split, temperature=args.temperature,
            pickle_path=pickle_path, max_tokens=args.max_tokens, experiment_id=args.exp_id, test_mode=args.test_mode)

def make_output_grid(test_input_grid, solution_code):
    namespace = {"__name__": "__not_main__"}
    solution_code = solution_code.replace("def main", "def solve")
    try:
        # seed.common 전체를 solution_code 안에서 import하도록 강제
        exec("from seeds.common import *\n" + solution_code, namespace)
        solve = namespace.get("solve")
        generated_output = solve(test_input_grid)
    except Exception as e:
        generated_output = solve(tuple(tuple(row) for row in test_input_grid['input']))
    
    return generated_output

import numpy as np

def levenshtein_distance_2d(arr1, arr2):
    """2D 배열을 1D로 펼쳐서 Levenshtein 거리(삽입, 삭제, 교체 최소 연산 수) 계산"""
    flat1 = arr1.flatten()
    flat2 = arr2.flatten()
    n, m = len(flat1), len(flat2)

    dp = np.zeros((n + 1, m + 1), dtype=int)

    # 초기화: 빈 문자열에서 만들기
    for i in range(n + 1):
        dp[i][0] = i  # 모두 삭제
    for j in range(m + 1):
        dp[0][j] = j  # 모두 삽입

    # 동적 계획법 수행
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = 0 if flat1[i - 1] == flat2[j - 1] else 1
            dp[i][j] = min(
                dp[i - 1][j] + 1,      # 삭제
                dp[i][j - 1] + 1,      # 삽입
                dp[i - 1][j - 1] + cost  # 교체
            )

    return dp[n][m]

def normalized_similarity(arr1, arr2):
    """Levenshtein 기반 유사도 점수 (0~1 스케일) 반환"""
    dist = levenshtein_distance_2d(arr1, arr2)
    max_len = max(arr1.size, arr2.size)
    return 1 - dist / max_len


def postprocessing_experiment_result_to_evaluate(args):
    assert args.answer_file is not None, "answer_file is required"

    answer_file_name = args.answer_file
    result_path  = f"experiment_results/results/{args.model_name}_exp_{args.exp_id}_{args.temperature}"
    os.makedirs(result_path, exist_ok=True)

    answer_file = f"{result_path}/{answer_file_name}"
    answer_data = []

    eval_file_name_list = os.listdir('experiment_results/data/evaluation')
    eval_data_list = []

    saving_file = answer_file.replace(".jsonl", f"_exec_results.jsonl")

    correct_count = 0
    total_count = len(eval_file_name_list)

    with open(answer_file, "r", encoding="utf-8") as f:
        lines = f.readlines()
        for line in lines:
            answer_data.append(json.loads(line))

        for problem_idx, p in enumerate(tqdm.tqdm(answer_data)):
            uid = p["uid"]
            print(f"Problem: {uid}")
            try:
                solution_code = p['solution_code']
                
                with open(os.path.join('experiment_results/data/evaluation', f"{uid}.json"), "r", encoding="utf-8") as f:
                    eval_data = json.load(f)

                test_input_grid = np.array(eval_data['test'][0]['input'])
                test_output_grid = np.array(eval_data['test'][0]['output'])

                generated_output = make_output_grid(test_input_grid, solution_code)
                generated_output = np.array(generated_output)

                correct_flag = np.array_equal(generated_output, test_output_grid)

                sim = normalized_similarity(generated_output, test_output_grid)

                with open(saving_file, "a", encoding="utf-8") as f:
                    f.write(json.dumps({
                        "uid": uid,
                        'solution_code': solution_code,
                        "thinking_process": p['thinking_process'],
                        "test_input_grid": test_input_grid.tolist(),
                        "correct_flag": correct_flag,
                        "similarity": sim,
                        "generated_output": generated_output.tolist(),
                        "test_output_grid": test_output_grid.tolist()
                    }, ensure_ascii=False) + "\n")
                
                with open(saving_file.replace('.jsonl', '.csv'), "a", newline='', encoding="utf-8") as f_csv:
                    writer = csv.DictWriter(f_csv, fieldnames=[
                        "uid", "solution_code", "thinking_process",
                        "test_input_grid", "correct_flag", "similarity",
                        "generated_output", "test_output_grid"
                    ])
                    
                    # 파일이 비어 있다면 헤더 쓰기
                    if f_csv.tell() == 0:
                        writer.writeheader()

                    writer.writerow({
                        "uid": uid,
                        "solution_code": solution_code,
                        "thinking_process": p['thinking_process'],
                        "test_input_grid": json.dumps(test_input_grid.tolist(), ensure_ascii=False),
                        "correct_flag": correct_flag,
                        "similarity": sim,
                        "generated_output": json.dumps(generated_output.tolist(), ensure_ascii=False),
                        "test_output_grid": json.dumps(test_output_grid.tolist(), ensure_ascii=False)
                    })
            except Exception as e:
                with open(f'{result_path}/missing_problem_index_exp_{args.exp_id}.jsonl', 'a') as f:
                    json.dump({"uid": uid, "error": str(e)}, f)
                    f.write('\n')
                # with open(saving_file, "a", encoding="utf-8") as f:
                #     sim =0
                #     correct_flag = False
                #     f.write(json.dumps({
                #         "uid": uid,
                #         'solution_code': solution_code,
                #         "thinking_process": p['thinking_process'],
                #         "test_input_grid": test_input_grid.tolist(),
                #         "correct_flag": correct_flag,
                #         "similarity": sim,
                #         "generated_output": generated_output.tolist(),
                #         "test_output_grid": test_output_grid.tolist()
                #     }, ensure_ascii=False) + "\n")
                
                # with open(saving_file.replace('.jsonl', '.csv'), "a", newline='', encoding="utf-8") as f_csv:
                #     writer = csv.DictWriter(f_csv, fieldnames=[
                #         "uid", "solution_code", "thinking_process",
                #         "test_input_grid", "correct_flag", "similarity",
                #         "generated_output", "test_output_grid"
                #     ])
                    
                #     # 파일이 비어 있다면 헤더 쓰기
                #     if f_csv.tell() == 0:
                #         writer.writeheader()

                #     writer.writerow({
                #         "uid": uid,
                #         "solution_code": solution_code,
                #         "thinking_process": p['thinking_process'],
                #         "test_input_grid": json.dumps(test_input_grid.tolist(), ensure_ascii=False),
                #         "correct_flag": correct_flag,
                #         "similarity": sim,
                #         "generated_output": json.dumps(generated_output.tolist(), ensure_ascii=False),
                #         "test_output_grid": json.dumps(test_output_grid.tolist(), ensure_ascii=False)
                #     })
                    continue

    print(f"Accepted: {correct_count}/{total_count}")

    # print(f"Accepted: {accepted}/{len(problem_answers)}")
    # # with open("correct_codes.json", "w") as f:
    # #     f.write(json.dumps(correct_codes))
    
    # print(f"Savings to {saving_file}")
    # with open(saving_file, "w") as f:
    #     f.write("\n".join(json.dumps(p) for p in problem_answers))

def write_csv(path, data_list, column_list, write_header=True):
    with open(path, "a", newline='', encoding="utf-8") as f_csv:
        writer = csv.DictWriter(f_csv, fieldnames=column_list)

        if write_header and f_csv.tell() == 0:
            writer.writeheader()

        for row in data_list:
            row_copy = copy.deepcopy(row)

            for key in column_list:
                if key in row_copy and isinstance(row_copy[key], (list, dict)):
                    row_copy[key] = json.dumps(row_copy[key], ensure_ascii=False)

            writer.writerow(row_copy)

def evaluation_task_and_analogy(uid, task, analogy, eval_model_name, max_tokens):
    user_prompt = EVALUATION_TASK_AND_ANALOGY.replace(f"{{uid}}", uid).replace(f"{{task}}", task).replace(f"{{analogy}}", analogy)
    if eval_model_name == "o3-mini":
        response = chat_with_retry(
            model=eval_model_name,
            messages=[{"role": "user", "content": user_prompt}],
            max_completion_tokens=max_tokens,
            top_p=1.0,
            n=1
        )
    else:
        response = chat_with_retry(
            model=eval_model_name,
            messages=[{"role": "user", "content": user_prompt}],
            max_completion_tokens=max_tokens,
            top_p=1.0,
            temperature=0.0,
            n=1
        )
    response_result = [c.message.content for c in response.choices]

    result = response.choices[0].message.content
    result_json = parse_json(result, mode='other')
    result_json['uid'] = uid
    result_json['task'] = task
    result_json['question'] = user_prompt
    result_json['response'] = response_result

    return result_json

def evaluation_ground_truth_analogy_and_generated_analogy(uid, task, ground_truth_analogy, generated_analogy, eval_model_name, max_tokens):
    user_prompt = EVALUATION_GROUND_TRUTH_ANALOGY_AND_GENERATED_ANALOGY.replace(f"{{uid}}", uid).replace(f"{{task}}", task).replace(f"{{ground_truth_analogy}}", ground_truth_analogy).replace(f"{{generated_analogy}}", generated_analogy)
    if eval_model_name == "o3-mini":
        response = chat_with_retry(
            model=eval_model_name,
            messages=[{"role": "user", "content": user_prompt}],
            max_completion_tokens=max_tokens,
            top_p=1.0,
            n=1
        )
    else:
        response = chat_with_retry(
            model=eval_model_name,
            messages=[{"role": "user", "content": user_prompt}],
            max_completion_tokens=max_tokens,
            top_p=1.0,
            temperature=0.0,
            n=1
        )
    response_result = [c.message.content for c in response.choices]

    result = response.choices[0].message.content
    result_json = parse_json(result, mode='other')
    result_json['uid'] = uid
    result_json['task'] = task
    result_json['question'] = user_prompt
    result_json['response'] = response_result

    return result_json


def evaluation_task_and_analogy_multiple_choice(correct_uid, incorrect_uids, task, analogy_dict, eval_model_name, max_tokens):
    options_index_list = [correct_uid] + incorrect_uids

    random.shuffle(options_index_list)

    correct_analogy_index = options_index_list.index(correct_uid)
    correct_analogy = analogy_dict[correct_uid]

    incorrect_analogy_indxes = [options_index_list.index(uid) for uid in incorrect_uids]
    incorrect_analogies = [analogy_dict[uid] for uid in incorrect_uids]

    options_list = [correct_analogy] + incorrect_analogies
    included_analogy_options = EVALUATION_TASK_AND_ANALOGY_MULTIPLE_CHOICE

    for i in range(len(options_list)):
        included_analogy_options = included_analogy_options.replace(f"{{analogy_{i+1}}}", options_list[i])
    
    user_prompt = included_analogy_options.replace(f"{{uid}}", correct_uid).replace(f"{{task}}", task)

    if eval_model_name == "o3-mini":
        response = chat_with_retry(
            model=eval_model_name,
            messages=[{"role": "user", "content": user_prompt}],
            max_completion_tokens=max_tokens,
            top_p=1.0,
            n=1
        )
    else:
        response = chat_with_retry(
            model=eval_model_name,
            messages=[{"role": "user", "content": user_prompt}],
            max_completion_tokens=max_tokens,
            top_p=1.0,
            temperature=0.0,
            n=1
        )

    response_result = [c.message.content for c in response.choices]

    result = response.choices[0].message.content
    result_json = parse_json(result, mode='other')
    result_json['uid'] = correct_uid
    result_json['question'] = user_prompt
    result_json['correct_analogy_index'] = correct_analogy_index+1
    result_json['correct_analogy'] = correct_analogy
    result_json['incorrect_analogy_indxes'] = incorrect_analogy_indxes
    result_json['incorrect_analogy'] = incorrect_analogies
    result_json['options'] = options_index_list
    result_json['options_list'] = options_list
    result_json['task'] = correct_analogy
    result_json['response'] = response_result
    result_json['finding_correct_analogy'] = 1 if result_json['answer'] == correct_analogy_index+1 else 0

    return result_json

def evaluation_alignment_task_anda_analogy(args):
    # TODO 실험 결과 파일 name list 가져오기
    eval = args.eval
    method = args.method
    model_name = args.model_name
    eval_model_name = args.eval_model_name
    temperature = args.temperature
    split = args.split
    experiment_id = args.exp_id
    max_tokens = args.max_tokens
    test_mode = args.test_mode
    test_count = 0

    target_file_name = f'experiment_results/results/{model_name}_exp_{experiment_id}_{temperature}'
    target_file_name_list = glob.glob(target_file_name+'/*.txt')
    base_result_path = f"experiment_results/alignment_result"
    result_task_and_analogy_file_name  = f"experiment_results/alignment_result/[{eval_model_name}]{model_name}_exp_{experiment_id}_{temperature}_task_and_analogy.jsonl"
    result_task_and_analogy_multiple_choice_file_name  = f"experiment_results/alignment_result/[{eval_model_name}]{model_name}_exp_{experiment_id}_{temperature}_task_and_analogy_multiple_choice.jsonl"
    pickle_path = f"postprocessing_dataset/{eval}_arcproblems.pkl"

    os.makedirs(base_result_path, exist_ok=True)

    all_result_task_and_analogy = []
    all_result_task_and_analogy_multiple_choice = []
    analogy_dict = {}

    for file_name in target_file_name_list:
        base_name = os.path.basename(file_name)
        if 'result'in base_name:
            with open(file_name, "r", encoding="utf-8") as f:
                base_name = os.path.basename(file_name)
                uid = base_name.split('[')[1].split(']')[0]
                text = f.read()
                analogy = parse_thinking_analogy(text)
                analogy_dict[uid] = analogy
        
    # TODO for loop를 사용해서 실험 결과 파일에서 task와 solution을 가져오기
    problems = load_arcproblems(pickle_path)
    pickle_path = f"postprocessing_dataset/{eval}_arcproblems.pkl"

    if problems is None:
        problems = load_arcproblems(pickle_path)

    evalated_task_and_analogy_file_name = set()
    if os.path.exists(result_task_and_analogy_file_name):
        with open(result_task_and_analogy_file_name, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    entry = json.loads(line)
                    evalated_task_and_analogy_file_name.add(entry.get("uid"))
                except json.JSONDecodeError:
                    continue
    print(f"➡️    {len(evalated_task_and_analogy_file_name)} problems already solved, will skip those.")
    
    if eval != "arc":
        task_and_analogy_problems = [p for p in problems if p.uid not in evalated_task_and_analogy_file_name]
    
    print(f"Should solve thn number of problems {len(task_and_analogy_problems)}")
    print(f"problems: {task_and_analogy_problems}")
    print(f"Start evaluation_task_and_analogy\n\n")

    # Step) evaluation_task_and_analogy
    # for arc_problem in tqdm.tqdm(task_and_analogy_problems):
    #     # Step) evaluation_task_and_analogy 
    #     try:
    #         if test_mode:
    #             if test_count == TEST_LIMIT:
    #                 break
    #             test_count += 1
    #         uid = arc_problem.uid
    #         analogy = analogy_dict.get(uid, None)
    #         target_task = ""
    #         for i in range(len(arc_problem.train_pairs)):
    #             target_task += f"Input-{i+1}:\n{arc_problem.train_pairs[i].x}\n"
    #             target_task += f"Output-{i+1}:\n{arc_problem.train_pairs[i].y}\n\n"

    #         result_task_and_analogy = evaluation_task_and_analogy(uid=uid, task=target_task, analogy=analogy, eval_model_name=eval_model_name, max_tokens=max_tokens)
    #         write_jsonl(result_task_and_analogy_file_name, [result_task_and_analogy])
    #         write_csv(result_task_and_analogy_file_name.replace('.jsonl', '.csv'), [result_task_and_analogy], column_list=result_task_and_analogy.keys(), write_header=True)
    #         all_result_task_and_analogy.append(result_task_and_analogy)

    #         # TODO csv 파일에 저장하기
    #     except Exception as e:
    #         ts = datetime.now().isoformat()
    #         uid = getattr(arc_problem, "uid", "unknown")
    #         stack = traceback.format_exc()

    #         # CSV에 로깅
    #         with open(LOG_FILE, "a", newline="", encoding="utf-8") as csvfile:
    #             writer = csv.writer(csvfile)
    #             writer.writerow([
    #                 ts, uid, eval, method, model_name, split or "",
    #                 str(e), stack
    #             ])

    #         print(f"❌ [UID:{uid}] Error logged to {LOG_FILE}: {e}")
    #         continue
    
    # Step) evaluation_task_and_analogy _multiple_choice
    evalated_task_and_analogy_multiple_choice_file_name = set()
    if os.path.exists(result_task_and_analogy_multiple_choice_file_name):
        with open(result_task_and_analogy_multiple_choice_file_name, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    entry = json.loads(line)
                    evalated_task_and_analogy_multiple_choice_file_name.add(entry.get("uid"))
                except json.JSONDecodeError:
                    continue
    
    if eval != "arc":
        multiple_choice_problems = [p for p in problems if p.uid not in evalated_task_and_analogy_multiple_choice_file_name]

    print(f"➡️    {len(evalated_task_and_analogy_multiple_choice_file_name)} problems already solved, will skip those.")
    print(f"Should solve thn number of problems {len(multiple_choice_problems)}")
    print(f"problems: {multiple_choice_problems}")

    print(f"Start evaluation_task_and_analogy_multiple_choice\n\n")
    test_count = 0
    for arc_problem in tqdm.tqdm(multiple_choice_problems):
        if test_mode:
            if test_count == TEST_LIMIT:
                break
        test_count += 1
        uid = arc_problem.uid
        analogy = analogy_dict.get(uid, None)
        target_task = ""
        for i in range(len(arc_problem.train_pairs)):
            target_task += f"Input-{i+1}:\n{arc_problem.train_pairs[i].x}\n"
            target_task += f"Output-{i+1}:\n{arc_problem.train_pairs[i].y}\n\n"
        try:
            filtered_list = list(filter(lambda x: x != uid, analogy_dict.keys()))
            wrong_uid = random.sample(filtered_list, 4)
            result_task_and_analogy_multiple_choice = evaluation_task_and_analogy_multiple_choice(correct_uid=uid, incorrect_uids=wrong_uid, task=target_task, analogy_dict=analogy_dict, eval_model_name=eval_model_name, max_tokens=max_tokens)
            write_jsonl(result_task_and_analogy_multiple_choice_file_name, [result_task_and_analogy_multiple_choice])
            write_csv(result_task_and_analogy_multiple_choice_file_name.replace('.jsonl', '.csv'), [result_task_and_analogy_multiple_choice], column_list=result_task_and_analogy_multiple_choice.keys(), write_header=True)
            all_result_task_and_analogy_multiple_choice.append(result_task_and_analogy_multiple_choice)
        except Exception as e:
            ts = datetime.now().isoformat()
            uid = getattr(arc_problem, "uid", "unknown")
            stack = traceback.format_exc()

            # CSV에 로깅
            with open(LOG_FILE, "a", newline="", encoding="utf-8") as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow([
                    ts, uid, eval, method, model_name, split or "",
                    str(e), stack
                ])

            print(f"❌ [UID:{uid}] Error logged to {LOG_FILE}: {e}")
            continue

    
    print(f"✅ Saved {len(all_result_task_and_analogy)} entries to {result_task_and_analogy_file_name}")
    print(f"✅ Saved {len(all_result_task_and_analogy_multiple_choice)} entries to {result_task_and_analogy_multiple_choice_file_name}")

def run_gifarc(args):
    eval = args.eval
    method = args.method
    model_name = args.model_name
    temperature = args.temperature
    split = args.split
    experiment_id = args.exp_id
    max_tokens = args.max_tokens
    test_mode = args.test_mode
    human_mode = args.human_mode
    human_result = args.human_result
    test_count = 0
    # jsonl_path = f"experiment_results/data/gifarc_data.jsonl"
    # pickle_path = f"experiment_results/data/gifarc_data.pkl"
    jsonl_path = f"experiment_results/data/quantitative_evaluation/selected_data_samples.jsonl"
    pickle_path = f"experiment_results/data/quantitative_evaluation/selected_data_samples.pkl"
    
    arc_problems = []
    jsonl_problems = []
    success_count = 0
    experiment_results_folder = 'experiment_results/selected_results'
    os.makedirs(experiment_results_folder, exist_ok=True)

    assert experiment_id in [1, 2, 3, 4], "experiment_id must be 1, 2, 3 or 4"

    filename_parts = [eval, method, model_name]
    if eval == "arc":
        filename_parts.append(split)

    if test_mode:
        filename_parts.insert(0, "test")
        test_count = 0
        
    saving_file_name = "_".join(filename_parts+[f'exp_{experiment_id}']) + ".jsonl"
    
    if human_mode:
        result_path  = f"experiment_results/selected_results/[{human_result}]"
    else:
        result_path  = f"experiment_results/selected_results/{args.model_name}_exp_{args.exp_id}_{args.temperature}"
    os.makedirs(result_path, exist_ok=True)
    saving_file = f"{result_path}/{saving_file_name}"

    solved_uids = set()
    if os.path.exists(saving_file):
        with open(saving_file, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    entry = json.loads(line)
                    solved_uids.add(entry.get("uid"))
                except json.JSONDecodeError:
                    continue
    print(f"➡️   {len(solved_uids)} problems already solved, will skip those.")

    if os.path.exists(pickle_path):
        with open(pickle_path, "rb") as f:
            arc_problems = pickle.load(f)
    else:
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                data = json.loads(line)
                data['uid'] = i
                jsonl_problems.append(data)

        for problem in jsonl_problems:
            train_pairs = []
            test_pairs = []
            if 'examples' not in problem:
                continue
            for i in range(len(problem['examples'])):
                if i <= 3:
                    train_pairs.append(IOPair(np.array(problem['examples'][i][0]), np.array(problem['examples'][i][1])))
                else:
                    test_pairs.append(IOPair(np.array(problem['examples'][i][0]), np.array(problem['examples'][i][1])))
                    break
            arc_problems.append(ArcProblem(uid=problem['uid'], train_pairs=train_pairs, test_pairs=test_pairs))
        
        if len(arc_problems) == 0:
            print(f"❌ No problems found in {jsonl_path}")

        random.shuffle(arc_problems)
        with open(pickle_path, "wb") as f:
            pickle.dump(arc_problems, f)
        
    if eval != "arc":
        problems = [p for p in arc_problems if p.uid not in solved_uids]
        success_count = len(problems)
    
    print(f"Should solve thn number of problems {len(problems)}")
    print(f"problems: {problems}")


    print(f"📥 Loaded {len(problems)} problems")
    print(f"💾 Saving to {saving_file}")
    print(f"⚙️  Max tokens: {max_tokens}")

    common_lib, _ = get_common_lib_from_file("seeds/common.py")
    all_results = []

    for arc_problem in tqdm.tqdm(problems):
        try:
            if test_mode:
                if test_count == TEST_LIMIT:
                    break
                if success_count == 100:
                    break
            result = solve_problem(arc_problem, model_name, common_lib, max_tokens, method=method, experiment_id=experiment_id, test_mode=test_mode, temperature=temperature, flag='selected')
            all_results.append(result)
            write_jsonl(saving_file, [result])
            success_count +=1 
            test_count += 1
        except Exception as e:
            ts = datetime.now().isoformat()
            uid = getattr(arc_problem, "uid", "unknown")
            stack = traceback.format_exc()

            # CSV에 로깅
            with open(LOG_FILE, "a", newline="", encoding="utf-8") as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow([
                    ts, uid, eval, method, model_name, split or "",
                    str(e), stack
                ])

            print(f"❌ [UID:{uid}] Error logged to {LOG_FILE}: {e}")
            continue
    print(f"✅ Saved {len(all_results)} entries to {saving_file}")

def embedding_similarity(ground_truth, generated, embedding_model_name):
    ground_truth_embedding_response = send_embedding_request(input=ground_truth, model=embedding_model_name)
    generated_embedding_response = send_embedding_request(input=generated, model=embedding_model_name)

    ground_truth_embedding = np.array(ground_truth_embedding_response)
    generated_embedding = np.array(generated_embedding_response)
    
    similarity = np.dot(ground_truth_embedding, generated_embedding) / (np.linalg.norm(ground_truth_embedding) * np.linalg.norm(generated_embedding))
    return similarity

def evaluation_alignment_gifarc_task_anda_analogy(args):
    # TODO 실험 결과 파일 name list 가져오기
    eval = args.eval
    method = args.method
    model_name = args.model_name
    eval_model_name = args.eval_model_name
    temperature = args.temperature
    split = args.split
    experiment_id = args.exp_id
    max_tokens = args.max_tokens
    test_mode = args.test_mode
    test_count = 0
    embedding_model_name=args.embedding_model_name
    human_mode = args.human_mode
    human_result = args.human_result

    # pickle_path = f"experiment_results/data/gifarc_data.pkl"
    # jsonl_path = f"experiment_results/data/gifarc_data.jsonl"
    jsonl_path = f"experiment_results/data/quantitative_evaluation/selected_data_samples.jsonl"
    pickle_path = f"experiment_results/data/quantitative_evaluation/selected_data_samples.pkl"

    if human_mode:
        result_task_and_analogy_file_name  = f"experiment_results/selected_results/[{human_result}-{eval_model_name}]task_and_analogy.jsonl"
    else:
        target_file_name = f'experiment_results/selected_results/{model_name}_exp_{experiment_id}_{temperature}'
        result_task_and_analogy_file_name  = f"experiment_results/selected_results/[{eval_model_name}]{model_name}_exp_{experiment_id}_{temperature}_task_and_analogy.jsonl"
    
    base_result_path = f"experiment_results/selected_results"
    if not human_mode:
        target_file_name_list = glob.glob(target_file_name+'/*.txt')
    else:
        target_file_name_list = f"experiment_results/selected_results/{human_result}/{human_result}.jsonl"
    os.makedirs(base_result_path, exist_ok=True)

    all_result_task_and_analogy = []
    analogy_dict = {}
    task_and_analogy_problems = []
    jsonl_problems = []

    with open(jsonl_path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            data = json.loads(line)
            data['uid'] = i
            jsonl_problems.append(data)

    if not human_mode:
        for file_name in target_file_name_list:
            base_name = os.path.basename(file_name)
            if 'result'in base_name:
                with open(file_name, "r", encoding="utf-8") as f:
                    base_name = os.path.basename(file_name)
                    uid = base_name.split('[')[1].split(']')[0]
                    text = f.read()
                    analogy = parse_thinking_analogy(text)
                    analogy_dict[uid] = analogy
    else:
        with open(target_file_name_list, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    entry = json.loads(line)
                    uid = entry.get("uid")
                    analogy = entry.get("thinking_analogy")
                    analogy_dict[uid] = analogy
                except json.JSONDecodeError:
                    continue

        
    problems = load_arcproblems(pickle_path)

    evalated_task_and_analogy_file_name = set()
    if os.path.exists(result_task_and_analogy_file_name):
        with open(result_task_and_analogy_file_name, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    entry = json.loads(line)
                    evalated_task_and_analogy_file_name.add(entry.get("uid"))
                except json.JSONDecodeError:
                    continue
    print(f"➡️    {len(evalated_task_and_analogy_file_name)} problems already solved, will skip those.")
    
    if eval != "arc":
        for p in problems:
            if p.uid not in evalated_task_and_analogy_file_name and str(p.uid) not in evalated_task_and_analogy_file_name:
                if str(p.uid) in analogy_dict.keys():
                    task_and_analogy_problems.append(p)
    
    print(f"Should solve thn number of problems {len(task_and_analogy_problems)}")
    print(f"problems: {task_and_analogy_problems}")
    print(f"Start evaluation_gifarc_task_and_analogy\n\n")

    # Step) evaluation_task_and_analogy
    for arc_problem in tqdm.tqdm(task_and_analogy_problems):
        # Step) evaluation_task_and_analogy 
        try:
            if test_mode:
                if test_count == TEST_LIMIT:
                    break
                test_count += 1
            uid = str(arc_problem.uid)
            analogy = analogy_dict.get(uid, None)
            target_task = ""
            for i in range(len(arc_problem.train_pairs)):
                target_task += f"Input-{i+1}:\n{arc_problem.train_pairs[i].x}\n"
                target_task += f"Output-{i+1}:\n{arc_problem.train_pairs[i].y}\n\n"

            for p in jsonl_problems:
                if p['uid'] == int(uid):
                    ground_truth = p['seeds'][-1].split('Description:')[1]

            result_task_and_analogy = evaluation_ground_truth_analogy_and_generated_analogy(uid=uid, task=target_task, ground_truth_analogy=ground_truth, generated_analogy=analogy, eval_model_name=eval_model_name, max_tokens=max_tokens)
            result_task_and_analogy['similarity']= embedding_similarity(ground_truth=ground_truth, generated=analogy, embedding_model_name=embedding_model_name)
            write_jsonl(result_task_and_analogy_file_name, [result_task_and_analogy])
            write_csv(result_task_and_analogy_file_name.replace('.jsonl', '.csv'), [result_task_and_analogy], column_list=result_task_and_analogy.keys(), write_header=True)
            all_result_task_and_analogy.append(result_task_and_analogy)

            # TODO csv 파일에 저장하기
        except Exception as e:
            ts = datetime.now().isoformat()
            uid = getattr(arc_problem, "uid", "unknown")
            stack = traceback.format_exc()

            # CSV에 로깅
            with open(LOG_FILE, "a", newline="", encoding="utf-8") as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow([
                    ts, uid, eval, method, model_name, split or "",
                    str(e), stack
                ])

            print(f"❌ [UID:{uid}] Error logged to {LOG_FILE}: {e}")
            continue
    
    print(f"✅ Saved {len(all_result_task_and_analogy)} entries to {result_task_and_analogy_file_name}")



def main(args):
    model_name = args.model_name
    max_tokens = args.max_tokens
    mode = args.mode
    method = args.method

    os.makedirs('experiment_results', exist_ok=True)

    # get_gifarc_examples(mode)
    # make_flat_information(model_name,max_tokens, mode)
    # filtering_analogy_and_solution_to_imporving(model_name, max_tokens)
    # imporving_analogy_and_solution(model_name, max_tokens, source_file_name='previous_few_shot_gifarc_info')
    # run_experiment(args)
    # postprocessing_experiment_result_to_evaluate(args)
    # evaluation_alignment_task_anda_analogy(args)
    if not args.human_mode:
        run_gifarc(args)
    evaluation_alignment_gifarc_task_anda_analogy(args)

    


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_id", type=int, default=3)
    parser.add_argument("--model_name", type=str, default="gpt-4.1-mini")
    parser.add_argument("--eval_model_name", type=str, default="o3-mini")
    parser.add_argument("--embedding_model_name", type=str, default="text-embedding-ada-002")
    parser.add_argument("--mode", type=str, default="previous")
    parser.add_argument("--max_tokens", type=int, default=32768)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--method", type=str, choices=["induction", "induction_simple", "transduction"], default="induction")
    parser.add_argument("--split", type=str, choices=["train", "validation"], default="validation")
    parser.add_argument("--eval", type=str, choices=["barc", "gifarc", "arc", 'arc2'], default="arc2")
    parser.add_argument("--answer_file", help="Path to the answer file")
    parser.add_argument("--test_mode", action="store_true", help="Run in test mode")
    parser.add_argument("--human_mode", action="store_true", help="Run in test mode")
    parser.add_argument("--human_result", type=str, default="unknown")
    args = parser.parse_args()
    
    main(args)