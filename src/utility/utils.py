import re
import ast
import json

class FunctionExtractor(ast.NodeVisitor):
    def __init__(self, code):
        self.functions = []
        self.code = code

    def visit_FunctionDef(self, node):
        func_name = node.name
        docstring = ast.get_docstring(node)
        func_code = ast.get_source_segment(self.code, node)
        # the api definition is just the source go to the function MINUS the actual implementation, so just the name, type signature/arguments, and docstring
        api_definition_parsed = ast.FunctionDef(name=node.name, args=node.args, decorator_list=node.decorator_list, returns=node.returns,
                                            type_comment=node.type_comment, body=node.body[:1] if docstring else node.body[:0])
        api_definition = ast.unparse(ast.fix_missing_locations(api_definition_parsed))
        if not docstring:
            api_definition = api_definition + "\n    pass"
        self.functions.append({
            'name': func_name,
            'docstring': docstring,
            'code': func_code,
            'api_definition': api_definition,
            "api_definition_parsed": api_definition_parsed
        })
        #self.generic_visit(node)# don't recurse

    def visit_ClassDef(self, node):
        # don't recurse into classes, which picks up methods
        pass

def extract_functions(code):
    tree = ast.parse(code)
    extractor = FunctionExtractor(code)
    extractor.visit(tree)
    return extractor.functions

def extract_class_definitions(code):
    class ClassExtractor(ast.NodeVisitor):
        def __init__(self):
            self.classes = []

        def visit_ClassDef(self, node):
            class_name = node.name
            docstring = ast.get_docstring(node)
            class_code = ast.get_source_segment(code, node)

            # now we visit all the methods, accomplished using the function visitor
            function_extractor = FunctionExtractor(code)
            function_extractor.visit(node)
            methods = [ f["api_definition_parsed"] for f in function_extractor.functions ]

            api_definition = ast.ClassDef(name=node.name, bases=node.bases, keywords=node.keywords, decorator_list=node.decorator_list,
                                          body=(node.body[:1] if docstring else node.body[:0])+methods)
            api_definition = ast.unparse(ast.fix_missing_locations(api_definition))
            self.classes.append({
                'name': class_name,
                'docstring': docstring,
                'code': class_code,
                'api_definition': api_definition
            })
            #self.generic_visit(node)
        
        def visit_FunctionDef(self, node):
            # don't recurse into functions
            pass

    tree = ast.parse(code)
    extractor = ClassExtractor()
    extractor.visit(tree)
    return extractor.classes

def extract_function_calls(function_code):
    class FunctionCallExtractor(ast.NodeVisitor):
        def __init__(self):
            self.called_functions = set()

        def visit_Call(self, node):
            if isinstance(node.func, ast.Name):
                self.called_functions.add(node.func.id)
            self.generic_visit(node)

    tree = ast.parse(function_code)
    extractor = FunctionCallExtractor()
    extractor.visit(tree)
    return list(extractor.called_functions)

def generate_html_grid(data, uid):
    color_map = {
        0: 'black',
        1: 'blue',
        2: 'red',
        3: 'green',
        4: 'yellow',
        5: 'grey',
        6: 'pink',
        7: 'orange',
        8: 'teal',
        9: 'maroon'
    }

    def array_to_html(array):
        html = '<table style="border-collapse: collapse;">'
        for row in array:
            html += '<tr>'
            for cell in row:
                color = color_map.get(cell, 'white')
                html += f'<td style="width: 20px; height: 20px; background-color: {color};"></td>'
            html += '</tr>'
        html += '</table>'
        return html

    html = f'<div id="img_{uid}">'
    for item in data:
        input_html = array_to_html(item['input'])
        output_html = array_to_html(item['output'])
        html += f'<div style="display: inline-block; margin: 10px;">'
        html += f'<div>Input:</div>{input_html}'
        html += f'<div>Output:</div>{output_html}'
        html += '</div>'
    html += '</div>'

    return html


def remove_trailing_code(code_str):
    lines = code_str.strip().split('\n')
    main_start = None

    for i, line in enumerate(lines):
        if line.strip().startswith('if __name__ == "__main__":'):
            main_start = i
            break

    if main_start is not None:
        # Remove the main block and any trailing non-function lines
        lines = lines[:main_start]

    # Ensure there are no trailing non-function lines
    while lines and lines[-1].strip() == "":
        lines.pop()

    return '\n'.join(lines).strip()

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

def get_concepts_from_lines(lines):
    concepts = []
    for i, line in enumerate(lines):
        if "# concepts:" in line:
            in_line_concepts = lines[i][12:]
            if in_line_concepts.strip() != "":
                concepts.extend(lines[i][12:].split(","))
            while i+1 < len(lines) and lines[i+1].startswith("# ") and not lines[i+1].startswith("# description:"):
                concepts.extend(lines[i+1][2:].split(","))
                i += 1
            concepts = [c.strip() for c in concepts]
            break
    if concepts == []:
        for i, line in enumerate(lines):
            if "concepts:" in line.lower():
                in_line_concepts = lines[i][12:]
                if in_line_concepts.strip() != "":
                    concepts.extend(lines[i][12:].split(","))
                while i+1 < len(lines) and lines[i+1].startswith("# ") and not lines[i+1].lower().startswith("description:"):
                    concepts.extend(lines[i+1][2:].split(","))
                    i += 1
                concepts = [c.strip() for c in concepts]
                break
    return concepts

def parse_code(paragraph):
    """
    This function extracts all Markdown code blocks from a given paragraph.
    Args:
        paragraph (str): The input paragraph containing the Markdown code blocks.
    Returns:
        list: A list of extracted code blocks.
    """
    # Regular expression to match Markdown code blocks
    code_block_pattern = re.compile(r"```python(.*?)```", re.DOTALL)

    # Find all code blocks in the paragraph
    matches = code_block_pattern.findall(paragraph)

    # Strip any leading/trailing whitespace from each code block
    code_blocks = [match.strip() for match in matches]

    if code_blocks:
        return code_blocks
    
    # assume that it does not begin with python
    code_block_pattern = re.compile(r"```(.*?)```", re.DOTALL)
    matches = code_block_pattern.findall(paragraph)
    code_blocks = [match.strip() for match in matches]
    return code_blocks

def parse_code_json_1(json_str):
    data = json.loads(json_str)
    return data['total_code']
def parse_code_json_2(json_str):
    import json
    # post processing markdown json
    if json_str.startswith("```json"):
        json_str = json_str[len("```json"):].strip()
    if json_str.endswith("```"):
        json_str = json_str[:json_str.rfind("```")].strip()
    
    data = json.loads(json_str)
    return data['total_code']

if __name__ == "__main__":
    code = """
def foo():
    \"""
    This is a docstring for foo.
    \"""
    return "foo"

def bar(x):
    \"""
    This is a docstring for bar.
    \"""
    return x * 2

def example_function(x):
    y = foo(x)
    z = bar(y)
    return z
"""

    expected_functions_output = [
        {
            'name': 'foo',
            'docstring': 'This is a docstring for foo.',
            'code': 'def foo():\n    """\n    This is a docstring for foo.\n    """\n    return "foo"', 
            'api_definition': 'def foo():\n    """\n    This is a docstring for foo.\n    """'
        },
        {
            'name': 'bar',
            'docstring': 'This is a docstring for bar.',
            'code': 'def bar(x):\n    """\n    This is a docstring for bar.\n    """\n    return x * 2', 
            'api_definition': 'def bar(x):\n    """\n    This is a docstring for bar.\n    """'
        },
        {
            'name': 'example_function',
            'docstring': None,
            'code': 'def example_function(x):\n    y = foo(x)\n    z = bar(y)\n    return z', 
            'api_definition': 'def example_function(x):\n    pass'
        }
    ]

    expected_calls_output = ['foo', 'bar']

    functions = extract_functions(code)
    assert functions == expected_functions_output, f"Expected {expected_functions_output}, but got {functions}"
    
    example_function_code = functions[2]['code']
    called_functions = extract_function_calls(example_function_code)
    assert sorted(called_functions) == sorted(expected_calls_output), f"Expected {expected_calls_output}, but got {called_functions}"
    
    print("All tests passed!")


import os
def ensure_dir(path: str) -> None:
    """
    Ensure dir
    """
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)
        print(f"Generated Folder: {path}")
    else:
        print(f"Folder that already exist: {path}")


import csv
import json
from pathlib import Path
from datetime import datetime, timezone
from typing import Dict, List, Any

FIELDNAMES = [
    "id",
    "step-name",
    "prev-step-id",
    "gif-id",
    "gen-model",
    "result_code",
    "result_path",
    "error_message",
    "createAt",
    "token-usage",
    "visualization_path",
]

def _normalize_record(rec: Dict[str, Any]) -> Dict[str, Any]:
    """ Key mapping by FIELDNAMES """
    row: Dict[str, Any] = {f: "" for f in FIELDNAMES}

    row["id"] = rec.get("id") or rec.get("step_uuid", "")
    row["step-name"] = rec.get("step_name") or rec.get("step-name", "")
    row["prev-step-id"] = rec.get("prev_step_id") or rec.get("prev-step-id", "")
    row["gif-id"] = rec.get("gif_id") or rec.get("gif-id", "")
    row["gen-model"] = rec.get("gen_model") or rec.get("gen-model", "")
    row["result_code"] = rec.get("result_code", "")
    row["result_path"] = rec.get("result_path", "")

    err = rec.get("error_message", "")
    if isinstance(err, list):
        row["error_message"] = json.dumps(err, ensure_ascii=False)
    else:
        row["error_message"] = str(err)

    # createAt: datetime→ISO, str→use it, if not curr UTC
    ts = rec.get("createAt") or rec.get("create_at") or rec.get("created_at")
    if isinstance(ts, datetime):
        row["createAt"] = ts.astimezone(timezone.utc).isoformat(timespec="seconds")
    elif ts:
        row["createAt"] = str(ts)
    else:
        row["createAt"] = datetime.now(timezone.utc).isoformat(timespec="seconds")

    row["token-usage"] = json.dumps(rec.get("token_usage", ""), ensure_ascii=False)

    row["visualization_path"] = rec.get("visualization_path", "")

    return row

def generate_metadata_csv_of_step_problem(
    records: List[Dict[str, Any]],
    output_csv: str | Path = None,
    encoding: str = "utf-8",
) -> Path:
    """
    Args
    ----
    records     : List of metadata dictionaries
    output_csv  : Path to the CSV file to save (if not provided, created in the same directory)
    encoding    : CSV encoding (default: 'utf-8')

    Returns
    -------
    Path object pointing to the written CSV file
    """
    output_csv = Path(output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    file_exists = output_csv.exists()

    with output_csv.open("a", newline="", encoding=encoding) as fp:
        writer = csv.DictWriter(fp, fieldnames=FIELDNAMES)
        if not file_exists:
            writer.writeheader()
        for rec in records:
            writer.writerow(_normalize_record(rec))

    return output_csv




