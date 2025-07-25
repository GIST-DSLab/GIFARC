import json
import argparse
import shutil
import numpy as np
from utility.utils import remove_trailing_code
from pygments import highlight
from pygments.lexers import PythonLexer
from pygments.formatters import HtmlFormatter
import hashlib
import base64
import os

def list_to_html_bullets(lst):
    if not lst:
        return ''
    if isinstance(lst, str):
        return '<ul style="font-size: 17px; line-height: 1.6;"><li>' + lst + '</li></ul>'
    return '<ul style="font-size: 17px; line-height: 1.6;">' + ''.join(f'<li>{item}</li>' for item in lst) + '</ul>'

def get_problem_gif_path(gif_id, data_folder="./data/GIF"):
    gif_path = os.path.join(data_folder, gif_id)
    if not os.path.exists(gif_path):
        print(f"GIF file not found: {gif_path}")
        return None
    return gif_path

def load_gif_base64_from_path(gif_path):
    with open(gif_path, "rb") as f:
        gif_data = f.read()
    return f"data:image/gif;base64,{base64.b64encode(gif_data).decode('utf-8')}"



def build_html_grid(examples_input_output, uid):
    def build_table(grid, title):
        num_rows = len(grid)
        num_cols = len(grid[0])
        table = f'''
        <div style="text-align:center; font-size:18px; margin-bottom:5px;">
            {title} ({num_rows}×{num_cols})
        </div>
        '''
        table += f'<table class="grid" style="border-collapse: collapse; table-layout: fixed; width: {24 * num_cols}px;">'
        for row in grid:
            table += "<tr>"
            for val in row:
                try:
                    val_int = int(val)
                except Exception:
                    val_int = -1
                color = {
                    0: '#000000', 1: '#0074D9', 2: '#FF4136', 3: '#2ECC40',
                    4: '#FFDC00', 5: '#AAAAAA', 6: '#F012BE', 7: '#FF851B',
                    8: '#7FDBFF', 9: '#870C25'
                }.get(val_int, "#FFFFFF")
                table += f'''
                <td style="
                    background: #FFFFFF; 
                    border: 3px solid #888888; 
                    width: 24px; 
                    height: 24px;
                    padding: 0;
                    position: relative;
                ">
                    <div style="
                        background: {color};
                        width: 100%;
                        height: 100%;
                        border-radius: 0px;
                    "></div>
                </td>
                '''
            table += "</tr>"
        table += "</table>"
        return table

    html = f'<div id="img_{uid}" class="centered-content">'
    for i, ex in enumerate(examples_input_output):
        input_grid = ex["input"]
        output_grid = ex["output"]
        input_table = build_table(input_grid, "Input")
        output_table = build_table(output_grid, "Output")
        html += f'''
        <div class="bubble">
            <b>Train Example {i+1}</b><br>
            <div class="sidebyside" style="display: flex; gap: 40px; margin-top: 10px; margin-bottom: 30px;">
                <div>{input_table}</div>
                <div>{output_table}</div>
            </div>
        </div>
        '''
    html += '</div>'
    return html


def extract_concepts_and_description(source_code):
    concepts = ""
    description = ""
    new_lines = []

    lines = source_code.splitlines()
    mode = None  # 'concepts' | 'description' | None

    for line in lines:
        line_strip = line.strip()
        if line_strip.startswith("#"):
            line_content = line_strip[1:].strip()

            if line_content.lower().startswith("concepts:"):
                mode = 'concepts'
                concepts += line_content[len("concepts:"):].strip()
                continue
            elif line_content.lower().startswith("description:"):
                mode = 'description'
                description += line_content[len("description:"):].strip()
                continue
            else:
                if mode == 'concepts':
                    concepts += " " + line_content
                    continue
                elif mode == 'description':
                    description += " " + line_content
                    continue

        mode = None
        new_lines.append(line)

    new_source_code = "\n".join(new_lines)
    return concepts.strip(), description.strip(), new_source_code


def highlight_code(code):
    formatter = HtmlFormatter(noclasses=True)
    highlighted_code = highlight(code, PythonLexer(), formatter)
    return f"<pre style='background:#f7f7f7; font-size:14px; line-height:1.6; padding:16px; border-radius:8px; overflow-x:auto; font-family:monospace; white-space: pre-wrap; word-break: break-word;'>{highlighted_code}</pre>"


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--jsonl", type=str, required=True)
    parser.add_argument("--gifid", type=str, required=True)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--size", type=int, default=100)
    parser.add_argument("--outdir", type=str, required=True)
    args = parser.parse_args()
    with open(args.jsonl, "r") as f:
        jsonl_file = f.read()
    if jsonl_file == "":
        raise Exception("file is empty")
    with open(args.jsonl, "r") as f:
        lines = f.readlines()
    
    print("Reading JSONL file...")
    from tqdm import tqdm
    data = [json.loads(line) for line in tqdm(lines)]

    try:
        print(f"Visualizing the subset from index {args.start} to {args.start + args.size - 1}")
        data = data[args.start:args.start + args.size]
    except IndexError:
        print(f"Error: Not enough problems in the JSONL file starting from index {args.start}")
        exit()

    total_problems = len(data)
    htmls = []

    all_uids = []
    for idx, problem in enumerate(data):
        code = problem["source"]
        seeds = problem["seeds"]
        
        code = remove_trailing_code(code)
        
        examples = problem["examples"]
        if len(examples) < 4:
            print(f"Skipping problem {idx} with less than 4 examples")
            continue
        input_grids = [np.array(example[0]) for example in examples[0:4]]
        output_grids = [np.array(example[1]) for example in examples[0:4]]

        # create unique ID for each problem
        hash_code = hashlib.md5(code.encode()).hexdigest()
        hash_examples = hashlib.md5(str(examples[0:4]).encode()).hexdigest()
        uid = f"{hash_code[0:8]}{hash_examples[0:8]}"
        all_uids.append(uid)

        examples_input_output = [ {"input": input_grid.tolist(), "output": output_grid.tolist()}
                                    for input_grid, output_grid in zip(input_grids, output_grids) 
                                    if isinstance(output_grid, np.ndarray) ]

        if len(examples_input_output) == 0:
            assert False, "No valid input-output examples found"

        grid_html = build_html_grid(examples_input_output, uid)
        code_html = highlight_code(code)

        # Encode the source code in base64
        code_base64 = base64.b64encode(code.encode()).decode()
        json_data = {
            "uid": uid,
            "examples": examples_input_output,
            "code": code,
            "metadata": {
                "source_file": os.path.basename(args.jsonl)
            }
        }
        json_data_base64 = base64.b64encode(json.dumps(json_data).encode()).decode()

        gif_path = get_problem_gif_path(args.gifid)
        if gif_path:
            gif_base64_src = load_gif_base64_from_path(gif_path)
            gif_html = f'''
            <div style="text-align: center; margin-top: 20px;">
                <h3>Associated GIF</h3>
                <img src="{gif_base64_src}" style="max-width: 400px;">
            </div>
            '''
        else:
            gif_html = ''



        def convert_seed_id(seed_id):
            ret = ""
            if seed_id.endswith(".py"):
                ret = seed_id[:-3]
            if "_" in ret:
                ret = ret.split("_")[0]
            return ret

        # read description jsonl file.
        with open(args.jsonl, "r", encoding="utf-8") as f:
            description_data = [json.loads(line) for line in f]

        
        for idx, problem in enumerate(data):
            description_obj = description_data[idx]
            
            visual_elements = description_obj.get("visual_elements", "")
            static_patterns = description_obj.get("static_patterns", "")
            dynamic_patterns = description_obj.get("dynamic_patterns", "")
            core_principles = description_obj.get("core_principles", "")
            intergrated = description_obj.get("intergrated", "")
        
        if intergrated:
            scenario = description_obj.get("scenario", "")
            objects = description_obj.get("objects", "")
            composite_objects = description_obj.get("composite_objects", "")
            interactions = description_obj.get("interactions", "")
            fundamental_principle = description_obj.get("fundamental_principle", "")
            similar_situations = description_obj.get("similar_situations", "")

            details_html = f"""
            <div style="margin-bottom: 30px;">
                <h2>Visual Info from GIF</h2>

                <h3>Visual Elements</h3>
                {list_to_html_bullets(visual_elements)}

                <h3>Static Patterns</h3>
                {list_to_html_bullets(static_patterns)}

                <h3>Dynamic Patterns</h3>
                {list_to_html_bullets(dynamic_patterns)}

                <h3>Core Principles</h3>
                {list_to_html_bullets(core_principles)}

                <h3>Scenario</h3>
                {list_to_html_bullets(scenario)}

                <h3>Objects</h3>
                {list_to_html_bullets(objects)}

                <h3>Composite Objects</h3>
                {list_to_html_bullets(composite_objects)}

                <h3>Interactions</h3>
                {list_to_html_bullets(interactions)}

                <h3>Fundamental Principle</h3>
                {list_to_html_bullets(fundamental_principle)}

                <h3>Similar Situations</h3>
                {list_to_html_bullets(similar_situations)}
            </div>
            """

        else:
            details_html = f"""
            <div style="margin-bottom: 30px;">
                <h2>Visual Info from GIF</h2>

                <h3>Visual Elements</h3>
                {list_to_html_bullets(visual_elements)}

                <h3>Static Patterns</h3>
                {list_to_html_bullets(static_patterns)}

                <h3>Dynamic Patterns</h3>
                {list_to_html_bullets(dynamic_patterns)}

                <h3>Core Principles</h3>
                {list_to_html_bullets(core_principles)}
            </div>
            """


        code = problem["source"]
        print(f"code: {code}")
        concepts, description, code_without_concepts = extract_concepts_and_description(code)
        print(f"Concepts: {concepts}")
        print(f"Description: {description}")
        code_without_concepts = remove_trailing_code(code_without_concepts)
        code_html = highlight_code(code_without_concepts)


        concepts_html = f"""
        <div style="margin-bottom: 20px;">
            <div style="font-size: 20px; font-weight: bold; margin-bottom: 10px;">Concepts</div>
            <div style="font-size: 17px; line-height: 1.6;">
                {concepts}
            </div>
        </div>

        <div style="margin-bottom: 30px;">
            <div style="font-size: 20px; font-weight: bold; margin-bottom: 10px;">Description</div>
            <div style="font-size: 17px; line-height: 1.8; text-align: justify;">
                {description}
            </div>
        </div>
        """

        problem_html = f"""
        <div class="problem" id="problem_{idx}" style="display: {'block' if idx == 0 else 'none'};">
            <h2>Problem UID {uid}</h2>
            {details_html}
            <hr>
            <h3>Concepts & Description</h3>
            {concepts_html} 
            <hr>
            <h3>Problem GIF</h3>
            {gif_html}
            <hr>
            {grid_html}
            <div style="text-align: center; margin-top: 20px;">
                <div style="font-size:32px">Problem Examples: <div style="font-size:17px">Considering the input/output examples, do they form a good ARC problem? A good ARC problem is one where you feel confident that you can explain the underlying transformation pattern to another person and where the problem is not overly trivial (although being easy is acceptable).</div></div>
                <button class="good-button" id="example_good_{idx}" onclick="annotate('example', 'good', {idx})">Good</button>
                <button class="ok-button" id="example_ok_{idx}" onclick="annotate('example', 'ok', {idx})">Ok</button>
                <button class="bad-button" id="example_bad_{idx}" onclick="annotate('example', 'bad', {idx})">Bad</button>
                <br>
                <div style="font-size:32px">Solution Code: <div style="font-size:17px">A solution code is "good" if the comment / code pair is consistent with a potential natural language transformation description</div></div>
                <button class="good-button" id="code_good_{idx}" onclick="annotate('code', 'good', {idx})">Good</button>
                <button class="ok-button" id="code_ok_{idx}" onclick="annotate('code', 'ok', {idx})">Ok</button>
                <button class="bad-button" id="code_bad_{idx}" onclick="annotate('code', 'bad', {idx})">Bad</button>
                <br>
                <button class="download-button" onclick="download_to_file('{uid}.py', '{code_base64}')">Download Source Code</button>
                <button class="download-button" onclick="download_with_annotations('{uid}.json', '{json_data_base64}', '{uid}')">Download JSON Data</button>
                <button class="download-button" onclick="download_div_to_image('img_{uid}', '{uid}.png')">Download Image</button>
            </div>
            {code_html}
        </div>
        """
        htmls.append(problem_html)

    all_uids_javascript_str = "const all_uids = " + json.dumps(all_uids) + ";"

    final_html = f"""
        <!DOCTYPE html>
        <html>
            <head>
                <meta charset="utf-8">
                <title>Code Visualization</title>
                <meta name="viewport" content="width=device-width, initial-scale=1.0">
                <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0-beta3/css/all.min.css">
                <link rel="stylesheet" href="./style.css">
                <script src="https://html2canvas.hertzen.com/dist/html2canvas.min.js"></script>
            </head>
            <body>
                <div class="navigation-arrow prev-arrow" onclick="prevProblem()">
                    <i class="fas fa-arrow-left"></i>
                </div>
                <div class="navigation-arrow next-arrow" onclick="nextProblem()">
                    <i class="fas fa-arrow-right"></i>
                </div>
                <div id="progress">0/{total_problems}</div>
                <div>
                    {"".join(htmls)}
                </div>
                <script>
                    {all_uids_javascript_str}
                    let currentProblem = 0;
                    let currentUid = all_uids[currentProblem];
                    const totalProblems = {total_problems};
                    let annotatedCount = 0;

                    const all_metrics = ['example', 'code'];
                </script>
                <script src="./script.js"></script>
            </body>
        </html>
        """

    file_name = args.jsonl.replace(".jsonl", f"_start_{args.start}_size_{args.start + args.size}.html")
    if args.outdir:
        file_name = os.path.normpath(os.path.join(args.outdir, os.path.basename(file_name)))

    print(f"Writing to {file_name}")
   
    print("Current working directory:", os.getcwd())
    full_path = os.path.join(os.getcwd(), 'generated_problems', 'visualized')
    print("Full Path:", full_path)
    print("Path exists:", os.path.exists(full_path))
    long_path_prefix = '\\\\?\\' if os.name == 'nt' else ''
    if not os.path.exists(full_path):
        os.makedirs(full_path)
    file_name = long_path_prefix + os.path.abspath(file_name)
    with open(file_name, "w", encoding="utf-8") as f:
        f.write(final_html)
    
    support_files = {
        "misc/visualization/visualization-script.js": os.path.join(long_path_prefix, "script.js"),
        "misc/visualization/visualization-style.css": os.path.join(long_path_prefix, "style.css")
    }

    for src, dest in support_files.items():
        if not os.path.exists(dest):
            shutil.copy2(src, dest)