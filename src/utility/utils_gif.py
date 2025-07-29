import os  
import base64
import cv2
from PIL import Image, ImageSequence
import ast
import tiktoken
import tempfile
from moviepy import VideoFileClip
import logging
from openai import AzureOpenAI  
import yaml
import re
import json




# initializes  openai API.
# @params api_key (str): OpenAI API key string
# @returns openai.Client: initialized OpenAI API client
def init_openai_api():
    

    deployment_image_processor_name = None
    deployment_data_generator_name = None

    if os.path.exists("configs/config.yaml"):
        with open("configs/config.yaml", "r") as file:
            config = yaml.safe_load(file)
            endpoint = config.get("AZURE_OPENAI_ENDPOINT")
            subscription_key = config.get("AZURE_OPENAI_API_KEY")
            deployment_image_processor_name = config.get("DEPLOYMENT_IMAGE_PROCESSOR_NAME")
            deployment_data_generator_name = config.get("DEPLOYMENT_DATA_GENERATOR_NAME")
    else:
        deployment_image_processor_name = config.get("DEPLOYMENT_IMAGE_PROCESSOR_NAME")
        deployment_data_generator_name = config.get("DEPLOYMENT_DATA_GENERATOR_NAME")
        endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")  
        subscription_key = os.getenv("AZURE_OPENAI_API_KEY")

    # Initialize Azure OpenAI Service client with key-based authentication    
    global client
    client = AzureOpenAI(  
        azure_endpoint=endpoint,  
        api_key=subscription_key,  
        api_version="2024-12-01-preview",
    )
    return deployment_image_processor_name, deployment_data_generator_name

def count_tokens_with_tiktoken(text: str, model: str = "gpt-4") -> int:
    """
    Calculate the number of tokens that the given text would encode into for the specified model.

    Args:
        text (str): The text to count tokens for (e.g., a base64-encoded string).
        model (str): The name of the model to use (e.g., 'gpt-4', 'gpt-3.5-turbo', etc.).

    Returns:
        int: The number of tokens.
    """
    try:
        encoding = tiktoken.encoding_for_model(model)
    except KeyError:
        print(f"[WARN] No Setted Encoder for Model:'{model}', will just use base encoder cl100k_base")
        encoding = tiktoken.get_encoding("cl100k_base") 

    tokens = encoding.encode(text)
    return len(tokens)

def call_api(model, message, max_completion_tokens=40_000, stop=None, stream=False, max_retries=3):
    """
    Call the OpenAI API to retrieve a response.

    Args:
        model (str): The model name to use (e.g., 'gpt-4', 'gpt-3.5-turbo', etc.).
        message (list): A list of message dictionaries (e.g., [{"role": "user", "content": "question"}]).
        max_completion_tokens (int): The maximum number of tokens to generate.
        stop (list): A list of stop tokens.
        stream (bool): Whether to stream the response.

    Returns:
        dict: The API response.
    """
    global client
    for i in range(max_retries):
        try:
            completion = client.chat.completions.create(
                model=model,
                messages=message,
                max_completion_tokens=max_completion_tokens,
                stop=stop,
                stream=stream
            )
        except Exception as e:
            error_str = str(e)
            if "potentially violating our usage policy." in error_str:
                # When prompt Blocked by policy
                continue
            else:
                raise e  # Other will be raise
    return completion


def check_file_size(file_path, max_size=4 * 1024 * 1024):
    compress_flag = False

    file_size = os.path.getsize(file_path)
    if file_size > max_size:
        scale_factor = max_size / file_size
        compress_file_path = file_path.replace('.gif', '_compressed.gif')
        file_path = compress_gif(file_path, compress_file_path, scale_factor=scale_factor)
        compress_flag = True

    return file_path, compress_flag


def compress_gif(input_path, output_path, scale_factor=0.5):

    try:
        im = Image.open(input_path)
        new_width = int(im.width * scale_factor)
        new_height = int(im.height * scale_factor)
        
        frames = []
        try:
            while True:
                frame = im.copy().resize((new_width, new_height), Image.Resampling.LANCZOS)
                frames.append(frame)
                im.seek(im.tell() + 1)
        except EOFError:
            # All frame done
            pass
        frames[0].save(output_path, save_all=True, append_images=frames[1:], optimize=True, loop=0)
        return output_path
    except Exception as e:
        print(f"Error compressing GIF: {e}")
        return input_path  # when it failed return original


def convert_webm_to_gif(webm_path):
    """
    Convert the given WebM file to a GIF and return the path to the temporary GIF file.
    To limit the total duration based on max_frames, you can use methods such as clip.subclip().
    """
    clip = VideoFileClip(webm_path)
    # If the full clip is too long, you can use only the needed segment (e.g., the first 3 seconds)
    # clip = clip.subclip(0, 3)
    
    temp_gif = tempfile.NamedTemporaryFile(suffix=".gif", delete=False)
    temp_gif.close()  # moviepy가 파일명만 사용하므로 close해 줍니다.
    clip.write_gif(temp_gif.name)
    clip.reader.close()
    if clip.audio is not None and clip.audio.reader is not None:
        clip.audio.reader.close_proc()
    return temp_gif.name

# Get the list of function names from a given file(dsl.py)
def extract_function_names_from_file(filepath):
    with open(filepath, "r", encoding="utf-8") as f:
        source = f.read()

    tree = ast.parse(source)

    # Extract all function definition nodes and get their names
    function_names = [node.name for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)]
    return function_names

# Extract the function names that are called in the code string
def extract_called_functions(code_str):
    tree = ast.parse(code_str)
    called = []

    class FunctionCallVisitor(ast.NodeVisitor):
        def visit_Call(self, node):
            if isinstance(node.func, ast.Name):  # calling a function directly
                called.append(node.func.id)
            elif isinstance(node.func, ast.Attribute):  # calling a method of an object
                called.append(node.func.attr)
            self.generic_visit(node)

    FunctionCallVisitor().visit(tree)
    return called

# Check if the code uses the functions defined in dsl.py
def check_code(info):
    code_info = info.get('code', None)

    if code_info is None:
        raise ValueError("Code or task information is missing in the input.")

    functions = extract_function_names_from_file('code/dsl.py')
    called_funcs = extract_called_functions(code_info)
    # defined_functions = [node.name for node in ast.walk(ast.parse(code_info)) if isinstance(node, ast.FunctionDef)]
    used_functions = [f for f in functions if f in called_funcs]

    info['used_dsls'] = {key: used_functions.count(key) for key in functions}

    # if len(defined_functions) > 1:
    #     info['used_dsls']['llm_defined'] = {key: defined_functions.count(key) for key in defined_functions if key!='solve'}
    # check the created function use the dsl function
    if len(used_functions) == 0:
        raise ValueError(f"""
                            Id: {info['id']}
                            Used_dsls not found in code: {code_info}
                        """)
    
    return info

# Make the output grid based on the code and task information
def make_output_grid(info):
    code_info = info.get('code', None)
    task_info = info.get('task', None)
    id_info = info.get('id', None)

    if code_info is None or task_info is None:
        raise ValueError("Code or task information is missing in the input.")

    # Check if the generated code is valid
    try:
        namespace = {"__name__": "__not_main__"}
        exec(code_info, namespace)

        for i, task in enumerate(task_info['train']):
            try:
                exec(code_info, namespace)
                solve = namespace.get("solve")
                generated_output = solve(task['input'])
            except Exception as e:
                generated_output = solve(tuple(tuple(row) for row in task['input']))

            info['task']['train'][i]['output'] = generated_output
            
        for i, task in enumerate(task_info['test']):
            try:
                exec(code_info, namespace)
                solve = namespace.get("solve")
                generated_output = solve(task['input'])
            except Exception as e:
                generated_output = solve(tuple(tuple(row) for row in task['input']))

            info['task']['test'][i]['output'] = generated_output

    except Exception as e:
        print(f"Error in {id_info}: {e}")
        raise ExecutionError(e, task['input'])

    return info

def direct_encode_gif_to_base64(gif_path):
    """
    gif_path: Path to the GIF file to encode.
    Returns:
        A base64-encoded string.
    """
    with open(gif_path, "rb") as f:
        gif_data = f.read()  # Read as  binary

    base64_str = base64.b64encode(gif_data).decode("utf-8")  # base64 + String
    return base64_str


def extract_key_frames_from_webm(video_path, num_frames=3):
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    selected_indices = [0, total_frames // 2, total_frames - 1]
    selected_indices = [min(idx, total_frames - 1) for idx in selected_indices]  # Check range

    base64_images = []
    for idx in selected_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        success, frame = cap.read()
        if not success:
            continue

        temp_path = f"./temp_frame_{idx}.png"
        cv2.imwrite(temp_path, frame)

        with open(temp_path, "rb") as f:
            encoded = base64.b64encode(f.read()).decode("utf-8")
            base64_images.append(encoded)

        os.remove(temp_path)
    cap.release()
    return base64_images

def extract_key_frames(gif_path):
    img = Image.open(gif_path)
    total_frames = img.n_frames
    selected_indices = [0, total_frames // 2, total_frames - 1]
    base64_images = []

    for i in selected_indices:
        img.seek(i)
        frame = img.convert("RGB")
        temp_path = f"./temp_frame_{i}.png"
        frame.save(temp_path)

        with open(temp_path, "rb") as f:
            encoded = base64.b64encode(f.read()).decode("utf-8")
            base64_images.append(encoded)

        os.remove(temp_path)
    return base64_images

def extract_key_frames_any(path):
    if path.endswith(".gif"):
        return extract_key_frames(path)
    elif path.endswith(".webm"):
        return extract_key_frames_from_webm(path)
    else:
        raise ValueError("This format is not supporting")
    
def filter_non_system_messages(total_history):
    """
    extracts non-system messages from the conversation history.
    """
    # Exclude system messages from total_history, then remove image_block from user messages
    non_system_history = []
    for msg in total_history:
        if msg.get("role") == "system":
            continue  # System messages are excluded here; they will be added separately later
        elif msg.get("role") == "user":
            # From user message content, remove type: image_url 
            new_content = [block for block in msg.get("content", []) if block.get("type") != "image_url"]
            # Only add the message if there is remaining content (skip if the list is empty)
            if new_content:
                # After generate copy change the content only
                new_msg = msg.copy()
                new_msg["content"] = new_content
                non_system_history.append(new_msg)
        else:
            non_system_history.append(msg)
    return non_system_history

def build_message_with_process(question, answer_format, response=None, total_history=None, max_history_count=5, base64_encoded=None):
    if total_history is None:
        total_history = []

    if response:
        # For o3-mini model, we need previouds message to be added to the message withouth image block
        message_assistant = {"role": "assistant", "content": [{"type": "text", "text": response}]}
        message_user  = {"role": "user", "content": [{"type": "text", "text": question}]}
        message_system  = {"role": "system", "content": [{"type": "text", "text": answer_format}]}
        total_history.extend([message_assistant, message_user, message_system])

        # Using the filter function to remove system messages from the history and image blocks from user messages
        non_system_history = filter_non_system_messages(total_history)
        n_history = non_system_history[-max_history_count:]
        n_history.extend([message_system])

        return n_history, total_history
    else:
        # For o1 model, we need to add the image block to the message
        image_block = {"type": "image_url", "image_url": {"url": f"data:image/gif;base64,{base64_encoded}"}}
        message_user = {"role": "user", "content": [{"type": "text", "text": question}, image_block]}
        message_system = {"role": "system", "content": [{"type": "text", "text": answer_format}]}
        total_history.extend([message_user, message_system])

        return total_history, total_history
    
def insert_values_into_question(target_str, keys, infos):
    for key in keys:
        if key != 'dsl' and key != 'arc_types':
            target_str = target_str.replace("{" + key + "_info}", str(infos[key]))
        else:
            target_str = target_str.replace("{" + key + "_info}", dsl_info if key == 'dsl' else arc_types_info)
    return target_str

def insert_values_into_json(target_dict, keys, infos):
    for key in keys:
        target_dict[key] = infos[key]
    return target_dict

def robust_parse_json(text: str):
    # Try various ways to parse string into JSON.
    
    try:
        cleaned_text = clean_llm_json_output(text)

        return json.loads(cleaned_text)
    except json.JSONDecodeError:
        try:
            return json.loads(cleaned_text.replace("'", '"'))
        except json.JSONDecodeError:
            try:
                return json.loads(cleaned_text.replace('"', "'"))
            except json.JSONDecodeError:
                try:
                    return ast.literal_eval(cleaned_text)
                except Exception:
                    return None

def clean_llm_json_output(text: str):
    # Remove Markdown code block if present
    text = re.sub(r"^```(?:json)?\n?", "", text.strip())
    text = re.sub(r"\n?```$", "", text.strip())
    return text

class ExecutionError(Exception):
    def __init__(self, error_info, target_input):
        self.error_info = error_info
        self.target_input = target_input

        message = json.dumps({
            "error_info": str(error_info),
            "target_input": target_input
        })
        super().__init__(message)
