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
from errors import *
import yaml
import re
import json

def init_openai_api():
    """
    OpenAI API 클라이언트를 초기화합니다.

    Args:
        api_key (str): OpenAI API 키

    Returns:
        openai.Client: 초기화된 OpenAI API 클라이언트
    """

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
    주어진 텍스트가 주어진 모델 기준으로 몇 개의 토큰으로 인코딩되는지 계산합니다.

    Args:
        text (str): 토큰 개수를 측정할 텍스트 (예: base64 문자열)
        model (str): 사용할 모델 이름 (예: gpt-4, gpt-3.5-turbo, etc.)

    Returns:
        int: 토큰 개수
    """
    try:
        encoding = tiktoken.encoding_for_model(model)
    except KeyError:
        print(f"[경고] '{model}'에 대한 인코딩이 등록되어 있지 않아 기본 인코딩 사용")
        encoding = tiktoken.get_encoding("cl100k_base")  # 대부분 모델에서 사용되는 기본값

    tokens = encoding.encode(text)
    return len(tokens)

def call_api(model, message, max_completion_tokens=40_000, stop=None, stream=False, max_retries=3):
    """
    OpenAI API를 호출하여 응답을 가져옵니다.
    
    Args:
        model (str): 사용할 모델 이름 (예: gpt-4, gpt-3.5-turbo 등)
        message (list): 메시지 리스트 (예: [{"role": "user", "content": "질문"}])
        max_completion_tokens (int): 최대 생성 토큰 수
        stop (list): 정지 토큰 리스트
        stream (bool): 스트리밍 여부

    Returns:
        dict: API 응답
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
                # 사용 정책 위반으로 프롬프트가 차단된 경우
                continue
            else:
                raise e  # 다른 오류는 그대로 발생시킴
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
    # GIF는 여러 프레임을 포함하므로 단순 리사이즈가 쉽지 않을 수 있음.
    # moviepy를 사용하는 방법도 고려해보세요.
    try:
        im = Image.open(input_path)
        new_width = int(im.width * scale_factor)
        new_height = int(im.height * scale_factor)
        
        # GIF의 경우, 최적화 작업은 프레임 별로 진행해야 할 수도 있음.
        frames = []
        try:
            while True:
                frame = im.copy().resize((new_width, new_height), Image.Resampling.LANCZOS)
                frames.append(frame)
                im.seek(im.tell() + 1)
        except EOFError:
            # 모든 프레임 처리 완료
            pass
        frames[0].save(output_path, save_all=True, append_images=frames[1:], optimize=True, loop=0)
        return output_path
    except Exception as e:
        print(f"Error compressing GIF: {e}")
        return input_path  # 실패 시 원본 리턴


def convert_webm_to_gif(webm_path):
    """
    주어진 webm 파일을 gif로 변환하여 임시 파일 경로를 리턴합니다.
    max_frames 값에 따라 전체 길이를 제한하고 싶다면 clip.subclip() 등의 방법으로 조절할 수 있습니다.
    """
    clip = VideoFileClip(webm_path)
    # 만약 전체 클립 길이가 길다면 필요한 구간만 사용할 수 있음 (예: 앞 3초)
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
    gif_path: 인코딩할 GIF 파일의 경로
    반환: base64 인코딩된 문자열
    """
    with open(gif_path, "rb") as f:
        gif_data = f.read()  # 바이너리로 읽기

    base64_str = base64.b64encode(gif_data).decode("utf-8")  # base64 인코딩 + 문자열로 변환
    return base64_str


def extract_key_frames_from_webm(video_path, num_frames=3):
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    selected_indices = [0, total_frames // 2, total_frames - 1]
    selected_indices = [min(idx, total_frames - 1) for idx in selected_indices]  # 범위 체크

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
        raise ValueError("지원하지 않는 파일 형식입니다.")
    
def filter_non_system_messages(total_history):
    """
    extracts non-system messages from the conversation history.
    """
    # 기존 total_history에서 system 메시지는 제외한 후, 사용자 메시지에서 image_block 제거
    non_system_history = []
    for msg in total_history:
        if msg.get("role") == "system":
            continue  # system 메시지는 여기서는 제외하고, 나중에 따로 추가함
        elif msg.get("role") == "user":
            # user 메시지의 content에서 image_url 타입을 제거
            new_content = [block for block in msg.get("content", []) if block.get("type") != "image_url"]
            # content가 남아있을 때만 메시지를 추가 (빈 리스트라면 skip)
            if new_content:
                # 복사본 생성 후 content만 교체
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
    """
    문자열을 JSON 객체로 파싱하기 위해 여러 방식을 시도합니다.
    실패 시 None을 반환합니다.
    """
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
