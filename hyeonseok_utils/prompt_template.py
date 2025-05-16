import base64
from utils_gif import *

def prompt_template_for_step_1_desc(gif_path, intergrated):
    base64_encoded = direct_encode_gif_to_base64(gif_path)
    if intergrated:
        with open("prompts/gif_intergrated.md", encoding="utf-8") as f:
            gif_prompt = f.read()
        with open("prompts/system_prompt_gif_intergrated.md", encoding="utf-8") as f: 
            gif_system_prompt = f.read()
    else:
        with open("prompts/gif.md", encoding="utf-8") as f:
            gif_prompt = f.read()
        with open("prompts/system_prompt_gif.md", encoding="utf-8") as f:
            gif_system_prompt = f.read()
    image_block = {"type": "image_url", "image_url": {"url": f"data:image/gif;base64,{base64_encoded}"}}
    message_user = [{"type": "text", "text": gif_prompt}, image_block]
    message_system = [{"type": "text", "text": gif_system_prompt}]
    return message_user, image_block, message_system