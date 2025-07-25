import os
def gif_list_load(data_dir):
    video_files = [
        os.path.join(data_dir, f)
        for f in os.listdir(data_dir)
        if f.endswith(".gif") or f.endswith(".webm")
    ]
    return video_files


