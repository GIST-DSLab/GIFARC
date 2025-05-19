import os
import re
def get_seeds_idx_ordered_content_from_files(seed_data_path):
    seeds = os.listdir(os.path.join(seed_data_path, "seeds") )
    # filter files with .py extension and 8 hex value characters in the file name
    pattern = r"[0-9a-f]{8}(_[a-zA-Z]+)?\.py"
    # get all files and its content
    seeds = [seed for seed in seeds if re.match(pattern, seed)]
    seeds_contents = []
    for seed in seeds:
        with open(os.path.join(seed_data_path, "seeds", seed)) as f:
            seeds_contents.append((seed, f.read()))
    
    return seeds, seeds_contents

def get_rng_offeset(raw_rng_offset, seeds):
    # print all files
    print(f"Using the following {len(seeds)} seeds:", ", ".join(seeds).replace(".py", ""))
    # derive a offset from rng_seed_offset by hashing it if the rng_seed_orig is not 0
    from hashlib import md5
    if raw_rng_offset != 0:

        rng_offset_str = md5(str(raw_rng_offset).encode()).hexdigest()[:7]
        print(rng_offset_str)
        # to integer
        rng_offset = int(rng_offset_str, 16)
    else:
        rng_offset = 0