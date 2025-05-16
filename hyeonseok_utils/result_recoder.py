import os 

def parse_step_code_result(
    output_dir, 
    file_name_json, 
    codes_and_seeds, 
    ensure_colors_exist, 
    ):
    
    if output_dir is not None: # join with the base path
        file_name_json = os.path.join(output_dir, os.path.basename(file_name_json))
    
    print(f"Writing to jsonl {file_name_json}")

    with open(file_name_json, "w") as f:
        # jsonl, one json per line
        import json
        for codes, seeds in codes_and_seeds:
            f.write(json.dumps({"code": [ensure_colors_exist(code[0]) for code in codes],
                                "seeds": seeds
                                }) + "\n")
    print(f"{len(codes_and_seeds)} codes written to {file_name_json}")
 