#!/bin/bash
# This resets the problem generation state by deleting everything---helpful if you want to start from scratch but be careful!
# rm -rf generated_descriptions generated_code generated_problems

# # First we generate natural language descriptions of problems
# # We do this in batches of 1000, producing 10 new descriptions with each batch, iterating through 10 different seeds
# # The final output we're after is 100k descriptions
# # Executing this script will produce files in generated_descriptions of the form:
# #           generated_descriptions/self_instruct_descriptions_fewshot_75_gpt-4_temp0.70_maxtokens1024_rng[0-9]*_used_concepts.jsonl
# mkdir -p generated_descriptions
MODEL="o3-mini"
TARGET="sequence"
# mkdir -p $MODEL-generated_descriptions/$TARGET
# python generate_descriptions.py --target $TARGET --samples 1 --outdir $MODEL-generated_descriptions/$TARGET --model o3-mini --num_generations 2 --max_tokens 40000 --batch_size 1 --num_descriptions 75 --rng_offset 777;

# # # # # Next we use the RAG model, using batching. 
# # # # # You don't have to keep tack of the intermediate files because it's smart about handling the batching for you
# # # # # # We do this both with suggesting functions, and without, to get diversity
mkdir -p $MODEL-generated_code
mkdir -p $MODEL-generated_code/$TARGET
for filename in `ls $MODEL-generated_descriptions/$TARGET/*jsonl`; do python generate_code.py                    --outdir $MODEL-generated_code/$TARGET --ignore_cache_samples --prompt_model o3-mini --max_tokens 40000 -n 2 -s 4 --nohtml --jsonl $filename; 
done
# # for filename in `ls generated_descriptions/*jsonl`; do python generate_code.py --suggest_function --outdir generated_code --ignore_cache_samples --prompt_model o3-mini --max_tokens 40000 -n 2 -s 4 --nohtml --jsonl $filename; done

# # # # # This will generate files of the form generated_code/self_instruct_code_fewshot_4_gpt-4o-mini_temp0.70_maxtokens2048_briefcommon_description_file_generated_descriptions_self_instruct_descriptions_fewshot_75_gpt-4_temp0.70_maxtokens1024_rng*.jsonl

# # # # # run problem generation script on everything in the directory, producing output in generated_problems
mkdir -p $MODEL-generated_problems
mkdir -p $MODEL-generated_problems/$TARGET
for filename in `ls $MODEL-generated_code/$TARGET/*jsonl`; do python generate_problems.py --jsonl $filename --outdir $MODEL-generated_problems/$TARGET --total_timeout 300; 
done
# # # if you want to you can visualize them like this, but this step is totally optional
mkdir -p $MODEL-generated_problems/visualized
mkdir -p $MODEL-generated_problems/visualized/$TARGET
for filename in `ls $MODEL-generated_problems/$TARGET/*jsonl`; do 
    python visualize_problems.py --jsonl $filename --outdir $MODEL-generated_problems/visualized/$TARGET;
done 