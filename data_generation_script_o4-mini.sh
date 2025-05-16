#!/bin/bash
# This resets the problem generation state by deleting everything---helpful if you want to start from scratch but be careful!
# rm -rf generated_descriptions generated_code generated_problems

# 대상 리스트 정의
TARGETS=("nature" "geometry" "pattern" "relational-pattern" "sequence")
MODEL="o4-mini"
# 각 대상에 대해 작업 반복
for TARGET in "${TARGETS[@]}"; do
  echo "Processing TARGET: $TARGET"
  
  # 디렉토리 생성
  mkdir -p $MODEL-generated_descriptions/$TARGET
  
  # 설명 생성
  python generate_descriptions.py --target $TARGET --samples 1 --outdir $MODEL-generated_descriptions/$TARGET --model $MODEL --num_generations 2 --max_tokens 40000 --batch_size 1 --num_descriptions 75 --rng_offset 777
  
  # 코드 생성 디렉토리 준비
  mkdir -p $MODEL-generated_code/$TARGET
  
  # 생성된 설명 파일에 대해 코드 생성
  for filename in `ls $MODEL-generated_descriptions/$TARGET/*jsonl`; do 
    python generate_code.py --outdir $MODEL-generated_code/$TARGET --ignore_cache_samples --prompt_model $MODEL --max_tokens 40000 -n 2 -s 4 --nohtml --jsonl $filename
  done
  
  # 문제 생성 디렉토리 준비
  mkdir -p $MODEL-generated_problems/$TARGET
  
  # 생성된 코드 파일에 대해 문제 생성
  for filename in `ls $MODEL-generated_code/$TARGET/*jsonl`; do 
    python generate_problems.py --jsonl $filename --outdir $MODEL-generated_problems/$TARGET --total_timeout 300
  done
  
  # 시각화 디렉토리 준비 (선택 사항)
  mkdir -p $MODEL-generated_problems/visualized/$TARGET
  
  # 생성된 문제 파일 시각화
  for filename in `ls $MODEL-generated_problems/$TARGET/*jsonl`; do 
    python visualize_problems.py --jsonl $filename --outdir $MODEL-generated_problems/visualized/$TARGET
  done
  
  echo "Completed processing TARGET: $TARGET"
  echo "----------------------------------------"
done