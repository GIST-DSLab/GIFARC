# Supervised fine-tuning and evaluation

This directory contains the GIFARC training, dataset-preparation, and evaluation scripts. The training script accepts a prepared SFT JSONL and writes its final model and tokenizer to the requested output directory. The evaluator defaults to transduction, which scores predicted output grids.

Use an environment with PyTorch, Transformers, PEFT, Datasets, Accelerate, tqdm, NumPy, tiktoken, and Weights & Biases. Transformers uses the cache configured in your environment.

## Train and evaluate

Run these commands from the repository root with paths to your own prepared inputs.

~~~bash
python reproducibility/sft/sft_train.py \
  --model_name nvidia/Mistral-NeMo-Minitron-8B-Base \
  --ground_truth_data_path /path/to/training.jsonl \
  --output_dir ./models/nemo-gifarc-seed777

python reproducibility/sft/sft_eval.py \
  --model_name nvidia/Mistral-NeMo-Minitron-8B-Base \
  --model_path ./models/nemo-gifarc-seed777 \
  --input_data_path /path/to/eval-input.jsonl \
  --ground_truth_data_path /path/to/eval-targets.jsonl \
  --output_path ./evaluation.jsonl \
  --temperature 0
~~~

Use matching, unique `(problem_id, test_index)` keys in the two evaluation files and the `messages` input format. Compare result keys with the input keys after a run. Legacy raw records mixed with message records can produce duplicate results, and missing target keys can leave rows unscored. The evaluator's completion message can appear after either case.

The historical Bash launchers expect their original dataset layouts. Set `GIFARC_TRAIN_OUTPUT_DIR` for the training output root or `GIFARC_CHECKPOINT_ROOT` for the evaluation checkpoint root. Both roots default to `./models` relative to the launch directory.

The optional `induction` and `both` modes execute generated Python in the evaluation process.
