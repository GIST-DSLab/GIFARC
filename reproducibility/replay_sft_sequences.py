"""Replay retained SFT sequence-building code with locally supplied pinned tokenizers.

Requires the matching saved messages-format training records, which are not
bundled. Loads local tokenizer files only, verifies their published hashes,
and requires transformers 4.53.2/tokenizers 0.21.2. No weights are loaded and no
model inference or training is performed. Historical executed counts are unknown.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
from pathlib import Path
import statistics

MANIFEST = Path(__file__).resolve().parent / 'summaries' / 'sft-sequence-audit' / 'current-pinned-tokenizer-reconstruction.json'
COLOR_NAMES = {0: 'Black', 1: 'Blue', 2: 'Red', 3: 'Green', 4: 'Yellow', 5: 'Gray', 6: 'Pink', 7: 'Orange', 8: 'Purple', 9: 'Brown'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, required=True, help='Locally supplied saved messages/answer_messages/ground_truth_output JSONL')
    parser.add_argument('--tokenizer-dir', type=Path, required=True, help='Local directory containing the pinned tokenizer files for the selected alias')
    parser.add_argument('--alias', choices=('nemo', 'qwen3_14b'), required=True)
    parser.add_argument('--seed', type=int, choices=(444, 555, 666), required=True)
    parser.add_argument('--dataset', choices=('gifarc_full', 'arc_agi2_base'), required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    reference = json.loads(MANIFEST.read_text(encoding='utf-8'))
    group = next(g for g in reference['summary'] if (g['tokenizer'], g['seed'], g['dataset']) == (args.alias, args.seed, args.dataset))
    if hashlib.sha256(args.data.read_bytes()).hexdigest() != group['saved_dataset_sha256']:
        raise ValueError('Input must match the saved training-file hash recorded for this audit group')
    for file in reference['tokenizer_files']:
        if file['alias'] == args.alias and 'sha256' in file:
            if hashlib.sha256((args.tokenizer_dir / file['filename']).read_bytes()).hexdigest() != file['sha256']:
                raise ValueError(f'Pinned tokenizer-file hash mismatch: {file["filename"]}')
    import tokenizers
    import transformers
    from transformers import AutoTokenizer
    import sft_sequence_logic as retained
    if transformers.__version__ != '4.53.2' or tokenizers.__version__ != '0.21.2':
        raise ValueError('Exact replay requires transformers 4.53.2 and tokenizers 0.21.2')
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_dir, local_files_only=True, trust_remote_code=False)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.model_max_length = 10**9
    retained.tokenizer, retained.max_length = tokenizer, 6144
    records = [json.loads(line) for line in args.data.read_text(encoding='utf-8').splitlines() if line.strip()]
    if len(records) != 1000 or any(not {'messages', 'answer_messages', 'ground_truth_output'} <= r.keys() or 'examples' in r for r in records):
        raise ValueError('Expected 1000 saved messages-format records; legacy source/examples branches are outside this replay')
    rows = []
    for start in range(0, len(records), 16):
        batch = records[start:start + 16]
        texts = [retained.convert_to_sft_format(r) for r in batch]
        result = retained.tokenize_function({'text': texts, 'original_data': batch})
        for offset, (record, text, ids) in enumerate(zip(batch, texts, result['input_ids'])):
            boundary = text.find('Assistant:') + len('Assistant:')
            input_part, output_part = text[:boundary], text[boundary:]
            input_length = len(tokenizer(input_part, add_special_tokens=True)['input_ids'])
            target_length = len(tokenizer(output_part, add_special_tokens=False)['input_ids'])
            raw_length = len(tokenizer(text, add_special_tokens=True)['input_ids'])
            decoded = tokenizer.decode(ids, skip_special_tokens=True)
            decoded_boundary = decoded.find('Assistant:')
            retained_target = decoded[decoded_boundary + len('Assistant:'):] if decoded_boundary >= 0 else ''
            answer = record['answer_messages'][0]['content']
            expected = tokenizer.decode(tokenizer(answer, add_special_tokens=False)['input_ids'], skip_special_tokens=True).strip()
            rendered = '\n'.join(' '.join(COLOR_NAMES[c] for c in row) for row in record['ground_truth_output'])
            if len(ids) != 6144:
                raise ValueError('Reconstructed array length differs from 6144')
            rows.append({
                'tokenizer': args.alias, 'seed': args.seed, 'dataset': args.dataset, 'line': start + offset + 1,
                'test_index': record.get('test_index'), 'raw_whole_sequence_tokens': raw_length,
                'raw_separate_input_tokens': input_length, 'raw_separate_target_tokens': target_length,
                'raw_separate_total_tokens': input_length + target_length, 'raw_whole_over_6144': raw_length > 6144,
                'reconstruction_input_shortened': input_length + target_length > 6144 and input_length > int(6144 * .7),
                'reconstruction_complete_target_text_preserved': expected in retained_target.strip(),
                'reconstruction_complete_rendered_gold_grid_preserved': rendered in retained_target,
                'reconstruction_complete_gold_grid_fence_preserved': f'```\n{rendered}\n```' in retained_target,
                'reconstruction_thinking_header_present': '# THINKING ANALOGY' in retained_target,
                'reconstruction_code_header_present': '# SOLUTION CODE' in retained_target,
                'reconstruction_grid_header_present': '# RESULT TEST OUTPUT GRID' in retained_target,
                'reconstruction_assistant_marker_present': decoded_boundary >= 0,
                'reconstruction_array_positions': len(ids),
                'reconstruction_nonpad_positions': sum(i != tokenizer.pad_token_id for i in ids),
            })
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x', newline='', encoding='utf-8') as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    incomplete = sum(not r['reconstruction_complete_rendered_gold_grid_preserved'] for r in rows)
    if incomplete != group['reconstruction_incomplete_rendered_gold_grid']:
        raise ValueError('Reconstruction count differs from included recorded audit')
    print(json.dumps({'tokenizer': args.alias, 'seed': args.seed, 'dataset': args.dataset, 'rows': len(rows),
                      'reconstruction_incomplete_rendered_gold_grid': incomplete,
                      'historical_executed_truncation_counts': None, 'label_masking_population_audit': False}, indent=2))


if __name__ == '__main__':
    main()
