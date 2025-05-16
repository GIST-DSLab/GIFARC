#!/usr/bin/env python3
"""
split_csv_batches.py
CSV를 m행 단위로 나누고,
지정한 열의 값만 추출해 batch_k.txt 로 저장.
"""
import csv
import os
import argparse
from pathlib import Path
from typing import List

def write_batch(lines: List[str], batch_idx: int, out_dir: Path) -> None:
    """batch_{k}.txt 파일로 저장"""
    out_dir.mkdir(parents=True, exist_ok=True)
    fname = out_dir / f"batch_{batch_idx}.txt"
    with fname.open("w", encoding="ISO-8859-1") as f:
        f.writelines(f"{line},\n" for line in lines)

def split_csv_to_batches(csv_path: Path, batch_size: int,
                         target_col, out_dir: Path) -> None:
    """CSV를 batch_size 단위로 나누어 target_col 값만 저장"""
    with csv_path.open(newline="", encoding="ISO-8859-1") as fp:
        reader = csv.DictReader(fp)
        batch_idx, bucket = 0, []

        for row_idx, row in enumerate(reader, start=1):
            bucket.append(row[target_col])
            if row_idx % batch_size == 0:
                write_batch(bucket, batch_idx, out_dir)
                batch_idx += 1
                bucket = []

        # 남은 레코드
        if bucket:
            write_batch(bucket, batch_idx, out_dir)

def main():
    p = argparse.ArgumentParser(
        description="CSV를 m행씩 끊어 지정 열만 batch_k.txt로 저장")
    p.add_argument("csv_path", help="입력 CSV 경로")
    p.add_argument("-m", "--batch-size", type=int, required=True,
                   help="배치 크기 (m)")
    p.add_argument("-c", "--column", required=True,
                   help="추출할 열 (헤더명 또는 0‑기준 인덱스)")
    p.add_argument("-o", "--output-dir", default="batches",
                   help="결과 텍스트 파일 저장 폴더 (기본: ./batches)")
    args = p.parse_args()

    csv_path = Path(args.csv_path).expanduser()
    out_dir  = Path(args.output_dir).expanduser()

    # 헤더 이름/인덱스 처리
    with csv_path.open(newline="", encoding="ISO-8859-1") as fp:
        headers = next(csv.reader(fp))           # 첫 줄만 읽어 헤더 확보
    if args.column.isdigit():                    # 숫자면 인덱스로 간주
        col_idx = int(args.column)
        if col_idx >= len(headers):
            raise IndexError(f"CSV에 열 {col_idx} 가 없습니다.")
        target_col = headers[col_idx]
    else:
        target_col = args.column
        if target_col not in headers:
            raise KeyError(f"CSV에 '{target_col}' 열이 없습니다.")

    split_csv_to_batches(csv_path, args.batch_size,
                         target_col, out_dir)

# for enc in ["cp949", "euc-kr", "ISO-8859-1", "latin1"]:
#     try:
#         df = pd.read_csv("all_gifs_metadata.csv", encoding=enc)
#         print(f"✅ 성공: {enc}, shape={df.shape}")
#         break              # 읽기 성공 시 반복 종료
#     except UnicodeDecodeError as e:
#         print(f"❌ {enc} 실패: {e}")

if __name__ == "__main__":
    main()
#    python data_batch_generation.py all_gifs_metadata.csv -m 300 -c id -o uuid_batchs