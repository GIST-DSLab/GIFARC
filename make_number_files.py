# make_number_files.py
from pathlib import Path
from typing import Union  # ← 추가

def write_number_files(
    start: int = 1,
    end: int = 400,
    chunk: int = 100,
    stem: str = "filename",
    out_dir: Union[str, Path] = ".",   # ← str | Path → Union[str, Path]
) -> None:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for chunk_start in range(start, end + 1, chunk):
        chunk_end = min(chunk_start + chunk - 1, end)
        numbers = ", ".join(str(n) for n in range(chunk_start, chunk_end + 1))
        fname = out_dir / f"{stem}_{chunk_start}_{chunk_end}.txt"
        fname.write_text(numbers, encoding="utf-8")
        print(f"✅ wrote {fname} ({chunk_end - chunk_start + 1} numbers)")

if __name__ == "__main__":
    write_number_files(start=1, end=400, chunk=100, stem="test_geometry")
    write_number_files(start=1, end=400, chunk=100, stem="test_nature")
    write_number_files(start=1, end=400, chunk=100, stem="test_pattern")
    write_number_files(start=1, end=400, chunk=100, stem="test_relational-pattern")
    write_number_files(start=1, end=400, chunk=100, stem="test_sequence")

