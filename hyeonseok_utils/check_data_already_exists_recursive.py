from pathlib import Path
from typing import Iterable, Tuple
import logging

# Function to check if the result file already exists
# 지정된 폴더 아래의 모든 하위 폴더를 검색하여
# 지정된 파일이 존재하는지 확인하는 함수
def check_data_path_already_exists_recursive(
    check_lists,
    output_base_folders,
    * ,
    recursive = True
):
    """
    Determine which items in `check_lists` already have results under any of
    the `output_base_folders`.

    Parameters
    ----------
    check_lists : Iterable[str | pathlib.Path]
        File or directory names (relative or absolute) you want to check.
        Example: ["task_001.json", "geometry_42", Path("foo/bar.txt")]

    output_base_folders : Iterable[str | pathlib.Path]
        One or more base directories in which results may be found.
        Example: ["./[intergrated]generated_descriptions", "./outputs"]

    recursive : bool, optional (default=True)
        If True, search recursively (`rglob`) inside each base folder.
        If False, look only at the top level inside each base folder.

    Returns
    -------
    Tuple[list[pathlib.Path], list[str]]
        (existing_results, missing_items)

        * existing_results : list of full `Path` objects where each item was found
        * missing_items    : list of item names (str) that were *not* found
    """
    # 절대 경로로 변환해 두기
    base_dirs = [Path(d).resolve() for d in output_base_folders]

    existing = []
    missing = []
    # print(base_dirs)
    for item in check_lists:
        item_name = str(Path(item))  # 경로가 들어와도 이름 부분만 비교
        found_path: Path | None = None
        # print(item)

        for base in base_dirs:
            # print(base)
            if recursive:
                candidates = base.rglob(item_name)
            else:
                candidates = base.glob(item_name)

            try:
                found_path = next(candidates)  # 첫 번째로 발견된 경로
                break
            except StopIteration:
                continue  # 이 base 폴더에는 없음 → 다음 base 폴더로

        if found_path:
            existing.append(found_path)
        else:
            missing.append(item_name)

    return existing, missing

def check_data_name_only_already_exists_recursive(
    check_lists,
    output_base_folders,
    * ,
    recursive = True
):
    """
    Determine which items in `check_lists` already have results under any of
    the `output_base_folders`.

    Parameters
    ----------
    check_lists : Iterable[str | pathlib.Path]
        File or directory names (relative or absolute) you want to check.
        Example: ["task_001.json", "geometry_42", Path("foo/bar.txt")]

    output_base_folders : Iterable[str | pathlib.Path]
        One or more base directories in which results may be found.
        Example: ["./[intergrated]generated_descriptions", "./outputs"]

    recursive : bool, optional (default=True)
        If True, search recursively (`rglob`) inside each base folder.
        If False, look only at the top level inside each base folder.

    Returns
    -------
    Tuple[list[pathlib.Path], list[str]]
        (existing_results, missing_items)

        * existing_results : list of full `Path` objects where each item was found
        * missing_items    : list of item names (str) that were *not* found
    """
    # 절대 경로로 변환해 두기
    base_dirs = [Path(d).resolve() for d in output_base_folders]

    existing = []
    missing = []
    # print(base_dirs)
    for item in check_lists:
        item_name = str(Path(item).name)  # 경로가 들어와도 이름 부분만 비교
        found_path: Path | None = None
        # print(item)

        for base in base_dirs:
            # print(base)
            if recursive:
                candidates = base.rglob(item_name)
            else:
                candidates = base.glob(item_name)

            try:
                found_path = next(candidates)  # 첫 번째로 발견된 경로
                break
            except StopIteration:
                continue  # 이 base 폴더에는 없음 → 다음 base 폴더로

        if found_path:
            existing.append(found_path)
        else:
            missing.append(item_name)

    return existing, missing