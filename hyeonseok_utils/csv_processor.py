from pathlib import Path
import pandas as pd

def list_file_names_without_ext(directory):
    """
    주어진 디렉토리 안의 모든 파일에 대해
    확장자를 제거한 파일명 리스트를 반환합니다.
    """
    p = Path(directory)
    return [f.stem for f in p.iterdir() if f.is_file()]

def load_names_from_csv(csv_path, column_name):
    """
    CSV 파일에서 지정한 컬럼(column_name)의 모든 값을
    문자열 리스트로 반환합니다.
    """
    df = pd.read_csv(csv_path, usecols=[column_name])
    # NaN 제외, 문자열로 변환
    return df[column_name].dropna().astype(str).tolist()

if __name__ == "__main__":
    # 사용 예시
    dir_path    = "path/to/your/directory"   # 파일 이름을 뽑을 폴더 경로
    csv_path    = "path/to/your/file.csv"    # 비교할 CSV 파일 경로
    column_name = "name"                     # CSV에서 이름이 들어있는 컬럼명

    # 1) 디렉토리에서 파일명(확장자 제외) 뽑기
    file_list = set(list_file_names_without_ext(dir_path))

    # 2) CSV에서 이름 리스트 불러오기
    csv_list = set(load_names_from_csv(csv_path, column_name))

    # 3) 교집합 & 차집합 계산
    intersection   = file_list & csv_list  # 두 리스트 모두에 있는 이름
    only_in_files  = file_list - csv_list  # 파일에는 있지만 CSV에는 없는 이름
    only_in_csv    = csv_list - file_list  # CSV에는 있지만 파일에는 없는 이름

    # 4) 결과 출력
    print("🔹 교집합 (Intersection):", intersection)
    print("🔹 파일 전용 (In files only):", only_in_files)
    print("🔹 CSV 전용  (In CSV only):", only_in_csv)
