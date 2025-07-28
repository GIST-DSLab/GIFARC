#!/usr/bin/env python3
import pandas as pd
from pandas.errors import EmptyDataError

def find_value_in_column(csv_path: str, column: str, value) -> tuple[bool, pd.DataFrame]:
    """
    csv_path: 검사할 CSV 파일 경로
    column:   검사할 열 이름
    value:    찾고자 하는 값
    반환:
      - is_unique: 해당 열에서 value가 정확히 1번만 등장하면 True
      - matches:   value가 등장하는 모든 행을 담은 DataFrame (없으면 빈 DataFrame)
    """
    try:
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        raise FileNotFoundError(f"파일을 찾을 수 없습니다: {csv_path}")
    except EmptyDataError:
        raise EmptyDataError(f"파일에 읽을 데이터가 없습니다: {csv_path}")

    if column not in df.columns:
        raise KeyError(f"지정한 열이 없습니다: {column}")

    matches = df[df[column] == value]
    clean = matches.where(pd.notna(matches), None)
    records: list[dict] = clean.to_dict(orient='records')

    is_unique = len(matches) == 1
    return { 'unique': is_unique,  "records": records }


def find_value_in_region(csv_path: str,
                         value,
                         row_indices: slice | list[int] = None,
                         cols: list[str] = None
                        ) -> tuple[bool, pd.DataFrame]:
    """
    csv_path: 검사할 CSV 파일 경로
    value:    찾고자 하는 값
    row_indices: 검사할 행 범위 (slice or list of ints), None이면 전체 행
    cols:     검사할 열 리스트, None이면 전체 열
    반환:
      - is_unique: 지정 영역에서 value가 정확히 1번만 등장하면 True
      - matches:   value가 등장하는 모든 행을 담은 DataFrame (없으면 빈 DataFrame)
    """
    try:
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        raise FileNotFoundError(f"파일을 찾을 수 없습니다: {csv_path}")
    except EmptyDataError:
        raise EmptyDataError(f"파일에 읽을 데이터가 없습니다: {csv_path}")

    sub = df
    if cols is not None:
        missing = set(cols) - set(df.columns)
        if missing:
            raise KeyError(f"다음 컬럼이 없습니다: {missing}")
        sub = sub[cols]

    if row_indices is not None:
        sub = sub.loc[row_indices]

    # Boolean mask over the sub-DataFrame
    mask = (sub == value).any(axis=1)
    matches = sub[mask]
    is_unique = len(matches) == 1
    return is_unique, matches

if __name__ == "__main__":
    path = "data.csv"
    val  = 12345

    # 예시1: user_id 열에서 값 찾기
    unique, rows = find_value_in_column(path, column="user_id", value=val)
    print(f"[열 검사] 유일 여부: {unique}")
    if not rows.empty:
        print("매칭된 행:")
        print(rows)

    # 예시2: 100~199번째 행, user_id와 order_id 열 영역에서 값 찾기
    unique_reg, rows_reg = find_value_in_region(
        csv_path=path,
        value=val,
        row_indices=slice(100, 200),
        cols=["user_id", "order_id"]
    )
    print(f"[영역 검사] 유일 여부: {unique_reg}")
    if not rows_reg.empty:
        print("매칭된 행:")
        print(rows_reg)