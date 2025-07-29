#!/usr/bin/env python3
import pandas as pd
from pandas.errors import EmptyDataError

def find_value_in_column(csv_path: str, column: str, value) -> tuple[bool, pd.DataFrame]:
    """
    csv_path: Path to the CSV file to check.
    column:   The name of the column to search.
    value:    The value to find.
    Returns:
      - is_unique: True if the value appears exactly once in the specified column.
      - matches:   DataFrame containing all rows where the value appears (empty if none).
    """

    try:
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        raise FileNotFoundError(f"Cannot found the file: {csv_path}")
    except EmptyDataError:
        raise EmptyDataError(f"Cannot found the data in file: {csv_path}")

    if column not in df.columns:
        raise KeyError(f"Cannot found the column: {column}")

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
    csv_path: Path to the CSV file to check.
    value:    The value to search for.
    row_indices: Row range to check (slice or list of ints); None means all rows.
    cols:     List of columns to check; None means all columns.

    Returns:
      - is_unique: True if the value appears exactly once in the specified region.
      - matches:   DataFrame containing all rows where the value appears (empty if none).
    """
    try:
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        raise FileNotFoundError(f"Cannot found the file: {csv_path}")
    except EmptyDataError:
        raise EmptyDataError(f"Cannot found the data in file: {csv_path}")

    sub = df
    if cols is not None:
        missing = set(cols) - set(df.columns)
        if missing:
            raise KeyError(f"No Next Column: {missing}")
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

    # Example 1: Find the value in the 'user_id' column
    unique, rows = find_value_in_column(path, column="user_id", value=val)
    print(f"[Check Column] Uniqueness: {unique}")
    if not rows.empty:
        print("Matched rows:")
        print(rows)

    # Example 2: Find the value in rows 100 to 199 and columns 'user_id' and 'order_id'
    unique_reg, rows_reg = find_value_in_region(
        csv_path=path,
        value=val,
        row_indices=slice(100, 200),
        cols=["user_id", "order_id"]
    )
    print(f"[Check Area] Uniqueness: {unique_reg}")
    if not rows_reg.empty:
        print("Matched rows:")
        print(rows_reg)