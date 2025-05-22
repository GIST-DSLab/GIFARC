from __future__ import annotations

import csv
import json
from pathlib import Path
from datetime import datetime, timezone
from typing import Dict, List, Any


FIELDNAMES = [
    "id",
    "step-name",
    "prev-step-id",
    "gif-id",
    "gen-model",
    "result_code",
    "result_path",
    "error_message",
    "createAt",
    "token-usage",
    "visualization_path",
]


def _normalize_record(rec: Dict[str, Any]) -> Dict[str, Any]:
    """사전 키를 FIELDNAMES 에 맞춰 정규화 & 직렬화"""
    row: Dict[str, Any] = {f: "" for f in FIELDNAMES}

    # 필수/주요 키 매핑
    row["id"] = rec.get("id") or rec.get("step_uuid", "")
    row["step-name"] = rec.get("step_name") or rec.get("step-name", "")
    row["prev-step-id"] = rec.get("prev_step_id") or rec.get("prev-step-id", "")
    row["gif-id"] = rec.get("gif_id") or rec.get("gif-id", "")
    row["gen-model"] = rec.get("gen_model") or rec.get("gen-model", "")
    row["result_code"] = rec.get("result_code", "")
    row["result_path"] = rec.get("result_path", "")

    # error_message: list → JSON 직렬화, str 은 그대로
    err = rec.get("error_message", "")
    if isinstance(err, list):
        row["error_message"] = json.dumps(err, ensure_ascii=False)
    else:
        row["error_message"] = str(err)

    # createAt: datetime→ISO, str→그대로, 없다면 현재 UTC
    ts = rec.get("createAt") or rec.get("create_at") or rec.get("created_at")
    if isinstance(ts, datetime):
        row["createAt"] = ts.astimezone(timezone.utc).isoformat(timespec="seconds")
    elif ts:
        row["createAt"] = str(ts)
    else:
        row["createAt"] = datetime.now(timezone.utc).isoformat(timespec="seconds")

    # token‑usage 는 어떤 타입이든 문자열로 저장
    row["token-usage"] = json.dumps(rec.get("token_usage", ""), ensure_ascii=False)

    row["visualization_path"] = rec.get("visualization_path", "")

    return row


def generate_metadata_csv_of_step_descriptions(
    records: List[Dict[str, Any]],
    output_csv: str | Path = "step_descriptions_metadata.csv",
    encoding: str = "utf-8",
) -> Path:
    """
    Args
    ----
    records     : 메타데이터 dict 리스트
    output_csv  : 저장할 CSV 경로 (없으면 동일 폴더에 생성)
    encoding    : CSV 인코딩 (기본 utf-8)

    Returns
    -------
    Path 객체 (작성된 CSV 경로)
    """
    output_csv = Path(output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    file_exists = output_csv.exists()

    with output_csv.open("a", newline="", encoding=encoding) as fp:
        writer = csv.DictWriter(fp, fieldnames=FIELDNAMES)
        if not file_exists:
            writer.writeheader()
        for rec in records:
            writer.writerow(_normalize_record(rec))

    return output_csv