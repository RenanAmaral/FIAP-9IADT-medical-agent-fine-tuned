"""Etapa 1, item 2 — Preprocessing: limpeza de texto e deduplicação.

Remove ruído (marcadores de OCR/copiar-colar), normaliza encoding e
espaçamento, e deduplica registros por hash do conteúdo textual.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import unicodedata
from pathlib import Path

NOISE_PATTERNS = [
    re.compile(r"<<[^>]*>>"),
    re.compile(r"\*\*\*+"),
    re.compile(r"[​‌‍﻿]"),
]


def normalize_encoding(text: str) -> str:
    return unicodedata.normalize("NFC", text)


def strip_noise(text: str) -> str:
    for pattern in NOISE_PATTERNS:
        text = pattern.sub(" ", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]+\n", "\n", text)
    return text.strip()


def clean_text(text: str) -> str:
    return strip_noise(normalize_encoding(text))


def _text_field(record: dict) -> str:
    for field in ("raw_text", "instruction", "output"):
        if field in record:
            return record[field]
    return json.dumps(record, ensure_ascii=False)


def _content_hash(record: dict) -> str:
    return hashlib.sha256(_text_field(record).strip().lower().encode("utf-8")).hexdigest()


def clean_records(records: list[dict]) -> tuple[list[dict], dict]:
    seen_hashes: set[str] = set()
    cleaned: list[dict] = []
    duplicates = 0

    for record in records:
        new_record = dict(record)
        for field in ("raw_text", "instruction", "input", "output"):
            if field in new_record and isinstance(new_record[field], str):
                new_record[field] = clean_text(new_record[field])

        digest = _content_hash(new_record)
        if digest in seen_hashes:
            duplicates += 1
            continue
        seen_hashes.add(digest)
        cleaned.append(new_record)

    stats = {
        "input_count": len(records),
        "output_count": len(cleaned),
        "duplicates_removed": duplicates,
    }
    return cleaned, stats


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def write_jsonl(records: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for record in records:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--in-file", type=Path, required=True)
    parser.add_argument("--out-file", type=Path, required=True)
    args = parser.parse_args()

    records = read_jsonl(args.in_file)
    cleaned, stats = clean_records(records)
    write_jsonl(cleaned, args.out_file)

    print(f"Limpeza concluída: {stats}")


if __name__ == "__main__":
    main()
