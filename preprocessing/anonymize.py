"""Etapa 1, item 3 — Anonimização de dados identificáveis (PII).

Estratégia principal: expressões regulares para padrões estruturados
brasileiros (CPF, RG, telefone, CEP, datas, número de prontuário) mais
heurísticas de campo (ex.: "Paciente: <nome>") para nomes próprios.

Estratégia opcional: se `spacy` e um modelo de NER em português estiverem
instalados localmente, ele é usado como camada adicional para capturar
nomes de pessoa (PER) que escapem das heurísticas de regex — ver
`_spacy_person_names`. Isso é totalmente opcional e o pipeline funciona
sem nenhuma dependência de rede (não faz download de modelo em tempo de
execução), o que é necessário neste ambiente sem acesso à internet.
"""

from __future__ import annotations

import argparse
import json
import logging
import re
from pathlib import Path

logger = logging.getLogger(__name__)

CPF_RE = re.compile(r"\b\d{3}\.\d{3}\.\d{3}-\d{2}\b")
RG_SHAPE_RE = re.compile(r"\b\d{1,2}\.\d{3}\.\d{3}-?[0-9Xx]?\b")
PHONE_RE = re.compile(r"\(?\d{2}\)?\s?9?\d{4}-?\d{4}\b")
CEP_RE = re.compile(r"\b\d{5}-\d{3}\b")
DATE_RE = re.compile(r"\b\d{2}/\d{2}/\d{4}\b")
PRONTUARIO_RE = re.compile(r"\bPRT-\d{6}\b")

# Regexes de forma: cobrem PII em texto livre, onde não há um rótulo de
# campo explícito antes do valor.
REGEX_MASKS = [
    (CPF_RE, "[CPF_REDACTED]"),
    (RG_SHAPE_RE, "[RG_REDACTED]"),
    (CEP_RE, "[CEP_REDACTED]"),
    (PHONE_RE, "[TELEFONE_REDACTED]"),
    (DATE_RE, "[DATA_REDACTED]"),
    (PRONTUARIO_RE, "[PRONTUARIO_REDACTED]"),
]

# Regexes de campo rotulado ("Rótulo: valor"): complementam as regexes de
# forma, cobrindo formatos de RG/telefone que variam (sem pontuação, com
# prefixo de país, etc.) mas que em registros hospitalares quase sempre
# aparecem atrás de um rótulo estruturado.
LABELED_FIELD_MASKS = [
    (re.compile(r"(Paciente:\s*)([^\n|]+)"), "[NOME_REDACTED]"),
    (re.compile(r"(Endereço:\s*)([^\n]+)"), "[ENDERECO_REDACTED]"),
    (re.compile(r"(RG:\s*)([A-Za-z0-9]+)"), "[RG_REDACTED]"),
    (re.compile(r"(CPF:\s*)([A-Za-z0-9.\-]+)"), "[CPF_REDACTED]"),
    (re.compile(r"(Telefone:\s*)([^\n|]+)"), "[TELEFONE_REDACTED]"),
    (re.compile(r"(Data de nascimento:\s*)([^\n|]+)"), "[DATA_REDACTED]"),
    (re.compile(r"(Prontuário\s*n[ºo°]?\s*)([A-Za-z0-9\-]+)"), "[PRONTUARIO_REDACTED]"),
]


def _spacy_person_names(text: str) -> list[str]:
    """Camada opcional de NER via spaCy. Retorna lista vazia se a biblioteca
    ou o modelo de português não estiverem disponíveis localmente — nunca
    tenta baixar nada da internet.
    """
    try:
        import spacy

        nlp = spacy.load("pt_core_news_sm")
    except Exception:
        return []

    doc = nlp(text)
    return [ent.text for ent in doc.ents if ent.label_ == "PER"]


def anonymize_text(text: str, use_ner: bool = False) -> tuple[str, list[str]]:
    found: list[str] = []
    masked = text

    def _mask_field(match: re.Match, label: str) -> str:
        found.append(label)
        return f"{match.group(1)}{label}"

    # Campos rotulados primeiro: são o sinal mais confiável em registros
    # semiestruturados e evitam falsos negativos de formatos que a regex de
    # forma não cobre (ex.: RG sem pontuação).
    for pattern, label in LABELED_FIELD_MASKS:
        masked = pattern.sub(lambda m, label=label: _mask_field(m, label), masked)

    for pattern, replacement in REGEX_MASKS:
        matches = pattern.findall(masked)
        if matches:
            found.extend([replacement] * len(matches))
        masked = pattern.sub(replacement, masked)

    if use_ner:
        for name in _spacy_person_names(masked):
            if name and name in masked:
                masked = masked.replace(name, "[NOME_REDACTED]")
                found.append("[NOME_REDACTED:ner]")

    return masked, found


def anonymize_record(record: dict, use_ner: bool = False) -> dict:
    new_record = dict(record)
    entities_found: list[str] = []

    for field in ("raw_text", "instruction", "output"):
        if field in new_record and isinstance(new_record[field], str):
            masked, found = anonymize_text(new_record[field], use_ner=use_ner)
            new_record[field] = masked
            entities_found.extend(found)

    new_record.pop("_pii_ground_truth", None)
    new_record["_anonymization"] = {
        "entities_masked": len(entities_found),
        "labels": sorted(set(entities_found)),
    }
    return new_record


def validate_against_ground_truth(records: list[dict], anonymized: list[dict]) -> dict:
    """Quando o dataset sintético traz `_pii_ground_truth` (gerado por
    generate_synthetic_data.py), mede se cada valor de PII conhecido deixou
    de aparecer literalmente no texto anonimizado — uma checagem de
    recall da anonimização para o relatório técnico.
    """
    leaked = 0
    checked = 0
    for original, clean in zip(records, anonymized):
        gt = original.get("_pii_ground_truth")
        if not gt:
            continue
        text = clean.get("raw_text", "")
        for value in gt.values():
            checked += 1
            if value and value in text:
                leaked += 1

    recall = None if checked == 0 else round(1 - leaked / checked, 4)
    return {"pii_values_checked": checked, "pii_values_leaked": leaked, "recall": recall}


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
    parser.add_argument("--use-ner", action="store_true")
    parser.add_argument("--validate", action="store_true")
    args = parser.parse_args()

    records = read_jsonl(args.in_file)
    anonymized = [anonymize_record(r, use_ner=args.use_ner) for r in records]
    write_jsonl(anonymized, args.out_file)

    total_entities = sum(r["_anonymization"]["entities_masked"] for r in anonymized)
    print(f"Anonimização concluída: {len(anonymized)} registros, {total_entities} entidades mascaradas")

    if args.validate:
        report = validate_against_ground_truth(records, anonymized)
        print(f"Validação contra ground truth sintético: {report}")


if __name__ == "__main__":
    main()
