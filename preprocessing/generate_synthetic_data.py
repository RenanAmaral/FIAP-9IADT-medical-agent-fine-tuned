"""Gera dados sintéticos que simulam o ambiente hospitalar interno (Etapa 1,
item 1 do passo a passo): perguntas frequentes de médicos, protocolos e
registros de prontuário/laudo com dados de identificação, usados depois para
demonstrar anonimização e curadoria.

Uso:
    python -m preprocessing.generate_synthetic_data --out-dir data/raw --seed 42
"""

from __future__ import annotations

import argparse
import json
import random
from datetime import date, timedelta
from pathlib import Path

from faker import Faker

from preprocessing.protocols_bank import PROTOCOLS

QUESTION_TEMPLATES = [
    "Paciente com {condicao}, qual a conduta recomendada segundo o protocolo interno?",
    "Quais exames devo solicitar antes de definir a conduta para um caso de {condicao}?",
    "Existe algum critério de encaminhamento urgente para {condicao} no nosso protocolo?",
    "Como o protocolo interno recomenda estratificar o risco em um caso de {condicao}?",
    "Quais são os sinais de alarme que exigem acionamento imediato da equipe em {condicao}?",
    "Segundo nosso protocolo, quando devo considerar internação em um quadro de {condicao}?",
]

# Ruído típico de fontes internas (OCR, copiar/colar de sistemas legados) usado
# deliberadamente para o preprocessing.clean remover na Etapa 1.
NOISE_SNIPPETS = ["  ", "\t", "***", "<<confidencial>>", "​", "  \n\n  "]


#: Data de referência do hospital simulado. Fixa de propósito: as idades dos
#: pacientes são calculadas a partir daqui, e não de "hoje".
REFERENCE_DATE = date(2026, 8, 25)

MIN_AGE_YEARS = 1
MAX_AGE_YEARS = 95


def _random_birth_date(rng: random.Random) -> date:
    """Data de nascimento determinística, ancorada em `REFERENCE_DATE`.

    Não usamos `Faker.date_of_birth`: ele calcula a idade a partir da data
    atual do sistema, então o mesmo seed produz datas diferentes conforme os
    dias passam — o dataset deixava de ser reprodutível de um dia para o
    outro, apesar do seed fixo.
    """
    dias = rng.randint(MIN_AGE_YEARS * 365, MAX_AGE_YEARS * 365)
    return REFERENCE_DATE - timedelta(days=dias)


def _inject_noise(text: str, rng: random.Random) -> str:
    if rng.random() < 0.4:
        snippet = rng.choice(NOISE_SNIPPETS)
        words = text.split(" ")
        pos = rng.randrange(len(words))
        words.insert(pos, snippet)
        text = " ".join(words)
    return text


def build_qa_pairs(rng: random.Random, n_per_protocol: int = 12) -> list[dict]:
    """Pares instrução/resposta 'limpos' baseados nos protocolos internos.

    Representam o conhecimento institucional (sem dados de paciente
    identificáveis) que forma o núcleo do dataset de fine-tuning.
    """
    pairs = []
    for protocol in PROTOCOLS:
        for _ in range(n_per_protocol):
            condicao = rng.choice(protocol["condicoes"])
            question = rng.choice(QUESTION_TEMPLATES).format(condicao=condicao)
            answer = (
                f"Com base no protocolo interno {protocol['id']} "
                f"({protocol['titulo']}):\n{protocol['texto']}\n\n"
                "Lembrete: esta é uma sugestão de apoio à decisão baseada em "
                "protocolo interno; a conduta final e qualquer prescrição "
                "dependem de validação do médico responsável."
            )
            pairs.append(
                {
                    "instruction": _inject_noise(question, rng),
                    "input": "",
                    "output": answer,
                    "especialidade": protocol["especialidade"],
                    "fonte": protocol["id"],
                    "tipo": "protocol_qa",
                }
            )
    rng.shuffle(pairs)
    return pairs


def build_hospital_records(fake: Faker, rng: random.Random, n_records: int = 80) -> list[dict]:
    """Registros simulados de prontuário/laudo COM dados de identificação,
    para demonstrar o pipeline de anonimização (Etapa 1, itens 2 e 3).
    """
    records = []
    for _ in range(n_records):
        protocol = rng.choice(PROTOCOLS)
        nome = fake.name()
        cpf = fake.cpf()
        rg = fake.rg()
        telefone = fake.phone_number()
        endereco = fake.address().replace("\n", ", ")
        nascimento = _random_birth_date(rng).strftime("%d/%m/%Y")
        prontuario = f"PRT-{rng.randint(100000, 999999)}"
        condicao = rng.choice(protocol["condicoes"])

        texto = (
            f"Paciente: {nome}\n"
            f"CPF: {cpf} | RG: {rg}\n"
            f"Data de nascimento: {nascimento}\n"
            f"Telefone: {telefone}\n"
            f"Endereço: {endereco}\n"
            f"Prontuário nº {prontuario}\n\n"
            f"Motivo da consulta: acompanhamento de {condicao}.\n"
            f"Especialidade: {protocol['especialidade']}.\n"
            f"Conduta registrada conforme {protocol['id']}."
        )
        records.append(
            {
                "raw_text": _inject_noise(texto, rng),
                "especialidade": protocol["especialidade"],
                "fonte": protocol["id"],
                "tipo": "registro_hospitalar",
                "_pii_ground_truth": {
                    "nome": nome,
                    "cpf": cpf,
                    "rg": rg,
                    "telefone": telefone,
                    "endereco": endereco,
                    "nascimento": nascimento,
                    "prontuario": prontuario,
                },
            }
        )
    return records


def write_jsonl(records: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for record in records:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")


def write_protocol_docs(out_dir: Path) -> None:
    protocols_dir = out_dir.parent / "protocols"
    protocols_dir.mkdir(parents=True, exist_ok=True)
    for protocol in PROTOCOLS:
        doc_path = protocols_dir / f"{protocol['id']}.md"
        content = f"# {protocol['titulo']}\n\n{protocol['texto']}\n"
        doc_path.write_text(content, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-per-protocol", type=int, default=12)
    parser.add_argument("--n-records", type=int, default=80)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    fake = Faker("pt_BR")
    Faker.seed(args.seed)

    qa_pairs = build_qa_pairs(rng, n_per_protocol=args.n_per_protocol)
    hospital_records = build_hospital_records(fake, rng, n_records=args.n_records)

    write_jsonl(qa_pairs, args.out_dir / "qa_pairs.jsonl")
    write_jsonl(hospital_records, args.out_dir / "registros_hospitalares.jsonl")
    write_protocol_docs(args.out_dir)

    print(f"Gerados {len(qa_pairs)} pares instrução/resposta em {args.out_dir / 'qa_pairs.jsonl'}")
    print(
        f"Gerados {len(hospital_records)} registros hospitalares sintéticos em "
        f"{args.out_dir / 'registros_hospitalares.jsonl'}"
    )
    print(f"Protocolos exportados como markdown em {args.out_dir.parent / 'protocols'}")


if __name__ == "__main__":
    main()
