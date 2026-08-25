"""Etapa 3, item 9 — Base estruturada simulando prontuários e registros de exames.

Usa SQLite (sugerido no enunciado) com três tabelas: `pacientes`,
`exames` e `evolucoes`. Todos os pacientes são fictícios e já entram no banco
com identificação pseudonimizada (código de paciente, sem nome real), de forma
coerente com a anonimização feita na Etapa 1.

O assistente consulta este banco por meio de funções tipadas
(`get_patient`, `get_pending_exams`, ...) que são expostas ao LangChain como
tools em `assistant/tools.py`. Optamos por tools com SQL parametrizado em vez
de um SQL Agent de texto livre: em contexto clínico, deixar o LLM escrever SQL
arbitrário sobre a base de prontuários é uma superfície de risco desnecessária
(injeção, leitura de outros pacientes), e o conjunto de consultas que o fluxo
precisa é pequeno e bem definido.
"""

from __future__ import annotations

import random
import sqlite3
from dataclasses import dataclass, field
from datetime import date, timedelta
from pathlib import Path

DEFAULT_DB_PATH = Path("assistant/hospital.db")

SCHEMA = """
CREATE TABLE IF NOT EXISTS pacientes (
    codigo_paciente TEXT PRIMARY KEY,
    idade INTEGER NOT NULL,
    sexo TEXT NOT NULL,
    condicao_principal TEXT NOT NULL,
    especialidade TEXT NOT NULL,
    alergias TEXT,
    comorbidades TEXT,
    ultima_atualizacao TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS exames (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    codigo_paciente TEXT NOT NULL,
    nome_exame TEXT NOT NULL,
    status TEXT NOT NULL CHECK (status IN ('pendente', 'concluido')),
    resultado TEXT,
    data_solicitacao TEXT NOT NULL,
    data_resultado TEXT,
    FOREIGN KEY (codigo_paciente) REFERENCES pacientes (codigo_paciente)
);

CREATE TABLE IF NOT EXISTS evolucoes (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    codigo_paciente TEXT NOT NULL,
    data TEXT NOT NULL,
    descricao TEXT NOT NULL,
    sinais_vitais TEXT,
    FOREIGN KEY (codigo_paciente) REFERENCES pacientes (codigo_paciente)
);

CREATE INDEX IF NOT EXISTS idx_exames_paciente ON exames (codigo_paciente);
CREATE INDEX IF NOT EXISTS idx_evolucoes_paciente ON evolucoes (codigo_paciente);
"""


@dataclass
class Patient:
    codigo_paciente: str
    idade: int
    sexo: str
    condicao_principal: str
    especialidade: str
    alergias: str
    comorbidades: str
    ultima_atualizacao: str

    def to_context_string(self) -> str:
        return (
            f"Paciente {self.codigo_paciente} | {self.idade} anos | sexo {self.sexo}\n"
            f"Condição principal: {self.condicao_principal}\n"
            f"Especialidade: {self.especialidade}\n"
            f"Comorbidades: {self.comorbidades or 'nenhuma registrada'}\n"
            f"Alergias: {self.alergias or 'nenhuma registrada'}\n"
            f"Última atualização do prontuário: {self.ultima_atualizacao}"
        )


@dataclass
class Exam:
    nome_exame: str
    status: str
    resultado: str | None
    data_solicitacao: str
    data_resultado: str | None


@dataclass
class Evolution:
    data: str
    descricao: str
    sinais_vitais: str | None


@dataclass
class PatientRecord:
    """Visão consolidada do paciente usada como contexto pelo assistente."""

    patient: Patient
    exames: list[Exam] = field(default_factory=list)
    evolucoes: list[Evolution] = field(default_factory=list)

    @property
    def exames_pendentes(self) -> list[Exam]:
        return [e for e in self.exames if e.status == "pendente"]

    def to_context_string(self) -> str:
        parts = [self.patient.to_context_string(), ""]

        if self.exames:
            parts.append("Exames registrados:")
            for exam in self.exames:
                if exam.status == "concluido":
                    parts.append(
                        f"- {exam.nome_exame}: CONCLUÍDO em {exam.data_resultado} "
                        f"— resultado: {exam.resultado}"
                    )
                else:
                    parts.append(
                        f"- {exam.nome_exame}: PENDENTE (solicitado em {exam.data_solicitacao})"
                    )
        else:
            parts.append("Exames registrados: nenhum.")

        if self.evolucoes:
            parts.append("")
            parts.append("Evoluções recentes:")
            for ev in self.evolucoes[:3]:
                vitals = f" | sinais vitais: {ev.sinais_vitais}" if ev.sinais_vitais else ""
                parts.append(f"- {ev.data}: {ev.descricao}{vitals}")

        return "\n".join(parts)


def connect(db_path: str | Path = DEFAULT_DB_PATH) -> sqlite3.Connection:
    db_path = Path(db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    return conn


def init_schema(conn: sqlite3.Connection) -> None:
    conn.executescript(SCHEMA)
    conn.commit()


def get_patient(conn: sqlite3.Connection, codigo_paciente: str) -> Patient | None:
    row = conn.execute(
        "SELECT * FROM pacientes WHERE codigo_paciente = ?", (codigo_paciente,)
    ).fetchone()
    return Patient(**dict(row)) if row else None


def get_exams(conn: sqlite3.Connection, codigo_paciente: str) -> list[Exam]:
    rows = conn.execute(
        "SELECT nome_exame, status, resultado, data_solicitacao, data_resultado "
        "FROM exames WHERE codigo_paciente = ? ORDER BY data_solicitacao DESC",
        (codigo_paciente,),
    ).fetchall()
    return [Exam(**dict(r)) for r in rows]


def get_pending_exams(conn: sqlite3.Connection, codigo_paciente: str) -> list[Exam]:
    rows = conn.execute(
        "SELECT nome_exame, status, resultado, data_solicitacao, data_resultado "
        "FROM exames WHERE codigo_paciente = ? AND status = 'pendente' "
        "ORDER BY data_solicitacao DESC",
        (codigo_paciente,),
    ).fetchall()
    return [Exam(**dict(r)) for r in rows]


def get_evolutions(conn: sqlite3.Connection, codigo_paciente: str, limit: int = 5) -> list[Evolution]:
    rows = conn.execute(
        "SELECT data, descricao, sinais_vitais FROM evolucoes "
        "WHERE codigo_paciente = ? ORDER BY data DESC LIMIT ?",
        (codigo_paciente, limit),
    ).fetchall()
    return [Evolution(**dict(r)) for r in rows]


def get_patient_record(conn: sqlite3.Connection, codigo_paciente: str) -> PatientRecord | None:
    """Contexto consolidado do paciente — é isso que a chain injeta no prompt
    para atender ao item 5 do checklist ("contextualização com dados
    atualizados do paciente").
    """
    patient = get_patient(conn, codigo_paciente)
    if patient is None:
        return None
    return PatientRecord(
        patient=patient,
        exames=get_exams(conn, codigo_paciente),
        evolucoes=get_evolutions(conn, codigo_paciente),
    )


def list_patients(conn: sqlite3.Connection) -> list[Patient]:
    rows = conn.execute("SELECT * FROM pacientes ORDER BY codigo_paciente").fetchall()
    return [Patient(**dict(r)) for r in rows]


# --------------------------------------------------------------------------
# Seed de dados sintéticos
# --------------------------------------------------------------------------

SEED_PATIENTS = [
    {
        "codigo_paciente": "PAC-0001",
        "idade": 62,
        "sexo": "M",
        "condicao_principal": "hipertensão arterial sistêmica",
        "especialidade": "cardiologia",
        "alergias": "",
        "comorbidades": "dislipidemia",
        "exames": [
            ("Creatinina", "concluido", "1,1 mg/dL (normal)"),
            ("Perfil lipídico", "concluido", "LDL 165 mg/dL (elevado)"),
            ("ECG de repouso", "concluido", "ritmo sinusal, sem alterações agudas"),
        ],
        "evolucoes": [
            ("Consulta de acompanhamento. PA 148/92 mmHg em duas aferições.", "PA 148/92 mmHg, FC 78 bpm"),
            ("Paciente refere adesão parcial à dieta hipossódica.", "PA 152/94 mmHg, FC 80 bpm"),
        ],
    },
    {
        "codigo_paciente": "PAC-0002",
        "idade": 54,
        "sexo": "F",
        "condicao_principal": "diabetes mellitus tipo 2",
        "especialidade": "endocrinologia",
        "alergias": "sulfa",
        "comorbidades": "obesidade grau I",
        # Exames pendentes: dispara o desvio condicional do grafo (Etapa 4).
        "exames": [
            ("Hemoglobina glicada (HbA1c)", "pendente", None),
            ("Microalbuminúria", "pendente", None),
            ("Glicemia de jejum", "concluido", "142 mg/dL (elevada)"),
        ],
        "evolucoes": [
            ("Retorno para ajuste terapêutico; aguardando resultado de HbA1c.", "PA 128/82 mmHg, IMC 32"),
        ],
    },
    {
        "codigo_paciente": "PAC-0003",
        "idade": 71,
        "sexo": "M",
        "condicao_principal": "pneumonia adquirida na comunidade",
        "especialidade": "pneumologia",
        "alergias": "penicilina",
        "comorbidades": "DPOC",
        # Sinais de gravidade: dispara o nó de alerta (Etapa 4).
        "exames": [
            ("Radiografia de tórax", "concluido", "consolidação em base direita"),
            ("Hemograma", "concluido", "leucocitose 18.000/mm³"),
            ("Lactato sérico", "concluido", "4,2 mmol/L (elevado)"),
        ],
        "evolucoes": [
            (
                "Paciente confuso, taquipneico, hipotenso. Suspeita de sepse de foco pulmonar.",
                "PA 92/58 mmHg, FR 28 irpm, SpO2 89%, Tax 38,9°C",
            ),
        ],
    },
    {
        "codigo_paciente": "PAC-0004",
        "idade": 29,
        "sexo": "F",
        "condicao_principal": "gestação de baixo risco (26 semanas)",
        "especialidade": "ginecologia",
        "alergias": "",
        "comorbidades": "",
        "exames": [
            ("Teste oral de tolerância à glicose (TOTG)", "pendente", None),
            ("Hemograma", "concluido", "sem alterações"),
            ("Sorologias", "concluido", "não reagentes"),
        ],
        "evolucoes": [
            ("Pré-natal de rotina, sem intercorrências. Movimentos fetais presentes.", "PA 110/70 mmHg"),
        ],
    },
    {
        "codigo_paciente": "PAC-0005",
        "idade": 45,
        "sexo": "M",
        "condicao_principal": "lombalgia aguda inespecífica",
        "especialidade": "ortopedia",
        "alergias": "",
        "comorbidades": "",
        "exames": [],
        "evolucoes": [
            ("Dor lombar há 5 dias após esforço, sem sinais de alarme (red flags).", "PA 124/78 mmHg"),
        ],
    },
]


def seed_database(conn: sqlite3.Connection, seed: int = 42) -> dict:
    """Popula o banco com os pacientes sintéticos. Idempotente: limpa as
    tabelas antes de inserir, para que rodar o seed duas vezes não duplique
    registros.
    """
    rng = random.Random(seed)
    today = date(2026, 8, 25)

    conn.executescript("DELETE FROM evolucoes; DELETE FROM exames; DELETE FROM pacientes;")

    for entry in SEED_PATIENTS:
        conn.execute(
            "INSERT INTO pacientes (codigo_paciente, idade, sexo, condicao_principal, "
            "especialidade, alergias, comorbidades, ultima_atualizacao) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (
                entry["codigo_paciente"],
                entry["idade"],
                entry["sexo"],
                entry["condicao_principal"],
                entry["especialidade"],
                entry["alergias"],
                entry["comorbidades"],
                today.isoformat(),
            ),
        )

        for nome, status, resultado in entry["exames"]:
            solicitacao = today - timedelta(days=rng.randint(2, 20))
            resultado_data = (
                (solicitacao + timedelta(days=rng.randint(1, 3))).isoformat()
                if status == "concluido"
                else None
            )
            conn.execute(
                "INSERT INTO exames (codigo_paciente, nome_exame, status, resultado, "
                "data_solicitacao, data_resultado) VALUES (?, ?, ?, ?, ?, ?)",
                (
                    entry["codigo_paciente"],
                    nome,
                    status,
                    resultado,
                    solicitacao.isoformat(),
                    resultado_data,
                ),
            )

        for offset, (descricao, vitais) in enumerate(entry["evolucoes"]):
            conn.execute(
                "INSERT INTO evolucoes (codigo_paciente, data, descricao, sinais_vitais) "
                "VALUES (?, ?, ?, ?)",
                (
                    entry["codigo_paciente"],
                    (today - timedelta(days=offset * 7)).isoformat(),
                    descricao,
                    vitais,
                ),
            )

    conn.commit()

    counts = {
        "pacientes": conn.execute("SELECT COUNT(*) FROM pacientes").fetchone()[0],
        "exames": conn.execute("SELECT COUNT(*) FROM exames").fetchone()[0],
        "evolucoes": conn.execute("SELECT COUNT(*) FROM evolucoes").fetchone()[0],
    }
    return counts


def build_database(db_path: str | Path = DEFAULT_DB_PATH, seed: int = 42) -> dict:
    conn = connect(db_path)
    try:
        init_schema(conn)
        return seed_database(conn, seed=seed)
    finally:
        conn.close()


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db-path", type=Path, default=DEFAULT_DB_PATH)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    counts = build_database(args.db_path, seed=args.seed)
    print(f"Base estruturada criada em {args.db_path}: {counts}")


if __name__ == "__main__":
    main()
