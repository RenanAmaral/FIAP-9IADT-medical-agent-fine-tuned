"""Etapa 5, item 13 — Logging detalhado para auditoria.

Registra, para cada interação, o conjunto exigido pelo enunciado:
timestamp, identificador da sessão, pergunta recebida, contexto recuperado,
resposta gerada, nós do grafo executados e eventuais bloqueios de segurança.

Formato: JSON Lines em `logs/audit.jsonl` (uma interação por linha), via o
módulo `logging` da biblioteca padrão com um formatter customizado. JSONL
torna o log diretamente consultável (`jq`, pandas) para auditoria, sem
parsing de texto livre.

Rastreabilidade opcional com LangSmith: se as variáveis de ambiente
`LANGCHAIN_TRACING_V2=true` e `LANGCHAIN_API_KEY` estiverem definidas, o
LangChain envia os traces automaticamente — este módulo não precisa de
nenhuma configuração adicional para isso, e o log local continua sendo a
fonte de auditoria própria do hospital.
"""

from __future__ import annotations

import json
import logging
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

DEFAULT_LOG_PATH = Path("logs/audit.jsonl")

_LOGGER_NAME = "medical_assistant.audit"


class JsonLinesFormatter(logging.Formatter):
    """Serializa o payload estruturado anexado ao LogRecord."""

    def format(self, record: logging.LogRecord) -> str:
        payload = getattr(record, "audit_payload", None)
        if payload is None:
            payload = {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "level": record.levelname,
                "message": record.getMessage(),
            }
        return json.dumps(payload, ensure_ascii=False, default=str)


def get_audit_logger(
    log_path: str | Path = DEFAULT_LOG_PATH,
    also_console: bool = False,
) -> logging.Logger:
    """Logger de auditoria com handler JSONL. Idempotente: chamar duas vezes
    com o mesmo caminho não duplica handlers (o que duplicaria linhas no log).
    """
    log_path = Path(log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger(f"{_LOGGER_NAME}.{log_path}")
    logger.setLevel(logging.INFO)
    logger.propagate = False

    already_attached = any(
        isinstance(h, logging.FileHandler) and Path(h.baseFilename) == log_path.resolve()
        for h in logger.handlers
    )
    if not already_attached:
        file_handler = logging.FileHandler(log_path, encoding="utf-8")
        file_handler.setFormatter(JsonLinesFormatter())
        logger.addHandler(file_handler)

    if also_console and not any(
        isinstance(h, logging.StreamHandler) and not isinstance(h, logging.FileHandler)
        for h in logger.handlers
    ):
        console = logging.StreamHandler()
        console.setFormatter(JsonLinesFormatter())
        logger.addHandler(console)

    return logger


@dataclass
class AuditRecord:
    """Registro de uma interação completa com o assistente."""

    session_id: str
    timestamp: str
    pergunta: str
    codigo_paciente: str | None = None
    contexto_recuperado: list[dict] = field(default_factory=list)
    contexto_paciente: str | None = None
    resposta: str | None = None
    fontes: list[dict] = field(default_factory=list)
    confianca: float | None = None
    grafo_nos_executados: list[str] = field(default_factory=list)
    bloqueios_seguranca: list[dict] = field(default_factory=list)
    alerta_emitido: bool = False
    requer_validacao_humana: bool = True
    llm_backend: str | None = None
    duracao_ms: float | None = None
    erro: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def new_session_id() -> str:
    return f"sess-{uuid.uuid4().hex[:12]}"


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def log_interaction(record: AuditRecord, log_path: str | Path = DEFAULT_LOG_PATH) -> None:
    """Persiste uma interação no log de auditoria."""
    logger = get_audit_logger(log_path)
    logger.info("interaction", extra={"audit_payload": record.to_dict()})


def read_audit_log(log_path: str | Path = DEFAULT_LOG_PATH) -> list[dict]:
    """Lê o log de auditoria de volta — usado nos testes e para a
    demonstração de rastreabilidade no vídeo.
    """
    log_path = Path(log_path)
    if not log_path.exists():
        return []
    with log_path.open(encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


__all__ = [
    "AuditRecord",
    "DEFAULT_LOG_PATH",
    "JsonLinesFormatter",
    "get_audit_logger",
    "log_interaction",
    "new_session_id",
    "read_audit_log",
    "utc_now_iso",
]
