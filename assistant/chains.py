"""Etapa 3 — Chain principal do assistente médico.

Orquestra, em ordem:

1. Guardrail de entrada (`security.guardrails.check_input_scope`).
2. Recuperação do contexto do paciente na base estruturada (SQLite).
3. Recuperação dos protocolos internos relevantes (RAG/FAISS).
4. Montagem do prompt no formato de treino e chamada à LLM.
5. Guardrail de saída + aviso de validação humana obrigatória.
6. Bloco de explicabilidade (fontes + grau de confiança).
7. Registro da interação no log de auditoria.

A LLM é injetada (`llm=`), então a mesma chain roda com o modelo fine-tuned,
com o modelo base ou com o stub offline — ver `assistant/llm.py`.
"""

from __future__ import annotations

import sqlite3
import time
from dataclasses import dataclass, field
from pathlib import Path

from langchain_core.language_models.llms import LLM

from assistant import database
from assistant.explainability import Explanation, build_explanation
from assistant.prompts import build_clinical_prompt
from assistant.rag import ProtocolRetriever, RetrievedChunk, build_context_block
from security import guardrails
from security.audit import AuditRecord, log_interaction, new_session_id, utc_now_iso


@dataclass
class AssistantResponse:
    """Resposta completa do assistente, já auditável."""

    resposta: str
    pergunta: str
    session_id: str
    codigo_paciente: str | None = None
    explanation: Explanation | None = None
    chunks: list[RetrievedChunk] = field(default_factory=list)
    bloqueado: bool = False
    motivo_bloqueio: str | None = None
    exames_pendentes: list[str] = field(default_factory=list)
    duracao_ms: float = 0.0

    @property
    def texto_completo(self) -> str:
        """Resposta + bloco de fontes/confiança, pronta para exibição."""
        if self.explanation is None:
            return self.resposta
        return f"{self.resposta}\n{self.explanation.to_block()}"


class MedicalAssistantChain:
    """Chain principal. Instanciar uma vez e reutilizar — a construção do
    índice FAISS e a conexão com o banco são feitas no `__init__`.
    """

    def __init__(
        self,
        llm: LLM,
        retriever: ProtocolRetriever | None = None,
        db_path: str | Path = database.DEFAULT_DB_PATH,
        protocols_dir: str | Path = "data/protocols",
        log_path: str | Path = "logs/audit.jsonl",
        top_k: int = 3,
        llm_backend_name: str = "unknown",
    ):
        self.llm = llm
        self.retriever = retriever or ProtocolRetriever.from_protocols(protocols_dir)
        self.db_path = Path(db_path)
        self.log_path = Path(log_path)
        self.top_k = top_k
        self.llm_backend_name = llm_backend_name

    def _connect(self) -> sqlite3.Connection:
        return database.connect(self.db_path)

    def get_patient_record(self, codigo_paciente: str) -> database.PatientRecord | None:
        conn = self._connect()
        try:
            return database.get_patient_record(conn, codigo_paciente)
        finally:
            conn.close()

    def run(
        self,
        pergunta: str,
        codigo_paciente: str | None = None,
        session_id: str | None = None,
        grafo_nos_executados: list[str] | None = None,
    ) -> AssistantResponse:
        started = time.perf_counter()
        session_id = session_id or new_session_id()
        bloqueios: list[dict] = []

        audit = AuditRecord(
            session_id=session_id,
            timestamp=utc_now_iso(),
            pergunta=pergunta,
            codigo_paciente=codigo_paciente,
            llm_backend=self.llm_backend_name,
            grafo_nos_executados=grafo_nos_executados or [],
        )

        # 1. Guardrail de entrada -------------------------------------------
        input_check = guardrails.check_input_scope(pergunta)
        if input_check.blocked:
            bloqueios.append(input_check.to_dict())
            duracao = (time.perf_counter() - started) * 1000
            audit.resposta = input_check.message
            audit.bloqueios_seguranca = bloqueios
            audit.duracao_ms = duracao
            log_interaction(audit, self.log_path)
            return AssistantResponse(
                resposta=input_check.message,
                pergunta=pergunta,
                session_id=session_id,
                codigo_paciente=codigo_paciente,
                bloqueado=True,
                motivo_bloqueio=input_check.reason.value,
                duracao_ms=duracao,
            )

        # 2. Contexto do paciente (base estruturada) ------------------------
        record = self.get_patient_record(codigo_paciente) if codigo_paciente else None
        contexto_paciente = record.to_context_string() if record else None
        exames_pendentes = [e.nome_exame for e in record.exames_pendentes] if record else []
        especialidade = record.patient.especialidade if record else None

        # 3. RAG sobre os protocolos internos -------------------------------
        # A consulta ao vector store combina a pergunta com a condição do
        # paciente, para que o contexto recuperado reflita o caso concreto e
        # não apenas o texto literal da pergunta.
        query = pergunta
        if record:
            query = f"{pergunta} {record.patient.condicao_principal}"
        chunks = self.retriever.retrieve(query, k=self.top_k, especialidade=especialidade)
        contexto_protocolos = build_context_block(chunks)

        # 4. Prompt + LLM ---------------------------------------------------
        prompt = build_clinical_prompt(
            pergunta=pergunta,
            contexto_protocolos=contexto_protocolos,
            contexto_paciente=contexto_paciente,
        )
        try:
            raw_answer = self.llm.invoke(prompt)
        except Exception as exc:  # pragma: no cover - depende do backend real
            duracao = (time.perf_counter() - started) * 1000
            audit.erro = f"{type(exc).__name__}: {exc}"
            audit.duracao_ms = duracao
            log_interaction(audit, self.log_path)
            raise

        answer = raw_answer.strip() if isinstance(raw_answer, str) else str(raw_answer)

        # 5. Guardrail de saída ---------------------------------------------
        output_check = guardrails.check_output_safety(answer)
        if output_check.blocked:
            bloqueios.append(output_check.to_dict())
            answer = output_check.message

        answer = guardrails.enforce_human_validation(answer)

        # 6. Explainability --------------------------------------------------
        explanation = build_explanation(
            chunks=chunks,
            codigo_paciente=codigo_paciente,
            tem_dados_paciente=record is not None,
            exames_pendentes=exames_pendentes,
        )

        duracao = (time.perf_counter() - started) * 1000

        # 7. Auditoria -------------------------------------------------------
        audit.contexto_recuperado = [
            {
                "protocol_id": c.protocol_id,
                "titulo": c.titulo,
                "score": c.score,
                "trecho": c.trecho[:500],
            }
            for c in chunks
        ]
        audit.contexto_paciente = contexto_paciente
        audit.resposta = answer
        audit.fontes = [s.to_dict() for s in explanation.sources]
        audit.confianca = explanation.confidence
        audit.bloqueios_seguranca = bloqueios
        audit.duracao_ms = duracao
        log_interaction(audit, self.log_path)

        return AssistantResponse(
            resposta=answer,
            pergunta=pergunta,
            session_id=session_id,
            codigo_paciente=codigo_paciente,
            explanation=explanation,
            chunks=chunks,
            bloqueado=bool(bloqueios),
            motivo_bloqueio=bloqueios[0]["reason"] if bloqueios else None,
            exames_pendentes=exames_pendentes,
            duracao_ms=duracao,
        )


def build_assistant(
    backend: str = "template",
    db_path: str | Path = database.DEFAULT_DB_PATH,
    protocols_dir: str | Path = "data/protocols",
    log_path: str | Path = "logs/audit.jsonl",
    **llm_kwargs,
) -> MedicalAssistantChain:
    """Atalho de construção usado pela CLI e pelo grafo da Etapa 4."""
    from assistant.llm import load_llm

    llm = load_llm(backend, **llm_kwargs)
    return MedicalAssistantChain(
        llm=llm,
        db_path=db_path,
        protocols_dir=protocols_dir,
        log_path=log_path,
        llm_backend_name=backend,
    )


__all__ = ["AssistantResponse", "MedicalAssistantChain", "build_assistant"]
