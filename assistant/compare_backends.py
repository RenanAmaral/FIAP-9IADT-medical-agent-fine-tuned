"""Compara o assistente COMPLETO rodando com o modelo base e com o fine-tuned.

Diferença em relação a `finetuning/evaluate.py`: lá os modelos são comparados
crus, recebendo só a pergunta. Aqui a comparação passa por toda a pilha do
assistente — RAG dos protocolos, dados do paciente, guardrails e
explainability. Os dois backends recebem **exatamente o mesmo contexto
recuperado**, então a diferença observada é só o que cada modelo faz com ele.

Essa é a pergunta que importa para este projeto: com o texto do protocolo
entregue no contexto, o modelo base já resolve? Se sim, o fine-tuning agrega
pouco; se não, ele se justifica.

Os modelos são carregados em sequência e liberados após o uso, para caber em
GPUs pequenas (o par não precisa estar na memória ao mesmo tempo).

Uso:
    python -m assistant.compare_backends \
        --paciente PAC-0003 \
        --pergunta "Qual a conduta para este paciente?" \
        --adapter-dir finetuning/adapters/medical-assistant-lora
"""

from __future__ import annotations

import argparse
import gc
import json
from dataclasses import dataclass
from pathlib import Path

from assistant.chains import MedicalAssistantChain
from assistant.database import DEFAULT_DB_PATH
from assistant.rag import ProtocolRetriever

#: Perguntas usadas quando nenhuma é informada — cobrem os três caminhos do
#: grafo e as especialidades com maior risco de confusão entre protocolos.
DEFAULT_CASES: list[tuple[str | None, str]] = [
    ("PAC-0003", "Qual a conduta para este paciente?"),
    ("PAC-0002", "Posso ajustar o tratamento agora?"),
    (None, "Qual o limiar de HbA1c para diagnóstico de diabetes tipo 2?"),
    (None, "Quais critérios do qSOFA indicam sepse?"),
]


@dataclass
class BackendAnswer:
    backend: str
    resposta: str
    fontes: list[str]
    confianca: float | None
    bloqueado: bool
    duracao_ms: float


@dataclass
class CaseComparison:
    pergunta: str
    codigo_paciente: str | None
    contexto_protocolos: list[str]
    respostas: list[BackendAnswer]

    def to_dict(self) -> dict:
        return {
            "pergunta": self.pergunta,
            "codigo_paciente": self.codigo_paciente,
            "contexto_protocolos": self.contexto_protocolos,
            "respostas": [a.__dict__ for a in self.respostas],
        }


def _run_backend(
    backend: str,
    cases: list[tuple[str | None, str]],
    retriever: ProtocolRetriever,
    db_path: Path,
    log_path: Path,
    base_model: str,
    adapter_dir: str | None,
) -> list[tuple[BackendAnswer, list[str]]]:
    """Roda todos os casos em um backend e libera o modelo em seguida."""
    from assistant.llm import load_llm

    llm_kwargs: dict = {}
    if backend in {"base", "finetuned"}:
        llm_kwargs["base_model"] = base_model
    if backend == "finetuned" and adapter_dir:
        llm_kwargs["adapter_dir"] = adapter_dir

    print(f"[{backend}] carregando modelo...")
    llm = load_llm(backend, **llm_kwargs)

    chain = MedicalAssistantChain(
        llm=llm,
        retriever=retriever,
        db_path=db_path,
        log_path=log_path,
        llm_backend_name=backend,
    )

    results = []
    for codigo, pergunta in cases:
        print(f"[{backend}] {pergunta[:60]}...")
        response = chain.run(pergunta, codigo_paciente=codigo)
        answer = BackendAnswer(
            backend=backend,
            resposta=response.resposta,
            fontes=[s.identificador for s in response.explanation.sources]
            if response.explanation
            else [],
            confianca=response.explanation.confidence if response.explanation else None,
            bloqueado=response.bloqueado,
            duracao_ms=response.duracao_ms,
        )
        contexto = [c.protocol_id for c in response.chunks]
        results.append((answer, contexto))

    # Libera o modelo antes de carregar o próximo: os dois não precisam estar
    # na memória ao mesmo tempo, e em GPU pequena não caberiam.
    del chain, llm
    gc.collect()
    try:
        import torch

        torch.cuda.empty_cache()
    except Exception:
        pass

    return results


def compare(
    backends: list[str],
    cases: list[tuple[str | None, str]],
    db_path: Path,
    protocols_dir: Path,
    log_path: Path,
    base_model: str,
    adapter_dir: str | None,
) -> list[CaseComparison]:
    # Um único retriever para todos os backends: garante que o contexto
    # recuperado seja idêntico e que a diferença observada venha só do modelo.
    retriever = ProtocolRetriever.from_protocols(protocols_dir)

    por_backend = {
        backend: _run_backend(
            backend, cases, retriever, db_path, log_path, base_model, adapter_dir
        )
        for backend in backends
    }

    comparisons = []
    for i, (codigo, pergunta) in enumerate(cases):
        contextos = {tuple(por_backend[b][i][1]) for b in backends}
        if len(contextos) > 1:
            print(
                f"[aviso] O contexto recuperado divergiu entre backends para "
                f"{pergunta!r}. A comparação deixa de ser controlada."
            )
        comparisons.append(
            CaseComparison(
                pergunta=pergunta,
                codigo_paciente=codigo,
                contexto_protocolos=list(next(iter(contextos))),
                respostas=[por_backend[b][i][0] for b in backends],
            )
        )
    return comparisons


def write_markdown(comparisons: list[CaseComparison], path: Path) -> None:
    lines = [
        "# Comparação do assistente: modelo base vs. fine-tuned",
        "",
        "Ambos os backends rodam a pilha completa do assistente e recebem o "
        "**mesmo contexto recuperado** (protocolos + dados do paciente). A "
        "diferença observada é só o que cada modelo faz com esse contexto.",
        "",
    ]
    for i, case in enumerate(comparisons, start=1):
        lines += [
            f"## Caso {i}",
            "",
            f"**Pergunta:** {case.pergunta}  ",
            f"**Paciente:** {case.codigo_paciente or '(nenhum)'}  ",
            f"**Protocolos recuperados:** {', '.join(case.contexto_protocolos) or '—'}",
            "",
        ]
        for answer in case.respostas:
            confianca = f"{answer.confianca:.0%}" if answer.confianca is not None else "—"
            lines += [
                f"### `{answer.backend}`",
                f"*confiança da recuperação: {confianca} · "
                f"{answer.duracao_ms:.0f} ms"
                + (" · **BLOQUEADO**" if answer.bloqueado else "")
                + "*",
                "",
                "```",
                answer.resposta.strip(),
                "```",
                "",
            ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--pergunta", help="Se omitido, usa o conjunto padrão de casos.")
    parser.add_argument("--paciente")
    parser.add_argument(
        "--backends",
        nargs="+",
        default=["base", "finetuned"],
        choices=["template", "base", "finetuned"],
        help="Backends a comparar, na ordem de exibição.",
    )
    parser.add_argument("--base-model", default="TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    parser.add_argument("--adapter-dir", default="finetuning/adapters/medical-assistant-lora")
    parser.add_argument("--db-path", type=Path, default=DEFAULT_DB_PATH)
    parser.add_argument("--protocols-dir", type=Path, default=Path("data/protocols"))
    parser.add_argument("--log-path", type=Path, default=Path("logs/audit.jsonl"))
    parser.add_argument("--output", type=Path, default=Path("docs/comparacao_backends.md"))
    args = parser.parse_args()

    if not args.db_path.exists():
        parser.error(
            f"Base estruturada não encontrada em {args.db_path}. "
            "Rode primeiro: python -m assistant.database"
        )

    cases = (
        [(args.paciente, args.pergunta)] if args.pergunta else DEFAULT_CASES
    )

    comparisons = compare(
        backends=args.backends,
        cases=cases,
        db_path=args.db_path,
        protocols_dir=args.protocols_dir,
        log_path=args.log_path,
        base_model=args.base_model,
        adapter_dir=args.adapter_dir,
    )

    write_markdown(comparisons, args.output)
    args.output.with_suffix(".json").write_text(
        json.dumps([c.to_dict() for c in comparisons], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print()
    for i, case in enumerate(comparisons, start=1):
        print("=" * 78)
        print(f"CASO {i}: {case.pergunta}")
        print(f"Protocolos recuperados: {', '.join(case.contexto_protocolos)}")
        for answer in case.respostas:
            print("-" * 78)
            print(f"[{answer.backend}]")
            print(answer.resposta.strip()[:800])
        print()

    print(f"Comparação salva em {args.output}")


if __name__ == "__main__":
    main()
