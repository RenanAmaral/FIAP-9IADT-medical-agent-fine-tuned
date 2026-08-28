"""CLI do fluxo clínico automatizado (Etapa 4).

Executa o grafo de decisão completo, mostrando o caminho percorrido entre os
nós — é a demonstração de "execução de um fluxo automatizado" exigida no
vídeo (item 18 do enunciado).

Exemplos:
    # Paciente estável -> caminho de sugestão
    python -m graphs.cli --paciente PAC-0001 --pergunta "Qual a conduta?"

    # Paciente com exames pendentes -> desvio para solicitação de exames
    python -m graphs.cli --paciente PAC-0002 --pergunta "Posso ajustar o tratamento?"

    # Paciente com critérios de risco -> nó de alerta
    python -m graphs.cli --paciente PAC-0003 --pergunta "Qual a conduta?"

    # Demonstra os três caminhos em sequência
    python -m graphs.cli --demo

    # Exporta o diagrama Mermaid do grafo
    python -m graphs.cli --diagrama docs/fluxo_langgraph.mmd
"""

from __future__ import annotations

import argparse
from pathlib import Path

from assistant.database import DEFAULT_DB_PATH
from graphs.clinical_flow import build_clinical_flow

DEMO_CASES = [
    ("PAC-0001", "Qual a conduta recomendada para este paciente?", "caminho normal → sugestão"),
    ("PAC-0002", "Posso ajustar o tratamento agora?", "desvio → exames pendentes"),
    ("PAC-0003", "Qual a conduta para este paciente?", "desvio → alerta de risco"),
]


def _print_state(state: dict, titulo: str = "") -> None:
    if titulo:
        print(f"\n{'#' * 78}\n# {titulo}\n{'#' * 78}")

    print(f"\nPergunta: {state.get('pergunta')}")
    print(f"Paciente: {state.get('codigo_paciente') or '(nenhum)'}")
    print(f"Sessão:   {state.get('session_id')}")

    print("\n--- Caminho percorrido no grafo ---")
    print("  " + " → ".join(state.get("trace", [])))

    print("\n--- Estado clínico avaliado ---")
    print(f"  Nível de risco:    {state.get('nivel_risco')}")
    criterios = state.get("criterios_risco") or []
    if criterios:
        for c in criterios:
            print(f"    • {c}")
    pendentes = state.get("exames_pendentes") or []
    print(f"  Exames pendentes:  {', '.join(pendentes) if pendentes else 'nenhum'}")
    print(f"  Alerta emitido:    {'SIM' if state.get('alerta_emitido') else 'não'}")
    print(f"  Status final:      {state.get('status_final')}")

    print("\n--- Resposta ---")
    print(state.get("resposta") or "(sem resposta)")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--pergunta")
    parser.add_argument("--paciente")
    parser.add_argument(
        "--backend", default="template", choices=["template", "base", "finetuned"]
    )
    parser.add_argument(
        "--base-model",
        default="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        help="Modelo base (usado com --backend base ou finetuned).",
    )
    parser.add_argument(
        "--adapter-dir",
        default="finetuning/adapters/medical-assistant-lora",
        help="Diretório dos adapters LoRA (usado com --backend finetuned). "
        "Se o treino gravou em outro caminho (ex.: Google Drive), passe-o aqui.",
    )
    parser.add_argument("--db-path", type=Path, default=DEFAULT_DB_PATH)
    parser.add_argument("--protocols-dir", type=Path, default=Path("data/protocols"))
    parser.add_argument("--log-path", type=Path, default=Path("logs/audit.jsonl"))
    parser.add_argument("--demo", action="store_true", help="Executa os três caminhos do grafo.")
    parser.add_argument(
        "--diagrama", type=Path, nargs="?", const=Path("docs/fluxo_langgraph.mmd"),
        help="Exporta o diagrama Mermaid do grafo e sai.",
    )
    args = parser.parse_args()

    if not args.diagrama and not args.demo and not args.pergunta:
        parser.error("informe --pergunta, ou use --demo ou --diagrama")

    if not args.db_path.exists():
        parser.error(
            f"Base estruturada não encontrada em {args.db_path}. "
            "Rode primeiro: python -m assistant.database"
        )

    # adapter_dir só faz sentido para o backend com fine-tuning; passá-lo aos
    # demais quebraria a construção da LLM.
    llm_kwargs: dict = {}
    if args.backend in {"base", "finetuned"}:
        llm_kwargs["base_model"] = args.base_model
    if args.backend == "finetuned":
        llm_kwargs["adapter_dir"] = args.adapter_dir

    flow = build_clinical_flow(
        backend=args.backend,
        db_path=args.db_path,
        protocols_dir=args.protocols_dir,
        log_path=args.log_path,
        **llm_kwargs,
    )

    if args.diagrama:
        args.diagrama.parent.mkdir(parents=True, exist_ok=True)
        args.diagrama.write_text(flow.to_mermaid(), encoding="utf-8")
        print(f"Diagrama Mermaid exportado para {args.diagrama}")
        return

    if args.demo:
        for codigo, pergunta, descricao in DEMO_CASES:
            state = flow.run(pergunta, codigo_paciente=codigo)
            _print_state(state, titulo=f"{codigo} — {descricao}")
        return

    _print_state(flow.run(args.pergunta, codigo_paciente=args.paciente))


if __name__ == "__main__":
    main()
