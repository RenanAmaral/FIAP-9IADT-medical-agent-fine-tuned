"""Inspeção do log de auditoria (Etapa 5, item 13).

Ferramenta de leitura do `logs/audit.jsonl` para demonstrar rastreabilidade —
é o que se mostra no vídeo ao falar de "logs e validação das respostas"
(item 18 do enunciado).

Uso:
    python -m security.inspect_logs                      # resumo geral
    python -m security.inspect_logs --bloqueios          # só interações bloqueadas
    python -m security.inspect_logs --alertas            # só fluxos com alerta
    python -m security.inspect_logs --sessao sess-abc123 # uma sessão específica
    python -m security.inspect_logs --detalhe -n 3       # últimas 3 em detalhe
"""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

from security.audit import DEFAULT_LOG_PATH, read_audit_log


def _print_summary(entries: list[dict]) -> None:
    if not entries:
        print("Log de auditoria vazio.")
        return

    bloqueadas = [e for e in entries if e.get("bloqueios_seguranca")]
    alertas = [e for e in entries if e.get("alerta_emitido")]
    fluxos = [e for e in entries if e.get("grafo_nos_executados")]

    print(f"Total de interações registradas: {len(entries)}")
    print(f"  Bloqueadas por guardrails:     {len(bloqueadas)}")
    print(f"  Com alerta de risco emitido:   {len(alertas)}")
    print(f"  Execuções de fluxo (grafo):    {len(fluxos)}")

    motivos = Counter(
        b.get("reason")
        for e in bloqueadas
        for b in e.get("bloqueios_seguranca", [])
        if b.get("reason")
    )
    if motivos:
        print("\nMotivos de bloqueio:")
        for motivo, n in motivos.most_common():
            print(f"  {motivo}: {n}")

    caminhos = Counter(
        " → ".join(e["grafo_nos_executados"]) for e in fluxos if e.get("grafo_nos_executados")
    )
    if caminhos:
        print("\nCaminhos percorridos no grafo:")
        for caminho, n in caminhos.most_common():
            print(f"  [{n}x] {caminho}")

    protocolos = Counter(
        f["identificador"]
        for e in entries
        for f in e.get("fontes", [])
        if f.get("tipo") == "protocolo"
    )
    if protocolos:
        print("\nProtocolos mais citados como fonte:")
        for protocolo, n in protocolos.most_common(5):
            print(f"  {protocolo}: {n}")

    confiancas = [e["confianca"] for e in entries if e.get("confianca") is not None]
    if confiancas:
        print(f"\nConfiança média de recuperação: {sum(confiancas) / len(confiancas):.1%}")


def _print_detail(entry: dict) -> None:
    print("=" * 78)
    print(f"Sessão:    {entry.get('session_id')}")
    print(f"Timestamp: {entry.get('timestamp')}")
    print(f"Paciente:  {entry.get('codigo_paciente') or '(nenhum)'}")
    print(f"Backend:   {entry.get('llm_backend')}")
    print(f"Pergunta:  {entry.get('pergunta')}")

    if entry.get("grafo_nos_executados"):
        print(f"Grafo:     {' → '.join(entry['grafo_nos_executados'])}")

    if entry.get("bloqueios_seguranca"):
        print("BLOQUEIOS:")
        for b in entry["bloqueios_seguranca"]:
            print(f"  - {b.get('reason')}: {b.get('matched_patterns', '')}")

    if entry.get("contexto_recuperado"):
        print("Contexto recuperado:")
        for c in entry["contexto_recuperado"]:
            print(f"  - {c.get('protocol_id')} (score {c.get('score', 0):.3f})")

    if entry.get("fontes"):
        print("Fontes citadas:")
        for f in entry["fontes"]:
            print(f"  - [{f.get('tipo')}] {f.get('identificador')}")

    if entry.get("confianca") is not None:
        print(f"Confiança: {entry['confianca']:.1%}")
    print(f"Alerta emitido: {entry.get('alerta_emitido')}")
    print(f"Requer validação humana: {entry.get('requer_validacao_humana')}")

    resposta = entry.get("resposta") or ""
    print(f"\nResposta ({len(resposta)} chars):")
    print(resposta[:600] + ("..." if len(resposta) > 600 else ""))
    print()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--log-path", type=Path, default=DEFAULT_LOG_PATH)
    parser.add_argument("--bloqueios", action="store_true", help="Só interações bloqueadas.")
    parser.add_argument("--alertas", action="store_true", help="Só fluxos com alerta emitido.")
    parser.add_argument("--sessao", help="Filtra por identificador de sessão.")
    parser.add_argument("--detalhe", action="store_true", help="Mostra as entradas em detalhe.")
    parser.add_argument("-n", type=int, default=5, help="Quantas entradas exibir em detalhe.")
    args = parser.parse_args()

    entries = read_audit_log(args.log_path)

    if args.sessao:
        entries = [e for e in entries if e.get("session_id") == args.sessao]
    if args.bloqueios:
        entries = [e for e in entries if e.get("bloqueios_seguranca")]
    if args.alertas:
        entries = [e for e in entries if e.get("alerta_emitido")]

    if args.detalhe or args.bloqueios or args.alertas or args.sessao:
        if not entries:
            print("Nenhuma entrada corresponde ao filtro.")
            return
        for entry in entries[-args.n :]:
            _print_detail(entry)
        print(f"({len(entries)} entrada(s) correspondem ao filtro; exibindo até {args.n})")
    else:
        _print_summary(entries)


if __name__ == "__main__":
    main()
