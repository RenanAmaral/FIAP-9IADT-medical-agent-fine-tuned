"""Etapa 3, item 8 — Integração da LLM customizada no LangChain.

Três backends, todos implementando a interface `LLM` do LangChain, então a
chain (`assistant/chains.py`) e o grafo (`graphs/`) funcionam com qualquer um
sem alteração — é o ponto de troca previsto no enunciado ("desenvolva usando o
modelo base e ao final substitua pelo modelo treinado"):

1. `load_finetuned_llm()` — modelo base + adapters LoRA da Etapa 2, servido via
   `HuggingFacePipeline`. É o backend de produção do projeto.
2. `load_base_llm()` — mesmo modelo base, sem os adapters. Usado para a
   comparação antes/depois.
3. `TemplateClinicalLLM` — **não é um modelo de linguagem**. É um stub
   determinístico que monta a resposta a partir do contexto já recuperado no
   prompt. Existe porque o ambiente de desenvolvimento deste projeto não tem
   GPU nem acesso à Hugging Face Hub: ele permite exercitar e testar
   ponta a ponta toda a orquestração (RAG, base estruturada, grafo LangGraph,
   guardrails, logging) sem depender do modelo. Os testes automatizados usam
   este backend. Ele NÃO deve ser usado na demonstração final do assistente —
   para isso use `load_finetuned_llm()`.
"""

from __future__ import annotations

import re
from typing import Any

from langchain_core.language_models.llms import LLM

from finetuning.config import SYSTEM_PROMPT


class TemplateClinicalLLM(LLM):
    """Stub determinístico para desenvolvimento e testes offline.

    Extrai os blocos de contexto que a chain já montou no prompt (trechos de
    protocolo e dados do paciente) e os reorganiza em uma resposta no formato
    esperado. Como não gera linguagem nova, é totalmente previsível — o que
    torna os testes da orquestração estáveis — mas também não demonstra
    nenhuma capacidade do modelo fine-tuned.
    """

    @property
    def _llm_type(self) -> str:
        return "template_clinical_stub"

    #: Delimita o bloco de protocolos dentro do prompt montado pela chain.
    _PROTOCOL_BLOCK_RE = re.compile(
        r"=== PROTOCOLOS INTERNOS RELEVANTES ===\n(.*?)(?=\n\nResponda à pergunta|\n<\|)",
        re.DOTALL,
    )
    _PROTOCOL_ENTRY_RE = re.compile(
        r"\[(PROT-[A-Z]+-\d+) — ([^\]]+)\]\n(.*?)(?=\n\n\[PROT-|\Z)", re.DOTALL
    )

    def _call(
        self,
        prompt: str,
        stop: list[str] | None = None,
        run_manager: Any = None,
        **kwargs: Any,
    ) -> str:
        block_match = self._PROTOCOL_BLOCK_RE.search(prompt)
        protocol_block = block_match.group(1) if block_match else ""
        entradas = self._PROTOCOL_ENTRY_RE.findall(protocol_block)

        pendentes = re.findall(r"- ([^:\n]+): PENDENTE", prompt)

        linhas = []
        if entradas:
            titulos = {pid: titulo for pid, titulo, _ in entradas}
            ids = ", ".join(f"{pid} ({titulo})" for pid, titulo in titulos.items())
            linhas.append(
                f"Com base nos protocolos internos {ids}, seguem as orientações "
                "aplicáveis ao caso apresentado."
            )
        else:
            linhas.append("Não localizei protocolo interno específico para esta consulta.")

        if pendentes:
            linhas.append(
                "Atenção: há exames pendentes no prontuário ("
                + ", ".join(dict.fromkeys(pendentes))
                + "). A conduta definitiva deve aguardar esses resultados."
            )

        # Recorta os trechos de protocolo já presentes no prompt para compor a
        # orientação, em vez de inventar conteúdo clínico. O corte é feito em
        # fronteira de linha para não truncar uma conduta no meio.
        if entradas:
            linhas.append("\nOrientações extraídas dos protocolos:")
            for pid, _titulo, trecho in entradas[:2]:
                linhas.append(f"[{pid}]")
                linhas.append(self._truncate_lines(trecho.strip(), max_chars=600))

        linhas.append(
            "\nEsta é uma sugestão de apoio à decisão baseada nos protocolos "
            "internos. Não constitui prescrição: a conduta final depende de "
            "validação do médico responsável."
        )
        return "\n".join(linhas)

    @staticmethod
    def _truncate_lines(text: str, max_chars: int) -> str:
        if len(text) <= max_chars:
            return text
        kept: list[str] = []
        total = 0
        for line in text.splitlines():
            if total + len(line) + 1 > max_chars:
                break
            kept.append(line)
            total += len(line) + 1
        return "\n".join(kept) if kept else text[:max_chars]


def _build_hf_pipeline_llm(
    base_model: str,
    adapter_dir: str | None,
    max_new_tokens: int = 512,
    temperature: float = 0.1,
):
    """Constrói um `HuggingFacePipeline` do LangChain sobre o modelo base,
    opcionalmente com os adapters LoRA aplicados.

    Imports tardios: `torch`/`transformers`/`peft` só são necessários quando
    de fato se usa um modelo real.
    """
    import torch
    from langchain_community.llms import HuggingFacePipeline
    from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

    from finetuning.hf_auth import dtype_kwarg, ensure_hf_login

    ensure_hf_login(verbose=False)

    tokenizer = AutoTokenizer.from_pretrained(base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        device_map="auto",
        **dtype_kwarg(torch.float16 if torch.cuda.is_available() else torch.float32),
    )

    if adapter_dir:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, adapter_dir)
        model = model.merge_and_unload()

    model.eval()

    text_pipeline = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        do_sample=temperature > 0,
        return_full_text=False,
        pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
    )
    return HuggingFacePipeline(pipeline=text_pipeline)


def load_finetuned_llm(
    base_model: str = "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    adapter_dir: str = "finetuning/adapters/medical-assistant-lora",
    **kwargs: Any,
):
    """Modelo fine-tuned (base + adapters LoRA da Etapa 2)."""
    return _build_hf_pipeline_llm(base_model, adapter_dir, **kwargs)


def load_base_llm(
    base_model: str = "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    **kwargs: Any,
):
    """Modelo base sem fine-tuning, para comparação antes/depois."""
    return _build_hf_pipeline_llm(base_model, adapter_dir=None, **kwargs)


def load_llm(backend: str = "template", **kwargs: Any) -> LLM:
    """Fábrica única usada pela CLI e pelo grafo.

    backend:
        "finetuned" — modelo com os adapters da Etapa 2 (produção)
        "base"      — modelo base sem fine-tuning (comparação)
        "template"  — stub offline determinístico (desenvolvimento/testes)
    """
    if backend == "finetuned":
        return load_finetuned_llm(**kwargs)
    if backend == "base":
        return load_base_llm(**kwargs)
    if backend == "template":
        return TemplateClinicalLLM()
    raise ValueError(
        f"Backend de LLM desconhecido: {backend!r}. Use 'finetuned', 'base' ou 'template'."
    )


__all__ = [
    "SYSTEM_PROMPT",
    "TemplateClinicalLLM",
    "load_base_llm",
    "load_finetuned_llm",
    "load_llm",
]
