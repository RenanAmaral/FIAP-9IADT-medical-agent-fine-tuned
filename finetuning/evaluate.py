"""Etapa 2 — Avaliação do modelo fine-tuned.

Combina métricas quantitativas (perplexidade, ROUGE) com uma comparação
qualitativa lado a lado das respostas do modelo base vs. fine-tuned em um
conjunto fixo de perguntas clínicas — exigido explicitamente no enunciado
para entrar no relatório técnico.

Como treinar, este script precisa de GPU/HF Hub e portanto é feito para
rodar no Colab/Kaggle logo após `finetuning/train.py` — ver
`finetuning/README.md`.

Uso:
    python -m finetuning.evaluate \
        --base-model TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
        --adapter-dir finetuning/adapters/medical-assistant-lora \
        --test-file data/processed/test.jsonl
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from finetuning.config import format_prompt
from finetuning.dataset import read_jsonl


def _load_pipeline(base_model: str, adapter_dir: str | None):
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from finetuning.hf_auth import dtype_kwarg, ensure_hf_login

    ensure_hf_login(verbose=False)

    tokenizer = AutoTokenizer.from_pretrained(base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Em GPU usamos fp16: para inferência a precisão extra do fp32 não muda o
    # resultado de forma relevante, mas dobra a memória e o tempo de geração.
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        device_map="auto",
        **dtype_kwarg(torch.float16 if torch.cuda.is_available() else torch.float32),
    )
    if adapter_dir:
        model = PeftModel.from_pretrained(model, adapter_dir)
    model.eval()
    return model, tokenizer


def _generate(model, tokenizer, prompt: str, max_new_tokens: int = 256) -> str:
    import torch

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
        )
    generated = output_ids[0][inputs["input_ids"].shape[1] :]
    return tokenizer.decode(generated, skip_special_tokens=True).strip()


def _perplexity(model, tokenizer, texts: list[str]) -> float:
    import torch

    losses = []
    for text in texts:
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=1024).to(model.device)
        with torch.no_grad():
            outputs = model(**inputs, labels=inputs["input_ids"])
        losses.append(outputs.loss.item())
    return math.exp(sum(losses) / len(losses)) if losses else float("nan")


def run_evaluation(
    base_model: str,
    adapter_dir: str | None,
    test_file: str,
    output_dir: str,
    max_examples: int = 20,
) -> dict:
    from rouge_score import rouge_scorer

    test_records = read_jsonl(test_file)[:max_examples]
    prompts = [format_prompt(r["instruction"], r.get("input", "")) for r in test_records]
    references = [r["output"] for r in test_records]
    full_texts = [
        format_prompt(r["instruction"], r.get("input", ""), r["output"]) for r in test_records
    ]

    print("Carregando modelo base...")
    base_model_obj, base_tokenizer = _load_pipeline(base_model, adapter_dir=None)
    print("Gerando respostas do modelo base...")
    base_outputs = [_generate(base_model_obj, base_tokenizer, p) for p in prompts]
    base_ppl = _perplexity(base_model_obj, base_tokenizer, full_texts)
    del base_model_obj

    print("Carregando modelo fine-tuned (base + adapter)...")
    ft_model_obj, ft_tokenizer = _load_pipeline(base_model, adapter_dir=adapter_dir)
    print("Gerando respostas do modelo fine-tuned...")
    ft_outputs = [_generate(ft_model_obj, ft_tokenizer, p) for p in prompts]
    ft_ppl = _perplexity(ft_model_obj, ft_tokenizer, full_texts)
    del ft_model_obj

    scorer = rouge_scorer.RougeScorer(["rouge1", "rougeL"], use_stemmer=True)

    def _avg_rouge(hyps: list[str]) -> dict:
        scores = [scorer.score(ref, hyp) for ref, hyp in zip(references, hyps)]
        return {
            metric: sum(s[metric].fmeasure for s in scores) / len(scores) if scores else 0.0
            for metric in ("rouge1", "rougeL")
        }

    comparison = []
    for record, prompt, base_out, ft_out in zip(test_records, prompts, base_outputs, ft_outputs):
        comparison.append(
            {
                "instruction": record["instruction"],
                "reference": record["output"],
                "base_model_response": base_out,
                "fine_tuned_response": ft_out,
            }
        )

    results = {
        "base_model": base_model,
        "adapter_dir": adapter_dir,
        "n_examples": len(test_records),
        "perplexity": {"base": base_ppl, "fine_tuned": ft_ppl},
        "rouge": {"base": _avg_rouge(base_outputs), "fine_tuned": _avg_rouge(ft_outputs)},
        "qualitative_comparison": comparison,
    }

    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    (out_path / "evaluation_results.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    _write_markdown_report(results, out_path / "evaluation_report.md")

    return results


def _write_markdown_report(results: dict, path: Path) -> None:
    lines = [
        "# Avaliação do Modelo Fine-tuned",
        "",
        f"Modelo base: `{results['base_model']}`  ",
        f"Adapter: `{results['adapter_dir']}`  ",
        f"Exemplos de teste avaliados: {results['n_examples']}",
        "",
        "## Métricas quantitativas",
        "",
        "| Métrica | Base | Fine-tuned |",
        "|---|---|---|",
        f"| Perplexidade | {results['perplexity']['base']:.2f} | {results['perplexity']['fine_tuned']:.2f} |",
        f"| ROUGE-1 | {results['rouge']['base']['rouge1']:.3f} | {results['rouge']['fine_tuned']['rouge1']:.3f} |",
        f"| ROUGE-L | {results['rouge']['base']['rougeL']:.3f} | {results['rouge']['fine_tuned']['rougeL']:.3f} |",
        "",
        "## Comparação qualitativa (amostra)",
        "",
    ]
    for i, item in enumerate(results["qualitative_comparison"][:5], start=1):
        lines += [
            f"### Exemplo {i}",
            f"**Pergunta:** {item['instruction']}",
            "",
            f"**Referência (protocolo):** {item['reference'][:300]}...",
            "",
            f"**Resposta do modelo base:** {item['base_model_response'][:300]}",
            "",
            f"**Resposta do modelo fine-tuned:** {item['fine_tuned_response'][:300]}",
            "",
        ]
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-model", required=True)
    parser.add_argument("--adapter-dir", required=True)
    parser.add_argument("--test-file", default="data/processed/test.jsonl")
    parser.add_argument("--output-dir", default="finetuning/eval_results")
    parser.add_argument("--max-examples", type=int, default=20)
    args = parser.parse_args()

    results = run_evaluation(
        base_model=args.base_model,
        adapter_dir=args.adapter_dir,
        test_file=args.test_file,
        output_dir=args.output_dir,
        max_examples=args.max_examples,
    )
    print(f"Perplexidade base={results['perplexity']['base']:.2f} "
          f"fine-tuned={results['perplexity']['fine_tuned']:.2f}")


if __name__ == "__main__":
    main()
