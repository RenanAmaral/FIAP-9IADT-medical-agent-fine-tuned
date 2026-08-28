"""Etapa 2 — Fine-tuning da LLM com LoRA/QLoRA.

Stack: transformers + peft + trl (SFTTrainer) + bitsandbytes.

Este script foi desenhado para rodar em um ambiente com GPU (Google Colab
ou Kaggle) — ver `finetuning/README.md` para o passo a passo. Ele NÃO é
executado no ambiente de desenvolvimento deste repositório, que não possui
GPU nem acesso à Hugging Face Hub; por isso os imports pesados
(torch/transformers/peft/trl/bitsandbytes) são feitos de forma tardia (lazy),
para que `finetuning.config` e `finetuning.dataset` continuem
testáveis/importáveis sem essas dependências.

Uso (Colab/Kaggle, com GPU):
    python -m finetuning.train --base-model TinyLlama/TinyLlama-1.1B-Chat-v1.0

Modo smoke-test (sanidade da pipeline, sem baixar nenhum modelo da internet):
    python -m finetuning.train --smoke-test
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from finetuning.config import LoraParams, TrainingConfig
from finetuning.dataset import load_training_dataset


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    defaults = TrainingConfig()
    parser.add_argument("--base-model", default=defaults.base_model)
    parser.add_argument("--train-file", default=defaults.train_file)
    parser.add_argument("--val-file", default=defaults.val_file)
    parser.add_argument("--output-dir", default=defaults.output_dir)
    parser.add_argument("--max-seq-length", type=int, default=defaults.max_seq_length)
    parser.add_argument("--learning-rate", type=float, default=defaults.learning_rate)
    parser.add_argument("--epochs", type=int, default=defaults.num_train_epochs)
    parser.add_argument("--batch-size", type=int, default=defaults.per_device_train_batch_size)
    parser.add_argument(
        "--grad-accum", type=int, default=defaults.gradient_accumulation_steps
    )
    parser.add_argument("--lora-r", type=int, default=defaults.lora.r)
    parser.add_argument("--lora-alpha", type=int, default=defaults.lora.lora_alpha)
    parser.add_argument("--no-4bit", action="store_true", help="Desativa QLoRA (usa LoRA em fp16/bf16 puro)")
    parser.add_argument(
        "--compute-dtype",
        default=defaults.bnb_compute_dtype,
        choices=["auto", "bfloat16", "float16", "float32"],
        help="Dtype de computação do QLoRA. 'auto' (padrão) usa bfloat16 só "
        "em GPUs que o suportam nativamente (A100/L4+) e float16 nas demais, "
        "como a T4.",
    )
    parser.add_argument("--seed", type=int, default=defaults.seed)
    parser.add_argument(
        "--resume",
        nargs="?",
        const="auto",
        default=None,
        metavar="CHECKPOINT",
        help="Retoma um treino interrompido. Sem valor, usa o checkpoint mais "
        "recente em --output-dir; com um caminho, usa aquele checkpoint. "
        "Útil quando o Colab desconecta no meio do treino.",
    )
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Roda um teste mecânico rápido com um modelo minúsculo e "
        "inicializado aleatoriamente (sem download), só para validar que "
        "a pipeline (tokenização, LoRA, treino, salvamento) executa sem erro.",
    )
    return parser


#: Aliases de parâmetros que mudaram de nome entre versões de `trl` /
#: `transformers`. Chave = nome canônico usado neste projeto; valor = nomes
#: alternativos aceitos por outras versões, em ordem de preferência.
#:
#: Isto existe porque o treino roda no Colab/Kaggle, onde as versões das
#: bibliotecas mudam sem aviso e sem que possamos fixá-las. Sem essa
#: resolução, uma atualização do `trl` derruba o script com
#: `TypeError: unexpected keyword argument`, no meio de um notebook, depois
#: de já ter baixado 2 GB de pesos.
#:
#: Casos conhecidos:
#: - `max_length`: chamava-se `max_seq_length` em trl < 0.20.
#: - `warmup_ratio`: removido em transformers 5.x, onde `warmup_steps` aceita
#:   um float em [0, 1) com a mesma semântica de proporção.
#: - `eval_strategy`: chamava-se `evaluation_strategy` em transformers < 4.41.
PARAM_ALIASES: dict[str, tuple[str, ...]] = {
    "max_length": ("max_seq_length",),
    "warmup_ratio": ("warmup_steps",),
    "eval_strategy": ("evaluation_strategy",),
}


def _accepted_field_names(config_cls) -> set[str]:
    """Nomes de parâmetro que a classe de configuração aceita nesta versão."""
    import dataclasses
    import inspect

    if dataclasses.is_dataclass(config_cls):
        names = {f.name for f in dataclasses.fields(config_cls) if f.init}
        if names:
            return names

    # Fallback para versões que não expõem a config como dataclass.
    return set(inspect.signature(config_cls.__init__).parameters) - {"self", "kwargs"}


def resolve_config_kwargs(
    config_cls, desired: dict[str, Any]
) -> tuple[dict[str, Any], list[str]]:
    """Traduz `desired` para os nomes que `config_cls` aceita nesta versão.

    Devolve `(kwargs_resolvidos, avisos)`. Um parâmetro sem correspondência é
    descartado e reportado no aviso, em vez de derrubar o treino — perder um
    hiperparâmetro secundário é preferível a perder a execução inteira.
    """
    accepted = _accepted_field_names(config_cls)
    resolved: dict[str, Any] = {}
    warnings: list[str] = []

    for name, value in desired.items():
        if name in accepted:
            resolved[name] = value
            continue

        for alias in PARAM_ALIASES.get(name, ()):
            if alias in accepted:
                resolved[alias] = value
                warnings.append(f"'{name}' não existe nesta versão; usando '{alias}'.")
                break
        else:
            warnings.append(
                f"'{name}' não é aceito por {config_cls.__name__} nesta versão "
                "e foi ignorado."
            )

    return resolved, warnings


def build_sft_config(config_cls, desired: dict[str, Any]):
    kwargs, warnings = resolve_config_kwargs(config_cls, desired)
    for message in warnings:
        print(f"[compat] {message}")
    return config_cls(**kwargs)


#: Abaixo disto os adapters LoRA praticamente não saem da inicialização e a
#: comparação antes/depois do relatório não mostra diferença nenhuma.
MIN_USEFUL_OPTIMIZER_STEPS = 50


def check_accelerator() -> None:
    """Verifica se há um acelerador compatível antes de baixar o modelo.

    O QLoRA deste projeto depende de `bitsandbytes`, cujos kernels de
    quantização em 4 bits existem apenas para CUDA (e ROCm). Não há backend
    TPU/XLA. Num runtime TPU o treino falharia — ou cairia para CPU e levaria
    horas — então é melhor avisar antes de baixar gigabytes de pesos.
    """
    try:
        import torch
    except ImportError:
        return

    if torch.cuda.is_available():
        name = torch.cuda.get_device_name(0)
        total_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"[gpu] {name} ({total_gb:.1f} GB) | bf16 nativo: {_supports_bf16()}")
        return

    is_tpu = False
    try:
        import torch_xla  # noqa: F401

        is_tpu = True
    except ImportError:
        pass

    if is_tpu:
        print(
            "[aviso] Runtime TPU detectado. O QLoRA deste projeto usa "
            "bitsandbytes, que não tem backend para TPU/XLA — o treino vai "
            "falhar ou cair para CPU.\n"
            "        Troque para um runtime GPU (Runtime > Change runtime "
            "type > T4 GPU). A T4 é folgada para este modelo."
        )
    else:
        print(
            "[aviso] Nenhuma GPU detectada. O treino em CPU é inviável para "
            "este modelo.\n"
            "        No Colab: Runtime > Change runtime type > T4 GPU."
        )


def _supports_bf16() -> bool:
    """`True` se a GPU tem suporte nativo a bfloat16 (Ampere/Ada em diante)."""
    try:
        import torch

        return bool(torch.cuda.is_available() and torch.cuda.is_bf16_supported())
    except Exception:
        return False


def resolve_compute_dtype(configured: str):
    """Resolve o dtype de computação do QLoRA.

    Com "auto", escolhe bfloat16 se a GPU o suportar nativamente e float16
    caso contrário. Isso importa na prática: a T4 do Colab gratuito é Turing
    e não tem bf16 nativo — pedir bfloat16 ali resulta em erro ou em
    emulação muito lenta, enquanto fp16 roda a plena velocidade.
    """
    import torch

    if configured == "auto":
        chosen = "bfloat16" if _supports_bf16() else "float16"
        print(f"[dtype] compute dtype do QLoRA: {chosen} (auto)")
        return getattr(torch, chosen)

    if configured == "bfloat16" and not _supports_bf16():
        print(
            "[aviso] bfloat16 pedido explicitamente, mas esta GPU não o "
            "suporta nativamente. Isso tende a ser muito lento — considere "
            "'auto' ou 'float16'."
        )

    return getattr(torch, configured)


def _report_training_plan(config: TrainingConfig, n_train: int) -> None:
    """Imprime o plano de treino e alerta se ele for curto demais para
    produzir um modelo mensuravelmente diferente do base."""
    effective_batch = config.per_device_train_batch_size * config.gradient_accumulation_steps
    steps_per_epoch = max(1, n_train // effective_batch)
    total_steps = steps_per_epoch * config.num_train_epochs

    print(
        f"[plano] {n_train} exemplos | batch efetivo {effective_batch} | "
        f"{steps_per_epoch} passos/época × {config.num_train_epochs} épocas = "
        f"~{total_steps} atualizações de peso"
    )

    if total_steps < MIN_USEFUL_OPTIMIZER_STEPS:
        print(
            f"[aviso] Apenas ~{total_steps} atualizações de peso. Com tão poucos "
            "passos os adapters LoRA mal saem da inicialização e o modelo "
            "treinado tende a ficar indistinguível do base na avaliação.\n"
            "        Aumente --epochs ou reduza --grad-accum. Como o dataset é "
            "pequeno, treinar mais custa poucos minutos."
        )


def build_config_from_args(args: argparse.Namespace) -> TrainingConfig:
    return TrainingConfig(
        base_model=args.base_model,
        train_file=args.train_file,
        val_file=args.val_file,
        output_dir=args.output_dir,
        max_seq_length=args.max_seq_length,
        learning_rate=args.learning_rate,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        load_in_4bit=not args.no_4bit,
        bnb_compute_dtype=args.compute_dtype,
        seed=args.seed,
        lora=LoraParams(r=args.lora_r, lora_alpha=args.lora_alpha),
    )


def find_latest_checkpoint(output_dir: str | Path) -> Path | None:
    """Último checkpoint salvo em `output_dir`, se houver.

    O Trainer salva em subpastas `checkpoint-<passo global>`; ordenamos pelo
    número do passo, não alfabeticamente (senão `checkpoint-90` viria depois
    de `checkpoint-135`).
    """
    output_dir = Path(output_dir)
    if not output_dir.is_dir():
        return None

    checkpoints = []
    for path in output_dir.glob("checkpoint-*"):
        if not path.is_dir():
            continue
        suffix = path.name.removeprefix("checkpoint-")
        if suffix.isdigit():
            checkpoints.append((int(suffix), path))

    if not checkpoints:
        return None
    return max(checkpoints)[1]


def resolve_resume_target(config: TrainingConfig, resume: str | None) -> str | bool | None:
    """Traduz a opção `--resume` no valor esperado por `Trainer.train`.

    `--resume` sem valor procura o checkpoint mais recente; com um caminho,
    usa aquele checkpoint. Se nada for encontrado, avisa e treina do zero em
    vez de falhar — retomar é uma conveniência, não um pré-requisito.
    """
    if resume is None:
        return None

    if resume != "auto":
        path = Path(resume)
        if not path.is_dir():
            raise FileNotFoundError(f"Checkpoint não encontrado: {path}")
        print(f"[resume] Retomando de {path}")
        return str(path)

    latest = find_latest_checkpoint(config.output_dir)
    if latest is None:
        print(
            f"[resume] Nenhum checkpoint em {config.output_dir} — "
            "iniciando o treino do zero."
        )
        return None

    print(f"[resume] Retomando do checkpoint mais recente: {latest}")
    return str(latest)


def run_training(
    config: TrainingConfig, smoke_test: bool = False, resume: str | None = None
) -> dict:
    import torch
    from datasets import Dataset
    from peft import LoraConfig, get_peft_model
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        BitsAndBytesConfig,
        set_seed,
    )
    from trl import SFTConfig, SFTTrainer

    set_seed(config.seed)

    train_records = load_training_dataset(config.train_file)
    val_records = load_training_dataset(config.val_file) if Path(config.val_file).exists() else []

    resume_from = None if smoke_test else resolve_resume_target(config, resume)

    if not smoke_test:
        check_accelerator()
        _report_training_plan(config, len(train_records))

    if smoke_test:
        from transformers import GPT2Config, GPT2LMHeadModel, GPT2TokenizerFast
        from tokenizers import ByteLevelBPETokenizer

        train_records = train_records[:8] or [{"text": "<|user|>\nteste\n<|assistant|>\nresposta\n<|end|>"}]

        tmp_dir = Path(config.output_dir) / "_smoke_tokenizer"
        tmp_dir.mkdir(parents=True, exist_ok=True)
        bpe = ByteLevelBPETokenizer()
        bpe.train_from_iterator([r["text"] for r in train_records], vocab_size=512, min_frequency=1)
        bpe.save_model(str(tmp_dir))
        tokenizer = GPT2TokenizerFast(
            vocab_file=str(tmp_dir / "vocab.json"), merges_file=str(tmp_dir / "merges.txt")
        )
        tokenizer.pad_token = tokenizer.eos_token or "<|endoftext|>"

        model = GPT2LMHeadModel(
            GPT2Config(
                vocab_size=len(tokenizer),
                n_positions=128,
                n_embd=32,
                n_layer=2,
                n_head=2,
            )
        )
        quantization_config = None
    else:
        # Autentica antes de qualquer download, para não esbarrar no limite
        # de taxa anônimo no meio de um arquivo de pesos de vários GB.
        from finetuning.hf_auth import ensure_hf_login

        ensure_hf_login()

        tokenizer = AutoTokenizer.from_pretrained(config.base_model)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        quantization_config = (
            BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=resolve_compute_dtype(config.bnb_compute_dtype),
                bnb_4bit_use_double_quant=True,
            )
            if config.load_in_4bit
            else None
        )
        model = AutoModelForCausalLM.from_pretrained(
            config.base_model,
            quantization_config=quantization_config,
            device_map="auto",
        )

    lora_config = LoraConfig(
        r=config.lora.r,
        lora_alpha=config.lora.lora_alpha,
        lora_dropout=config.lora.lora_dropout,
        bias=config.lora.bias,
        target_modules=list(config.lora.target_modules) if not smoke_test else ["c_attn"],
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)

    train_dataset = Dataset.from_list(train_records)
    eval_dataset = Dataset.from_list(val_records) if val_records else None

    sft_config = build_sft_config(
        SFTConfig,
        {
            "output_dir": config.output_dir,
            "max_length": 32 if smoke_test else config.max_seq_length,
            "learning_rate": config.learning_rate,
            "num_train_epochs": 1 if smoke_test else config.num_train_epochs,
            "per_device_train_batch_size": 2 if smoke_test else config.per_device_train_batch_size,
            "gradient_accumulation_steps": 1 if smoke_test else config.gradient_accumulation_steps,
            "warmup_ratio": config.warmup_ratio,
            "weight_decay": config.weight_decay,
            "logging_steps": 1 if smoke_test else config.logging_steps,
            "save_strategy": "no" if smoke_test else config.save_strategy,
            "eval_strategy": "no"
            if (smoke_test or eval_dataset is None)
            else config.eval_strategy,
            "dataset_text_field": "text",
            "report_to": [],
            "seed": config.seed,
        },
    )

    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
    )

    train_result = trainer.train(resume_from_checkpoint=resume_from)

    Path(config.output_dir).mkdir(parents=True, exist_ok=True)
    trainer.save_model(config.output_dir)
    tokenizer.save_pretrained(config.output_dir)

    metrics = dict(train_result.metrics)
    if eval_dataset is not None:
        metrics.update({f"eval_{k}": v for k, v in trainer.evaluate().items()})

    hyperparams_path = Path(config.output_dir) / "hyperparameters.json"
    hyperparams_path.write_text(
        json.dumps({"config": config.hyperparameters_dict(), "metrics": metrics}, indent=2),
        encoding="utf-8",
    )

    return metrics


def main() -> None:
    args = _build_arg_parser().parse_args()
    config = build_config_from_args(args)
    metrics = run_training(config, smoke_test=args.smoke_test, resume=args.resume)
    print(json.dumps(metrics, indent=2, default=str))


if __name__ == "__main__":
    main()
