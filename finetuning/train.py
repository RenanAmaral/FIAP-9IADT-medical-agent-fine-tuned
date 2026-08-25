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
from dataclasses import asdict
from pathlib import Path

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
    parser.add_argument("--seed", type=int, default=defaults.seed)
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Roda um teste mecânico rápido com um modelo minúsculo e "
        "inicializado aleatoriamente (sem download), só para validar que "
        "a pipeline (tokenização, LoRA, treino, salvamento) executa sem erro.",
    )
    return parser


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
        seed=args.seed,
        lora=LoraParams(r=args.lora_r, lora_alpha=args.lora_alpha),
    )


def run_training(config: TrainingConfig, smoke_test: bool = False) -> dict:
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
        tokenizer = AutoTokenizer.from_pretrained(config.base_model)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        quantization_config = (
            BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=getattr(torch, config.bnb_compute_dtype),
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

    sft_config = SFTConfig(
        output_dir=config.output_dir,
        max_seq_length=32 if smoke_test else config.max_seq_length,
        learning_rate=config.learning_rate,
        num_train_epochs=1 if smoke_test else config.num_train_epochs,
        per_device_train_batch_size=2 if smoke_test else config.per_device_train_batch_size,
        gradient_accumulation_steps=1 if smoke_test else config.gradient_accumulation_steps,
        warmup_ratio=config.warmup_ratio,
        weight_decay=config.weight_decay,
        logging_steps=1 if smoke_test else config.logging_steps,
        save_strategy="no" if smoke_test else config.save_strategy,
        eval_strategy="no" if (smoke_test or eval_dataset is None) else config.eval_strategy,
        dataset_text_field="text",
        report_to=[],
        seed=config.seed,
    )

    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
    )

    train_result = trainer.train()

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
    metrics = run_training(config, smoke_test=args.smoke_test)
    print(json.dumps(metrics, indent=2, default=str))


if __name__ == "__main__":
    main()
