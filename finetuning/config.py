"""Configuração e contrato de prompt do fine-tuning (Etapa 2).

Este módulo é o "contrato de dados" combinado entre as duas frentes do
projeto (sugestão do enunciado): define o formato exato de entrada/saída da
LLM. `assistant/` (Etapa 3) importa `SYSTEM_PROMPT` e `format_prompt` daqui
para garantir que o assistente conversa com o modelo fine-tuned exatamente
no formato em que ele foi treinado.
"""

from __future__ import annotations

from dataclasses import dataclass, field

SYSTEM_PROMPT = (
    "Você é um assistente virtual médico de apoio a profissionais de saúde "
    "de um hospital. Seu papel é auxiliar em condutas clínicas e responder "
    "dúvidas com base nos protocolos internos e nos dados do paciente "
    "fornecidos.\n"
    "Regras obrigatórias:\n"
    "1. Nunca prescreva medicação ou tratamento de forma direta e definitiva; "
    "sempre apresente como sugestão condicionada à validação de um médico.\n"
    "2. Sempre indique a necessidade de validação humana antes de qualquer "
    "conduta ser executada.\n"
    "3. Recuse educadamente perguntas fora do escopo dos protocolos internos "
    "do hospital.\n"
    "4. Sempre que possível, cite a fonte (protocolo, documento ou registro) "
    "que embasou sua resposta."
)

PROMPT_TEMPLATE = (
    "<|system|>\n{system_prompt}\n"
    "<|user|>\n{instruction}{input_block}\n"
    "<|assistant|>\n{output}"
)


def format_prompt(instruction: str, input_text: str = "", output: str = "", system_prompt: str = SYSTEM_PROMPT) -> str:
    """Monta o prompt no formato usado tanto no treino (Etapa 2) quanto na
    inferência (Etapa 3), garantindo consistência entre as duas etapas.

    Quando `output` é vazio, o prompt termina logo após a tag
    `<|assistant|>`, pronto para geração.
    """
    input_block = f"\n{input_text}" if input_text else ""
    return PROMPT_TEMPLATE.format(
        system_prompt=system_prompt,
        instruction=instruction.strip(),
        input_block=input_block,
        output=output.strip(),
    ).rstrip() + ("" if not output else "\n<|end|>")


@dataclass
class LoraParams:
    r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    bias: str = "none"
    target_modules: tuple[str, ...] = (
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    )


@dataclass
class TrainingConfig:
    # Modelo base pequeno o suficiente para GPU única (T4/Colab), conforme
    # recomendado no enunciado para times com recursos limitados. Trocável
    # via --base-model por LLaMA 3 8B, Mistral 7B etc. quando houver GPU maior.
    base_model: str = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

    train_file: str = "data/processed/train.jsonl"
    val_file: str = "data/processed/val.jsonl"
    output_dir: str = "finetuning/adapters/medical-assistant-lora"

    max_seq_length: int = 1024
    learning_rate: float = 2e-4

    # O dataset é pequeno (~59 exemplos de treino). Com batch efetivo 16 e 3
    # épocas seriam apenas ~9 atualizações de peso — os adapters LoRA mal
    # sairiam da inicialização e a comparação antes/depois não mostraria
    # diferença alguma. Com batch efetivo 4 e 12 épocas são ~180 passos, que
    # é o suficiente para o modelo aprender o formato de resposta
    # institucional, e ainda assim o treino leva poucos minutos numa T4.
    num_train_epochs: int = 12
    per_device_train_batch_size: int = 4
    gradient_accumulation_steps: int = 1
    warmup_ratio: float = 0.03
    weight_decay: float = 0.01
    logging_steps: int = 10
    save_strategy: str = "epoch"
    eval_strategy: str = "epoch"

    load_in_4bit: bool = True

    # "auto" resolve para bfloat16 em GPUs que o suportam nativamente
    # (Ampere/Ada em diante: A100, L4, RTX 30xx+) e para float16 nas demais.
    # A T4 do Colab gratuito é Turing e NÃO tem bf16 nativo — fixar
    # "bfloat16" ali causa erro ou emulação muito lenta. Ver
    # `resolve_compute_dtype` em train.py.
    bnb_compute_dtype: str = "auto"

    seed: int = 42

    lora: LoraParams = field(default_factory=LoraParams)

    def hyperparameters_dict(self) -> dict:
        d = {k: v for k, v in self.__dict__.items() if k != "lora"}
        d["lora"] = self.lora.__dict__
        return d
