# finetuning/

Implementa a Etapa 2 do Tech Challenge: fine-tuning da LLM com LoRA/QLoRA.

## Por que isso não roda no ambiente de desenvolvimento deste repositório

Este projeto foi desenvolvido em um ambiente sandbox **sem GPU e sem acesso
à Hugging Face Hub** (rede corporativa restrita). Fine-tuning real de uma
LLM exige as duas coisas. Por isso:

- O código em `train.py` e `evaluate.py` é completo e correto, mas os
  imports pesados (`torch`, `transformers`, `peft`, `trl`, `bitsandbytes`)
  são carregados de forma tardia (lazy), então `finetuning.config` e
  `finetuning.dataset` (contrato de prompt + carregamento do dataset)
  puderam ser desenvolvidos e testados com `pytest` sem GPU.
- O treinamento real deve ser feito no **Google Colab** ou **Kaggle**
  (sugestão do próprio enunciado), seguindo os passos abaixo.

## Qual runtime usar: GPU, não TPU

**Use `T4 GPU`.** O runtime TPU (v5e-1 etc.) do Colab não serve para este
projeto, e a razão é de software, não de potência.

O QLoRA aqui depende do `bitsandbytes` para a quantização em 4 bits, e os
kernels dele existem apenas para CUDA e ROCm — o pacote não tem backend
TPU/XLA. Num runtime TPU o treino falha, ou cai para CPU e leva horas.
`check_accelerator()` detecta isso e avisa antes de baixar o modelo.

Mesmo desativando a quantização (`--no-4bit`), a TPU não compensaria: o
modelo tem 1,1 B de parâmetros e o treino são ~168 passos sobre 59 exemplos.
A vantagem da TPU aparece em lotes grandes com formas fixas; aqui, o custo de
compilação XLA — que recompila a cada novo formato de sequência — dominaria um
treino que na T4 leva poucos minutos.

| Runtime | Serve? | Observação |
|---|---|---|
| **T4 GPU** | ✅ | Recomendado. 16 GB, folgado para este modelo. |
| L4 / A100 | ✅ | Mais rápido (e com bf16 nativo), mas exige Colab Pro. |
| v5e-1 TPU | ❌ | `bitsandbytes` não tem backend TPU/XLA. |
| CPU | ❌ | Inviável para treino. |

### Precisão numérica na T4

A T4 é Turing e **não tem suporte nativo a bfloat16** — isso só aparece a
partir de Ampere (A100) e Ada (L4). Por isso `bnb_compute_dtype` tem o
padrão `"auto"`, que resolve para `bfloat16` em GPUs que o suportam e
`float16` nas demais. Fixar `bfloat16` na T4 causaria erro ou emulação muito
lenta. Para forçar: `--compute-dtype float16`.

Consumo de memória na T4 (16 GB) com os padrões:

| Item | Aproximado |
|---|---|
| TinyLlama 1.1B em 4-bit NF4 | ~0,7 GB |
| Adapters LoRA (r=16) | poucas dezenas de MB |
| Ativações (batch 4 × 1024 tokens) | ~2–3 GB |

Sobra folga suficiente para subir o batch ou trocar por um modelo maior.

## Se o Colab desconectar no meio do treino

O Colab derruba a sessão por inatividade ou por limite de uso. Quando a
máquina é reciclada, **tudo em `/content/` é perdido** — inclusive os
checkpoints.

### Retomar

O treino salva um checkpoint por época (`save_strategy="epoch"`), em
`<output-dir>/checkpoint-<passo>`. Para continuar de onde parou:

```bash
python -m finetuning.train --resume
```

Sem valor, `--resume` procura o checkpoint mais recente em `--output-dir`
(comparando o número do passo, não a ordem alfabética — `checkpoint-135` é
mais recente que `checkpoint-90`). Também aceita um caminho explícito:

```bash
python -m finetuning.train --resume finetuning/adapters/medical-assistant-lora/checkpoint-135
```

Se não houver checkpoint nenhum, o script avisa e treina do zero, em vez de
falhar. Um caminho explícito inexistente, ao contrário, levanta erro — ali o
silêncio esconderia um engano do usuário.

**Primeiro confirme que os arquivos sobreviveram:**

```bash
ls finetuning/adapters/medical-assistant-lora/
```

Se a máquina foi reciclada, a pasta não existe mais e não há o que retomar.

### O caminho tem que bater entre treino e avaliação

Se você treinou com `--output-dir` apontando para o Drive, **use o mesmo
caminho** em `--adapter-dir` na avaliação e no assistente:

```bash
OUT='/content/drive/MyDrive/tech-challenge-fase3/medical-assistant-lora'

python -m finetuning.train    --output-dir  "$OUT" --resume
python -m finetuning.evaluate --adapter-dir "$OUT" --base-model TinyLlama/TinyLlama-1.1B-Chat-v1.0
python -m graphs.cli --backend finetuned --adapter-dir "$OUT" --paciente PAC-0003 --pergunta "Qual a conduta?"
```

`resolve_adapter_dir` (em `finetuning/paths.py`) valida esse caminho antes de
carregar qualquer modelo e explica o que está errado. Sem essa validação, um
caminho inexistente era repassado ao PEFT, que o interpretava como
identificador de repositório da Hugging Face e devolvia
`HFValidationError: Repo id must be in the form 'repo_name' or
'namespace/repo_name'` — uma mensagem que não diz nada sobre a causa real.
Pior: o erro só aparecia **depois** de carregar o modelo base e gerar todas as
respostas dele, desperdiçando minutos.

A validação também cobre o caso de um treino interrompido: se o diretório não
tem `adapter_config.json` mas contém checkpoints, o checkpoint mais recente é
usado automaticamente, com aviso.

### Evitar o problema: salvar no Drive

A proteção real é gravar os checkpoints no Google Drive, que persiste entre
sessões:

```python
from google.colab import drive
drive.mount('/content/drive')
OUTPUT_DIR = '/content/drive/MyDrive/tech-challenge-fase3/medical-assistant-lora'
```

```bash
python -m finetuning.train --output-dir "$OUTPUT_DIR" --resume
```

Com isso, uma desconexão custa no máximo a época em andamento. O notebook já
vem com essas células.

## Conflito com o torchao pré-instalado no Colab

O Colab traz `torchao 0.10` pré-instalado. O PEFT recente exige `>= 0.16` e
sua função `is_torchao_available()` **levanta `ImportError`** ao encontrar uma
versão anterior, em vez de retornar `False`. Como este projeto não usa
torchao para nada, a saída é removê-lo:

```bash
pip uninstall -y torchao
```

O detalhe curioso é que **o treino não é afetado**: com o modelo em 4 bits, o
`dispatch_bnb_4bit` do PEFT casa antes e a cadeia de dispatchers nunca alcança
o `dispatch_torchao`. O erro só aparece na avaliação e na inferência, que
carregam o modelo sem quantização.

`check_torchao_conflict()` (em `finetuning/environment.py`) detecta a
incompatibilidade **antes** de carregar qualquer modelo e explica a correção,
em vez de deixar o processo estourar depois de vários minutos de carregamento
e geração. A célula de setup do notebook já remove o pacote.

## Token da Hugging Face

Sem um token, os downloads da Hub são anônimos e compartilham o limite de taxa
do IP — no Colab, isso significa dividir a cota com todos os outros notebooks
que saem pelo mesmo IP. É a origem do aviso:

```
Warning: You are sending unauthenticated requests to the HF Hub.
```

O download de modelos públicos ainda funciona, mas fica mais lento e pode
falhar com HTTP 429 no meio de um arquivo de vários GB. Para modelos de
licença restrita (Llama 3, Gemma), o token é **obrigatório**.

Crie um token de leitura em https://huggingface.co/settings/tokens e:

- **No Colab:** abra **Secrets** (🔑), adicione `HF_TOKEN` e ative
  *Notebook access*.
- **Localmente:** `export HF_TOKEN=hf_xxx`

`finetuning/hf_auth.py` detecta o token automaticamente (variável de
ambiente, secrets do Colab ou login já persistido) e é chamado por `train.py`,
`evaluate.py` e `assistant/llm.py` antes de qualquer download. Sem token, o
fluxo continua — apenas com o aviso e sujeito ao limite de taxa.

## Como rodar no Google Colab

1. Abra um notebook com GPU (Runtime > Change runtime type > GPU, T4 é
   suficiente para os modelos sugeridos).
2. Clone o repositório e instale as dependências:
   ```bash
   !git clone <url-do-repo> && cd FIAP-9IADT-medical-agent-fine-tuned
   !pip install -r requirements.txt
   ```
3. Gere o dataset (ou use o já versionado em `data/processed/`):
   ```bash
   !python -m preprocessing.run_pipeline
   ```
4. Treine:
   ```bash
   !python -m finetuning.train --base-model TinyLlama/TinyLlama-1.1B-Chat-v1.0
   ```
5. Avalie (métricas quantitativas + comparação antes/depois):
   ```bash
   !python -m finetuning.evaluate \
       --base-model TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
       --adapter-dir finetuning/adapters/medical-assistant-lora \
       --test-file data/processed/test.jsonl
   ```
6. Baixe a pasta `finetuning/adapters/medical-assistant-lora/` (contém os
   adapters LoRA + `hyperparameters.json`) e `finetuning/eval_results/`
   (métricas + comparação qualitativa) para usar no relatório técnico e
   trocar o modelo base pelo fine-tuned no assistente (Etapa 3).

Um notebook pronto com essas células está em
`finetuning/notebooks/colab_finetune.ipynb`.

### Atualizando um notebook já aberto no Colab

São **duas** coisas independentes, e confundi-las é a causa mais comum de
"corrigi o bug mas o erro continua":

| O que | Como atualizar |
|---|---|
| **O notebook** (`.ipynb`) | O Colab carregou um snapshot do GitHub e não o atualiza sozinho. *File > Open notebook > GitHub* e abra o arquivo novamente. Cópias salvas no Drive são arquivos separados e não recebem as atualizações. |
| **O código clonado em `/content/`** | É o que de fato roda. A célula de clone do notebook é idempotente: se o diretório já existe, ela faz `git pull` em vez de falhar. Basta executá-la de novo. |

Rodar `!git clone` uma segunda vez falha com *"destination path already
exists"* — por isso a célula testa a existência do diretório antes. Ela também
imprime o commit atual (`git log --oneline -1`), útil para confirmar que a
versão carregada é a esperada.

Em caso de dúvida, *Runtime > Disconnect and delete runtime* e rodar tudo do
zero recria a máquina limpa, com clone atualizado.

## Modelo base

Padrão: `TinyLlama/TinyLlama-1.1B-Chat-v1.0` — recomendado no próprio
enunciado para times com recursos limitados (treina em GPU única, T4 do
Colab gratuito). Trocável via `--base-model` por `meta-llama/Meta-Llama-3-8B`,
`mistralai/Mistral-7B-v0.1`, `microsoft/Phi-3-mini-4k-instruct` etc.,
dependendo da GPU disponível.

## Hiperparâmetros (padrão, ajustáveis via CLI)

| Parâmetro | Padrão | Flag |
|---|---|---|
| Learning rate | 2e-4 | `--learning-rate` |
| Épocas | 12 | `--epochs` |
| Batch size (por device) | 4 | `--batch-size` |
| Gradient accumulation | 1 | `--grad-accum` |
| LoRA r | 16 | `--lora-r` |
| LoRA alpha | 32 | `--lora-alpha` |
| Quantização | 4-bit (QLoRA) | `--no-4bit` desativa |

### Por que 12 épocas e não 3

O dataset tem ~59 exemplos de treino. Com o padrão inicial (3 épocas, batch
efetivo 16) o treino rendia apenas **~9 atualizações de peso** — os adapters
LoRA mal saíam da inicialização e o modelo treinado ficaria indistinguível do
base na avaliação, esvaziando a comparação antes/depois que o relatório
precisa mostrar.

Com batch efetivo 4 e 12 épocas são **~180 atualizações**. O script imprime o
plano antes de começar e avisa se o número de passos cair abaixo do útil:

```
[plano] 59 exemplos | batch efetivo 4 | 15 passos/época × 12 épocas = ~180 atualizações de peso
```

### Tempo real de treino

Medido numa T4 do Colab: **~10 s por passo**, ou seja **~30 minutos** para os
180 passos. Some o download do modelo e a avaliação e o notebook completo fica
em torno de **40 minutos**.

Vale saber que o modelo converge bem antes do fim. Numa execução real, na
**época 9** a perda de validação já estava em 0,0146 com 99,6% de acurácia por
token — o modelo memorizou os 59 exemplos, o que é esperado num dataset
pequeno. As últimas épocas não acrescentam nada mensurável: se o treino for
interrompido depois da época 8 ou 9, o checkpoint correspondente já serve para
a avaliação e a demonstração.

Todos os hiperparâmetros efetivamente usados e as métricas de treino/
avaliação são salvos em `finetuning/adapters/<nome>/hyperparameters.json`
ao final do treino (item 7 do passo a passo).

## Compatibilidade entre versões de `trl` / `transformers`

O treino roda no Colab/Kaggle, onde as versões das bibliotecas mudam sem aviso
e sem que possamos fixá-las. Alguns parâmetros foram renomeados entre versões,
e passar o nome errado derruba o script **depois** de já ter baixado
gigabytes de pesos:

| Nome canônico no projeto | Nome alternativo | Onde mudou |
|---|---|---|
| `max_length` | `max_seq_length` | `SFTConfig`, trl < 0.20 |
| `warmup_ratio` | `warmup_steps` (aceita float como proporção) | transformers 5.x |
| `eval_strategy` | `evaluation_strategy` | transformers < 4.41 |
| `dtype` | `torch_dtype` | transformers 5.0 |

`resolve_config_kwargs` (em `train.py`) e `dtype_kwarg` (em `hf_auth.py`)
resolvem esses nomes **em runtime**, inspecionando o que a versão instalada
aceita. Um parâmetro sem correspondência é descartado com aviso em vez de
interromper o treino — perder um hiperparâmetro secundário é preferível a
perder a execução inteira.

O caso do `dtype` merece atenção especial: como `from_pretrained` recebe
`**kwargs`, passar o nome errado **não levanta erro** — em transformers 4.x um
`dtype=` seria simplesmente ignorado e o modelo carregaria em fp32, o dobro da
memória e cerca de metade da velocidade, sem nenhum sinal de que algo deu
errado. Por isso a escolha é feita pelo número de versão, não por tentativa.

`tests/test_finetuning_compat.py` cobre as duas convenções de API.

## Contrato de prompt (`finetuning/config.py`)

`SYSTEM_PROMPT` e `format_prompt()` definem o formato exato usado tanto no
treino quanto na inferência — é o ponto de integração combinado entre a
Etapa 2 (fine-tuning) e a Etapa 3 (assistente). Qualquer mudança nesse
formato deve ser feita aqui e vale para as duas frentes.

## Smoke test (validação mecânica da pipeline, sem GPU/internet)

```bash
pip install torch transformers peft trl tokenizers
python -m finetuning.train --smoke-test
```

Isso treina um modelo GPT-2 minúsculo, **inicializado aleatoriamente e com
tokenizer treinado localmente** (nenhum download da internet), por uma
época em poucos exemplos, só para validar que a pipeline de tokenização +
LoRA + `SFTTrainer` + salvamento executa sem erros antes de gastar
tempo/GPU real no Colab. Não produz um modelo útil.

## Avaliação (item 7 e caixa "Avaliação do modelo" do enunciado)

`finetuning/evaluate.py` gera, para o mesmo conjunto de perguntas de
`data/processed/test.jsonl`:

- **Perplexidade** do modelo base vs. fine-tuned.
- **ROUGE-1 / ROUGE-L** das respostas geradas contra a resposta de
  referência (o texto do protocolo).
- **Comparação qualitativa lado a lado** (pergunta, referência, resposta
  base, resposta fine-tuned) para um subconjunto de exemplos.

Saída em `finetuning/eval_results/evaluation_report.md` e
`evaluation_results.json` — usados diretamente na seção de avaliação do
relatório técnico (`docs/relatorio_tecnico.md`).
