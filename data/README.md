# Dados

Todo o conteúdo desta pasta é **sintético**, gerado por
`preprocessing/generate_synthetic_data.py` para simular o ambiente de um
hospital (protocolos internos, perguntas frequentes de médicos, registros de
prontuário). Nenhum dado real de paciente é usado neste projeto.

## Como gerar

```bash
python -m preprocessing.run_pipeline
```

Isso executa a Etapa 1 completa (geração -> limpeza -> anonimização ->
curadoria) e popula as pastas abaixo. Os arquivos gerados **não são
versionados no Git** (ver `.gitignore`); o pipeline é reprodutível a partir do
código com `--seed` fixo.

## Estrutura

```
data/
  raw/                            # saída bruta da geração sintética
    qa_pairs.jsonl                # pares instrução/resposta (sem PII)
    registros_hospitalares.jsonl  # registros simulados COM dados de identificação
  protocols/                      # protocolos internos em Markdown (fonte do RAG, Etapa 3)
    PROT-*.md
  processed/
    qa_pairs.clean.jsonl                    # após preprocessing.clean
    registros_hospitalares.clean.jsonl
    registros_hospitalares.anonimizado.jsonl # após preprocessing.anonymize
    train.jsonl / val.jsonl / test.jsonl     # splits finais para fine-tuning (Etapa 2)
    curation_report.md                       # critérios e estatísticas de curadoria
    manifest.json                            # resumo de todas as etapas do pipeline
```

## Schema de `train.jsonl` / `val.jsonl` / `test.jsonl`

```json
{
  "instruction": "Paciente com hipertensão, qual a conduta recomendada segundo o protocolo interno?",
  "input": "",
  "output": "Com base no protocolo interno PROT-CARD-001 (...)",
  "especialidade": "cardiologia",
  "fonte": "PROT-CARD-001",
  "tipo": "protocol_qa"
}
```

Esse é o formato consumido por `finetuning/train.py` (Etapa 2).

## Anonimização

`registros_hospitalares.jsonl` simula prontuários com nome, CPF, RG,
telefone, endereço, data de nascimento e número de prontuário. O script
`preprocessing/anonymize.py` mascara esses campos (regex + campos rotulados,
com camada opcional de NER via spaCy se disponível localmente) antes de
qualquer uso posterior. A validação de recall da anonimização (comparando
contra o valor original conhecido, já que os dados são sintéticos) é
registrada em `manifest.json`.
