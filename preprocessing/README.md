# preprocessing/

Implementa a Etapa 1 do Tech Challenge: preparação dos dados.

| Script | Responsabilidade |
|---|---|
| `protocols_bank.py` | Banco de protocolos clínicos internos sintéticos (fonte de verdade para dataset e RAG). |
| `generate_synthetic_data.py` | Gera pares instrução/resposta e registros hospitalares sintéticos com PII. |
| `clean.py` | Limpeza de texto: normalização de encoding, remoção de ruído, deduplicação. |
| `anonymize.py` | Mascaramento de PII (CPF, RG, telefone, endereço, datas, prontuário, nome). |
| `curate.py` | Filtragem, balanceamento por especialidade e split train/val/test. |
| `run_pipeline.py` | Orquestra as quatro etapas acima de ponta a ponta. |

## Rodar tudo

```bash
python -m preprocessing.run_pipeline --seed 42
```

## Rodar uma etapa isolada

```bash
python -m preprocessing.generate_synthetic_data --out-dir data/raw --seed 42
python -m preprocessing.clean --in-file data/raw/qa_pairs.jsonl --out-file data/processed/qa_pairs.clean.jsonl
python -m preprocessing.anonymize --in-file data/processed/registros_hospitalares.clean.jsonl \
    --out-file data/processed/registros_hospitalares.anonimizado.jsonl --validate
python -m preprocessing.curate --in-file data/processed/qa_pairs.clean.jsonl --out-dir data/processed
```

## Critérios de curadoria

Documentados automaticamente em `data/processed/curation_report.md` a cada
execução (mínimo de tokens por resposta, teto de tokens por exemplo, fator de
balanceamento por especialidade, proporção do split). Ver também
`data/README.md` para o schema final do dataset.
