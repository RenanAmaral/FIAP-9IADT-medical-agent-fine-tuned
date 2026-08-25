# Relatório de Curadoria do Dataset

## Critérios aplicados
- Resposta mínima: 15 tokens (descarta respostas incompletas).
- Tamanho máximo (instrução + resposta): 800 tokens.
- Balanceamento por especialidade: fator máximo 1.5x em relação à especialidade menos representada (undersampling).
- Split: 80% treino / 10% validação / 10% teste, estratificado por especialidade.

## Estatísticas de filtragem
- Registros de entrada: 92
- Descartados por resposta incompleta: 0
- Descartados por excesso de tokens: 0
- Restantes após filtragem: 92

## Balanceamento por especialidade
- infectologia: 9 -> 9
- pneumologia: 14 -> 10
- endocrinologia: 7 -> 7
- cardiologia: 21 -> 10
- neurologia: 10 -> 10
- ortopedia: 9 -> 9
- ginecologia: 11 -> 10
- pediatria: 11 -> 10

## Tamanho final dos splits
- train: 59
- val: 8
- test: 8
