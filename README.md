# Projeto Final CD — Reduzindo acidentes fatais em rodovias federais

Projeto Final de **Ciência de Dados** (UFAL · Instituto de Computação · 2026.2).

## O problema

Todo ano, milhares de pessoas morrem em acidentes nas rodovias federais brasileiras. A Polícia Rodoviária Federal (PRF), as concessionárias e as seguradoras têm recursos limitados de fiscalização, sinalização e obras. A pergunta de negócio:

> **Quais combinações de condições (causa, tipo de acidente, horário, clima, traçado da via, trecho, veículos envolvidos) mais elevam o risco de um acidente ser fatal, e onde e quando concentrar fiscalização, sinalização e campanhas para reduzir mortes?**

Usamos dados reais e recentes (2024 e 2025) dos Dados Abertos da PRF para (1) entender os padrões, (2) prever o risco e (3) recomendar ações concretas apoiadas em evidência.

## A base de dados

| Item | Valor |
|---|---|
| Nome | Acidentes em rodovias federais (BAT, Boletim de Acidente de Trânsito) |
| Fonte | [Dados Abertos da PRF](https://www.gov.br/prf/pt-br/acesso-a-informacao/dados-abertos/dados-abertos-da-prf) |
| Período | 2024 e 2025 (anos completos) |
| Tamanho bruto | 145.731 acidentes · 30 colunas (+ 391.077 linhas de pessoas/veículos em outro arquivo) |
| Tamanho tratado | **145.342 linhas × 42 colunas** |
| Variável-alvo | `acidente_fatal` (1 se houve ao menos uma morte): 7,05% de positivos |

Por que essa base, como ela se compara às alternativas e como ela atende às regras do projeto: [`docs/ESCOLHA_DA_BASE.md`](docs/ESCOLHA_DA_BASE.md).

## Equipe e pastas

| Pessoa | Papel | Pasta principal | Entrega |
|---|---|---|---|
| 1 | Engenharia de dados e pré-processamento | `src/p1_engenharia_dados/`, `docs/` | Dataset limpo + seção "Base de Dados" |
| 2 | EDA e estatística | `src/p2_eda/`, `notebooks/` | Gráficos + seção "Estatística Descritiva e Inferência" |
| 3 | Modelagem preditiva | `src/p3_modelagem/` | Modelos + seções de Métodos/Métricas/Avaliação |
| 4 | Explicabilidade, negócios e pitch | `src/p4_explicabilidade/`, `apresentacao/` | SHAP + 3 insights + pitch |
| 5 | Integração, SBC e extra | `src/p5_integracao/`, `artigo/` | Artigo (≤ 12 págs), ZIP, dashboard |

Cada pasta de `src/` tem um `LEIA-ME.md` com entrada e saída esperadas.

## Estrutura

```
Projeto-Final-CD/
├── data/
│   ├── raw/          # bruto (baixado por script, NÃO versionado)
│   ├── interim/      # base legível gerada pelo pipeline (NÃO versionada: é grande)
│   └── processed/    # base de modelagem + dicionário de dados (versionado)
├── docs/             # engenharia de dados: README, plano, escolha da base, figuras, tabelas
├── notebooks/        # exploração (um notebook por pessoa: p2_..., p3_...)
├── reports/figuras/  # figuras finais usadas no artigo e nos slides
├── src/              # código por pessoa (p1_ ... p5_)
├── tests/            # testes automatizados do pipeline
├── artigo/           # template SBC
└── apresentacao/     # slides
```

## Como rodar

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt

python -m src.p1_engenharia_dados.baixar_base         # baixa os 4 arquivos da PRF para data/raw/
python -m src.p1_engenharia_dados.executar_pipeline   # refaz o tratamento (dados + docs/)
pytest                                                # testes do pipeline
```

O tratamento de dados está especificado em [`docs/PLANO_IMPLEMENTACAO_ENGENHARIA_DE_DADOS.md`](docs/PLANO_IMPLEMENTACAO_ENGENHARIA_DE_DADOS.md) e explicado, com números e gráficos, em [`docs/README.md`](docs/README.md) (README da parte de Engenharia de Dados e Pré-processamento).

**Para as demais pessoas:** a base pronta para usar está em `data/processed/acidentes_prf_modelagem.csv` (ou `.parquet`), com o dicionário em `data/processed/dicionario_de_dados.csv`. O alvo é `acidente_fatal` (última coluna). Não há divisão treino/teste nem escalonamento/one-hot/imputação: isso é da Pessoa 3, dentro de um `Pipeline` do sklearn.

## Fluxo de trabalho (Git)

- `main` só recebe código revisado, via Pull Request.
- Uma branch por tarefa: `p1/limpeza`, `p2/eda`, `p3/modelos`...
- Commits curtos, no imperativo: `Adiciona tratamento de nulos`.
- Não versionar dados brutos. Dados tratados (`data/processed/`) são versionados para todos usarem a mesma versão.
- Quem consome dados de outra pessoa não altera o arquivo dela: abre uma issue ou PR.

## Cronograma e dependências

`Pessoa 1` (base limpa) → `Pessoa 2` (EDA) → `Pessoa 3` (modelos) → `Pessoa 4` (SHAP, insights, pitch) → `Pessoa 5` (artigo, ZIP, extra).
A Pessoa 1 destrava todo o projeto.
