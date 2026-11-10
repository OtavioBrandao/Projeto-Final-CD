# Projeto Final CD — Reduzindo readmissões hospitalares em 30 dias

Projeto Final de **Ciência de Dados** (UFAL · Instituto de Computação · 2026.2).

## O problema

Hospitais são penalizados financeiramente quando pacientes voltam a ser internados em até 30 dias após a alta (readmissão precoce). Em pacientes diabéticos isso é ainda mais comum e caro. A pergunta de negócio:

> **Quais pacientes têm maior risco de readmissão em 30 dias, e que ação preventiva o hospital deve priorizar para reduzi-la?**

Usamos dados reais de 130 hospitais dos EUA (1999–2008) para (1) entender os padrões, (2) prever o risco e (3) recomendar ações concretas apoiadas em evidência.

## A base de dados

| Item | Valor |
|---|---|
| Nome | Diabetes 130-US Hospitals for Years 1999-2008 |
| Fonte | [UCI Machine Learning Repository (id 296)](https://archive.ics.uci.edu/dataset/296/diabetes+130-us+hospitals+for+years+1999-2008) |
| Tamanho | 101.766 internações · 50 colunas (brutas) → **69.970 linhas × 54 colunas** após o tratamento (Pessoa 1) |
| Variável-alvo | `readmitted` → binarizada em `readmit_30d` (readmitido em < 30 dias): 8,97% de positivos na base tratada |
| Licença | CC BY 4.0 |
| Referência | Strack et al. (2014), *BioMed Research International* |

Por que essa base, e como ela se compara às alternativas: [`docs/ESCOLHA_DA_BASE.md`](docs/ESCOLHA_DA_BASE.md).

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
│   ├── interim/      # etapas intermediárias do tratamento
│   └── processed/    # dataset final + dicionário de dados (versionado)
├── docs/             # explicação do tratamento (Pessoa 1): README, figuras, tabelas
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

python -m src.p1_engenharia_dados.baixar_base         # baixa a base para data/raw/
python -m src.p1_engenharia_dados.executar_pipeline   # refaz o tratamento (dados + docs/)
pytest                                                # testes do pipeline
```

O pipeline de tratamento (Pessoa 1) está especificado em [`docs/PLANO_IMPLEMENTACAO_ENGENHARIA_DE_DADOS.md`](docs/PLANO_IMPLEMENTACAO_ENGENHARIA_DE_DADOS.md) e explicado, com números e gráficos, em [`docs/README.md`](docs/README.md).

**Para as demais pessoas:** a base pronta para usar está em `data/processed/diabetes_modelagem.csv` (ou `.parquet`), com o dicionário em `data/processed/dicionario_de_dados.csv`. O alvo é `readmit_30d` (última coluna). Não há divisão treino/teste nem escalonamento/one-hot: isso é da Pessoa 3, dentro de um `Pipeline` do sklearn.

## Fluxo de trabalho (Git)

- `main` só recebe código revisado, via Pull Request.
- Uma branch por tarefa: `p1/limpeza`, `p2/eda`, `p3/modelos`...
- Commits curtos, no imperativo: `Adiciona tratamento de nulos`.
- Não versionar dados brutos. Dados tratados (`data/processed/`) são versionados para todos usarem a mesma versão.
- Quem consome dados de outra pessoa não altera o arquivo dela: abre uma issue ou PR.

## Cronograma e dependências

`Pessoa 1` (base limpa) → `Pessoa 2` (EDA) → `Pessoa 3` (modelos) → `Pessoa 4` (SHAP, insights, pitch) → `Pessoa 5` (artigo, ZIP, extra).
A Pessoa 1 destrava todo o projeto.
