# Modelagem: fatalidade em acidentes registrados pela PRF

Classificação binária de acidentes registrados em rodovias federais: `acidente_fatal = 1`
quando houve ao menos uma morte e `0` caso contrário. Os modelos estimam a chance de
fatalidade **condicional a um acidente registrado**, não a probabilidade de um acidente acontecer.

## Dados e cenários

A entrada é [acidentes_prf_modelagem.csv](../data/processed/acidentes_prf_modelagem.csv),
produzida pela engenharia de dados. São 73.006 acidentes de 2024 e 72.336 de 2025;
o teste contém 5.104 acidentes fatais (7,056%).

- **A — retrospectivo, 34 atributos:** localização, calendário, características do
  acidente, causa, veículos e condições da via. Parte dessas informações é observada
  após a ocorrência e não está disponível para uma previsão antecipada.
- **B — localização e calendário, 12 atributos:** dia da semana, UF, BR, km, latitude,
  longitude, hora, faixa horária, mês, fim de semana, feriado e região.

As listas ordenadas `FEATURES_A` e `FEATURES_B`, os cortes e os limiares estão em
[configuracao.py](../src/p3_modelagem/configuracao.py). O alvo, consequências como
mortos e feridos, identificadores, ano, atributos sensíveis dos condutores e
indicadores globais de outliers não entram como preditores.

O pré-processamento fica dentro dos pipelines: imputação numérica pela mediana,
One-Hot para categorias, ignorando categorias novas, e passagem direta de binários.
A regressão logística também padroniza as variáveis numéricas. Todas as transformações
aprendidas são ajustadas somente no treino de cada rodada.

## Separação temporal e modelos

2024 foi usado para desenvolver e selecionar os modelos. A validação temporal usa
três janelas expansivas, sem embaralhamento:

| Rodada | Treino em 2024 | Validação em 2024 |
|---|---|---|
| 1 | janeiro a junho | julho e agosto |
| 2 | janeiro a agosto | setembro e outubro |
| 3 | janeiro a outubro | novembro e dezembro |

Foram comparados Dummy com estratégia prior, regressão logística, Random Forest e
XGBoost nos dois cenários. Além das configurações padrão, foram avaliados pesos de
classe: `balanced` na regressão/Random Forest e negativos/positivos **do treino** no
XGBoost. São 42 ajustes ao todo. Não houve reamostragem nem busca adicional de hiperparâmetros.

Os parâmetros explícitos estão em [modelos.py](../src/p3_modelagem/modelos.py):
regressão com max_iter=2000; Random Forest com 100 árvores, profundidade 12 e
min_samples_leaf=3; XGBoost com 100 árvores, profundidade 4, learning_rate=0,1,
objective=binary:logistic, eval_metric=logloss, hist e CPU. A semente é 42;
Random Forest e XGBoost usam n_jobs=2.

O XGBoost padrão foi selecionado em ambos os cenários pela AP média das validações.
Cada pipeline definitivo foi treinado novamente com **todo 2024** e avaliado em 2025.
Os limiares abaixo foram escolhidos por maximização de F1 nas previsões de validação
de 2024 e mantidos no teste:

| Cenário | Limiar |
|---|---|
| A | 0.2066718339920044 |
| B | 0.101788230240345 |

A regra é `probabilidade >= limiar`. O baseline final foi treinado uma vez:
suas probabilidades constantes independem das features. As métricas dele estão
salvas; somente os XGBoost A/B precisam ser serializados.

## Métricas e resultados

A métrica principal é **Average Precision (AP)**, que resume o ranking na curva
Precision-Recall e é adequada à classe positiva pouco frequente. AP não é acurácia
nem precision em um limiar. Também foram calculados ROC-AUC, precision, recall, F1,
Brier Score, Log Loss, matriz de confusão e tempos de treino/inferência.
Brier e Log Loss menores indicam melhor qualidade probabilística.
Precision fica indefinida quando não há positivos previstos.

| Resultado | A | B | Dummy |
|---|---:|---:|---:|
| AP média das validações de 2024 | 0,378798 | 0,155628 | 0,070921 |
| AP no teste de 2025 | 0,377379 | 0,153435 | 0,070560 |
| ROC-AUC em 2025 | 0,850367 | 0,685863 | 0,500000 |
| Precision em 2025 | 0,400614 | 0,158239 | indefinida |
| Recall em 2025 | 0,434757 | 0,350705 | 0 |
| F1 em 2025 | 0,416988 | 0,218080 | 0 |
| Brier em 2025 | 0,053186 | 0,063204 | 0,065581 |
| Log Loss em 2025 | 0,192457 | 0,239816 | 0,255084 |

Precision, recall e F1 dos XGBoost usam os limiares selecionados em 2024; o Dummy usa
0,5. Os CSVs também trazem a comparação descritiva dos XGBoost em 0,5.

A AP caiu 0,001418 em A e 0,002194 em B em relação à média das validações de 2024.
No teste, A e B obtiveram respectivamente 5,35 e 2,17 vezes a AP do Dummy, igual à
prevalência de 2025. Os modelos não foram reajustados após esses resultados.

As validações de 2024 usaram tamanhos de treino diferentes; as médias entre cortes
não equivalem às métricas de todas as previsões reunidas. Precision/recall/F1 de
ajuste do limiar em 2024 são otimistas, pois foram medidos nas previsões usadas para
escolhê-lo. A avaliação temporal de 2025 é independente dessa escolha.
A diferença entre A/B ou entre anos não identifica causas.

## Resultados para o artigo e a apresentação

| Arquivo ou pasta | Uso |
|---|---|
| [p3_resumo_validacao_2024.csv](tabelas/p3_resumo_validacao_2024.csv) | Médias e dispersão dos modelos entre os três cortes |
| [p3_validacao_2024.csv](tabelas/p3_validacao_2024.csv) | Métricas individuais por rodada |
| [p3_previsoes_validacao_2024.csv.gz](tabelas/p3_previsoes_validacao_2024.csv.gz) | Previsões fora do treino para reproduzir curvas e limiares |
| [p3_metricas_globais_2024.csv](tabelas/p3_metricas_globais_2024.csv) | Métricas das previsões reunidas, distintas das médias por rodada |
| [p3_comparacao_limiares.csv](tabelas/p3_comparacao_limiares.csv) | Comparação entre 0,5 e limiar ajustado em 2024 |
| [p3_calibracao_2024.csv](tabelas/p3_calibracao_2024.csv) | Frequência observada por faixa de probabilidade |
| [metricas_2025.csv](p3_final/metricas_2025.csv) | Métricas finais e matrizes de confusão em colunas TN/FP/FN/TP |
| [previsoes_2025.csv.gz](p3_final/previsoes_2025.csv.gz) | Probabilidades e classes finais por observação |
| [comparacao_2024_2025.csv](p3_final/comparacao_2024_2025.csv) | Comparação entre validação e teste |
| [calibracao_2025.csv](p3_final/calibracao_2025.csv) | Probabilidades médias, frequências fatais e contagens por faixa |
| [figuras de 2024](figuras/) e [figuras de 2025](p3_final/figuras/) | AP, Precision-Recall, ROC, calibração e matrizes de confusão |

A calibração é aproximada, em faixas fixas de 0,1. Há diferenças entre probabilidade
média e frequência observada; faixas superiores têm poucas observações e são instáveis.
Não houve recalibração.

## Como executar

Execute os comandos na raiz do repositório, em um ambiente com as dependências de
[requirements.txt](../requirements.txt). Os modelos salvos foram produzidos com
Python 3.12.4, numpy 2.4.2, pandas 3.0.0, scikit-learn 1.7.2, XGBoost 3.4.1,
matplotlib 3.10.8 e joblib 1.5.2. Use versões compatíveis para carregá-los.

Os resultados já estão prontos. Para recriar **somente os gráficos a partir dos CSVs**,
sem treinamento ou inferência:

```bash
python -m src.p3_modelagem.executar_validacao --somente-relatorios
python -m src.p3_modelagem.executar_avaliacao_final --somente-relatorios
```

Para reproduzir os experimentos quando necessário, escreva em outras pastas:

```bash
python -m src.p3_modelagem.executar_validacao --saida reports/reproducao_2024
python -m src.p3_modelagem.executar_avaliacao_final --saida reports/reproducao_2025 --modelos models/reproducao
```

Esses dois últimos comandos treinam modelos; não são necessários para consumir os
resultados publicados. O comando final recusa sobrescrever modelos ou resultados existentes.
O argumento `--csv` permite indicar outra cópia da base com o mesmo esquema.
`--reports-2024` indica a pasta de validação para a comparação final, quando necessário.

Testes com dados sintéticos:

```bash
python -m pytest tests/test_modelagem_p3.py tests/test_modelos_p3.py tests/test_avaliacao_p3.py tests/test_avaliacao_final_p3.py -q
```

## Pessoa 4: modelos e pré-processamento para SHAP

Os pipelines completos estão em [models/](../models/). As interfaces
`carregar_pipeline` e `prever` continuam em
[executar_avaliacao_final.py](../src/p3_modelagem/executar_avaliacao_final.py).

```python
from src.p3_modelagem.configuracao import FEATURES_POR_CENARIO, LIMIARES
from src.p3_modelagem.dados import carregar_base, selecionar_xy
from src.p3_modelagem.executar_avaliacao_final import carregar_pipeline

cenario = "A"  # ou "B"
pipeline = carregar_pipeline(cenario)
features_originais = FEATURES_POR_CENARIO[cenario]
limiar = LIMIARES[cenario]
preprocessamento = pipeline.named_steps["preprocessamento"]
estimador = pipeline.named_steps["modelo"]
nomes_transformados = preprocessamento.get_feature_names_out()

base = carregar_base()
X, y = selecionar_xy(base.loc[base.ano.eq(2024)], cenario)
X_transformado = preprocessamento.transform(X)
```

As colunas de `X_transformado` correspondem, na mesma ordem, a `nomes_transformados`.
Use `transform` do objeto ajustado, sem novo `fit`. Carregue apenas arquivos joblib confiáveis.

## Pessoa 5: previsões e métricas

```python
import pandas as pd
from src.p3_modelagem.configuracao import LIMIARES

previsoes = pd.read_csv("reports/p3_final/previsoes_2025.csv.gz")
metricas = pd.read_csv("reports/p3_final/metricas_2025.csv")
previsoes_A = previsoes.loc[previsoes.modelo.eq("xgboost_A")]
```

`linha_original` é a posição original no CSV de entrada, começando em zero.
A chave é `modelo + linha_original`; `cenario` sozinho não distingue B do Dummy.
As colunas `y_real`, `probabilidade`, `limiar`, `previsao` e `previsao_05`
permitem integrar resultados sem nova inferência. Na tabela de métricas,
`tipo_limiar` distingue o limiar selecionado, 0,5 descritivo e baseline.

Para outras observações, `prever(pipeline, base, cenario)` devolve probabilidades
e classes pelo limiar selecionado, preservando o índice. A função consulta
`classes_ == 1` para extrair a probabilidade positiva. O `pipeline.predict`
usa a decisão padrão do estimador e não aplica esses limiares.

## Limitações

- 2025 foi reservado para a avaliação final da modelagem, mas já havia sido usado em
  auditorias e análises exploratórias. Portanto, não é um conjunto completamente cego.
- A engenharia de dados anterior foi produzida sobre 2024/2025 conjuntamente.
  Excluir indicadores globais de outliers não elimina a dependência desse processamento.
- A base contém somente acidentes registrados pela PRF, sem medida de exposição ao
  tráfego e sem situações em que não houve acidente.
- O cenário B não estima risco absoluto de acidente por trecho nem prevê sozinho
  demanda de fiscalização. O cenário A depende de informações posteriores ao acidente.
- A seleção de modelos e limiares em 2024 pode apresentar otimismo. Três cortes de um
  único ano não demonstram superioridade estatística ou generalização ampla.
- Não foram feitos testes independentes de generalização espacial nem análise causal.
  Associações preditivas não demonstram efeito de intervenções de segurança viária.
