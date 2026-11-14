# Rascunho para o artigo SBC

> Resultados do SHAP: `reports/p4/tabelas/p4_shap_*.csv`. Os demais números vêm de `reports/p4/tabelas/` e de `reports/README_modelagem.md`.

## Resultados e Discussão

**Desempenho preditivo.** Treinados em 2024 e avaliados em 2025, os modelos XGBoost atingiram AP de 0,377 (cenário A, retrospectivo)
e 0,153 (cenário B, localização e calendário), contra 0,071 do baseline, que equivale à prevalência de fatais (7,06%); o ROC-AUC foi de 0,850 e 0,686.
Como a classe positiva é rara, interpretamos o desempenho também pela concentração de risco (Figura X): ao priorizar os 10% de acidentes
com maior risco previsto, o cenário A cobre 50,7% dos acidentes fatais (lift de 5,1) e o cenário B, 26,3% (lift de 2,6).
O cenário B é, portanto, útil para ordenar prioridades, mas modesto; sua utilidade não deve ser superestimada.

**Explicabilidade.** Para entender o que sustenta essas previsões, calculamos valores SHAP (TreeSHAP exato, calculado pelo próprio XGBoost) sobre uma amostra de 10.000 acidentes de 2025,
somando as colunas One-Hot de cada variável original. Os valores estão em log-odds. No cenário A, o tipo de acidente é a variável de maior peso (média do |SHAP| de 0,50; 18% do total), seguido de caminhão envolvido (0,31), área urbana (0,26), número de veículos (0,24) e motocicleta (0,23). Por tema, veículos e pessoas somam 34% da importância, tipo e causa do acidente 22%, localização 20%, horário 11% e via 10%; clima e calendário pesam pouco (1% e 2%). No cenário B, a localização responde por 64% (latitude 29%, longitude 12%, além de km, BR e UF) e o horário por 30% (a hora sozinha, 24%); o calendário pesa 6%, sobretudo fim de semana, e feriado praticamente não pesa. Dos nove sinais da EDA testados no cenário A, oito concordam em direção com o SHAP. Os multiplicadores de odds foram 5,5 para colisão frontal, 2,0 para caminhão, 1,7 para trecho rural, 1,6 para madrugada ou noite, 1,6 para Norte/Nordeste, 1,5 para pedestre e 1,3 para pista simples; chuva ficou em 0,90, abaixo de 1, como na EDA. No cenário B, madrugada ou noite (1,8) e Norte/Nordeste (1,8) também se confirmam. Os efeitos de pista simples e pedestre são menores que seus riscos relativos porque parte do efeito é atribuída a outras variáveis (colisão frontal, área rural e o tipo atropelamento). A exceção é a motocicleta (multiplicador de 1,58, contra RR bruto de 0,79), discutida adiante.
O SHAP descreve o que o modelo aprendeu e não implica causalidade.

**Padrões e implicações.** Três padrões sustentam recomendações práticas. (i) Acidentes em pista simples, à noite e no Norte/Nordeste
representam 7,1% dos acidentes e 19,4% dos acidentes fatais, com letalidade de 19,3% (RR 3,16), estável em 2024 e 2025. (ii) Colisões frontais em
pista simples somam 5,9% dos acidentes e 26,1% dos acidentes fatais, com letalidade de 31,4%. (iii) Acidentes com pedestres são 4,2% do total e 16,9% dos acidentes fatais,
com letalidade de 28,0%, chegando a 36,7% em trechos rurais. Em contraste, chuva e feriado não elevaram a letalidade por acidente (RR de 0,87 e 1,09), sugerindo menor retorno de campanhas orientadas apenas por esses fatores.
Já a motocicleta ilustra a diferença entre efeito bruto e controlado: seu RR bruto é 0,79, mas sobe para 1,06, 1,26 e 1,49 ao controlar, respectivamente, o tipo de acidente,
a área/pista e o número de veículos e a presença de caminhão (Mantel-Haenszel). O SHAP, que é condicional às demais variáveis, concorda (multiplicador de odds de 1,58).

**Limitações.** (a) Os resultados são associações em dados observacionais; o SHAP não identifica efeito de intervenções. (b) A base contém apenas acidentes
registrados, sem medida de exposição ao tráfego; assim, as taxas são por acidente, não por veículo-km. (c) O cenário A usa `tipo_acidente` e `grupo_causa`, definidos
após o atendimento, e serve para explicar, não para alertar. (d) 2025 foi usado em auditorias e análises exploratórias anteriores, logo não é um conjunto completamente cego.
(e) Seleção de modelos e limiares em um único ano de desenvolvimento pode ser otimista.

## Conclusão

Este trabalho mostrou que a letalidade dos acidentes em rodovias federais se concentra em combinações específicas de condição e local:
pista simples, período noturno, Norte/Nordeste, colisão frontal e envolvimento de pedestres. Um modelo XGBoost treinado em 2024 e testado em 2025
ordena os acidentes por risco de forma útil (AP de 0,377 no cenário A, 5,35 vezes o baseline), e a análise de explicabilidade confirmou os principais fatores da estatística (colisão frontal, caminhão, período noturno e Norte/Nordeste) e mostrou que o modelo sem informações do acidente se apoia sobretudo em localização e horário.
Traduzimos esses padrões em três ações priorizáveis, cada uma com métrica de acompanhamento: fiscalização noturna focada, tratamento viário
de trechos com colisão frontal e proteção de pedestres. A principal limitação é a ausência de dados de exposição e de desenho causal.
Como trabalho futuro, propomos ranquear trechos (BR/km), incluir volume de tráfego e testar as intervenções.