# Pessoa 3 — Etapa 03: validação temporal de 2024

Relatório gerado por `python -m src.p3_modelagem.executar_validacao`.

## Protocolo e rastreabilidade

CSV: `\data\processed\acidentes_prf_modelagem.csv`;

Desenvolvimento completo: 73006 registros de 2024. Semente: 42.

O carregador existente lê e valida o contrato do CSV completo; a separação existente isola 2024. O objeto reservado de 2025 é descartado antes dos experimentos, sem inferência, métricas, seleção ou inspeção de resultados de 2025.

Treino janeiro–junho / validação julho–agosto; janeiro–agosto / setembro–outubro; janeiro–outubro / novembro–dezembro. Cortes gerados uma única vez e reutilizados nos dois cenários e em todas as variantes. Pipeline novo por ajuste, pré-processamento ajustado apenas no treino; estimadores descartados após inferência. Tempo de inferência inclui transformação e predict_proba; tempo de treino inclui todo o fit.

| rodada | n_treino | n_validacao | positivos_treino | positivos_validacao |
| --- | --- | --- | --- | --- |
| 1 | 35160 | 12537 | 2452 | 909 |
| 2 | 47697 | 12599 | 3361 | 869 |
| 3 | 60296 | 12710 | 4230 | 906 |

Execuções concluídas: 42; falhas: 0.

Configurações e pesos efetivos de cada treino, versões, avisos, falhas e hashes dos índices estão em `p3_execucao_2024.json`.

## Comparação no limiar 0,5

Médias não ponderadas entre três cortes. DP populacional (ddof=0), apenas descritivo, sem interpretação de intervalo de confiança. Campos NaN/CSV vazio indicam métricas indefinidas; a coluna `indefinidas` explica os motivos. Médias e DP ignoram valores indefinidos; os campos *_n_definidos no resumo informam quantos cortes contribuíram. Precision é indefinida quando não há positivos previstos; recall é indefinido sem positivos reais; F1 é zero quando há positivos reais mas nenhum previsto; AP é indefinida sem positivos reais e ROC-AUC exige ambas as classes. Brier e Log Loss são calculados com as probabilidades originais, sem recalibração. AP usa average_precision_score.

| cenario | algoritmo | variante | ap_media | ap_dp | roc_auc_media | precision_media | precision_n_definidos | recall_media | f1_media | brier_media | log_loss_media | tempo_treino_total_s |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| A | dummy | padrao | 0.070921 | 0.001464 | 0.500000 | indefinida | 0 | 0.000000 | 0.000000 | 0.065892 | 0.256026 | 0.962514 |
| A | random_forest | padrao | 0.353601 | 0.028346 | 0.830738 | 0.933333 | 3 | 0.003005 | 0.005984 | 0.056090 | 0.205210 | 16.812448 |
| A | random_forest | ponderado | 0.339779 | 0.023181 | 0.828883 | 0.269152 | 3 | 0.583936 | 0.368160 | 0.135649 | 0.442977 | 16.247120 |
| A | regressao_logistica | padrao | 0.333317 | 0.028684 | 0.833365 | 0.537360 | 3 | 0.122849 | 0.199833 | 0.055729 | 0.201274 | 1.878910 |
| A | regressao_logistica | ponderado | 0.320475 | 0.024589 | 0.835543 | 0.209125 | 3 | 0.721227 | 0.324159 | 0.157873 | 0.493183 | 2.373474 |
| A | xgboost | padrao | 0.378798 | 0.028134 | 0.850139 | 0.611457 | 3 | 0.121741 | 0.203025 | 0.053357 | 0.192962 | 2.023372 |
| A | xgboost | ponderado | 0.373224 | 0.028249 | 0.850705 | 0.231373 | 3 | 0.723471 | 0.350561 | 0.144253 | 0.453019 | 1.745036 |
| B | dummy | padrao | 0.070921 | 0.001464 | 0.500000 | indefinida | 0 | 0.000000 | 0.000000 | 0.065892 | 0.256026 | 0.380203 |
| B | random_forest | padrao | 0.153434 | 0.011051 | 0.681503 | indefinida | 0 | 0.000000 | 0.000000 | 0.063730 | 0.242142 | 9.255777 |
| B | random_forest | ponderado | 0.150623 | 0.010001 | 0.680084 | 0.135613 | 3 | 0.445485 | 0.207864 | 0.191991 | 0.571662 | 10.960695 |
| B | regressao_logistica | padrao | 0.142408 | 0.004248 | 0.668934 | 0.166667 | 2 | 0.000367 | 0.000731 | 0.064000 | 0.243486 | 0.966598 |
| B | regressao_logistica | ponderado | 0.141212 | 0.004788 | 0.668277 | 0.111844 | 3 | 0.606688 | 0.188850 | 0.229240 | 0.653204 | 1.656540 |
| B | xgboost | padrao | 0.155628 | 0.009253 | 0.688005 | 0.000000 | 1 | 0.000000 | 0.000000 | 0.063489 | 0.240476 | 0.956085 |
| B | xgboost | ponderado | 0.151928 | 0.008795 | 0.684316 | 0.119750 | 3 | 0.597626 | 0.199506 | 0.215316 | 0.620079 | 1.356297 |

## Rodadas individuais

| cenario | algoritmo | variante | rodada | prevalencia | ap | roc_auc | precision | recall | f1 | tp | fp | tn | fn |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| A | dummy | padrao | 1 | 0.072505 | 0.072505 | 0.500000 | indefinida | 0.000000 | 0.000000 | 0 | 0 | 11628 | 909 |
| A | dummy | padrao | 2 | 0.068974 | 0.068974 | 0.500000 | indefinida | 0.000000 | 0.000000 | 0 | 0 | 11730 | 869 |
| A | dummy | padrao | 3 | 0.071282 | 0.071282 | 0.500000 | indefinida | 0.000000 | 0.000000 | 0 | 0 | 11804 | 906 |
| A | regressao_logistica | padrao | 1 | 0.072505 | 0.352951 | 0.840617 | 0.591133 | 0.132013 | 0.215827 | 120 | 83 | 11545 | 789 |
| A | regressao_logistica | padrao | 2 | 0.068974 | 0.292758 | 0.821704 | 0.459091 | 0.116226 | 0.185491 | 101 | 119 | 11611 | 768 |
| A | regressao_logistica | padrao | 3 | 0.071282 | 0.354241 | 0.837776 | 0.561856 | 0.120309 | 0.198182 | 109 | 85 | 11719 | 797 |
| A | random_forest | padrao | 1 | 0.072505 | 0.372848 | 0.832122 | 1.000000 | 0.001100 | 0.002198 | 1 | 0 | 11628 | 908 |
| A | random_forest | padrao | 2 | 0.068974 | 0.313524 | 0.825420 | 0.800000 | 0.004603 | 0.009153 | 4 | 1 | 11729 | 865 |
| A | random_forest | padrao | 3 | 0.071282 | 0.374432 | 0.834673 | 1.000000 | 0.003311 | 0.006601 | 3 | 0 | 11804 | 903 |
| A | xgboost | padrao | 1 | 0.072505 | 0.403598 | 0.854617 | 0.637931 | 0.122112 | 0.204986 | 111 | 63 | 11565 | 798 |
| A | xgboost | padrao | 2 | 0.068974 | 0.339454 | 0.841537 | 0.555556 | 0.115075 | 0.190658 | 100 | 80 | 11650 | 769 |
| A | xgboost | padrao | 3 | 0.071282 | 0.393341 | 0.854263 | 0.640884 | 0.128035 | 0.213431 | 116 | 65 | 11739 | 790 |
| B | dummy | padrao | 1 | 0.072505 | 0.072505 | 0.500000 | indefinida | 0.000000 | 0.000000 | 0 | 0 | 11628 | 909 |
| B | dummy | padrao | 2 | 0.068974 | 0.068974 | 0.500000 | indefinida | 0.000000 | 0.000000 | 0 | 0 | 11730 | 869 |
| B | dummy | padrao | 3 | 0.071282 | 0.071282 | 0.500000 | indefinida | 0.000000 | 0.000000 | 0 | 0 | 11804 | 906 |
| B | regressao_logistica | padrao | 1 | 0.072505 | 0.139031 | 0.663329 | 0.333333 | 0.001100 | 0.002193 | 1 | 2 | 11626 | 908 |
| B | regressao_logistica | padrao | 2 | 0.068974 | 0.139794 | 0.667541 | 0.000000 | 0.000000 | 0.000000 | 0 | 2 | 11728 | 869 |
| B | regressao_logistica | padrao | 3 | 0.071282 | 0.148399 | 0.675932 | indefinida | 0.000000 | 0.000000 | 0 | 0 | 11804 | 906 |
| B | random_forest | padrao | 1 | 0.072505 | 0.143091 | 0.670147 | indefinida | 0.000000 | 0.000000 | 0 | 0 | 11628 | 909 |
| B | random_forest | padrao | 2 | 0.068974 | 0.148460 | 0.682743 | indefinida | 0.000000 | 0.000000 | 0 | 0 | 11730 | 869 |
| B | random_forest | padrao | 3 | 0.071282 | 0.168751 | 0.691618 | indefinida | 0.000000 | 0.000000 | 0 | 0 | 11804 | 906 |
| B | xgboost | padrao | 1 | 0.072505 | 0.145711 | 0.678911 | 0.000000 | 0.000000 | 0.000000 | 0 | 1 | 11627 | 909 |
| B | xgboost | padrao | 2 | 0.068974 | 0.153195 | 0.688413 | indefinida | 0.000000 | 0.000000 | 0 | 0 | 11730 | 869 |
| B | xgboost | padrao | 3 | 0.071282 | 0.167980 | 0.696691 | indefinida | 0.000000 | 0.000000 | 0 | 0 | 11804 | 906 |
| A | regressao_logistica | ponderado | 1 | 0.072505 | 0.337743 | 0.841956 | 0.215820 | 0.729373 | 0.333082 | 663 | 2409 | 9219 | 246 |
| A | regressao_logistica | ponderado | 2 | 0.068974 | 0.285701 | 0.825604 | 0.196773 | 0.715765 | 0.308685 | 622 | 2539 | 9191 | 247 |
| A | regressao_logistica | ponderado | 3 | 0.071282 | 0.337980 | 0.839068 | 0.214781 | 0.718543 | 0.330709 | 651 | 2380 | 9424 | 255 |
| A | random_forest | ponderado | 1 | 0.072505 | 0.352712 | 0.829652 | 0.288313 | 0.575358 | 0.384135 | 523 | 1291 | 10337 | 386 |
| A | random_forest | ponderado | 2 | 0.068974 | 0.307225 | 0.821764 | 0.244389 | 0.563867 | 0.340988 | 490 | 1515 | 10215 | 379 |
| A | random_forest | ponderado | 3 | 0.071282 | 0.359401 | 0.835232 | 0.274752 | 0.612583 | 0.379357 | 555 | 1465 | 10339 | 351 |
| A | xgboost | ponderado | 1 | 0.072505 | 0.398596 | 0.854989 | 0.236285 | 0.724972 | 0.356409 | 659 | 2130 | 9498 | 250 |
| A | xgboost | ponderado | 2 | 0.068974 | 0.333813 | 0.841748 | 0.220183 | 0.718067 | 0.337024 | 624 | 2210 | 9520 | 245 |
| A | xgboost | ponderado | 3 | 0.071282 | 0.387263 | 0.855377 | 0.237649 | 0.727373 | 0.358250 | 659 | 2114 | 9690 | 247 |
| B | regressao_logistica | ponderado | 1 | 0.072505 | 0.137439 | 0.662307 | 0.113386 | 0.621562 | 0.191785 | 565 | 4418 | 7210 | 344 |
| B | regressao_logistica | ponderado | 2 | 0.068974 | 0.138230 | 0.665678 | 0.105730 | 0.590334 | 0.179339 | 513 | 4339 | 7391 | 356 |
| B | regressao_logistica | ponderado | 3 | 0.071282 | 0.147967 | 0.676845 | 0.116417 | 0.608168 | 0.195425 | 551 | 4182 | 7622 | 355 |
| B | random_forest | ponderado | 1 | 0.072505 | 0.141463 | 0.668102 | 0.133830 | 0.414741 | 0.202362 | 377 | 2440 | 9188 | 532 |
| B | random_forest | ponderado | 2 | 0.068974 | 0.145871 | 0.684176 | 0.134609 | 0.461450 | 0.208420 | 401 | 2578 | 9152 | 468 |
| B | random_forest | ponderado | 3 | 0.071282 | 0.164536 | 0.687972 | 0.138400 | 0.460265 | 0.212809 | 417 | 2596 | 9208 | 489 |
| B | xgboost | ponderado | 1 | 0.072505 | 0.145860 | 0.670990 | 0.117179 | 0.583058 | 0.195140 | 530 | 3993 | 7635 | 379 |
| B | xgboost | ponderado | 2 | 0.068974 | 0.145559 | 0.688063 | 0.116107 | 0.597238 | 0.194418 | 519 | 3951 | 7779 | 350 |
| B | xgboost | ponderado | 3 | 0.071282 | 0.164365 | 0.693895 | 0.125965 | 0.612583 | 0.208961 | 555 | 3851 | 7953 | 351 |

## Previsões reunidas e limiares

As validações julho–dezembro são disjuntas. As métricas abaixo usam todas as previsões reunidas; não são médias das métricas dos cortes. O baseline constante do Dummy por rodada tem AP igual à prevalência da respectiva validação; ao reunir rodadas, suas constantes podem variar e o AP global pode diferir da prevalência global.

| cenario | algoritmo | variante | n_validacao | prevalencia | ap | roc_auc | brier | log_loss |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| A | dummy | padrao | 37846 | 0.070919 | 0.070024 | 0.494053 | 0.065891 | 0.256022 |
| A | random_forest | padrao | 37846 | 0.070919 | 0.350117 | 0.830730 | 0.056088 | 0.205204 |
| A | random_forest | ponderado | 37846 | 0.070919 | 0.338222 | 0.828817 | 0.135657 | 0.442999 |
| A | regressao_logistica | padrao | 37846 | 0.070919 | 0.331238 | 0.833515 | 0.055727 | 0.201267 |
| A | regressao_logistica | ponderado | 37846 | 0.070919 | 0.318280 | 0.835668 | 0.157865 | 0.493166 |
| A | xgboost | padrao | 37846 | 0.070919 | 0.377599 | 0.850189 | 0.053355 | 0.192957 |
| A | xgboost | ponderado | 37846 | 0.070919 | 0.371607 | 0.850784 | 0.144254 | 0.453028 |
| B | dummy | padrao | 37846 | 0.070919 | 0.070024 | 0.494053 | 0.065891 | 0.256022 |
| B | random_forest | padrao | 37846 | 0.070919 | 0.151465 | 0.681364 | 0.063727 | 0.242130 |
| B | random_forest | ponderado | 37846 | 0.070919 | 0.149393 | 0.679824 | 0.191986 | 0.571650 |
| B | regressao_logistica | padrao | 37846 | 0.070919 | 0.141335 | 0.668799 | 0.063997 | 0.243477 |
| B | regressao_logistica | ponderado | 37846 | 0.070919 | 0.139646 | 0.668140 | 0.229219 | 0.653157 |
| B | xgboost | padrao | 37846 | 0.070919 | 0.154057 | 0.687859 | 0.063486 | 0.240464 |
| B | xgboost | ponderado | 37846 | 0.070919 | 0.150863 | 0.684144 | 0.215296 | 0.620036 |

Busca exata nos escores únicos observados (inclui probabilidades pequenas). Maximiza F1; empates com atol=1e-12 e rtol=0 favorecem o maior limiar. A decisão é p >= limiar. O limiar e seu F1 são ajustados nas mesmas validações: esses valores são otimistas e não constituem avaliação independente. Também existe otimismo de seleção ao comparar modelos em 2024.

| cenario | algoritmo | variante | tipo_limiar | limiar | precision | recall | f1 | fp | tp | fn |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| A | dummy | padrao | padrao_0.5 | 0.500000 | indefinida | 0.000000 | 0.000000 | 0 | 0 | 2684 |
| A | dummy | padrao | candidato_f1_ajuste | 0.069738 | 0.070919 | 1.000000 | 0.132445 | 35162 | 2684 | 0 |
| A | random_forest | padrao | padrao_0.5 | 0.500000 | 0.888889 | 0.002981 | 0.005941 | 1 | 8 | 2676 |
| A | random_forest | padrao | candidato_f1_ajuste | 0.162292 | 0.360750 | 0.444486 | 0.398264 | 2114 | 1193 | 1491 |
| A | random_forest | ponderado | padrao_0.5 | 0.500000 | 0.268539 | 0.584203 | 0.367946 | 4271 | 1568 | 1116 |
| A | random_forest | ponderado | candidato_f1_ajuste | 0.620641 | 0.396785 | 0.404620 | 0.400664 | 1651 | 1086 | 1598 |
| A | regressao_logistica | padrao | padrao_0.5 | 0.500000 | 0.534846 | 0.122951 | 0.199939 | 287 | 330 | 2354 |
| A | regressao_logistica | padrao | candidato_f1_ajuste | 0.158404 | 0.321628 | 0.494784 | 0.389843 | 2801 | 1328 | 1356 |
| A | regressao_logistica | ponderado | padrao_0.5 | 0.500000 | 0.208981 | 0.721311 | 0.324071 | 7328 | 1936 | 748 |
| A | regressao_logistica | ponderado | candidato_f1_ajuste | 0.726399 | 0.307058 | 0.499255 | 0.380250 | 3024 | 1340 | 1344 |
| A | xgboost | padrao | padrao_0.5 | 0.500000 | 0.611215 | 0.121833 | 0.203169 | 208 | 327 | 2357 |
| A | xgboost | padrao | candidato_f1_ajuste | 0.206672 | 0.404091 | 0.448957 | 0.425344 | 1777 | 1205 | 1479 |
| A | xgboost | ponderado | padrao_0.5 | 0.500000 | 0.231301 | 0.723547 | 0.350542 | 6454 | 1942 | 742 |
| A | xgboost | ponderado | candidato_f1_ajuste | 0.727440 | 0.386892 | 0.477273 | 0.427356 | 2030 | 1281 | 1403 |
| B | dummy | padrao | padrao_0.5 | 0.500000 | indefinida | 0.000000 | 0.000000 | 0 | 0 | 2684 |
| B | dummy | padrao | candidato_f1_ajuste | 0.069738 | 0.070919 | 1.000000 | 0.132445 | 35162 | 2684 | 0 |
| B | random_forest | padrao | padrao_0.5 | 0.500000 | indefinida | 0.000000 | 0.000000 | 0 | 0 | 2684 |
| B | random_forest | padrao | candidato_f1_ajuste | 0.105978 | 0.176433 | 0.311848 | 0.225363 | 3907 | 837 | 1847 |
| B | random_forest | ponderado | padrao_0.5 | 0.500000 | 0.135657 | 0.445231 | 0.207953 | 7614 | 1195 | 1489 |
| B | random_forest | ponderado | candidato_f1_ajuste | 0.533491 | 0.158991 | 0.338301 | 0.216319 | 4803 | 908 | 1776 |
| B | regressao_logistica | padrao | padrao_0.5 | 0.500000 | 0.200000 | 0.000373 | 0.000744 | 4 | 1 | 2683 |
| B | regressao_logistica | padrao | candidato_f1_ajuste | 0.110087 | 0.157707 | 0.330104 | 0.213443 | 4732 | 886 | 1798 |
| B | regressao_logistica | ponderado | padrao_0.5 | 0.500000 | 0.111820 | 0.606930 | 0.188848 | 12939 | 1629 | 1055 |
| B | regressao_logistica | ponderado | candidato_f1_ajuste | 0.598248 | 0.146919 | 0.379285 | 0.211797 | 5911 | 1018 | 1666 |
| B | xgboost | padrao | padrao_0.5 | 0.500000 | 0.000000 | 0.000000 | 0.000000 | 1 | 0 | 2684 |
| B | xgboost | padrao | candidato_f1_ajuste | 0.101788 | 0.161270 | 0.372951 | 0.225172 | 5206 | 1001 | 1683 |
| B | xgboost | ponderado | padrao_0.5 | 0.500000 | 0.119710 | 0.597615 | 0.199465 | 11795 | 1604 | 1080 |
| B | xgboost | ponderado | candidato_f1_ajuste | 0.578428 | 0.153012 | 0.399404 | 0.221259 | 5934 | 1072 | 1612 |

## Ponderação e calibração aproximada

### Cenário A

- regressao_logistica: ΔAP=-0.012842, Δrecall=+0.598378, Δprecision=-0.328235, ΔF1=+0.124325, ΔBrier=+0.102144, ΔLogLoss=+0.291909; DP(AP) padrão=0.028684, ponderado=0.024589.

- random_forest: ΔAP=-0.013822, Δrecall=+0.580931, Δprecision=-0.664182, ΔF1=+0.362176, ΔBrier=+0.079559, ΔLogLoss=+0.237767; DP(AP) padrão=0.028346, ponderado=0.023181.

- xgboost: ΔAP=-0.005573, Δrecall=+0.601730, Δprecision=-0.380084, ΔF1=+0.147536, ΔBrier=+0.090896, ΔLogLoss=+0.260057; DP(AP) padrão=0.028134, ponderado=0.028249.

### Cenário B

- regressao_logistica: ΔAP=-0.001196, Δrecall=+0.606321, Δprecision=-0.054823, ΔF1=+0.188119, ΔBrier=+0.165240, ΔLogLoss=+0.409718; DP(AP) padrão=0.004248, ponderado=0.004788.

- random_forest: ΔAP=-0.002811, Δrecall=+0.445485, Δprecision=indefinida, ΔF1=+0.207864, ΔBrier=+0.128261, ΔLogLoss=+0.329520; DP(AP) padrão=0.011051, ponderado=0.010001.

- xgboost: ΔAP=-0.003700, Δrecall=+0.597626, Δprecision=+0.119750, ΔF1=+0.199506, ΔBrier=+0.151827, ΔLogLoss=+0.379603; DP(AP) padrão=0.009253, ponderado=0.008795.

Deltas são ponderado menos padrão; Brier/Log Loss menores são melhores. Pesos balanced em LR/RF e scale_pos_weight=negativos/positivos do treino em XGBoost. Sem reamostragem. As faixas fixas de largura 0,1 no CSV/gráfico de calibração permitem observar discrepâncias entre probabilidade média e frequência fatal; faixas pouco povoadas são instáveis. Brier/Log Loss avaliam qualidade probabilística e não demonstram sozinhos calibração perfeita.

## Candidatas preliminares para revisão

- Cenário A: **xgboost / padrao**, AP médio=0.378798 ± 0.028134 (Dummy=0.070921; razão=5.34×). Brier=0.053357; Log Loss=0.192962. Limiar candidato=0.206671833992; precision de ajuste=0.404091, recall=0.448957, F1=0.425344, FP=1777. Recomendação preliminar pelo maior AP médio completo; estabilidade e probabilidades devem ser revistas nas tabelas acima. O F1 ajustado não participa da escolha do modelo.

AP nas rodadas 1/2/3: 0.403598, 0.339454, 0.393341.

A segunda configuração por AP médio foi xgboost / ponderado (0.373224 ± 0.028249); diferença observada de 0.005573. Esses três cortes não demonstram superioridade estatística rigorosa. A decisão permanece preliminar.

- Cenário B: **xgboost / padrao**, AP médio=0.155628 ± 0.009253 (Dummy=0.070921; razão=2.19×). Brier=0.063489; Log Loss=0.240476. Limiar candidato=0.10178823024; precision de ajuste=0.161270, recall=0.372951, F1=0.225172, FP=5206. Recomendação preliminar pelo maior AP médio completo; estabilidade e probabilidades devem ser revistas nas tabelas acima. O F1 ajustado não participa da escolha do modelo.

AP nas rodadas 1/2/3: 0.145711, 0.153195, 0.167980.

A segunda configuração por AP médio foi random_forest / padrao (0.153434 ± 0.011051); diferença observada de 0.002194. Esses três cortes não demonstram superioridade estatística rigorosa. A decisão permanece preliminar.

A diferença observada entre as candidatas A e B é ΔAP=0.223169 (razão A/B=2.43×). A usa informações retrospectivas do acidente; B estima fatalidade condicional a um acidente registrado com atributos disponíveis antecipadamente. B não prevê a ocorrência de acidentes. A diferença de AP entre cenários sugere dependência de informações posteriores à ocorrência, mas não é uma decomposição causal nem isola o efeito de cada atributo.

## Verificação da entrega

| item | resultado |
| --- | --- |
| testes_novos | 28 aprovados (16,03 s) |
| suite_completa | 162 aprovados (92,39 s) |
| testes_apos_ajuste_apresentacao | 2 aprovados |
| integridade_P1_P2_dados | 32 arquivos com SHA-256 idêntico ao início (excluídos caches Python) |
| git | branch p3/modelagem; HEAD 9143289; nenhum arquivo rastreado alterado |
| resultados_reais | 42 ajustes, 14 configurações, 529844 previsões; métricas recalculadas (atol=1e-7 para float32/float64) e alinhadas ao índice original de 2024 |
| falhas_avisos | 0 falhas; 0 avisos registrados nos 42 ajustes |

## Artefatos e reprodução

- `tabelas/p3_validacao_2024.csv`: 42 linhas esperadas, métricas individuais e tempos.

- `tabelas/p3_resumo_validacao_2024.csv`: médias, DP, contagens de valores definidos e tempo total.

- `tabelas/p3_metricas_globais_2024.csv`: métricas das validações reunidas.

- `tabelas/p3_previsoes_validacao_2024.csv.gz`: índice original, ano/mês, rodada, alvo, probabilidade e classe a 0,5; CSV comprimido para evitar duplicação volumosa.

- `tabelas/p3_comparacao_limiares.csv` e `tabelas/p3_calibracao_2024.csv`.

- `figuras/p3_comparacao_ap.png`, `figuras/p3_precision_recall_validacao.png`, `figuras/p3_calibracao_validacao.png`.

- `p3_execucao_2024.json`: proveniência, parâmetros, avisos, falhas e hashes.

Reexecutar com `.venv/Scripts/python -m src.p3_modelagem.executar_validacao`; `--somente-relatorios` regenera gráficos e relatório dos resultados salvos sem novos ajustes. Sementes fixas garantem repetibilidade numérica dentro das tolerâncias de ponto flutuante; tempos dependem da máquina. Versões do ambiente estão no JSON. Testes: `.venv/Scripts/python -m pytest -q`.

## Limitações e próxima decisão

Três cortes consecutivos de um único ano não estabelecem generalização para outros anos. Pesos e limiares alteram o compromisso entre falsos positivos e perdas de fatais; não foi informado um custo operacional. O limiar por F1 é apenas candidato para revisão. Não há recalibração, SHAP, dashboard, busca extensa ou treinamento definitivo. A etapa encerra com decisões preliminares: após a revisão, congelar cenário, algoritmo, variante e limiar antes de uma única avaliação futura em 2025. Nenhuma avaliação de 2025 foi realizada.