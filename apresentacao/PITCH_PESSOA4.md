# Pitch comercial (5 min)

### Roteiro 

Titulo: Reduzindo mortes nas rodovias federais

#### Slide 2:
	Titulo: Problema 
	Visual: Número de mortes totais nos dois anos registrados (7 em cada 100 acidentes). Incluindo gráfico da figura7 p2.
	Fala:  Entre 2024 e 2025 foram registrados 145 mil acidentes, onde 10.240 deles apresentou fatalidade. Alagoas em especial apresenta uma taxa de 12,8% de fatalidades em acidentes, quase o dobro da média nacional.  Os recursos limitados de fiscalização, sinalização e obras apresentam a pergunta do problema: onde e quando agir primeiro?
	
#### Slide 3:
	Titulo: O que os dados mostram
	Visual: Plot fig1 p2 (recorte).
	Fala: Observando a concentração nos dados, viagens feitas de noite (ou madrugada), em uma pista simples e no norte/nordeste representam 7% dos acidentes, porém elas apresentam 19% das fatalidades totais da base. Ou seja, 1 em cada 5 acidentes nessas condições terminam em morte, contra a taxa de 7% na média. Outro grupo se destaca (não está no slide), colisões frontais em pistas simples representam 6% dos acidentes totais e 26% das fatalidades totais. Além desses grupos, outros dois fatores tornam o acidente fatal: quando há pedestres e quando há caminhões envolvidos. Todos esses padrões são similares em 2024 e 2025.  
#### Slide 4:
	Titulo: O modelo
	Visual: figura p4 curva ganho 2025 ( o gráfico vai do acidente mais arriscado até o menos arriscado, o modelo do XGBOOST faz o ranking de risco)
	Fala:  Transformando os dados em ferramenta, foram treinados dois modelos. O treino foi feito com os registros de 2024 e o teste com os de 2025. Cada modelo analisa um acidente e estima o risco de morte. O primeiro modelo utiliza informações como o tipo de acidente, veículos envolvidos, causa, etc e consegue cobrir 50,7% de todas as mortes. Um segundo modelo mais realista, utilizando apenas os dados de local e hora, cobre 26%. Como o primeiro modelo utiliza informações coletadas pela PRF após o acidente ele seria usado para explicar, enquanto o segundo, mesmo sendo mais fraco, poderia ser usado antes para prever uma fatalidade.
	
#### Slide 5-6:
	Titulo: Por que o modelo diz isso (SHAP)
	Visual: Gráficos de importância e do caso especifico do modelo A e B
	Fala: Utilizando a técnica SHAP, que mostra quanto cada informação influenciou o risco, podemos explicar cada modelo.
		(slide com importâncias): No modelo completo, o que mais pesa nas previsões é o tipo de acidente, principalmente a colisão frontal. Depois vêm a presença de caminhão e o acidente ocorrer fora de área urbana.
		(slide do caso real): Com um acidente real de 2025 no Maranhão às 4 da manhã, só com o local e hora o modelo simples estima 42% de chance de morte, taxa elevada comparada a média de 7%, já com o modelo mais complexo a estimativa foi de 90%.
		O SHAP confirmou os **principais** fatores que a estatística já tinha apontado: colisão frontal, caminhão, noite e Norte e Nordeste. São dois métodos independentes chegando à mesma direção.
#### Slide 7-10:
	Titulo: insights: Três ações
	Visual: Os três insights de maneira reduzida (problema/causa, ação, como medir o resultado)
	Fala: Pensando no problema principal "onde e quando agir primeiro?", foram obtidos três insights/ações sobre os dados.
		Primeiro: na situação de pista simples no Norte e Nordeste, é recomendado uma fiscalização noturna mais focada e presente. 
		Segundo: colisões frontais em pistas simples são só 6% dos acidentes totais, mas 26% das fatalidades, a recomendação seria a melhoria da infraestrutura, como separadores frontais e faixas de ultrapassagem.
		Terceiro: pedestres aparecem em 4% dos acidentes mas em 17% dos acidentes mortais. Quase 3 em cada 10 são fatais, e em trecho rural, mais de 1 em cada 3. É recomendado melhorias na iluminação. travessias mais seguras e redução de velocidade em vias com maior taxa de atropelamento.
	O teste seria comparar a queda nesses trechos com trechos parecidos que não receberam a ação. Só assim dá para separar o efeito da ação de uma melhora geral.



## Os 3 insights acionáveis

### Insight 1 — Onde e quando agir primeiro: pista simples, à noite, no Norte/Nordeste
- **Percebemos que** acidentes em pista simples, entre 18h e 5h, no Norte/Nordeste são 7,1% dos acidentes e 19,3% deles são fatais, contra 7,05% na média (RR 3,16). Representam **19,4% das fatalidades** totais. O padrão se repete em 2024 (19,1%) e 2025 (19,5%).
- **Por isso recomendamos** concentrar a fiscalização noturna (ultrapassagem indevida, álcool, velocidade) nesses trechos, em vez de distribuí-la uniformemente.
- **Sustentado por:** EDA (RR de noite 1,94 e de pista simples 2,16, que sobrevivem à estratificação pelo CMH) e SHAP: no cenário B, localização e horário somam 94% da importância (64% e 30%); no A, madrugada/noite multiplica as odds por 1,64, Norte/Nordeste por 1,59 e pista simples por 1,28 (os três confirmam a EDA).
- **Medido por:** taxa de acidentes fatais no segmento-alvo (linha de base: 19,3%) e mortes/mês no segmento.

### Insight 2 — Colisão frontal em pista simples: o maior "retorno por intervenção"
- **Percebemos que** colisões frontais em pista simples são só **5,9% dos acidentes, mas  31,4% são fatais (RR 5,67)**. Representam 26,1% das fatalidades totais.
- **Por isso recomendamos** priorizar separador central, faixas de ultrapassagem e sinalização horizontal/refletiva nos trechos que mais acumulam esse tipo.
- **Sustentado por:** EDA (tipo de acidente é a variável mais associada ao alvo, V de Cramér 0,305) e SHAP do cenário A: `tipo_acidente` é a variável de maior peso (|SHAP| médio 0,50; 18% do total) e Colisão frontal multiplica as odds por 5,5, o maior efeito entre os sinais testados..
- **Medido por:** nº de colisões frontais fatais por trecho tratado vs. trechos de controle.

### Insight 3 — Pedestres: pouco volume, altíssima letalidade
- **Percebemos que** pedestre aparece em 4,2% dos acidentes mas 28,0% são fatais (RR 4,58), sendo 16,9% das fatalidades totais. Em trecho rural a letalidade é de 36,7% e à noite/madrugada de 35,9%.
- **Por isso recomendamos** iluminação, travessias seguras (passarelas/semáforos) e redução de velocidade nos trechos com histórico de atropelamento, começando pelos noturnos.
- **Sustentado por:** EDA e SHAP do cenário A: odds ×1,51 para pedestre envolvido. O efeito total é maior, porque parte dele é atribuída a `tipo_acidente = Atropelamento de Pedestre`.
- **Medido por:** mortes por atropelamento nos trechos tratados.

## O modelo como ferramenta de decisão

| Priorizando os… | Modelo A (retrospectivo) | Modelo B (local + calendário) |
|---|---|---|
| 5% de maior risco | 33,2% das mortes (lift 6,6×) | 16,0% (lift 3,2×) |
| **10%** de maior risco | **50,7%** das mortes (lift 5,1×) | **26,3%** (lift 2,6×) |
| 20% de maior risco | 69,1% (lift 3,5×) | 41,6% (lift 2,1×) |

Leitura honesta: **A explica** (usa tipo e causa, só conhecidos após o atendimento); **B prioriza** com informação disponível antes (modelo modesto: AP 0,153 vs. 0,071 do baseline, ROC-AUC 0,69). Ambos estimam a chance de morte **dado que o acidente ocorreu**, não a chance de o acidente ocorrer.
