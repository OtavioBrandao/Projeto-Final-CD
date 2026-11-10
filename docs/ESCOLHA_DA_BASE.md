# Escolha da base de dados

**Base escolhida:** Diabetes 130-US Hospitals for Years 1999-2008 (UCI ML Repository, id 296).

## Como a escolhemos

Comparamos cinco candidatas contra os 6 critérios do enunciado e contra o que a Pessoa 4 precisa (explicabilidade com SHAP, 3 insights acionáveis e um pitch convincente).

| Candidata | Linhas | Alvo / problema | Dá trabalho de preparação? | Acesso | Ponto fraco |
|---|---|---|---|---|---|
| **Diabetes 130 Hospitals (UCI)** | 101.766 | Readmissão em < 30 dias (reduzir custo e penalidade) | **Sim:** valores `?`, colunas quase vazias, categorias inconsistentes, pacientes repetidos, códigos CID-9 | Direto, sem login, CC BY 4.0 | Dados de 1999–2008 (EUA) |
| Olist (Kaggle, e-commerce BR) | ~100 mil pedidos em 9 tabelas | Atraso na entrega → nota baixa | Sim (junção de tabelas) | Exige conta Kaggle; CC BY-NC-SA | Muito usada em trabalhos; alto risco de repetir com outro grupo |
| Online Shoppers Intention (UCI) | 12.330 | Compra ou não na sessão | Pouca (base bem comportada) | Direto | Pequena e "limpa demais" para o critério 5 |
| Bank Marketing (UCI) | 45.211 | Cliente assina depósito a prazo | Média | Direto | Clássica de sala de aula; pouco original |
| Inside Airbnb (Rio de Janeiro) | ~dezenas de milhares | Prever preço/ocupação | Sim (muitos nulos, texto) | Direto, mas muda a cada snapshot | Difícil reproduzir e citar uma versão fixa; alvo menos "decisão" |

Fontes consultadas: páginas do UCI (datasets 296 e 468), Fairlearn, OpenML, Kaggle/Olist e Inside Airbnb.

## Checagem dos critérios do enunciado

| # | Critério | Atende? | Evidência |
|---|---|---|---|
| 1 | > 5.000 linhas e ≥ 10 colunas úteis | Sim | 101.766 internações; dezenas de variáveis clínicas e de utilização |
| 2 | Numéricas + categóricas + resultado principal | Sim | Numéricas (`time_in_hospital`, `num_medications`...), categóricas (`age`, `race`, medicamentos, diagnósticos), alvo `readmitted` |
| 3 | Problema de negócio claro | Sim | Reduzir readmissões em 30 dias (custo e penalidade a hospitais) |
| 4 | Potencial de modelagem | Sim | Classificação binária desbalanceada: pede F1 e AUC, não só acurácia |
| 5 | Exige preparação | Sim | `?` como nulo, colunas com muita ausência, `Unknown/Invalid`, óbitos/hospice, encontros repetidos do mesmo paciente, códigos CID-9 de alta cardinalidade |
| 6 | ≥ 3 insights acionáveis | Sim (a confirmar nos dados) | Hipóteses abaixo |

Pontos do enunciado que **não** violamos: não é "Hotel Booking". **A confirmar pela equipe:** que não é a base usada na Lista 2.

## Hipóteses iniciais (para a Parte 1 e para a Pessoa 4)

1. Pacientes com muitas internações prévias no último ano têm maior risco de voltar em 30 dias → programa de acompanhamento pós-alta para esse grupo.
2. A alta para determinados destinos (por exemplo, transferência para outra unidade) está associada a mais readmissões → protocolo de alta e transição de cuidado.
3. Mudança de medicação e medição de HbA1c durante a internação podem estar associadas ao risco → revisar o manejo glicêmico antes da alta.

Todas precisam de teste estatístico antes de virarem afirmação (o enunciado exige).

## Riscos e limitações (para falar com honestidade na banca)

- Dados de 1999–2008: práticas clínicas mudaram; recomendamos o método, não os números absolutos.
- Contém `race`, `gender` e `age`: tratar como atributos sensíveis na discussão de ética e viés (não usar para discriminar).
- Vários encontros por paciente: sem tratamento, o modelo "decora" pacientes e as métricas ficam infladas (vazamento). A Pessoa 1 resolve isso na limpeza.

## Referências

- Strack, B. et al. (2014). *Impact of HbA1c Measurement on Hospital Readmission Rates: Analysis of 70,000 Clinical Database Patient Records.* BioMed Research International. DOI 10.1155/2014/781670.
- Clore, J., Cios, K., DeShazo, J., Strack, B. (2014). *Diabetes 130-US Hospitals for Years 1999-2008* [Dataset]. UCI ML Repository. DOI 10.24432/C5230J.
