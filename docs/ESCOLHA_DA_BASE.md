# Escolha da base de dados

**Base escolhida:** Acidentes em rodovias federais do Brasil, **PRF (Polícia Rodoviária Federal), 2024 e 2025**.

> Os números abaixo vêm do tratamento de dados (`docs/tabelas/metricas.csv`, gerado por `python -m src.p1_engenharia_dados.executar_pipeline`). A comparação com as alternativas foi medida em 10/11/2026, baixando cada arquivo e perfilando com `pandas`.

## 1. A escolha em resumo

- **Fonte:** [Dados Abertos da PRF](https://www.gov.br/prf/pt-br/acesso-a-informacao/dados-abertos/dados-abertos-da-prf) (base BAT, *Boletim de Acidente de Trânsito*; atualização mensal).
- **Arquivos usados:** `datatran2024.csv` e `datatran2025.csv` (um acidente por linha) e a versão "agrupada por pessoa" de cada ano (uma pessoa/veículo por linha).
- **Tamanho:** **145.731 acidentes × 30 colunas** e **391.077 linhas** de pessoas/veículos.
- **Pergunta de negócio:** *quais combinações de condições (causa, tipo, horário, clima, traçado, trecho, veículos envolvidos) mais elevam o risco de um acidente ser **fatal**, e onde e quando a PRF, as concessionárias ou as seguradoras devem concentrar fiscalização, sinalização e campanhas?*
- **Alvo:** `acidente_fatal` (1 se houve ao menos uma morte): **7,16%** dos acidentes no bruto (10.435 de 145.731) e **7,05%** na base tratada. Classe desbalanceada: pede F1/AUC, não só acurácia.
- **Por que vende bem:** é um problema brasileiro e recente, com "cliente" óbvio (PRF, Ministério dos Transportes, concessionárias, seguradoras), decisão concreta (onde reforçar o quê, e quando) e métrica de sucesso clara (queda de acidentes fatais no trecho ou horário atendido).

## 2. Regras do PDF do projeto e checagem

Fonte: *Projeto (2026-2).pdf* e *Planejamento - Projeto Final de Ciência de Dados.pdf*.

### 2.1 Restrições

| Regra | Situação |
|---|---|
| Não usar as bases das Listas 1 e 2 (`creditcard.csv`, *Medical Appointment No Shows*) | **OK**, não é nenhuma delas |
| Não usar *Hotel Booking* | **OK** |
| Planejamento: "sem usar bases antigas" | **OK**, dados de 2024 e 2025 |
| Origem clara e referenciada | **OK**, órgão público, página oficial, base BAT |
| Bases não devem se repetir entre grupos (casos pontuais podem ser discutidos) | **Risco que não conseguimos medir:** acidentes da PRF são um tema conhecido no Brasil, mas os arquivos de 2024 e 2025 são recentes. **Precisamos combinar com os outros grupos** |

### 2.2 Os 6 critérios da base

| # | Critério do PDF | Atende? | Evidência medida |
|---|---|---|---|
| 1 | Preferencialmente > 5.000 linhas e ≥ 10 colunas úteis (sem IDs) | **Sim** | 145.731 linhas; 42 colunas na base de modelagem |
| 2 | Numéricas e categóricas, resultado principal, exploração estatística e visual | **Sim** | 10 numéricas, 11 categóricas e 20 binárias; alvo `acidente_fatal`; tempo, local, causa, clima, traçado, veículos e pessoas |
| 3 | Problema de negócio claro | **Sim** | Onde e quando reduzir acidentes fatais (seção 1) |
| 4 | Potencial para modelagem | **Sim** | Classificação binária desbalanceada; também dá regressão (nº de feridos), agrupamento (perfis de trecho), regras de associação (causa × clima × tipo) e série temporal (acidentes por dia) |
| 5 | Necessidade de preparação (não pode estar "perfeita") | **Sim** | Seção 4 |
| 6 | ≥ 3 insights acionáveis | **Provável, a confirmar nos testes da Pessoa 2** | Prévia na seção 5 |

**Ponto de atenção honesto (critérios 2 e 4):** das 13 colunas numéricas do arquivo de acidentes, 7 são **consequência** do acidente (`pessoas`, `mortos`, `feridos_*`, `ilesos`, `ignorados`, `feridos`). Elas definem o alvo e **não podem entrar como atributos** (vazamento). Os preditores numéricos naturais são `km`, `latitude`, `longitude` e `veiculos`; o resto vem de **transformação** (hora, calendário, grupo de causa, traçado em colunas, resumo das pessoas). O modelo pode ter desempenho moderado, e isso também é um resultado válido para discutir.

## 3. Alternativas testadas

| Candidata | Tamanho medido | Recência | Alvo | Preparação | Ponto forte | Ponto fraco |
|---|---|---|---|---|---|---|
| **PRF acidentes (escolhida)** | 145.731 × 30 (2024–25) | 2025 completo | Fatal: 7,16% | Moderada | Brasil, "cliente" claro, integra com o arquivo por pessoa | Preditores numéricos naturais escassos; arquivos no Google Drive (download frágil) |
| NYC *Motor Vehicle Collisions* (NYC Open Data) | 85.559 × 29 (só 2025) | registros até 06/11/2026 | Lesão ou morte: 43,74% (equilibrada) | **Alta**: 369 tipos de veículo, até 99,4% de nulos nos campos de veículos 3 a 5 | Muito recente, API estável, classes equilibradas | EUA; atributos ainda mais "descritivos"; revisões retroativas dos dados |
| UK STATS19 2024 (*Dept. for Transport*) | 100.927 × 44 (+ tabelas de veículos e vítimas) | 2024 | Gravidade: fatal 1,49% | Média (códigos `-1` para ausente) | Documentação excelente, ano fechado (reprodutível), 3 tabelas para integrar | Reino Unido; poucos fatais |

## 4. Preparação necessária (critério 5), com evidência

| Problema encontrado | Evidência | Tratamento |
|---|---|---|
| Arquivos em `latin1`, separador `;`, vírgula decimal | Latitude como `-23,48586772`; acentos quebram em UTF-8 | Leitura com `encoding="latin1"`, `decimal=","` |
| Dados de pessoas em outro arquivo | 391.077 linhas de pessoas para 145.731 acidentes (casam 100%) | Integração: resumo por acidente e join pelo `id` |
| "Não sei" escrito de várias formas | `Ignorado` em 1.933 acidentes (clima), `Não Informado` em 356 (sentido da via), `NA` | Unificar em `Nao_informado` |
| Valores impossíveis | Idade de até 2024 anos; condutores "com 0 ano"; ano de fabricação 0 | Viram nulo (sem inventar valor) |
| Contagens que não fecham | 7.672 acidentes (5,28%) com `mortos + feridos + ilesos + ignorados ≠ pessoas` | Sinalizar |
| Eventos que não são acidentes de trânsito | Suicídio presumido e sinistro pessoal | Remover |
| Traçado com valores compostos | 898 combinações (ex.: `Reta;Declive`) | Separar em colunas 0/1 |
| Alta cardinalidade | 1.943 municípios, 69 causas | Agrupar causas; região no lugar de município |
| Outliers | `veiculos` chega a 82; `km` até 1.470 | Sinalizar (não remover) |
| Integrações externas | Calendário de feriados e região do IBGE | Novas colunas (`eh_feriado`, `regiao`) |

**Coisas que verificamos e não eram problema:** as 593 latitudes positivas estão todas em AP e RR (ao norte do Equador, portanto válidas); `id` é único; não há rótulos duplicados por acento ou caixa.

## 5. Prévia dos padrões (descritiva, ainda sem teste de hipótese)

Taxa de acidentes fatais por grupo na base tratada (taxa geral: 7,05%):

| Grupo | Taxa de fatais |
|---|---|
| Madrugada / Noite / Manhã | 11,53% / 9,16% / 4,87% |
| Pista simples / dupla | 9,77% / 4,69% |
| Com pedestre envolvido | 28,04% |
| Com caminhão / com bicicleta | 12,28% / 12,43% |
| Região Norte / Nordeste / Sudeste | 11,24% / 10,23% / 5,50% |
| Trecho urbano / rural | 4,43% / 9,01% |

Hipóteses para a Pessoa 2 testar (e a Pessoa 4 transformar em insights no formato do PDF: *percebemos que... recomendamos... sustentado por... sucesso medido por... critério objetivo*):

1. **Madrugada, noite e pista simples** concentram mais acidentes fatais → reforçar sinalização, iluminação e fiscalização nesses horários e tipos de pista.
2. **Pedestres, ciclistas e caminhões** envolvidos elevam muito o risco → ações de travessia e segurança de usuários vulneráveis; atenção a trechos de alto tráfego de carga.
3. **Norte e Nordeste** têm taxa de fatais cerca do dobro da do Sudeste e do Sul → priorizar recursos por região e por trecho (Alagoas tem 1.337 acidentes na base tratada).

Atenção: causas como álcool (5,18%) e velocidade incompatível (5,28%) têm taxa de fatais **abaixo** da média, embora sejam causas frequentes. Isso precisa ser analisado em volume (quantos acidentes) e em gravidade (que fração é fatal) antes de qualquer recomendação.

## 6. Riscos e limitações (para falar com honestidade na banca)

- Só entram acidentes **registrados** pela PRF em **rodovias federais**; vias urbanas e rodovias estaduais ficam de fora (viés de registro).
- O alvo é raro e os preditores são limitados; os resultados mostram **associação, não causa**.
- Causa e tipo do acidente são definidos pela PRF depois do atendimento: servem para explicar o risco e orientar decisões, não para alertar antes do acidente.
- Os dados são revistos mensalmente: congelamos 2024 e 2025 (anos completos) e citamos a data do download. O ano de 2026 é parcial e ficou de fora.
- Na versão por pessoa, `sexo` e `idade` são atributos sensíveis: usar para auditar equidade, não para discriminar.

## 7. Descartadas

| Base | Motivo |
|---|---|
| *Credit Card Fraud* e *Medical Appointment No Shows* | Usadas nas Listas 1 e 2: proibidas pelo PDF |
| *Hotel Booking* | Proibida pelo PDF |
| Olist (2016–2018), Bank Marketing (2008–2010), Online Shoppers (2018) | Antigas; Bank Marketing e Olist são clássicas de sala de aula |
| IBM Telco Churn | Base fictícia (amostra da IBM), muito usada |
| KKBox churn | Dados de 2017 |

## 8. Referências

- Polícia Rodoviária Federal. *Dados Abertos da PRF: Acidentes* (BAT). <https://www.gov.br/prf/pt-br/acesso-a-informacao/dados-abertos/dados-abertos-da-prf>
- NYC Open Data. *Motor Vehicle Collisions - Crashes* (`h9gi-nx95`). <https://data.cityofnewyork.us/resource/h9gi-nx95>
- UK Department for Transport. *Road Safety Data (STATS19)*. <https://data.dft.gov.uk/road-accidents-safety-data/dft-road-casualty-statistics-collision-2024.csv>
- Documentos do projeto: *Projeto (2026-2).pdf* e *Planejamento - Projeto Final de Ciência de Dados.pdf*.
