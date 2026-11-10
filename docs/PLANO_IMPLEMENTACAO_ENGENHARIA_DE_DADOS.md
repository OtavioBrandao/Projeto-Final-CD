# Plano de Implementação — Engenharia de Dados e Pré-processamento

Este documento descreve **o que o tratamento de dados faz e por quê**, para que o grupo todo entenda a base que vai usar. Os números (linhas removidas, % de nulos etc.) ficam em [`README.md`](README.md) e em [`tabelas/`](tabelas/), gerados pelo código; aqui ficam as regras.

## 1. Objetivo e entregáveis

Transformar os acidentes de rodovias federais da PRF (2024 e 2025) em uma base limpa e pronta para as Pessoas 2, 3 e 4, cumprindo o que o projeto exige: limpeza, tratamento de nulos, remoção de inconsistências, integração de dados, transformação de variáveis, criação de colunas, análise de outliers e redução/seleção de atributos.

| Entregável | Caminho |
|---|---|
| Pipeline em código | `src/p1_engenharia_dados/` |
| Base legível (com `id`, consequências e `trecho`) | `data/interim/acidentes_prf_limpo.csv` (gerada; não versionada por ser grande) |
| Base para modelagem | `data/processed/acidentes_prf_modelagem.csv` (+ `.parquet`) |
| Dicionário de dados | `data/processed/dicionario_de_dados.csv` |
| Auditoria antes/depois | `docs/tabelas/*.csv` |
| README da parte de Engenharia de Dados (estudo e apresentação) | `docs/README.md` + `docs/figuras/*.png` |
| Texto da seção "Base de Dados" do artigo | `docs/secao_base_de_dados.md` |
| Testes | `tests/test_pipeline_p1.py` |

## 2. Regras de ouro

1. **Sem vazamento.** O tratamento entrega limpeza e engenharia de atributos *sem estimar nada dos dados* (sem escalonar, sem one-hot, sem imputar pela média, sem limitar outliers). Isso é feito pela Pessoa 3 dentro de um `sklearn.Pipeline`, depois do split. Aqui só entram regras determinísticas e baseadas em conhecimento do domínio.
2. **Consequência não é atributo.** `mortos`, `feridos_*`, `ilesos`, `ignorados`, `feridos`, `pessoas` e `classificacao_acidente` descrevem o resultado do acidente. Saem da base de modelagem.
3. **Reprodutível.** Um comando refaz tudo: `python -m src.p1_engenharia_dados.executar_pipeline`. Sem passos manuais. Rodar duas vezes gera arquivos idênticos.
4. **Toda decisão é registrada** com: problema → evidência (número) → decisão → motivo → impacto. Isso alimenta a documentação e a fala da apresentação.
5. **Não apagar informação à toa.** Prefira marcar (flag) ou anular um valor impossível a remover a linha, exceto onde a remoção é necessária (D01, D02).
6. **Linhas removidas são contadas em um funil** (quantas linhas entram e saem em cada etapa).

## 3. Código: padrões

- Python 3.11+, *type hints* em todas as funções.
- **Docstrings em português no estilo Google** (compatíveis com `pydoc`): resumo de uma linha, descrição, `Args`, `Returns`, `Raises` quando houver. Também docstring de módulo no topo de cada arquivo.
- **Comentários explicam o porquê** de cada decisão. Cada regra referencia o ID da decisão (ex.: `# D05`).
- Constantes (listas de colunas, agrupamentos, limiares) em `configuracao.py`, com comentário da origem.
- Funções pequenas e puras (recebem DataFrame, devolvem DataFrame). Sem `print`: usar `logging`.

```
src/p1_engenharia_dados/
├── baixar_base.py          # download dos 4 arquivos da PRF
├── configuracao.py         # caminhos, constantes, agrupamentos, limiares
├── carregar.py             # leitura (latin1, ';', vírgula decimal) e join ocorrências x pessoas
├── auditoria.py            # perfil de qualidade (nulos, rótulos "não sei", distintos)
├── limpeza.py              # inconsistências, valores impossíveis, filtros
├── engenharia_atributos.py # novas colunas, agrupamentos e resumo de pessoas
├── outliers.py             # detecção (IQR) e sinalização, sem alterar valores
├── selecao.py              # seleção de colunas + dicionário de dados
├── relatorios.py           # tabelas e figuras de docs/
├── documentacao.py         # gera docs/README.md e docs/secao_base_de_dados.md
└── executar_pipeline.py    # orquestra tudo
```

## 4. Etapas do pipeline

### 4.1 Carga e auditoria inicial (antes de mexer em qualquer coisa)

- Os quatro CSVs usam `sep=";"`, `encoding="latin1"` e vírgula decimal. Só `NA` e campo vazio viram nulo (`keep_default_na=False`).
- Ler também uma cópia em texto puro, só para contar `NA` e rótulos "não sei" na auditoria.
- Gerar `docs/tabelas/auditoria_inicial.csv`: por coluna → tipo, % de nulos, rótulos "Ignorado/Não Informado", valores distintos, exemplo, observação.

### 4.2 Integração de dados

| ID | O que integra | Como |
|---|---|---|
| I01 | Arquivo de **pessoas** (idade, sexo, tipo de envolvido, tipo e ano do veículo) × arquivo de **ocorrências** | Resumir as pessoas **por acidente** (F11) e juntar pelo `id` (join à esquerda, `validate="one_to_one"`). Nenhum acidente pode ser perdido |
| I02 | **Calendário** de feriados nacionais | Fixos + móveis calculados pela Páscoa (Carnaval, Sexta-feira Santa, Corpus Christi) + 20/11 a partir de 2024 → `eh_feriado` |
| I03 | **Região do Brasil** (IBGE) | Mapa UF → região → `regiao` |

### 4.3 Inconsistências e filtros (cada filtro vira uma linha do funil)

| ID | Problema | Regra |
|---|---|---|
| D01 | Acidente possivelmente duplicado | Checar **depois** de juntar as pessoas; remover só os idênticos em todos os campos (exceto `id`). Dois registros iguais só no cabeçalho podem ser acidentes distintos |
| D02 | Suicídio presumido e "sinistro pessoal de trânsito" | **Remover.** Não são acidentes de trânsito; nenhuma ação de segurança viária os evitaria e eles inflariam a taxa de fatais |
| D03 | Contagens de vítimas que não fecham (`mortos + feridos + ilesos + ignorados ≠ pessoas`) | **Sinalizar** em `contagens_inconsistentes` (só no interim). Não corrigir: não sabemos qual campo está errado; o alvo usa só `mortos` |
| D04 | "Não sei" escrito de várias formas (`Ignorado`, `Não Informado`, `NA`, vazio) | Unificar em `Nao_informado`; tirar espaços das pontas |
| D05 | Idades impossíveis (ex.: 2024; condutor com 0 ano, que é "não informado") | Idade fora de 0–110, ou condutor com menos de 10 anos, vira **nulo** |
| D06 | Ano de fabricação impossível (0, 1900, futuro) | Fora de 1950 até o ano do acidente + 1 vira **nulo** |
| D07 | Coordenadas fora do Brasil | Conferir a caixa do Brasil e a UF; latitude positiva só vale em AP, RR, AM e PA. Anular só as impossíveis |

### 4.4 Valores ausentes

A auditoria mostrou que o problema desta base **não são nulos declarados**, e sim rótulos "não sei" e valores impossíveis. A regra é: nunca inventar valor.

- Categóricas: `Nao_informado` (D04).
- Numéricas impossíveis: **nulo** (D05, D06, D07). Os nulos que sobram na base de modelagem (`idade_media_condutores`, `pct_condutores_homens`, `idade_media_veiculos`) são tratados pela Pessoa 3 **dentro do `Pipeline`** (ex.: `SimpleImputer(add_indicator=True)`).

> **Diferença em relação ao planejamento do grupo:** o planejamento pede uma base "sem nulos". Preencher antes do split com média ou mediana vazaria informação do teste. Por isso os poucos nulos que sobram ficam documentados em `docs/README.md` e a imputação fica para o `Pipeline`.

### 4.5 Atributos novos

| ID | Colunas | Como | Por quê |
|---|---|---|---|
| F01 | `acidente_fatal` | `1` se `mortos > 0` | Alvo binário |
| F02 | `hora`, `faixa_horaria` | Hora do `horario`; Madrugada (0–5), Manha (6–11), Tarde (12–17), Noite (18–23) | O risco muda ao longo do dia |
| F03 | `mes`, `ano`, `fim_de_semana` | Da `data_inversa` (fim de semana pela data, não pelo texto) | Sazonalidade e rotina de viagem. `ano` serve para validação temporal |
| F04 | `eh_feriado` | Calendário (I02) | Feriados concentram viagens longas |
| F05 | `regiao` | Mapa IBGE (I03) | Reduz 27 UFs a 5 regiões |
| F06 | `grupo_causa` | As 69 causas em 9 grupos (atenção/reação, velocidade/distância, álcool/substâncias, manobra/infração, pedestre, defeito mecânico/carga, condição da via, clima/ambiente, animais) | Um gestor age sobre grupos de causa |
| F07 | `via_reta`, `via_curva`, `via_aclive`, `via_declive`, `via_intersecao`, `via_rotatoria_retorno`, `via_obras_desvio`, `via_ponte_viaduto_tunel` | `tracado_via` ("Reta;Declive") separado em colunas 0/1 | Centenas de combinações viram 8 colunas |
| F08 | `area_urbana` | `uso_solo == "Sim"` | Urbano × rural |
| F09 | `br` | Código numérico vira categoria (`BR-116`) | É um nome, não uma quantidade |
| F10 | `trecho` (só no interim) | `br` + faixa de 10 km | Achar trechos críticos |
| F11 | `envolve_moto`, `envolve_caminhao`, `envolve_onibus`, `envolve_bicicleta`, `envolve_pedestre`, `idade_media_condutores`, `pct_condutores_homens`, `idade_media_veiculos` | Pessoas resumidas por acidente (I01) | Quem e o quê se envolveu |

Regras: não criar atributo com o alvo nem com a consequência do acidente; nomes em português, `snake_case`.

### 4.6 Outliers (`outliers.py`)

- Para `veiculos`, `km`, `idade_media_condutores` e `idade_media_veiculos`: limites IQR (1,5×), proporção de outliers e boxplots.
- **Não remover nem limitar** valores: um acidente com muitos veículos é real (e grave). Criar colunas `*_outlier` (0/1) e documentar. Se a Pessoa 3 quiser *winsorizar*, faz no `Pipeline` com limites estimados só no treino.
- Documentar a cauda longa (assimetria) e sugerir `log1p` como opção para modelos lineares.

### 4.7 Seleção e redução de atributos (`selecao.py`)

Saem da base de modelagem, sempre com motivo registrado em `docs/tabelas/colunas_removidas.csv`:

1. **Identificador e colunas técnicas:** `id`, `ano_arquivo`.
2. **Consequências do acidente** (vazamento): `classificacao_acidente`, `pessoas`, `mortos`, `feridos_leves`, `feridos_graves`, `ilesos`, `ignorados`, `feridos`.
3. **Administrativas:** `regional`, `delegacia`, `uop` (quem atendeu; redundantes com a UF).
4. **Substituídas:** `causa_acidente` (→ `grupo_causa`), `tracado_via` (→ `via_*`), `uso_solo` (→ `area_urbana`), `municipio` (→ `uf`, `regiao`), `data_inversa` e `horario` (→ calendário e hora).
5. **Quase constantes:** um único valor em ≥ 99,9% das linhas.
6. **Redundantes:** pares numéricos com |r| > 0,9 (listados em `correlacoes.csv`).

`dicionario_de_dados.csv`: `coluna`, `descricao_negocio` (linguagem simples), `tipo` (numerica / categorica / binaria / alvo), `origem` (original / derivada), `sensivel` (idade e sexo dos condutores), `observacao`.

### 4.8 Saídas finais

- `data/interim/acidentes_prf_limpo.csv`: legível, com `id`, consequências e `trecho`.
- `data/processed/acidentes_prf_modelagem.csv` e `.parquet`: sem `id`, sem consequências, alvo `acidente_fatal` na última coluna.
- `random_state=42` onde houver aleatoriedade. Não gerar split treino/teste (é da Pessoa 3).

## 5. Testes (`tests/test_pipeline_p1.py`, `pytest`)

- Rótulos "não sei" unificados; nenhum `NA`/`Ignorado`/`Não Informado` na saída (D04).
- D01 e D02 aplicados; `id` único; apenas 2024 e 2025.
- Regras pequenas testadas com exemplos: idade (D05), ano de fabricação (D06), coordenadas (D07), feriados, faixas do dia, traçado, resumo de pessoas.
- `acidente_fatal` contém só 0/1 e a proporção de 1 bate com o cálculo direto no CSV bruto.
- Nenhuma coluna de consequência nem `id` na base de modelagem; alvo na última coluna.
- Nulos só nas 3 colunas documentadas; valores dentro dos limites plausíveis.
- Valores originais de `km` e `veiculos` intactos (outliers só sinalizados).
- Colunas do dicionário = colunas do arquivo de modelagem.
- Funil de linhas: `linhas_entrada − removidas = linhas_saida`.
- Rodar o pipeline duas vezes produz o mesmo hash dos arquivos finais.
- Todo número do `docs/README.md` e da seção do artigo existe em `docs/tabelas/*.csv`.

## 6. README da parte de Engenharia de Dados (`docs/README.md`)

É a base da apresentação (cerca de **3 minutos**, Bloco 2 do cronograma). Linguagem simples, direta. Tudo com **números gerados pelo código**. Estrutura:

1. Resumo em 30 segundos.
2. Diagrama do pipeline (Mermaid, renderiza no GitHub) com linhas × colunas em cada etapa.
3. Funil de linhas (tabela + gráfico).
4. Mapa de ausentes e valores impossíveis (gráfico + tabela).
5. Tabela "O que estava errado → o que fizemos → por quê" (ID, problema, evidência, decisão, motivo, impacto).
6. Antes × depois (alvo, outliers, causas) e prévia descritiva de padrões.
7. Atributos criados.
8. Cuidados: vazamento, nulos que sobram, atributos sensíveis, limitações e o que verificamos e **não** era problema.
9. O que entregamos para o grupo.
10. Roteiro de fala de 3 minutos e 10 perguntas prováveis da banca.
11. Como reproduzir.

## 7. Critérios de pronto (checklist)

- [ ] `python -m src.p1_engenharia_dados.executar_pipeline` roda do zero sem erro.
- [ ] `pytest` passa.
- [ ] Todos os arquivos da seção 1 existem.
- [ ] Todo número do `docs/README.md` confere com `docs/tabelas/*.csv` (teste automático).
- [ ] `python -m pydoc src.p1_engenharia_dados.limpeza` mostra a documentação das funções.
- [ ] `docs/README.md` renderiza no GitHub (Mermaid e imagens).
- [ ] Pull Request aberto para as Pessoas 2 e 3 revisarem.

## 8. Onde cada exigência do PDF do projeto está atendida

Fonte: *Projeto (2026-2).pdf* e *Planejamento - Projeto Final de Ciência de Dados.pdf*.

| Exigência | Onde |
|---|---|
| Base com > 5.000 linhas e ≥ 10 colunas úteis | [`ESCOLHA_DA_BASE.md`](ESCOLHA_DA_BASE.md) |
| Base recente ("sem usar bases antigas") | 2024 e 2025 |
| Não usar bases das Listas 1 e 2 nem *Hotel Booking* | [`ESCOLHA_DA_BASE.md`](ESCOLHA_DA_BASE.md) |
| Limpeza, nulos, inconsistências, integração, transformação, seleção, outliers | Seções 4.1 a 4.7 e `docs/README.md` |
| Entregável: dataset limpo | `data/processed/` |
| Entregável: texto da seção "Base de Dados" do artigo | `docs/secao_base_de_dados.md` |
| Dataset "sem nulos" (planejamento) | Parcial: ver nota na seção 4.4 |
