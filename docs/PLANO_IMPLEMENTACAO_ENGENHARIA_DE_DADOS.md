# Plano de Implementação — Engenharia de Dados e Pré-processamento

## 1. Objetivo e entregáveis

Transformar `data/raw/diabetic_data.csv` (+ `IDS_mapping.csv`) em uma base limpa e pronta para as Pessoas 2, 3 e 4, cumprindo o que o projeto exige: limpeza, tratamento de nulos, remoção de inconsistências, integração de dados, transformação de variáveis, criação de colunas, análise de outliers e redução/seleção de atributos.

| Entregável | Caminho |
|---|---|
| Pipeline em código | `src/p1_engenharia_dados/` |
| Base legível (categorias com nome) | `data/interim/diabetes_limpo.csv` |
| Base para modelagem | `data/processed/diabetes_modelagem.csv` (+ `.parquet`) |
| Dicionário de dados | `data/processed/dicionario_de_dados.csv` |
| Auditoria antes/depois | `docs/tabelas/*.csv` |
| Documentação para estudo/apresentação | `docs/README.md` + `docs/figuras/*.png` |
| Texto da seção "Base de Dados" do artigo | `docs/secao_base_de_dados.md` |
| Testes | `tests/test_pipeline_p1.py` |

## 2. Regras de ouro

1. **Sem vazamento.** A Pessoa 1 entrega limpeza e engenharia de atributos *sem estimar nada dos dados* (sem escalonar, sem codificar one-hot, sem imputar pela média, sem limitar outliers). Isso é feito pela Pessoa 3 dentro de um `sklearn.Pipeline`, depois do split. Aqui só entram regras determinísticas e baseadas em conhecimento do domínio.
2. **Reprodutível.** Um comando refaz tudo: `python -m src.p1_engenharia_dados.executar_pipeline`. Sem passos manuais. Rodar duas vezes gera arquivos idênticos.
3. **Toda decisão é registrada** com: problema → evidência (número) → decisão → motivo → impacto. Isso alimenta a documentação e a fala da apresentação.
4. **Não apagar informação à toa.** Prefira marcar (flag) a remover, exceto onde a remoção é necessária (ver 4.3).
5. **Linhas removidas são contadas em um funil** (quantas linhas entram e saem em cada etapa).

## 3. Código: padrões obrigatórios

- Python 3.11+, *type hints* em todas as funções.
- **Docstrings em português no estilo Google** (compatíveis com `pydoc`/`python -m pydoc`): resumo de uma linha, descrição, `Args`, `Returns`, `Raises` quando houver. Também docstring de módulo no topo de cada arquivo.
- **Comentários abundantes**: explique o *porquê* de cada decisão, não só o *o quê*. Cada regra de limpeza referencia o ID da decisão (ex.: `# D05 — ver docs/README.md`).
- Constantes (listas de colunas, códigos de alta, limiares) no topo, nomeadas, com comentário da origem.
- Funções pequenas e puras (recebem DataFrame, devolvem DataFrame). Sem `print` solto: usar `logging`.
- Estrutura sugerida (um arquivo por responsabilidade):

```
src/p1_engenharia_dados/
├── baixar_base.py          # (já existe) download do UCI
├── configuracao.py         # caminhos, constantes, limiares
├── carregar.py             # leitura do CSV e do IDS_mapping (3 tabelas num arquivo só)
├── auditoria.py            # perfil de qualidade (nulos, tipos, cardinalidade, duplicados)
├── limpeza.py              # inconsistências, nulos, duplicados, filtros
├── engenharia_atributos.py # novas colunas e agrupamentos (CID-9, idade, medicamentos...)
├── outliers.py             # detecção (IQR) e sinalização, sem alterar valores
├── selecao.py              # remoção de colunas constantes/redundantes + dicionário de dados
├── relatorios.py           # gera tabelas e figuras de docs/
└── executar_pipeline.py    # orquestra tudo e escreve os arquivos de saída
```

## 4. Etapas do pipeline

### 4.1 Carga e auditoria inicial (antes de mexer em qualquer coisa)
- Ler o CSV tratando `?` como nulo (`na_values=["?"]`). Guardar também a cópia sem esse tratamento para provar que os `?` existiam.
- Gerar `docs/tabelas/auditoria_inicial.csv`: por coluna → tipo, % nulos, nº de valores distintos, exemplo, observação.
- Registrar linhas, colunas, nº de pacientes únicos, nº de encontros duplicados por paciente.

### 4.2 Integração de dados (`IDS_mapping.csv`)
- O arquivo traz **três tabelas** separadas por linha em branco: `admission_type_id`, `discharge_disposition_id`, `admission_source_id`. Parsear cada uma.
- Fazer *join* para criar colunas legíveis (`tipo_admissao`, `destino_alta`, `origem_admissao`). Mostrar no docs o "antes (códigos) → depois (nomes)".
- Valores como `NULL`, `Not Available`, `Not Mapped` no mapeamento devem ser unificados em uma única categoria `Nao_informado`.

### 4.3 Inconsistências e filtros (cada um vira uma linha do funil)
| ID | Problema | Regra |
|---|---|---|
| D01 | `gender = Unknown/Invalid` | Remover (muito poucos registros; confirmar a contagem) |
| D02 | Pacientes que **faleceram ou foram para hospice** na alta (`discharge_disposition_id` ∈ {11, 13, 14, 19, 20, 21}) | Remover: não podem ser readmitidos, então enviesariam o alvo |
| D03 | **Mesmo paciente com vários encontros** (`patient_nbr` repetido) | Manter só o **primeiro encontro** por paciente (menor `encounter_id`). Evita vazamento e viola independência das amostras |
| D04 | Categorias equivalentes escritas de formas diferentes | Padronizar texto (strip, mesma caixa) e unir `NULL`/`Not Available`/`Not Mapped` |

Confirme os códigos de D02 lendo `IDS_mapping.csv` (não confie apenas nesta lista). Registre quantas linhas cada filtro removeu.

### 4.4 Valores ausentes
Decidir **coluna por coluna**, com base no % de nulos medido em 4.1:

| ID | Coluna | Decisão sugerida | Motivo |
|---|---|---|---|
| D05 | `weight` | Remover coluna | Quase toda ausente: imputar seria inventar dado |
| D06 | `payer_code`, `medical_specialty` | Manter, nulo → categoria `Desconhecido` + flag `*_ausente` | Ausência pode ser informativa (ex.: paciente sem convênio) |
| D07 | `race` | `?` → `Desconhecido` | Evita perder linhas; é atributo sensível, tratar com cuidado |
| D08 | `diag_1`, `diag_2`, `diag_3` | Nulo → `Sem_diagnostico` | Ausência de diagnóstico secundário é real, não erro |
| D09 | `max_glu_serum`, `A1Cresult` com valor `None` | Renomear para `Nao_medido` | **Não é nulo**: significa que o exame não foi feito. Ponto didático importante |

Atenção: nenhuma imputação estatística (média/mediana/modelo) aqui. Se sobrar nulo numérico, documentar e deixar para a Pessoa 3 imputar no Pipeline.

### 4.5 Transformação de variáveis e engenharia de atributos
| ID | Nova coluna | Como | Por quê (insight esperado) |
|---|---|---|---|
| F01 | `readmit_30d` | `1` se `readmitted == "<30"`, senão `0` | Alvo binário do projeto; manter `readmitted` original no interim |
| F02 | `idade_media` | Ponto médio da faixa `[0-10)`, `[10-20)`... | Variável numérica ordinal em vez de 10 categorias |
| F03 | `grupo_diag_1/2/3` | Agrupar CID-9 em: Circulatório (390–459, 785), Respiratório (460–519, 786), Digestivo (520–579, 787), Diabetes (250.xx), Lesões (800–999), Musculoesquelético (710–739), Geniturinário (580–629, 788), Neoplasias (140–239), Outros (inclui códigos `V`/`E`) | Reduz ~800 códigos a ~9 grupos interpretáveis (agrupamento usado no artigo de Strack et al.) |
| F04 | `total_visitas_previas` | `number_outpatient + number_emergency + number_inpatient` | Resume utilização prévia do sistema |
| F05 | `usuario_frequente` | `1` se `number_inpatient >= 1` (confirmar limiar nos dados) | Candidato a insight de acompanhamento |
| F06 | `n_meds_alterados` | Nº de medicamentos com `Up` ou `Down` | Intensidade da mudança de tratamento |
| F07 | `n_meds_em_uso` | Nº de medicamentos diferentes de `No` | Complexidade terapêutica |
| F08 | `a1c_medido`, `glicose_medida` | `1` se exame diferente de `Nao_medido` | HbA1c é o tema do artigo-fonte |
| F09 | `grupo_destino_alta` | Casa / Transferência / Outros | Reduz cardinalidade, mantém significado clínico |
| F10 | `exames_por_dia`, `procedimentos_por_dia` | `num_lab_procedures / time_in_hospital` etc. | Intensidade de cuidado, normalizada pela estadia |
| F11 | `especialidade_agrupada` | Top-N especialidades + `Outras` | `medical_specialty` tem dezenas de níveis raros |

Não criar atributos que usem o alvo ou informação posterior à alta (vazamento). Manter nomes em português, `snake_case`.

### 4.6 Outliers (`outliers.py`)
- Para `time_in_hospital`, `num_lab_procedures`, `num_procedures`, `num_medications`, `number_outpatient`, `number_emergency`, `number_inpatient`, `number_diagnoses`: calcular limites IQR (1,5×) e proporção de outliers; gerar boxplots antes.
- **Não remover nem limitar** valores: são clinicamente plausíveis (pacientes graves). Criar colunas `*_outlier` (0/1) e documentar. Se a Pessoa 3 quiser *winsorizar*, ela faz no Pipeline com limites estimados só no treino.
- Documentar a distribuição assimétrica (cauda longa) e sugerir transformação `log1p` como opção para modelos lineares.

### 4.7 Seleção e redução de atributos (`selecao.py`)
- Remover identificadores (`encounter_id`, `patient_nbr`) da base de modelagem (manter no interim para rastreio).
- Remover colunas constantes ou quase constantes (ex.: alguns medicamentos com 99,9%+ `No`; verificar `examide`, `citoglipton`). Listar com a % de dominância.
- Remover colunas redundantes: códigos numéricos já substituídos por versões legíveis; pares com correlação |r| > 0,9 entre numéricas (listar e justificar).
- Gerar `dicionario_de_dados.csv`: `coluna`, `descricao_negocio` (linguagem simples, útil para a Pessoa 4), `tipo` (numerica / categorica / binaria / alvo), `origem` (original / derivada), `sensivel` (sim para race, gender, age), `observacao`.

### 4.7.1 Saídas finais
- `data/interim/diabetes_limpo.csv`: legível, com ID e colunas originais úteis.
- `data/processed/diabetes_modelagem.csv` e `.parquet`: sem IDs, com alvo `readmit_30d`, sem a coluna `readmitted` original (evita vazamento).
- Fixar `random_state=42` onde houver aleatoriedade. Não gerar split treino/teste (é da Pessoa 3).

## 5. Testes (`tests/test_pipeline_p1.py`, `pytest`)
Verificar, no mínimo:
- Nenhum `?` restante; nenhum nulo em colunas que a regra manda preencher.
- `patient_nbr` único na saída (D03). Nenhum código de alta de óbito/hospice (D02). Nenhum `Unknown/Invalid` em `gender` (D01).
- `readmit_30d` contém só 0/1 e a proporção de 1 bate com a calculada direto do CSV bruto (após os filtros).
- Colunas do dicionário = colunas do arquivo de modelagem.
- Rodar o pipeline duas vezes produz o mesmo hash do arquivo final.
- Funil de linhas: `linhas_entrada - removidas = linhas_saida`.

## 6. Documentação (`docs/README.md`) — será a base da apresentação

Público: a própria Pessoa 1, que vai explicar para a turma em cerca de **3 minutos** (Bloco 2 do cronograma). Linguagem simples, direta, sem enrolação. Tudo com **números gerados pelo código**.

Estrutura obrigatória:
1. **Resumo em 30 segundos**: de onde vem a base, o que estava errado, o que fizemos, o que ficou.
2. **Diagrama do pipeline** (bloco Mermaid, renderiza no GitHub): bruto → auditoria → integração → inconsistências → nulos → atributos → outliers → seleção → base final, com o nº de linhas/colunas em cada etapa.
3. **Funil de linhas** (tabela + gráfico de barras): quantas linhas saíram em D01, D02, D03 e por quê.
4. **Mapa de ausentes** (gráfico de barras horizontais do % de nulos por coluna, antes) e tabela das decisões D05–D09.
5. **Tabela "O que estava errado → o que fizemos → por quê"** com colunas: ID, problema, evidência (número), decisão, motivo, impacto. Esta é a tabela mais importante.
6. **Antes × depois** de pelo menos 3 variáveis (histograma/boxplot lado a lado), incluindo uma de outliers e a do alvo (balanceamento de classes).
7. **Atributos criados** (tabela F01–F11 com a intuição de negócio de cada um).
8. **Cuidados**: vazamento (por que mantemos só o 1º encontro; por que não escalonamos aqui), atributos sensíveis, limitações dos dados (1999–2008).
9. **O que entregamos para o resto do grupo**: arquivos, dicionário, como carregar em 2 linhas de Python.
10. **Roteiro de fala de 3 minutos** (tópicos curtos) e **10 perguntas prováveis da banca com resposta curta** (ex.: "por que removeu pacientes que morreram?", "por que não imputou a média?", "por que só o primeiro encontro?", "os outliers não atrapalham?").
11. **Como reproduzir** (comandos).

Regras de estilo: frases curtas; cada figura com título que já diga a conclusão; tabelas em vez de parágrafos quando possível; figuras salvas em `docs/figuras/` em PNG (dpi ≥ 150) com paleta consistente e rótulos em português.

Também gerar `docs/secao_base_de_dados.md`: texto corrido (~1 página) para a seção "Base de Dados" do artigo SBC: origem, contexto, nº de registros, variáveis, e as técnicas de pré-processamento (limpeza, ausentes, inconsistências, integração, transformação, seleção), citando Strack et al. (2014) e o DOI do UCI.

## 7. Critérios de pronto (checklist)
- [ ] `python -m src.p1_engenharia_dados.executar_pipeline` roda do zero sem erro.
- [ ] `pytest` passa.
- [ ] Todos os arquivos da seção 1 existem.
- [ ] Todo número do `docs/README.md` confere com `docs/tabelas/*.csv`.
- [ ] `python -m pydoc src.p1_engenharia_dados.limpeza` mostra a documentação das funções.
- [ ] README de `docs/` renderiza no GitHub (Mermaid e imagens).
- [ ] Commit em branch `p1/limpeza` e Pull Request aberto para as Pessoas 2 e 3 revisarem.
