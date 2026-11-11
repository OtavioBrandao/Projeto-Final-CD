# Pessoa 3 — Etapas 01 e 02: contrato temporal e pipelines de classificação

Esta etapa prepara a classificação binária de **acidentes registrados** como
fatais ou não fatais. O alvo `acidente_fatal` é inteiro: 1 quando há pelo menos
uma morte, 0 caso contrário. Sua origem foi conferida em
`p1_engenharia_dados/engenharia_atributos.py`: `(mortos > 0)`.

Na Etapa 01 não houve treinamento ou transformação aprendida. A Etapa 02,
documentada ao final, acrescenta pipelines e testes funcionais pequenos,
sem experimentos comparativos definitivos. A entrada é o CSV versionado em
`data/processed/acidentes_prf_modelagem.csv`; a base e os códigos das Pessoas 1
e 2 não são modificados.

As listas e divisões da Etapa 01 continuam inalteradas. A documentação abaixo
registra aquela entrega; a seção **Etapa 02** descreve as novas fábricas.

## Cenários e listas explícitas

As listas ordenadas estão em `configuracao.py`. A seleção considera a origem
e a disponibilidade da informação, sem consultar rankings de associação da
EDA ou resultados do alvo de 2025.

### A — Retrospectivo (34 atributos)

Estima a probabilidade de fatalidade usando características observadas no
acidente. Informações dos envolvidos, clima, causa e tipo são descritivas e
não incorporam diretamente a contagem de mortos. Causa e tipo são definidos
após o atendimento; este cenário não é uma previsão operacional prévia.

```text
dia_semana, uf, br, km, tipo_acidente, fase_dia, sentido_via,
condicao_metereologica, tipo_pista, veiculos, latitude, longitude,
envolve_moto, envolve_caminhao, envolve_onibus, envolve_bicicleta,
envolve_pedestre, idade_media_veiculos, hora, faixa_horaria, mes,
fim_de_semana, eh_feriado, regiao, grupo_causa, via_reta, via_curva,
via_aclive, via_declive, via_intersecao, via_rotatoria_retorno,
via_obras_desvio, via_ponte_viaduto_tunel, area_urbana
```

### B — Informações antecipadas (12 atributos)

Estima a fatalidade **condicional à ocorrência de um acidente registrado** em
um local e período planejados. Não estima a ocorrência de acidentes. Os valores
de localização e calendário representam condições que podem ser especificadas
antes de uma ação de fiscalização, sem conhecer os futuros envolvidos.

```text
dia_semana, uf, br, km, latitude, longitude, hora, faixa_horaria, mes,
fim_de_semana, eh_feriado, regiao
```

| Atributos incluídos | Justificativa e disponibilidade |
|---|---|
| `uf`, `br`, `km`, `latitude`, `longitude`, `regiao` (A/B) | Local a planejar; região é um mapeamento fixo de UF. `km` deve ser interpretado com `br` e `uf`, nunca como identificador isolado de lugar. Coordenadas são posições observadas usadas como representação de um local planejável; não existe prova de precisão operacional do trecho. |
| `dia_semana`, `hora`, `faixa_horaria`, `mes`, `fim_de_semana`, `eh_feriado` (A/B) | Calendário/horário planejáveis; derivações fixas da data e hora, sem aprendizado. Faixa, fim de semana e feriado seguem as definições da Pessoa 1. |
| `tipo_acidente`, `grupo_causa` (só A) | Descrevem mecanismo e causa atribuída pela PRF após o acidente. Não disponíveis antecipadamente. |
| `veiculos`, `envolve_moto`, `envolve_caminhao`, `envolve_onibus`, `envolve_bicicleta`, `envolve_pedestre`, `idade_media_veiculos` (só A) | Descrevem os envolvidos efetivos; resumos de pessoas não usam `estado_fisico` nem vítimas. Não é possível conhecer esses envolvidos antes do acidente. |
| `condicao_metereologica` (só A) | Tempo observado no acidente; não é uma previsão meteorológica previamente arquivada. |
| `fase_dia`, `sentido_via` (só A) | Estados registrados no boletim. Não há derivação astronômica da fase nem demonstração de sentido previamente planejado nesta base. |
| `tipo_pista`, `via_reta`, `via_curva`, `via_aclive`, `via_declive`, `via_intersecao`, `via_rotatoria_retorno`, `via_ponte_viaduto_tunel`, `area_urbana` (só A) | Infraestrutura potencialmente estável, mas extraída do registro do acidente; sem inventário externo vigente antes da ocorrência. A escolha conservadora é excluí-la de B até existir comprovação. |
| `via_obras_desvio` (só A) | Estado temporário observado; sem histórico de obras disponível no instante de planejamento. |

`br` é sempre entregue em X como texto, inclusive se a entrada usar códigos
numéricos. Isso apenas declara seu tipo categórico; não cria um codificador,
não aprende categorias e não impõe ordenação numérica às rodovias. Meses e
horas permanecem numéricos nesta etapa; o tratamento cíclico será uma decisão
posterior, validada apenas em 2024.

## Exclusões e incertezas

| Colunas | Decisão |
|---|---|
| `acidente_fatal` | Exclusivamente y; nunca preditor. |
| `ano` | Exclusivamente metadado para divisão temporal. |
| `veiculos_outlier`, `km_outlier`, `idade_media_condutores_outlier`, `idade_media_veiculos_outlier` | Excluídos de A e B: os limites IQR foram calculados na base conjunta 2024/2025, conforme `executar_pipeline.py` e `outliers.py` da Pessoa 1. Valores originais são mantidos. |
| `idade_media_condutores`, `pct_condutores_homens` | Existem na base, mas excluídos dos dois cenários. São sensíveis e a documentação da Pessoa 1 orienta reservá-los à análise de equidade. Essa decisão normativa, sem uso de associações com o alvo, pode ser revisada explicitamente no futuro; não são consequências nem vazamento direto. |
| `mortos`, `feridos`, `feridos_leves`, `feridos_graves`, `ilesos`, `ignorados`, `pessoas`, `classificacao_acidente`, `estado_fisico`, `contagens_inconsistentes` | Consequências, gravidade ou diagnósticos das contagens de vítimas. Já ausentes no CSV processado; proibidos como preditores e rejeitados se aparecerem como novas colunas. |
| `id`, `id_veiculo`, `ano_arquivo`, `regional`, `delegacia`, `uop` | Identificadores, origem administrativa ou responsável pelo atendimento. Ausentes no CSV; não usados como preditores. |

Todos os atributos de A que não constam na lista B são excluídos de B pelas
justificativas de disponibilidade da tabela anterior. Não há colunas derivadas
das consequências entre os 34 preditores de A. No CSV, a derivação direta de
`mortos` identificada é o próprio alvo, que fica separado.

O contrato é fechado para as 42 colunas atuais, inclusive as excluídas de X.
Novas colunas não são aceitas silenciosamente: é necessária revisão explícita
do contrato e dos testes. Ausência de qualquer coluna, duplicação de nomes,
índices duplicados/nulos, tipos incompatíveis, valores não finitos, alvo inválido,
meses/horas inválidos ou anos inesperados produzem `ValueError` com contexto.
Não é exigida uma contagem fixa de registros nas funções de produção.

## Auditoria conferida diretamente no CSV

Conferência em 11/11/2026, no commit-base
`af342ea48a8031b3f823b4452ad46c0ec436233a`.
SHA-256 do CSV:
`518f7d54e8f0093befd41939b47117344ee2983ab977d47124794c3212f05552`.

Dimensão: **145.342 linhas × 42 colunas**. Distribuição total: **135.102 não
fatais**, **10.240 fatais**, taxa **7,04545%**. Estes são resultados descritivos
da auditoria, não critérios para escolher atributos, limites ou modelos.

| Ano | Registros | Não fatais | Fatais |
|---|---:|---:|---:|
| 2024 — desenvolvimento | 73.006 | 67.870 | 5.136 |
| 2025 — teste final reservado | 72.336 | 67.232 | 5.104 |

Abaixo, os tipos inferidos por `pandas.read_csv` no ambiente da verificação
(pandas 3.0.0). O carregador da Pessoa 3 declara `br` como `string`; em outras
versões do pandas textos podem aparecer como `object`. A validação usa tipos
semânticos, não o nome exato do dtype de texto.

| Coluna | Tipo observado | Nulos |
|---|---|---:|
| dia_semana | str | 0 |
| uf | str | 0 |
| br | str | 0 |
| km | float64 | 0 |
| tipo_acidente | str | 0 |
| fase_dia | str | 0 |
| sentido_via | str | 0 |
| condicao_metereologica | str | 0 |
| tipo_pista | str | 0 |
| veiculos | int64 | 0 |
| latitude | float64 | 0 |
| longitude | float64 | 0 |
| envolve_moto | int64 | 0 |
| envolve_caminhao | int64 | 0 |
| envolve_onibus | int64 | 0 |
| envolve_bicicleta | int64 | 0 |
| envolve_pedestre | int64 | 0 |
| idade_media_condutores | float64 | 7.009 |
| pct_condutores_homens | float64 | 5.037 |
| idade_media_veiculos | float64 | 5.545 |
| hora | int64 | 0 |
| faixa_horaria | str | 0 |
| mes | int64 | 0 |
| ano | int64 | 0 |
| fim_de_semana | int64 | 0 |
| eh_feriado | int64 | 0 |
| regiao | str | 0 |
| grupo_causa | str | 0 |
| via_reta | int64 | 0 |
| via_curva | int64 | 0 |
| via_aclive | int64 | 0 |
| via_declive | int64 | 0 |
| via_intersecao | int64 | 0 |
| via_rotatoria_retorno | int64 | 0 |
| via_obras_desvio | int64 | 0 |
| via_ponte_viaduto_tunel | int64 | 0 |
| area_urbana | int64 | 0 |
| veiculos_outlier | int64 | 0 |
| km_outlier | int64 | 0 |
| idade_media_condutores_outlier | int64 | 0 |
| idade_media_veiculos_outlier | int64 | 0 |
| acidente_fatal | int64 | 0 |

Os nulos não são preenchidos ou removidos. Além das três médias/percentual
acima, o contrato aceita coordenadas nulas, como previsto no dicionário para
posições impossíveis; na versão auditada não há coordenadas nulas.
`Nao_informado` permanece uma categoria, sem imputação.

## Divisões temporais e identidade das linhas

Todos os registros de 2024 são desenvolvimento, todos os de 2025 são teste
final reservado. Não há sorteio, estratificação, exclusão ou reamostragem.
As proporções de classes podem variar naturalmente por período.

| Rodada | Treino em 2024 | N treino | Fatais treino | Validação em 2024 | N validação | Fatais validação |
|---|---|---:|---:|---|---:|---:|
| 1 | Janeiro–junho | 35.160 | 2.452 | Julho–agosto | 12.537 | 909 |
| 2 | Janeiro–agosto | 47.697 | 3.361 | Setembro–outubro | 12.599 | 869 |
| 3 | Janeiro–outubro | 60.296 | 4.230 | Novembro–dezembro | 12.710 | 906 |

Treino e validação são disjuntos **dentro de cada rodada**; nenhum deles inclui
2025. Janelas expansivas reutilizam meses de validações anteriores nos treinos
posteriores, conforme o protocolo solicitado. Janeiro–junho só participa de
treinos e não recebe previsões de validação interna.

Os meses/anos são extraídos da data pela Pessoa 1 e conferidos aqui como
inteiros nos domínios esperados. A ordem original é preservada, mesmo se a
entrada não estiver ordenada cronologicamente; os cortes usam máscaras por mês,
sem necessidade de recuperar o dia do acidente.

O carregador atribui o índice `linha_original`: posição do registro no CSV,
começando em zero e sem contar o cabeçalho. O índice acompanha X, y, desenvolvimento,
teste e cortes, sem entrar em X. As referências de cada `CorteTemporal` são
**rótulos para `.loc`**, não posições para `.iloc`. Reutilize os mesmos objetos
de cortes em todos os cenários e algoritmos. Não redefina índices após separar.

Essa identidade vale para o mesmo conteúdo e ordem do CSV. Não é o ID original
da PRF, já removido pela Pessoa 1, e muda se a base for regenerada/reordenada.
Guarde o hash da entrada com resultados futuros; não é possível recuperar uma
identidade administrativa persistente apenas a partir deste CSV.

## Interface reutilizável

- `carregar_base(caminho=CSV_MODELAGEM)`: lê/valida o CSV e nomeia o índice.
- `validar_esquema(base)`: valida o contrato sem modificar o DataFrame.
- `selecionar_xy(base, cenario)`: retorna cópias de X/y para `"A"` ou `"B"`.
- `separar_desenvolvimento_teste(base)`: retorna cópias `(dados_2024, dados_2025)`;
  exige ambos os anos presentes.
- `gerar_cortes_temporais(dados_2024)`: retorna três `CorteTemporal`, com rodada
  e índices de treino/validação; rejeita 2025 e partições vazias.

Exemplo, executado a partir da raiz do repositório:

```python
from src.p3_modelagem.dados import (
    carregar_base, separar_desenvolvimento_teste,
    gerar_cortes_temporais, selecionar_xy,
)

base = carregar_base()
desenvolvimento, teste_final = separar_desenvolvimento_teste(base)
cortes = gerar_cortes_temporais(desenvolvimento)

for cenario in ("A", "B"):
    X, y = selecionar_xy(desenvolvimento, cenario)
    for corte in cortes:
        X_treino, y_treino = X.loc[corte.treino], y.loc[corte.treino]
        X_validacao, y_validacao = X.loc[corte.validacao], y.loc[corte.validacao]
        # Etapa 01 termina com os dados separados. Nenhum ajuste aqui.

# teste_final permanece reservado para a avaliação final futura.
```

## Limitações metodológicas

- A base contém somente acidentes registrados pela PRF. Não há exposição ao
  tráfego nem exemplos de trechos/períodos sem acidentes: B não calcula risco
  absoluto de acidente ou quantidade esperada de mortes em uma rodovia.
- O teste de 2025 está reservado para avaliação futura, mas já foi descrito na
  auditoria e na EDA anterior; não é um teste completamente cego. Nenhum ranking,
  efeito ou teste de associação dessas análises foi usado na seleção desta etapa.
- A Pessoa 1 calculou dominância e correlações na base conjunta. A documentação
  e as tabelas registram zero exclusões por esses critérios nesta versão. Ainda
  existe dependência da engenharia anterior feita com os dois anos; nenhum
  procedimento de seleção global é reexecutado aqui. Os quatro indicadores
  IQR globais são expressamente excluídos.
- Causa, caracterização dos envolvidos e preenchimento podem depender do
  atendimento e da gravidade, mesmo sem uma derivação direta do alvo. A é
  retrospectivo; associações futuras não demonstrarão causalidade.
- A exclusão de infraestrutura de B é conservadora. Sua incorporação futura
  requer fonte externa ou prova de disponibilidade anterior, sem consultar
  desempenho em 2025 para tomar essa decisão.
- Sem a data completa não validamos dia da semana/feriado contra a data nem
  verificamos ordem dentro do mês; os cortes mensais não dependem disso.
- O feriado herda a convenção da Pessoa 1 (inclui Carnaval e datas móveis),
  não representa todos os calendários estaduais/municipais.
- Geografia pode permitir memorização de locais e `km` não possui o mesmo
  significado entre UFs/rodovias. A representação conjunta e a generalização
  espacial precisam ser avaliadas em etapas futuras, com desenvolvimento de 2024.

## Testes

Na raiz, com as dependências de `requirements.txt` disponíveis:

```powershell
python -m pytest tests/test_modelagem_p3.py -q
python -m pytest -q
```

Os testes usam uma base sintética pequena, com meses intercalados, índices não
sequenciais e contagens de fatais conhecidas. Verificam limites, tipos, erros,
integridade das linhas, ausência de reamostragem, listas autorizadas e
determinismo. Alterar valores de 2025 não altera o desenvolvimento/cortes.
Testes de integração carregam o CSV real e conferem contagens da versão auditada,
dicionário e cortes. Nenhum teste modifica arquivos das Pessoas 1 e 2.

No ambiente Windows desta execução, o alias `python` da Microsoft Store não
funcionou. Os comandos reais usam o executável Python 3.12 instalado:

```powershell
& 'C:/Users/victo/AppData/Local/Programs/Python/Python312/python.exe' -m pytest tests/test_modelagem_p3.py -q
& 'C:/Users/victo/AppData/Local/Programs/Python/Python312/python.exe' -m pytest -q
```

Resultados reais em 11/11/2026: **45 testes da Pessoa 3 aprovados em 5,12 s**;
**96 testes da suíte completa aprovados em 76,93 s**, incluindo os 31 da Pessoa 1
e os 20 da Pessoa 2. Nenhuma falha ou teste ignorado. Os dados brutos necessários
à suíte da Pessoa 1 estavam presentes. Python 3.12, pandas 3.0.0 e pytest 9.1.1.

## Etapa 02 — Pré-processamento e modelos

`preprocessamento.py` e `modelos.py` criam objetos novos e **não ajustados**.
Não leem dados, não escolhem divisões, não executam experimentos na importação
e não exportam modelos. Reutilizam o contrato de `configuracao.py`; a seleção
e as divisões continuam sendo responsabilidade das interfaces de `dados.py`.

### Partição e transformações

| Cenário | Numéricas | Categóricas | Binárias | Total |
|---|---:|---:|---:|---:|
| A | 7 | 11 | 16 | 34 |
| B | 5 | 5 | 2 | 12 |

`identificar_tipos(cenario)` cruza as listas semânticas do contrato com uma
lista explícita de numéricas: `km`, `veiculos`, `latitude`, `longitude`,
`idade_media_veiculos`, `hora`, `mes`. Rejeita atributos sem tipo, ambiguidade,
duplicações e preditores proibidos. Cada feature deve pertencer a um único
grupo. Mês/hora continuam numéricos; não há engenharia nova.

`criar_preprocessamento(cenario, escalonar=False)` retorna um `ColumnTransformer`:

- Numéricas: `SimpleImputer(strategy="median", keep_empty_features=True)`.
  Se uma coluna estiver inteiramente ausente no treino, permanece na matriz
  com preenchimento zero, comportamento do imputador. A Regressão Logística
  acrescenta `StandardScaler`; Dummy e árvores não recebem escalonamento.
- Categóricas: conversão textual sem aprendizado, incluindo `br`, normalização
  de `None`/`pd.NA`/`NaN`, imputação constante `__ausente__` e
  `OneHotEncoder(handle_unknown="ignore", sparse_output=True)`. Categorias
  inéditas geram zeros no bloco correspondente, sem mudar a dimensionalidade.
  `Nao_informado` continua sendo uma categoria observada.
- Binárias: passagem direta dos valores 0/1 validados por `selecionar_xy`;
  não há imputação, One-Hot ou escalonamento redundante nesse grupo.

O `ColumnTransformer` usa seletores explícitos e `remainder="drop"`: extras
não entram na matriz do estimador; para validar a base, use sempre
`selecionar_xy`. Features obrigatórias ausentes geram erro. O parâmetro
`sparse_threshold=1.0` evita densificação quando a densidade estimada é menor
que 1. Uma entrada pequena com uma única categoria em cada coluna pode resultar
em saída densa. As oito combinações reais testadas produziram matrizes esparsas.

### Fábricas e configurações iniciais

Use `criar_estimador(algoritmo)` ou, preferencialmente,
`criar_pipeline(cenario, algoritmo)`. Os nomes aceitos estão em `ALGORITMOS`:

| Algoritmo | Configuração inicial |
|---|---|
| `dummy` | `DummyClassifier(strategy="prior")`; probabilidades = proporções do treino. |
| `regressao_logistica` | `LogisticRegression(max_iter=2000, random_state=42)`; solver e regularização padrão (`lbfgs`, L2, C=1). |
| `random_forest` | 100 árvores, profundidade máxima 12, mínimo de 3 amostras por folha, `n_jobs=2`, `random_state=42`. |
| `xgboost` | `binary:logistic`, `logloss`, `tree_method="hist"`, `device="cpu"`, 100 árvores, profundidade 4, taxa 0,1, `n_jobs=2`, `random_state=42`. |

Pesos de classe e limiares permanecem nos padrões; não há busca ou escolha
automática de vencedor. O XGBoost é importado apenas ao ser solicitado. Se não
estiver disponível, a fábrica emite `ImportError` com instrução de instalação;
os outros três algoritmos continuam disponíveis, sem substituição simulada.

### Uso com a divisão da Etapa 01

O exemplo abaixo mostra **uma** janela. Não é executado automaticamente e não
autoriza treinar em toda a base ou avaliar 2025 nesta entrega.

```python
import numpy as np
from src.p3_modelagem.dados import (
    carregar_base, separar_desenvolvimento_teste,
    gerar_cortes_temporais, selecionar_xy,
)
from src.p3_modelagem.modelos import criar_pipeline

base = carregar_base()
desenvolvimento, teste_final = separar_desenvolvimento_teste(base)
corte = gerar_cortes_temporais(desenvolvimento)[0]
X, y = selecionar_xy(desenvolvimento, "B")
pipeline = criar_pipeline("B", "regressao_logistica")
pipeline.fit(X.loc[corte.treino], y.loc[corte.treino])
classes = pipeline.predict(X.loc[corte.validacao])
indice_positivo = np.flatnonzero(pipeline.classes_ == 1).item()
p_fatal = pipeline.predict_proba(X.loc[corte.validacao])[:, indice_positivo]
# teste_final continua reservado; use um pipeline novo em cada futuro treino.
```

Com as duas classes no treino, `classes_` é `[0, 1]` e a coluna 1 corresponde
a `P(acidente_fatal=1)`. O treino deve conter ambas as classes, como nos testes.
O pipeline não recebe `ano` como feature e não pode deduzir o período sozinho:
cabe ao chamador usar os cortes. Imputadores, vocabulários e escalonadores são
aprendidos no `fit` desse treino; `predict`/`predict_proba` apenas transformam.

### Ambiente e reprodução dos testes

Em 11/11/2026 foi criado `.venv` local, ignorado pelo Git, com acesso às
dependências Python 3.12 já existentes (`--system-site-packages`). XGBoost não
estava instalado; a primeira tentativa de download foi bloqueada pela rede
do sandbox. A instalação autorizada fora do sandbox concluiu com XGBoost
3.4.1 **somente nesse ambiente virtual**. Nenhuma dependência global ou arquivo
`requirements.txt` foi alterado. Versões usadas: numpy 2.4.2, pandas 3.0.0,
scipy 1.17.0 e scikit-learn 1.7.2.

```powershell
# Para recriar em outro ambiente: instalar requirements.txt no venv escolhido.
& '.venv/Scripts/python.exe' -m pytest tests/test_modelos_p3.py -q -s
& '.venv/Scripts/python.exe' -m pytest -q
```

Os 38 testes novos verificam ajuste e inferência para os quatro algoritmos
em A/B, cenários inválidos, partição exata das features, exclusão de alvo/ano,
mediana conhecida (3), média escalonada conhecida (3,8), categorias inéditas,
nulos textuais/numéricos, coluna inteiramente ausente no treino, binários
inalterados, saída esparsa, probabilidade positiva, prevalência conhecida do
Dummy (0,2), sinal conhecido para os demais algoritmos, determinismo e ausência
de mutação das entradas. A indisponibilidade do XGBoost é simulada apenas para
testar o erro e a independência dos outros modelos.

Resultados em 11/11/2026: execução isolada da Etapa 02 com **38 aprovados em
9,00 s**; última execução da suíte completa, após a revisão dos testes, com
**134 aprovados em 65,97 s** (96 anteriores + 38 novos). Nenhuma falha ou teste
ignorado. `git diff --check` também passou.

As sentinelas de validação (`longitude=1e9`, categoria exclusiva) não alteram
medianas ou vocabulários. Mudar atributos e alvos sintéticos de 2025 não muda
os dados de desenvolvimento nem as probabilidades dos modelos. Estimadores
novos rejeitam previsão antes de `fit`. Na Random Forest, as árvores e classes
previstas são idênticas; probabilidades admitem tolerância absoluta de `1e-14`
pela ordem da soma paralela (diferenças observadas da ordem de `1e-16`).

### Verificação funcional com dados reais de 2024

`test_integracao_real_2024` reutiliza carga, separação, cortes e seleção da
Etapa 01. Usa somente a primeira janela: amostra aleatória fixa de **384**
registros de janeiro–junho para treino e **128** de julho–agosto para inferência.
Não há balanceamento; o treino contém **355 não fatais e 29 fatais**. As duas
classes estão presentes também na validação. Nenhum registro real de 2025
entra no treinamento, transformação aprendida ou inferência desta verificação.

As oito combinações algoritmo/cenário treinaram e produziram previsões e
probabilidades válidas. Em A, a amostra de treino tem 15 valores ausentes;
em B, zero. As matrizes transformadas da validação têm **128 × 168** em A e
**128 × 110** em B, todas esparsas. Na execução isolada, cada combinação levou
aproximadamente **0,04–0,23 s** (fit, inferência e verificações). Não houve
aviso de convergência ou falha de tipos, dimensionalidade ou memória. O pico
de memória não foi medido; a amostra não comprova custo no conjunto completo.

Não foram calculadas métricas comparativas ou ajustados hiperparâmetros a
partir desses resultados. Os testes descartam seus objetos ao terminar;
nenhum modelo definitivo é exportado. A suíte anterior ainda faz a auditoria
descritiva de 2025 documentada na Etapa 01, sem treinamento nesse ano.

### Limite da entrega

A Etapa 03 recebe pipelines novos por cenário/algoritmo, compatíveis com os
mesmos cortes temporais e aptos a produzir classes e probabilidades. Não foram
executadas validação comparativa completa, otimização, balanceamento, ajuste
de limiares, seleção de vencedor, avaliação final em 2025, curvas ou matrizes
finais, SHAP ou dashboard. O contrato anterior e as limitações metodológicas
acima permanecem vigentes. Esta entrega permanece local em `p3/modelagem`,
sem commit, push, PR, merge ou rebase.
