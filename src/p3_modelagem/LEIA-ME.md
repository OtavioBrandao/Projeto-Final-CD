# Pessoa 3 — Etapa 01: contrato de dados e divisão temporal

Esta etapa prepara a classificação binária de **acidentes registrados** como
fatais ou não fatais. O alvo `acidente_fatal` é inteiro: 1 quando há pelo menos
uma morte, 0 caso contrário. Sua origem foi conferida em
`p1_engenharia_dados/engenharia_atributos.py`: `(mortos > 0)`.

Não há treinamento, imputação, codificação, escalonamento, seleção de limiares
ou métricas de modelos nesta entrega. A entrada é o CSV versionado em
`data/processed/acidentes_prf_modelagem.csv`; a base e os códigos das Pessoas 1
e 2 não são modificados.

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
