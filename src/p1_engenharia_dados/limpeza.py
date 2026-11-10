"""Limpeza determinística: inconsistências (D01–D04) e valores ausentes (D05–D09).

Todas as funções são **puras**: recebem um DataFrame e devolvem um DataFrame
novo, sem alterar o original. Elas só aplicam regras fixas, baseadas no
conhecimento do domínio. Nada aqui é "aprendido" dos dados (nada de média,
mediana ou modelo para imputar), para não haver vazamento: essas estimativas
pertencem ao ``Pipeline`` do sklearn da Pessoa 3, depois da divisão treino/teste.
"""

from __future__ import annotations

import logging

import pandas as pd
from pandas.api.types import is_numeric_dtype

from src.p1_engenharia_dados import configuracao as cfg

logger = logging.getLogger(__name__)


def colunas_de_texto(df: pd.DataFrame) -> list[str]:
    """Lista as colunas não numéricas (texto) de um DataFrame.

    Usamos ``is_numeric_dtype`` em vez de ``select_dtypes("object")`` porque, no
    pandas 3, texto passa a ter dtype próprio (``str``) e o filtro por
    ``object`` deixaria de encontrá-lo.

    Args:
        df: DataFrame a inspecionar.

    Returns:
        Nomes das colunas de texto, na ordem original.
    """
    return [c for c in df.columns if not is_numeric_dtype(df[c])]


# ---------------------------------------------------------------------------
# D04 — padronização de texto
# ---------------------------------------------------------------------------


def padronizar_texto(df: pd.DataFrame) -> pd.DataFrame:
    """Padroniza o texto das categorias (decisão D04).

    * tira espaços das pontas e colapsa espaços repetidos em todas as colunas
      de texto (evita que ``"Male "`` e ``"Male"`` virem categorias diferentes);
    * unifica grafias diferentes da mesma categoria, conforme
      ``cfg.PADRONIZACAO_CATEGORIAS`` (ex.: ``readmitted`` usa ``NO`` enquanto o
      resto da base usa ``No``).

    Nulos são preservados.

    Args:
        df: Base com colunas de texto.

    Returns:
        Cópia do DataFrame com o texto padronizado.
    """
    saida = df.copy()
    for coluna in colunas_de_texto(saida):
        saida[coluna] = saida[coluna].str.strip().str.replace(r"\s+", " ", regex=True)
    for coluna, trocas in cfg.PADRONIZACAO_CATEGORIAS.items():
        if coluna in saida.columns:
            saida[coluna] = saida[coluna].replace(trocas)
    return saida


def contar_celulas_alteradas(antes: pd.DataFrame, depois: pd.DataFrame) -> int:
    """Conta quantas células de texto mudaram entre duas versões da base.

    Args:
        antes: Base antes da padronização.
        depois: Base depois da padronização (mesmas linhas e colunas).

    Returns:
        Número de células de texto com valor diferente (nulo contra nulo conta
        como igual).
    """
    total = 0
    for coluna in colunas_de_texto(antes):
        a, d = antes[coluna], depois[coluna]
        total += int((~((a == d) | (a.isna() & d.isna()))).sum())
    return total


# ---------------------------------------------------------------------------
# D01, D02, D03 — filtros de linhas
# ---------------------------------------------------------------------------


def remover_genero_invalido(df: pd.DataFrame) -> pd.DataFrame:
    """D01 — remove registros com ``gender = Unknown/Invalid``.

    São pouquíssimos registros e não há como recuperar o valor correto. Perder
    algumas linhas é melhor do que criar uma categoria de gênero artificial.

    Args:
        df: Base com a coluna ``gender``.

    Returns:
        DataFrame sem os registros de gênero inválido (índice refeito).
    """
    return df.loc[df["gender"] != cfg.GENERO_INVALIDO].reset_index(drop=True)


def manter_primeiro_encontro(df: pd.DataFrame) -> pd.DataFrame:
    """D03 — mantém só o primeiro encontro de cada paciente.

    "Primeiro" = menor ``encounter_id``. Motivos:

    1. **Independência das amostras:** vários encontros do mesmo paciente são
       parecidos entre si; tratá-los como linhas independentes infla as métricas.
    2. **Vazamento:** se o mesmo paciente cair no treino e no teste, o modelo
       "reconhece" o paciente em vez de aprender o padrão.

    Args:
        df: Base com ``patient_nbr`` e ``encounter_id``.

    Returns:
        Uma linha por paciente, ordenada por ``encounter_id``.
    """
    return (
        df.sort_values("encounter_id", kind="stable")
        .drop_duplicates(subset="patient_nbr", keep="first")
        .reset_index(drop=True)
    )


def identificar_codigos_obito_hospice(tabela_alta: pd.DataFrame) -> frozenset[int]:
    """Descobre, no ``IDS_mapping``, os códigos de alta por óbito ou hospice.

    Em vez de confiar só na lista do plano, lemos a descrição original de cada
    código e procuramos "expired" ou "hospice".

    Args:
        tabela_alta: Tabela ``discharge_disposition_id`` de
            ``carregar.carregar_ids_mapping``.

    Returns:
        Conjunto de códigos de alta que indicam óbito ou hospice.
    """
    casam = tabela_alta["descricao_original"].str.contains(
        cfg.PADRAO_OBITO_HOSPICE, case=False, regex=True
    )
    return frozenset(int(c) for c in tabela_alta.loc[casam, "codigo"])


def remover_obito_hospice(df: pd.DataFrame, codigos: frozenset[int]) -> pd.DataFrame:
    """D02 — remove encontros que terminaram em óbito ou hospice.

    Quem faleceu ou foi para cuidados paliativos **não pode ser readmitido**.
    Mantê-los faria o alvo ``readmit_30d = 0`` por um motivo que não tem a ver
    com a qualidade do atendimento, enviesando a taxa de readmissão para baixo.

    Args:
        df: Base com ``discharge_disposition_id``.
        codigos: Códigos de alta a remover.

    Returns:
        DataFrame sem esses encontros (índice refeito).
    """
    return df.loc[~df["discharge_disposition_id"].isin(codigos)].reset_index(drop=True)


# ---------------------------------------------------------------------------
# D05–D09 — valores ausentes
# ---------------------------------------------------------------------------


def tratar_ausentes(df: pd.DataFrame) -> pd.DataFrame:
    """Aplica as decisões D05–D09 sobre valores ausentes.

    =====  ==========================  =====================================
    ID     Coluna                      Decisão
    =====  ==========================  =====================================
    D05    ``weight``                  removida (quase toda ausente)
    D06    ``payer_code``,             nulo -> ``Desconhecido`` + coluna
           ``medical_specialty``       ``*_ausente`` (0/1)
    D07    ``race``                    nulo -> ``Desconhecido``
    D08    ``diag_1/2/3``              nulo -> ``Sem_diagnostico``
    D09    ``max_glu_serum``,          ``None`` -> ``Nao_medido``
           ``A1Cresult``               (não era nulo: exame não feito)
    =====  ==========================  =====================================

    Nenhuma imputação estatística é feita aqui.

    Args:
        df: Base com nulos reais (``NaN``) nas colunas acima.

    Returns:
        Cópia do DataFrame tratada.
    """
    saida = df.copy()

    # D05 — weight: 96,9% ausente. Preencher seria inventar o peso de quase todos.
    saida = saida.drop(columns=["weight"])

    # D06 — a falta de convênio/especialidade pode ter relação com o desfecho
    # (ex.: paciente sem plano de saúde). Guardamos a informação "estava ausente"
    # numa coluna própria ANTES de preencher o texto.
    for coluna in ("payer_code", "medical_specialty"):
        saida[f"{coluna}_ausente"] = saida[coluna].isna().astype("int8")
        saida[coluna] = saida[coluna].fillna(cfg.ROTULO_DESCONHECIDO)

    # D07 — race: só ~2% ausente. Categoria própria evita perder linhas.
    # Atributo sensível: a categoria "Desconhecido" NÃO deve ser usada para
    # discriminar ninguém (ver seção de ética na documentação).
    saida["race"] = saida["race"].fillna(cfg.ROTULO_DESCONHECIDO)

    # D08 — um paciente pode simplesmente não ter 2º ou 3º diagnóstico.
    # A ausência é um fato clínico, não um erro de coleta.
    for coluna in cfg.COLUNAS_DIAGNOSTICO:
        saida[coluna] = saida[coluna].fillna(cfg.ROTULO_SEM_DIAGNOSTICO)

    # D09 — "None" aqui significa "exame não realizado". Renomear deixa claro
    # que é uma categoria real (e evita confusão com nulo).
    for coluna in ("max_glu_serum", "A1Cresult"):
        saida[coluna] = saida[coluna].replace({"None": cfg.ROTULO_NAO_MEDIDO})

    return saida


def nulos_restantes(df: pd.DataFrame) -> pd.Series:
    """Lista as colunas que ainda têm nulos depois do tratamento.

    Args:
        df: Base tratada.

    Returns:
        Série com a contagem de nulos, apenas das colunas com pelo menos um.
    """
    contagem = df.isna().sum()
    return contagem[contagem > 0]
