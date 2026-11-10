"""Engenharia de atributos (F01–F11): novas colunas e agrupamentos.

Regra de ouro: **só usamos informação disponível até a alta do paciente** e
nunca usamos o alvo para criar atributos (isso seria vazamento). Todas as
transformações são regras fixas (somas, razões, agrupamentos do conhecimento
clínico); nada é "aprendido" dos dados, com uma exceção consciente e
documentada: a lista das ``N`` especialidades mais frequentes (F11) usa apenas
contagem de frequência, sem olhar o alvo.

Cada função recebe um DataFrame e devolve uma **cópia** com a(s) coluna(s) nova(s).
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from src.p1_engenharia_dados import configuracao as cfg

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# F01 — alvo
# ---------------------------------------------------------------------------


def criar_alvo(df: pd.DataFrame) -> pd.DataFrame:
    """F01 — cria ``readmit_30d`` (1 se readmitido em menos de 30 dias).

    A coluna original ``readmitted`` (``<30``, ``>30``, ``No``) continua no
    interim para rastreio, mas **sai da base de modelagem**: ela contém a
    própria resposta e deixá-la como atributo seria vazamento.

    Args:
        df: Base com ``readmitted`` já padronizada.

    Returns:
        Cópia com a coluna ``readmit_30d`` (0/1).
    """
    saida = df.copy()
    saida[cfg.COLUNA_ALVO] = (saida["readmitted"] == cfg.CLASSE_POSITIVA_READMITIDO).astype("int8")
    return saida


# ---------------------------------------------------------------------------
# F02 — idade
# ---------------------------------------------------------------------------


def criar_idade_media(df: pd.DataFrame) -> pd.DataFrame:
    """F02 — converte a faixa etária (``[60-70)``) no seu ponto médio (65).

    Transforma 10 categorias em uma variável numérica ordinal, o que modelos
    lineares e árvores aproveitam melhor.

    Args:
        df: Base com a coluna ``age`` no formato ``[inicio-fim)``.

    Returns:
        Cópia com a coluna ``idade_media``.

    Raises:
        ValueError: Se algum valor de ``age`` não seguir o formato esperado.
    """
    saida = df.copy()
    limites = saida["age"].str.extract(r"^\[(\d+)-(\d+)\)$").astype(float)
    if limites.isna().any().any():
        raise ValueError("Há faixas etárias fora do formato [inicio-fim).")
    saida["idade_media"] = (limites[0] + limites[1]) / 2
    return saida


# ---------------------------------------------------------------------------
# F03 — grupos de diagnóstico (CID-9)
# ---------------------------------------------------------------------------


def agrupar_cid9(codigos: pd.Series) -> pd.Series:
    """Agrupa códigos CID-9 em 9 grandes grupos clínicos (agrupamento de Strack et al.).

    Os ~800 códigos distintos viram poucos grupos interpretáveis:

    ======================  =====================================
    Grupo                   Faixa CID-9 (parte inteira do código)
    ======================  =====================================
    Circulatorio            390–459 e 785
    Respiratorio            460–519 e 786
    Digestivo               520–579 e 787
    Diabetes                250 (250.xx)
    Lesoes                  800–999
    Musculoesqueletico      710–739
    Geniturinario           580–629 e 788
    Neoplasias              140–239
    Outros                  todo o resto, inclusive códigos ``V`` e ``E``
    ======================  =====================================

    ``Sem_diagnostico`` (D08) e nulos continuam como ``Sem_diagnostico``.

    Args:
        codigos: Série com os códigos (texto), ex.: ``"250.83"``, ``"V57"``.

    Returns:
        Série com o nome do grupo, mesmo índice de ``codigos``.
    """
    texto = codigos.astype("string")
    # Códigos V (fatores de saúde) e E (causas externas) não são numéricos.
    nao_numerico = texto.str.match(r"^[VvEe]").fillna(False).astype(bool)
    # astype(float): o tipo anulável do pandas vira float comum (NA -> NaN).
    numero = pd.to_numeric(texto.mask(nao_numerico), errors="coerce").astype("float64")
    inteiro = np.floor(numero)  # 250.83 -> 250

    condicoes = [
        inteiro.between(390, 459) | (inteiro == 785),
        inteiro.between(460, 519) | (inteiro == 786),
        inteiro.between(520, 579) | (inteiro == 787),
        inteiro == 250,
        inteiro.between(800, 999),
        inteiro.between(710, 739),
        inteiro.between(580, 629) | (inteiro == 788),
        inteiro.between(140, 239),
    ]
    nomes = [
        cfg.GRUPO_CIRCULATORIO,
        cfg.GRUPO_RESPIRATORIO,
        cfg.GRUPO_DIGESTIVO,
        cfg.GRUPO_DIABETES,
        cfg.GRUPO_LESOES,
        cfg.GRUPO_MUSCULOESQUELETICO,
        cfg.GRUPO_GENITURINARIO,
        cfg.GRUPO_NEOPLASIAS,
    ]
    grupos = pd.Series(
        np.select([c.to_numpy(dtype=bool) for c in condicoes], nomes, default=cfg.GRUPO_OUTROS),
        index=codigos.index,
    )
    sem_diagnostico = codigos.isna() | (codigos == cfg.ROTULO_SEM_DIAGNOSTICO)
    return grupos.mask(sem_diagnostico, cfg.ROTULO_SEM_DIAGNOSTICO)


def criar_grupos_diagnostico(df: pd.DataFrame) -> pd.DataFrame:
    """F03 — cria ``grupo_diag_1``, ``grupo_diag_2`` e ``grupo_diag_3``.

    Args:
        df: Base com ``diag_1``, ``diag_2``, ``diag_3``.

    Returns:
        Cópia com as três colunas de grupo.
    """
    saida = df.copy()
    for i, coluna in enumerate(cfg.COLUNAS_DIAGNOSTICO, start=1):
        saida[f"grupo_diag_{i}"] = agrupar_cid9(saida[coluna])
    return saida


# ---------------------------------------------------------------------------
# F04, F05 — utilização prévia do sistema
# ---------------------------------------------------------------------------


def criar_visitas_previas(df: pd.DataFrame) -> pd.DataFrame:
    """F04 e F05 — resume o uso prévio do hospital.

    * ``total_visitas_previas`` = ambulatório + emergência + internações no ano anterior.
    * ``usuario_frequente`` = 1 se já houve internação prévia
      (``number_inpatient >= cfg.LIMIAR_USUARIO_FREQUENTE``).

    Args:
        df: Base com ``number_outpatient``, ``number_emergency`` e ``number_inpatient``.

    Returns:
        Cópia com as duas colunas novas.
    """
    saida = df.copy()
    saida["total_visitas_previas"] = (
        saida["number_outpatient"] + saida["number_emergency"] + saida["number_inpatient"]
    )
    saida["usuario_frequente"] = (saida["number_inpatient"] >= cfg.LIMIAR_USUARIO_FREQUENTE).astype(
        "int8"
    )
    return saida


# ---------------------------------------------------------------------------
# F06, F07 — medicamentos
# ---------------------------------------------------------------------------


def criar_contagens_medicamentos(df: pd.DataFrame) -> pd.DataFrame:
    """F06 e F07 — conta medicamentos alterados e em uso.

    * ``n_meds_alterados``: quantos tiveram dose aumentada (``Up``) ou reduzida (``Down``).
    * ``n_meds_em_uso``: quantos são diferentes de ``No`` (``Steady``, ``Up`` ou ``Down``).

    Devem ser calculadas **antes** de remover colunas quase constantes (seleção),
    para que nenhum medicamento raro deixe de ser contado.

    Args:
        df: Base com as 23 colunas de medicamentos.

    Returns:
        Cópia com as duas colunas novas.
    """
    saida = df.copy()
    meds = saida[list(cfg.COLUNAS_MEDICAMENTOS)]
    saida["n_meds_alterados"] = meds.isin(["Up", "Down"]).sum(axis=1).astype("int64")
    saida["n_meds_em_uso"] = (meds != "No").sum(axis=1).astype("int64")
    return saida


# ---------------------------------------------------------------------------
# F08 — exames
# ---------------------------------------------------------------------------


def criar_flags_exames(df: pd.DataFrame) -> pd.DataFrame:
    """F08 — indica se HbA1c e glicose foram medidas na internação.

    A HbA1c é o tema do artigo-fonte (Strack et al., 2014): medir (ou não) o
    exame diz algo sobre o cuidado dado ao paciente.

    Args:
        df: Base com ``A1Cresult`` e ``max_glu_serum`` (D09 já aplicada).

    Returns:
        Cópia com ``a1c_medido`` e ``glicose_medida`` (0/1).
    """
    saida = df.copy()
    saida["a1c_medido"] = (saida["A1Cresult"] != cfg.ROTULO_NAO_MEDIDO).astype("int8")
    saida["glicose_medida"] = (saida["max_glu_serum"] != cfg.ROTULO_NAO_MEDIDO).astype("int8")
    return saida


# ---------------------------------------------------------------------------
# F09 — destino de alta
# ---------------------------------------------------------------------------


def agrupar_destino_alta(df: pd.DataFrame) -> pd.DataFrame:
    """F09 — agrupa o destino de alta em Casa / Transferencia / Outros / Nao_informado.

    Reduz ~25 destinos a 4 grupos sem perder o sentido clínico. Diferença em
    relação ao plano: mantemos ``Nao_informado`` como grupo próprio (é uma
    fatia relevante das linhas) em vez de misturá-lo com ``Outros``, para não
    esconder que não sabemos o destino.

    Args:
        df: Base com ``discharge_disposition_id`` e ``destino_alta``.

    Returns:
        Cópia com a coluna ``grupo_destino_alta``.
    """
    saida = df.copy()
    codigo = saida["discharge_disposition_id"]
    saida["grupo_destino_alta"] = np.select(
        [
            (saida["destino_alta"] == cfg.ROTULO_NAO_INFORMADO).to_numpy(),
            codigo.isin(cfg.CODIGOS_ALTA_CASA).to_numpy(),
            codigo.isin(cfg.CODIGOS_ALTA_TRANSFERENCIA).to_numpy(),
        ],
        [cfg.ROTULO_NAO_INFORMADO, "Casa", "Transferencia"],
        default="Outros",
    )
    return saida


# ---------------------------------------------------------------------------
# F10 — intensidade de cuidado
# ---------------------------------------------------------------------------


def criar_intensidade_cuidado(df: pd.DataFrame) -> pd.DataFrame:
    """F10 — exames e procedimentos por dia de internação.

    Normaliza pelo tempo de estadia: 40 exames em 2 dias é bem diferente de 40
    exames em 10 dias.

    Args:
        df: Base com ``num_lab_procedures``, ``num_procedures`` e ``time_in_hospital``.

    Returns:
        Cópia com ``exames_por_dia`` e ``procedimentos_por_dia``.

    Raises:
        ValueError: Se houver ``time_in_hospital`` menor que 1 (divisão inválida).
    """
    if (df["time_in_hospital"] < 1).any():
        raise ValueError("time_in_hospital < 1 impede o cálculo das razões por dia.")
    saida = df.copy()
    casas = cfg.CASAS_DECIMAIS_RAZOES
    saida["exames_por_dia"] = (saida["num_lab_procedures"] / saida["time_in_hospital"]).round(casas)
    saida["procedimentos_por_dia"] = (saida["num_procedures"] / saida["time_in_hospital"]).round(
        casas
    )
    return saida


# ---------------------------------------------------------------------------
# F11 — especialidade médica
# ---------------------------------------------------------------------------


def calcular_top_especialidades(especialidades: pd.Series, n: int) -> list[str]:
    """Lista as ``n`` especialidades mais frequentes (ignorando ``Desconhecido``).

    O desempate é por nome (ordem alfabética) para o resultado ser sempre o mesmo.

    Args:
        especialidades: Coluna ``medical_specialty`` já tratada (D06).
        n: Quantas manter.

    Returns:
        Lista de nomes, da mais para a menos frequente.
    """
    contagem = especialidades[especialidades != cfg.ROTULO_DESCONHECIDO].value_counts()
    tabela = pd.DataFrame({"nome": contagem.index.astype(str), "n": contagem.to_numpy()})
    tabela = tabela.sort_values(["n", "nome"], ascending=[False, True], kind="stable")
    return tabela["nome"].head(n).tolist()


def agrupar_especialidades(df: pd.DataFrame, top: list[str]) -> pd.DataFrame:
    """F11 — mantém as especialidades do ``top`` e junta o resto em ``Outras``.

    ``medical_specialty`` tem dezenas de níveis, muitos com poucas linhas, o que
    atrapalha modelos e gráficos. ``Desconhecido`` (D06) continua separado, pois
    "não informado" é diferente de "especialidade rara".

    Args:
        df: Base com ``medical_specialty``.
        top: Especialidades a manter (ver :func:`calcular_top_especialidades`).

    Returns:
        Cópia com a coluna ``especialidade_agrupada``.
    """
    saida = df.copy()
    especialidade = saida["medical_specialty"]
    saida["especialidade_agrupada"] = np.select(
        [especialidade.isin(top).to_numpy(), (especialidade == cfg.ROTULO_DESCONHECIDO).to_numpy()],
        [especialidade.to_numpy(dtype=object), cfg.ROTULO_DESCONHECIDO],
        default=cfg.ROTULO_OUTRAS,
    )
    return saida


# ---------------------------------------------------------------------------
# Orquestração das etapas F01–F11
# ---------------------------------------------------------------------------


def criar_atributos(df: pd.DataFrame, top_especialidades: list[str]) -> pd.DataFrame:
    """Aplica F01–F11 na ordem.

    Args:
        df: Base limpa (após D01–D09).
        top_especialidades: Lista de especialidades mantidas em F11.

    Returns:
        DataFrame com todas as colunas derivadas.
    """
    saida = criar_alvo(df)
    saida = criar_idade_media(saida)
    saida = criar_grupos_diagnostico(saida)
    saida = criar_visitas_previas(saida)
    saida = criar_contagens_medicamentos(saida)
    saida = criar_flags_exames(saida)
    saida = agrupar_destino_alta(saida)
    saida = criar_intensidade_cuidado(saida)
    saida = agrupar_especialidades(saida, top_especialidades)
    return saida
