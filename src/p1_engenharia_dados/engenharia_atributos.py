"""Engenharia de atributos (F01–F11): novas colunas, agrupamentos e resumo de pessoas.

Regra de ouro: **nunca usamos o alvo (nem a consequência do acidente) para
criar atributos** (isso seria vazamento). Todas as transformações são regras
fixas (extração de hora, agrupamentos do conhecimento do domínio, resumos por
acidente); nada é "aprendido" dos dados.

Cada função recebe um DataFrame e devolve uma **cópia** com a(s) coluna(s) nova(s).
"""

from __future__ import annotations

import logging
from datetime import date, timedelta

import numpy as np
import pandas as pd
from dateutil.easter import easter

from src.p1_engenharia_dados import configuracao as cfg

logger = logging.getLogger(__name__)

# Tipos de texto usados nas colunas de pessoas (após D04).
_CONDUTOR = "Condutor"
_PEDESTRE = "Pedestre"


# ---------------------------------------------------------------------------
# F11 — resumo do arquivo de pessoas (integração I01)
# ---------------------------------------------------------------------------


def resumir_pessoas(pessoas: pd.DataFrame) -> pd.DataFrame:
    """F11 — resume o arquivo "por pessoa" em **uma linha por acidente**.

    Colunas criadas (todas descrevem *quem e o quê* se envolveu, e não o que
    aconteceu com as pessoas, então não vazam o alvo):

    * ``envolve_moto``, ``envolve_caminhao``, ``envolve_onibus``,
      ``envolve_bicicleta``, ``envolve_pedestre``: 1 se há esse tipo de veículo
      (ou pedestre) no acidente;
    * ``idade_media_condutores``: média de idade dos condutores (idades
      impossíveis já foram anuladas em D05);
    * ``pct_condutores_homens``: % de condutores homens entre os de sexo informado;
    * ``idade_media_veiculos``: idade média dos veículos, em anos (ano do
      acidente menos ano de fabricação; D06 já anulou anos impossíveis).

    Acidente sem condutor de idade válida fica com nulo na coluna (não
    inventamos valor; a Pessoa 3 trata no ``Pipeline``).

    Args:
        pessoas: Uma linha por pessoa/veículo, **já limpa** (D04, D05, D06).

    Returns:
        DataFrame com ``id`` e as colunas acima, uma linha por acidente.
    """
    ids = pd.Index(pessoas[cfg.COLUNA_ID].unique(), name=cfg.COLUNA_ID).sort_values()
    resumo = pd.DataFrame(index=ids)

    # --- veículos: um por (acidente, id_veiculo), para não contar o veículo uma vez por ocupante.
    veiculos = pessoas.loc[pessoas["tipo_veiculo"] != cfg.ROTULO_NAO_INFORMADO].drop_duplicates(
        subset=[cfg.COLUNA_ID, "id_veiculo"]
    )

    for nome, grupo in (
        ("envolve_moto", cfg.VEICULOS_MOTO),
        ("envolve_caminhao", cfg.VEICULOS_CAMINHAO),
        ("envolve_onibus", cfg.VEICULOS_ONIBUS),
        ("envolve_bicicleta", cfg.VEICULOS_BICICLETA),
    ):
        # 1 se o acidente aparece ao menos uma vez com esse tipo de veículo.
        resumo[nome] = ids.isin(veiculos.loc[veiculos["tipo_veiculo"].isin(grupo), cfg.COLUNA_ID]).astype("int8")
    resumo["envolve_pedestre"] = ids.isin(
        pessoas.loc[pessoas["tipo_envolvido"] == _PEDESTRE, cfg.COLUNA_ID]
    ).astype("int8")

    # --- condutores: idade média e % de homens (só onde o sexo é conhecido).
    condutores = pessoas.loc[pessoas["tipo_envolvido"] == _CONDUTOR]
    resumo["idade_media_condutores"] = condutores.groupby(cfg.COLUNA_ID)["idade"].mean().round(
        cfg.CASAS_DECIMAIS
    )
    sexo_conhecido = condutores.loc[condutores["sexo"].isin(["Masculino", "Feminino"])]
    resumo["pct_condutores_homens"] = (
        100 * (sexo_conhecido["sexo"] == "Masculino").groupby(sexo_conhecido[cfg.COLUNA_ID]).mean()
    ).round(cfg.CASAS_DECIMAIS)

    # --- veículos: idade em anos.
    ano_acidente = veiculos["data_inversa"].str[:4].astype(int)
    idade_veiculo = (ano_acidente - veiculos["ano_fabricacao_veiculo"]).clip(lower=0)
    resumo["idade_media_veiculos"] = idade_veiculo.groupby(veiculos[cfg.COLUNA_ID]).mean().round(
        cfg.CASAS_DECIMAIS
    )
    return resumo.reset_index()


# ---------------------------------------------------------------------------
# F01 — alvo
# ---------------------------------------------------------------------------


def criar_alvo(ocorrencias: pd.DataFrame) -> pd.DataFrame:
    """F01 — cria ``acidente_fatal`` (1 se houve ao menos uma morte).

    As colunas de consequência (``mortos``, ``feridos_*``, ``classificacao_acidente``...)
    continuam no interim para rastreio, mas **saem da base de modelagem**:
    usá-las como atributo seria vazamento.

    Args:
        ocorrencias: Base com ``mortos``.

    Returns:
        Cópia com a coluna ``acidente_fatal`` (0/1).
    """
    saida = ocorrencias.copy()
    saida[cfg.COLUNA_ALVO] = (saida["mortos"] > 0).astype("int8")
    return saida


# ---------------------------------------------------------------------------
# F02, F03, F04 — tempo e calendário
# ---------------------------------------------------------------------------


def _nome_da_faixa(hora: int) -> str:
    """Devolve o nome da faixa horária de uma hora (0 a 23).

    Args:
        hora: Hora do dia.

    Returns:
        ``Madrugada``, ``Manha``, ``Tarde`` ou ``Noite``.
    """
    nome = cfg.FAIXAS_HORARIAS[0][1]
    for inicio, candidato in cfg.FAIXAS_HORARIAS:
        if hora >= inicio:
            nome = candidato
    return nome


def criar_variaveis_de_tempo(ocorrencias: pd.DataFrame) -> pd.DataFrame:
    """F02 e F03 — hora, faixa do dia, mês, ano e fim de semana.

    Args:
        ocorrencias: Base com ``data_inversa`` (``AAAA-MM-DD``) e ``horario`` (``HH:MM:SS``).

    Returns:
        Cópia com ``hora`` (0 a 23), ``faixa_horaria``, ``mes``, ``ano`` e ``fim_de_semana`` (0/1).

    Raises:
        ValueError: Se algum horário estiver fora do formato esperado.
    """
    saida = ocorrencias.copy()
    hora = saida["horario"].str.extract(r"^(\d{1,2}):\d{2}")[0].astype(float)
    if hora.isna().any() or not hora.between(0, 23).all():
        raise ValueError("Há horários fora do formato HH:MM:SS (0 a 23 horas).")
    saida["hora"] = hora.astype("int64")
    saida["faixa_horaria"] = saida["hora"].map(_nome_da_faixa)
    data = pd.to_datetime(saida["data_inversa"], format="%Y-%m-%d")
    saida["mes"] = data.dt.month.astype("int64")
    saida["ano"] = data.dt.year.astype("int64")
    # Segunda = 0 ... domingo = 6. Calculado pela data (e não pelo texto "sábado")
    # para não depender de acento nem de caixa.
    saida["fim_de_semana"] = (data.dt.day_of_week >= 5).astype("int8")
    return saida


def feriados_nacionais(anos: list[int]) -> set[date]:
    """Monta o conjunto de feriados nacionais (e Carnaval) dos anos dados.

    Fixos: ``cfg.FERIADOS_FIXOS`` (+ 20/11 a partir de 2024). Móveis: calculados
    a partir da Páscoa (Carnaval, Sexta-feira Santa e Corpus Christi).

    Args:
        anos: Anos a cobrir.

    Returns:
        Conjunto de datas.
    """
    datas: set[date] = set()
    for ano in anos:
        datas |= {date(ano, m, d) for m, d in cfg.FERIADOS_FIXOS}
        if ano >= 2024:
            datas.add(date(ano, *cfg.FERIADO_CONSCIENCIA_NEGRA))
        pascoa = easter(ano)
        datas |= {pascoa + timedelta(days=dias) for dias in cfg.FERIADOS_MOVEIS_DIAS_APOS_PASCOA}
    return datas


def criar_feriado(ocorrencias: pd.DataFrame) -> pd.DataFrame:
    """F04 — indica se o acidente ocorreu em feriado nacional (integração I02).

    Args:
        ocorrencias: Base com ``data_inversa``.

    Returns:
        Cópia com ``eh_feriado`` (0/1).
    """
    saida = ocorrencias.copy()
    data = pd.to_datetime(saida["data_inversa"], format="%Y-%m-%d")
    feriados = feriados_nacionais(sorted(data.dt.year.unique().tolist()))
    saida["eh_feriado"] = data.dt.date.isin(feriados).astype("int8")
    return saida


# ---------------------------------------------------------------------------
# F05, F06, F07, F08, F09, F10 — local, causa e via
# ---------------------------------------------------------------------------


def criar_regiao(ocorrencias: pd.DataFrame) -> pd.DataFrame:
    """F05 — região do Brasil a partir da UF (integração I03).

    Args:
        ocorrencias: Base com ``uf``.

    Returns:
        Cópia com ``regiao`` (Norte, Nordeste, Centro-Oeste, Sudeste, Sul).

    Raises:
        ValueError: Se houver UF fora da tabela do IBGE.
    """
    saida = ocorrencias.copy()
    saida["regiao"] = saida["uf"].map(cfg.REGIAO_POR_UF)
    if saida["regiao"].isna().any():
        raise ValueError(f"UF sem região: {sorted(saida.loc[saida['regiao'].isna(), 'uf'].unique())}")
    return saida


def agrupar_causas(ocorrencias: pd.DataFrame) -> pd.DataFrame:
    """F06 — agrupa as 69 causas em poucos grupos que um gestor entende.

    Causa fora de ``cfg.GRUPOS_CAUSA`` vai para ``Outras`` (e o pipeline avisa).

    Args:
        ocorrencias: Base com ``causa_acidente``.

    Returns:
        Cópia com ``grupo_causa``.
    """
    saida = ocorrencias.copy()
    grupo_da_causa = {causa: grupo for grupo, causas in cfg.GRUPOS_CAUSA.items() for causa in causas}
    saida["grupo_causa"] = saida["causa_acidente"].map(grupo_da_causa).fillna(cfg.ROTULO_OUTRAS)
    n_outras = int((saida["grupo_causa"] == cfg.ROTULO_OUTRAS).sum())
    if n_outras:
        nomes = sorted(saida.loc[saida["grupo_causa"] == cfg.ROTULO_OUTRAS, "causa_acidente"].unique())
        logger.warning("%d acidentes com causa fora dos grupos: %s", n_outras, nomes)
    return saida


def criar_flags_tracado(ocorrencias: pd.DataFrame) -> pd.DataFrame:
    """F07 — transforma ``tracado_via`` (ex.: ``Reta;Declive``) em colunas 0/1.

    O traçado vem com vários valores colados por ``;`` (centenas de combinações).
    Separá-los em colunas deixa o modelo (e o gestor) ver "curva", "aclive",
    "interseção" separadamente.

    Args:
        ocorrencias: Base com ``tracado_via``.

    Returns:
        Cópia com as colunas ``via_*`` (0/1).
    """
    saida = ocorrencias.copy()
    pedacos = saida["tracado_via"].fillna("").str.split(";").map(lambda xs: {x.strip() for x in xs if x.strip()})
    desconhecidos = set().union(*pedacos) - set(cfg.FLAGS_TRACADO) - {cfg.ROTULO_NAO_INFORMADO}
    if desconhecidos:
        logger.warning("Traçados sem coluna correspondente: %s", sorted(desconhecidos))
    for coluna in dict.fromkeys(cfg.FLAGS_TRACADO.values()):  # ordem estável, sem repetição
        textos = {t for t, c in cfg.FLAGS_TRACADO.items() if c == coluna}
        saida[coluna] = pedacos.map(lambda ps, ts=textos: int(bool(ps & ts))).astype("int8")
    return saida


def criar_area_urbana(ocorrencias: pd.DataFrame) -> pd.DataFrame:
    """F08 — ``area_urbana`` = 1 quando ``uso_solo`` é "Sim" (trecho urbano).

    Args:
        ocorrencias: Base com ``uso_solo`` (``Sim``/``Não``).

    Returns:
        Cópia com ``area_urbana`` (0/1).
    """
    saida = ocorrencias.copy()
    saida["area_urbana"] = (saida["uso_solo"] == "Sim").astype("int8")
    return saida


def criar_rodovia_e_trecho(ocorrencias: pd.DataFrame) -> pd.DataFrame:
    """F09 e F10 — rodovia como categoria e "trecho" (rodovia + faixa de km).

    ``br`` é um **código** (116, 101...), não uma quantidade: somar ou tirar
    média de BRs não faz sentido. Vira texto (``BR-116``). O ``trecho`` junta a
    rodovia a uma faixa de ``cfg.KM_POR_TRECHO`` km (``BR-116_km230``) e permite
    estudar onde os acidentes se concentram.

    Args:
        ocorrencias: Base com ``br`` e ``km``.

    Returns:
        Cópia com ``br`` (texto) e ``trecho``.
    """
    saida = ocorrencias.copy()
    saida["br"] = "BR-" + saida["br"].astype("int64").astype(str)
    faixa = (np.floor(saida["km"] / cfg.KM_POR_TRECHO) * cfg.KM_POR_TRECHO).astype("int64")
    saida["trecho"] = saida["br"] + "_km" + faixa.astype(str)
    return saida


# ---------------------------------------------------------------------------
# Orquestração
# ---------------------------------------------------------------------------


def criar_atributos(ocorrencias: pd.DataFrame) -> pd.DataFrame:
    """Aplica F01–F10 na ordem (F11 já veio da integração com pessoas).

    Args:
        ocorrencias: Base limpa (após D01–D07) e integrada com o resumo de pessoas.

    Returns:
        DataFrame com todas as colunas derivadas.
    """
    saida = criar_alvo(ocorrencias)
    saida = criar_variaveis_de_tempo(saida)
    saida = criar_feriado(saida)
    saida = criar_regiao(saida)
    saida = agrupar_causas(saida)
    saida = criar_flags_tracado(saida)
    saida = criar_area_urbana(saida)
    saida = criar_rodovia_e_trecho(saida)
    return saida
