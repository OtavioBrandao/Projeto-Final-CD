"""Seleção e redução de atributos + dicionário de dados.

Monta a base de modelagem removendo colunas que não devem (ou não precisam)
chegar ao modelo, sempre registrando **o que saiu e por quê**:

1. identificador (``id``) e colunas técnicas;
2. colunas de **consequência** do acidente (vítimas, classificação): definem o
   alvo, então usá-las como atributo seria vazamento;
3. colunas administrativas da PRF e colunas substituídas por uma versão mais
   útil (causa -> grupo de causa, traçado -> colunas 0/1, município -> região...);
4. colunas constantes ou quase constantes (um único valor em >= 99,9% das linhas);
5. colunas numéricas redundantes (|correlação de Pearson| > 0,9).

Nenhum dos critérios usa o alvo, portanto não há vazamento de informação.
"""

from __future__ import annotations

import logging

import pandas as pd
from pandas.api.types import is_numeric_dtype

from src.p1_engenharia_dados import configuracao as cfg

logger = logging.getLogger(__name__)

#: Colunas removidas por serem consequência do acidente (vazamento).
MOTIVO_RESULTADO = "consequencia do acidente: define o alvo, seria vazamento"

#: ``{coluna_removida: (substituta ou "-", motivo)}`` para as colunas que saem da modelagem
#: mas que mantemos no interim.
COLUNAS_SUBSTITUIDAS: dict[str, tuple[str, str]] = {
    "id": ("-", "identificador do acidente; so serve para rastrear linhas"),
    "ano_arquivo": ("-", "coluna tecnica (de qual arquivo a linha veio)"),
    "data_inversa": ("mes, ano, fim_de_semana, eh_feriado", "data em texto; os atributos de calendario resumem"),
    "horario": ("hora, faixa_horaria", "horario em texto; a hora resume"),
    "causa_acidente": ("grupo_causa", "69 causas; o grupo resume"),
    "tracado_via": ("via_*", "valores compostos; viraram colunas 0/1"),
    "uso_solo": ("area_urbana", "Sim/Nao; virou coluna 0/1"),
    "municipio": ("uf, regiao", "quase 2 mil municipios; uf e regiao ja descrevem o local"),
    "trecho": ("br, km", "alto numero de trechos; use na analise de concentracao"),
    "contagens_inconsistentes": ("-", "diagnostico das contagens de vitimas (consequencia)"),
    **{c: ("-", MOTIVO_RESULTADO) for c in cfg.COLUNAS_RESULTADO},
    **{c: ("uf, regiao", "coluna administrativa (quem atendeu); redundante com a UF") for c in cfg.COLUNAS_ADMINISTRATIVAS},
}  # fmt: skip

#: Colunas criadas pelo pipeline (as demais são originais da PRF).
COLUNAS_DERIVADAS: frozenset[str] = frozenset(
    {
        cfg.COLUNA_ALVO, "hora", "faixa_horaria", "mes", "ano", "fim_de_semana", "eh_feriado",
        "regiao", "grupo_causa", "area_urbana", "trecho", "contagens_inconsistentes",
        "envolve_moto", "envolve_caminhao", "envolve_onibus", "envolve_bicicleta",
        "envolve_pedestre", "idade_media_condutores", "pct_condutores_homens",
        "idade_media_veiculos",
        *dict.fromkeys(cfg.FLAGS_TRACADO.values()),
        *(f"{c}_outlier" for c in cfg.COLUNAS_OUTLIER),
    }
)  # fmt: skip

#: Colunas sensíveis (ética e viés): usar para análise de equidade, não para discriminar.
COLUNAS_SENSIVEIS: frozenset[str] = frozenset({"idade_media_condutores", "pct_condutores_homens"})


# ---------------------------------------------------------------------------
# Detecção
# ---------------------------------------------------------------------------


def calcular_dominancia(df: pd.DataFrame, colunas: list[str], limiar: float) -> pd.DataFrame:
    """Mede, para cada coluna, quanto o valor mais frequente domina.

    Coluna com dominância >= ``limiar`` é "quase constante": quase todas as
    linhas têm o mesmo valor, então ela quase não ajuda a separar acidentes e só
    aumenta a dimensão do problema.

    Args:
        df: Base a inspecionar.
        colunas: Colunas candidatas.
        limiar: Fração a partir da qual a coluna é considerada quase constante.

    Returns:
        Uma linha por coluna, ordenada da maior para a menor dominância, com
        ``quase_constante`` (0/1).
    """
    linhas = []
    for coluna in colunas:
        frequencias = df[coluna].value_counts(normalize=True, dropna=False)
        linhas.append(
            {
                "coluna": coluna,
                "valor_dominante": str(frequencias.index[0]),
                "pct_dominancia": round(100 * float(frequencias.iloc[0]), 3),
                "n_distintos": int(df[coluna].nunique(dropna=False)),
                "quase_constante": int(float(frequencias.iloc[0]) >= limiar),
            }
        )
    tabela = pd.DataFrame(linhas)
    return tabela.sort_values(["pct_dominancia", "coluna"], ascending=[False, True]).reset_index(
        drop=True
    )


def calcular_correlacoes(df: pd.DataFrame, colunas: list[str], limiar: float) -> pd.DataFrame:
    """Lista os pares de colunas numéricas por correlação de Pearson (|r|).

    Só entram colunas numéricas com mais de 2 valores (as sinalizações 0/1 não
    fazem sentido nesse cálculo). Nulos são ignorados par a par.

    Args:
        df: Base a inspecionar.
        colunas: Colunas candidatas.
        limiar: |r| acima do qual um par é considerado redundante.

    Returns:
        Todos os pares, ordenados por |r| decrescente, com ``acima_do_limiar`` (0/1).
    """
    numericas = [c for c in colunas if is_numeric_dtype(df[c]) and df[c].nunique() > 2]
    corr = df[numericas].corr(method="pearson")
    pares = []
    for i, a in enumerate(numericas):
        for b in numericas[i + 1 :]:
            r = float(corr.loc[a, b])
            pares.append(
                {
                    "coluna_a": a,
                    "coluna_b": b,
                    "r_pearson": round(r, 4),
                    "r_absoluto": round(abs(r), 4),
                    "acima_do_limiar": int(abs(r) > limiar),
                }
            )
    return (
        pd.DataFrame(pares)
        .sort_values(["r_absoluto", "coluna_a", "coluna_b"], ascending=[False, True, True])
        .reset_index(drop=True)
    )


# ---------------------------------------------------------------------------
# Seleção
# ---------------------------------------------------------------------------


def selecionar_para_modelagem(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Gera a base de modelagem e documenta cada coluna removida.

    Args:
        df: Base completa (todas as colunas originais úteis + derivadas).

    Returns:
        Tupla ``(modelagem, removidas, dominancia, correlacoes)``:

        * ``modelagem``: DataFrame final, com o alvo na última coluna;
        * ``removidas``: tabela ``coluna``, ``motivo``, ``detalhe``;
        * ``dominancia`` e ``correlacoes``: tabelas de evidência.
    """
    removidas: list[dict[str, str]] = []
    restantes = list(df.columns)

    def remover(coluna: str, motivo: str, detalhe: str) -> None:
        """Tira ``coluna`` da lista de restantes e registra o motivo."""
        restantes.remove(coluna)
        removidas.append({"coluna": coluna, "motivo": motivo, "detalhe": detalhe})

    # 1) Identificador, técnicas, consequências, administrativas e substituídas.
    for coluna, (substituta, motivo) in COLUNAS_SUBSTITUIDAS.items():
        if coluna in restantes:
            if coluna in cfg.COLUNAS_RESULTADO:
                tipo = "consequencia"
            elif coluna in cfg.COLUNAS_ADMINISTRATIVAS:
                tipo = "administrativa"
            elif coluna in ("id", "ano_arquivo"):
                tipo = "identificador_ou_tecnica"
            else:
                tipo = "substituida"
            detalhe = motivo if substituta == "-" else f"{motivo} -> usar {substituta}"
            remover(coluna, tipo, detalhe)

    # 2) Colunas constantes / quase constantes (o alvo nunca é removido aqui).
    candidatas = [c for c in restantes if c != cfg.COLUNA_ALVO]
    dominancia = calcular_dominancia(df, candidatas, cfg.LIMIAR_DOMINANCIA)
    for linha in dominancia.loc[dominancia["quase_constante"] == 1].itertuples():
        remover(
            linha.coluna,
            "quase_constante",
            f"valor '{linha.valor_dominante}' em {str(linha.pct_dominancia).replace('.', ',')}% das linhas",
        )

    # 3) Redundância entre numéricas: de cada par muito correlacionado, sai a 2ª coluna.
    candidatas = [c for c in restantes if c != cfg.COLUNA_ALVO]
    correlacoes = calcular_correlacoes(df, candidatas, cfg.LIMIAR_CORRELACAO)
    for linha in correlacoes.loc[correlacoes["acima_do_limiar"] == 1].itertuples():
        if linha.coluna_a in restantes and linha.coluna_b in restantes:
            remover(
                linha.coluna_b,
                "redundante",
                f"|r| = {linha.r_absoluto} com {linha.coluna_a} (mantida)",
            )

    # Alvo sempre na última coluna: facilita separar X e y.
    ordem = [c for c in restantes if c != cfg.COLUNA_ALVO] + [cfg.COLUNA_ALVO]
    modelagem = df[ordem].copy()
    logger.info(
        "Seleção: %d colunas -> %d colunas (%d removidas)",
        df.shape[1], modelagem.shape[1], len(removidas),
    )  # fmt: skip
    return modelagem, pd.DataFrame(removidas), dominancia, correlacoes


# ---------------------------------------------------------------------------
# Dicionário de dados
# ---------------------------------------------------------------------------

#: ``{coluna: (descricao_negocio, observacao)}``. Linguagem simples, para quem for usar a base.
DESCRICOES: dict[str, tuple[str, str]] = {
    "dia_semana": ("Dia da semana do acidente.", "Texto como na PRF."),
    "uf": ("Estado (UF) onde o acidente ocorreu.", ""),
    "br": ("Rodovia federal onde ocorreu (ex.: BR-116).", "F09. É um código, não uma quantidade."),
    "km": ("Quilômetro da rodovia onde ocorreu.", "Cauda longa (rodovias longas)."),
    "tipo_acidente": ("Tipo do acidente (colisão traseira, saída de pista, atropelamento...).", "Só é conhecido após o atendimento."),
    "fase_dia": ("Fase do dia: amanhecer, pleno dia, anoitecer ou plena noite.", ""),
    "sentido_via": ("Sentido da via (crescente ou decrescente).", "Nao_informado = PRF não registrou (D04)."),
    "condicao_metereologica": ("Condição do tempo no momento do acidente.", "Nome da coluna mantido como na PRF (com grafia original). Nao_informado = Ignorado (D04)."),
    "tipo_pista": ("Tipo da pista: simples, dupla ou múltipla.", ""),
    "veiculos": ("Quantidade de veículos envolvidos.", "Cauda longa."),
    "latitude": ("Latitude do acidente.", "Nula quando impossível (D07)."),
    "longitude": ("Longitude do acidente.", "Nula quando impossível (D07)."),
    "hora": ("Hora do acidente (0 a 23).", "F02."),
    "faixa_horaria": ("Faixa do dia: Madrugada (0-5h), Manha (6-11h), Tarde (12-17h) ou Noite (18-23h).", "F02."),
    "mes": ("Mês do acidente (1 a 12).", "F03."),
    "ano": ("Ano do acidente (2024 ou 2025).", "Use para validação temporal; evite como atributo."),
    "fim_de_semana": ("1 se o acidente foi em sábado ou domingo.", "F03."),
    "eh_feriado": ("1 se foi em feriado nacional (inclui Carnaval).", "F04. Calendário integrado (I02)."),
    "regiao": ("Região do Brasil (Norte, Nordeste, Centro-Oeste, Sudeste, Sul).", "F05. Tabela do IBGE (I03)."),
    "grupo_causa": ("Grupo da causa do acidente (atenção do condutor, velocidade, álcool, manobra, via...).", "F06. A causa é definida pela PRF após o atendimento."),
    "via_reta": ("1 se o trecho é reta.", "F07."),
    "via_curva": ("1 se o trecho tem curva.", "F07."),
    "via_aclive": ("1 se o trecho tem aclive (subida).", "F07."),
    "via_declive": ("1 se o trecho tem declive (descida).", "F07."),
    "via_intersecao": ("1 se o local é uma interseção de vias.", "F07."),
    "via_rotatoria_retorno": ("1 se o local é rotatória ou retorno regulamentado.", "F07."),
    "via_obras_desvio": ("1 se o trecho está em obras ou tem desvio temporário.", "F07."),
    "via_ponte_viaduto_tunel": ("1 se o local é ponte, viaduto ou túnel.", "F07."),
    "area_urbana": ("1 se o acidente foi em trecho urbano.", "F08."),
    "envolve_moto": ("1 se há motocicleta (ou similar) no acidente.", "F11. Vem do arquivo por pessoa."),
    "envolve_caminhao": ("1 se há caminhão ou caminhão-trator no acidente.", "F11."),
    "envolve_onibus": ("1 se há ônibus ou micro-ônibus no acidente.", "F11."),
    "envolve_bicicleta": ("1 se há bicicleta no acidente.", "F11."),
    "envolve_pedestre": ("1 se há pedestre envolvido.", "F11."),
    "idade_media_condutores": ("Idade média dos condutores envolvidos, em anos.", "Sensível. Nula quando não há idade válida (D05); imputar só no Pipeline."),
    "pct_condutores_homens": ("Percentual de condutores homens entre os de sexo informado.", "Sensível. Nula quando o sexo não consta; imputar só no Pipeline."),
    "idade_media_veiculos": ("Idade média dos veículos, em anos (ano do acidente menos ano de fabricação).", "Nula quando o ano de fabricação é inválido (D06); imputar só no Pipeline."),
    cfg.COLUNA_ALVO: ("ALVO: 1 se o acidente teve ao menos uma morte; 0 caso contrário.", "F01. Não use como atributo."),
}  # fmt: skip


def _descricao_coluna(coluna: str) -> tuple[str, str]:
    """Devolve ``(descricao_negocio, observacao)`` de uma coluna.

    Args:
        coluna: Nome da coluna da base de modelagem.

    Returns:
        Par com a descrição em linguagem simples e uma observação.

    Raises:
        KeyError: Se a coluna não tiver descrição (impede coluna sem documentação).
    """
    if coluna.endswith("_outlier") and coluna[: -len("_outlier")] in cfg.COLUNAS_OUTLIER:
        base = coluna[: -len("_outlier")]
        return (
            f"1 se '{base}' está fora dos limites do IQR (valor atípico).",
            "Só sinaliza; o valor original NÃO foi alterado. Opcional para modelagem.",
        )
    if coluna in DESCRICOES:
        return DESCRICOES[coluna]
    raise KeyError(f"Coluna sem descrição no dicionário de dados: {coluna}")


def classificar_tipo(serie: pd.Series, coluna: str) -> str:
    """Classifica a coluna em ``alvo``, ``binaria``, ``numerica`` ou ``categorica``.

    Args:
        serie: Valores da coluna.
        coluna: Nome da coluna.

    Returns:
        Uma das quatro classes acima.
    """
    if coluna == cfg.COLUNA_ALVO:
        return "alvo"
    valores = set(serie.dropna().unique())
    if is_numeric_dtype(serie):
        return "binaria" if valores <= {0, 1} else "numerica"
    return "binaria" if len(valores) <= 2 else "categorica"


def montar_dicionario(modelagem: pd.DataFrame) -> pd.DataFrame:
    """Gera o dicionário de dados da base de modelagem.

    Args:
        modelagem: Base final (saída de :func:`selecionar_para_modelagem`).

    Returns:
        Uma linha por coluna: ``coluna``, ``descricao_negocio``, ``tipo``,
        ``origem`` (original/derivada), ``sensivel`` (sim/nao), ``observacao``.
    """
    linhas = []
    for coluna in modelagem.columns:
        descricao, observacao = _descricao_coluna(coluna)
        linhas.append(
            {
                "coluna": coluna,
                "descricao_negocio": descricao,
                "tipo": classificar_tipo(modelagem[coluna], coluna),
                "origem": "derivada" if coluna in COLUNAS_DERIVADAS else "original",
                "sensivel": "sim" if coluna in COLUNAS_SENSIVEIS else "nao",
                "observacao": observacao,
            }
        )
    return pd.DataFrame(linhas)
