"""Auditoria de qualidade dos dados (perfil por coluna e resumos gerais).

A auditoria **não altera** nenhum dado. Ela mede o problema antes de resolvê-lo,
gerando as evidências (números) que depois aparecem na tabela
"o que estava errado -> o que fizemos -> por quê" da documentação.
"""

from __future__ import annotations

import logging

import pandas as pd
from pandas.api.types import is_numeric_dtype

from src.p1_engenharia_dados import configuracao as cfg

logger = logging.getLogger(__name__)

#: Observações fixas para colunas cuja situação já conhecemos pelo plano.
#: As demais recebem uma observação automática (ver :func:`_observacao`).
OBSERVACOES_FIXAS: dict[str, str] = {
    "encounter_id": "Identificador do encontro (sai da base de modelagem).",
    "patient_nbr": "Identificador do paciente; repete quando há vários encontros (D03).",
    "weight": "Quase toda ausente: imputar seria inventar dado (D05).",
    "payer_code": "Ausência pode ser informativa, ex.: paciente sem convênio (D06).",
    "medical_specialty": "Ausência pode ser informativa (D06); dezenas de níveis raros (F11).",
    "race": "Atributo sensível; '?' vira 'Desconhecido' (D07).",
    "diag_1": "Código CID-9; agrupado em F03.",
    "diag_2": "Ausência de diagnóstico secundário é real, não erro (D08).",
    "diag_3": "Ausência de diagnóstico secundário é real, não erro (D08).",
    "max_glu_serum": "'None' = exame não realizado, NÃO é nulo (D09).",
    "A1Cresult": "'None' = exame não realizado, NÃO é nulo (D09).",
    "gender": "Contém 'Unknown/Invalid' em poucos registros (D01).",
    "discharge_disposition_id": "Contém códigos de óbito/hospice (D02).",
    "readmitted": "Alvo original com 3 classes ('NO' em maiúsculas: D04); vira readmit_30d (F01).",
}


def _tipo_simplificado(serie: pd.Series) -> str:
    """Classifica uma coluna como ``numerica`` ou ``texto``.

    Args:
        serie: Coluna a classificar.

    Returns:
        ``"numerica"`` se o dtype for numérico, senão ``"texto"``.
    """
    return "numerica" if is_numeric_dtype(serie) else "texto"


def _observacao(coluna: str, n_distintos: int) -> str:
    """Escolhe a observação da auditoria para uma coluna.

    Args:
        coluna: Nome da coluna.
        n_distintos: Número de valores distintos (sem nulos).

    Returns:
        Texto curto explicando a situação da coluna.
    """
    if coluna in OBSERVACOES_FIXAS:
        return OBSERVACOES_FIXAS[coluna]
    if n_distintos <= 1:
        return "Constante: não traz informação (candidata a remoção)."
    if coluna in cfg.COLUNAS_MEDICAMENTOS:
        return "Medicamento (No/Steady/Up/Down); base dos atributos F06 e F07."
    return ""


def perfil_qualidade(df: pd.DataFrame, df_texto: pd.DataFrame) -> pd.DataFrame:
    """Gera o perfil de qualidade por coluna (``auditoria_inicial.csv``).

    Args:
        df: Base lida com ``?`` já convertido em nulo.
        df_texto: Mesma base lida como texto puro (para contar ``?`` e ``None``).

    Returns:
        Uma linha por coluna com: tipo, nulos (n e %), quantidade de ``?`` no
        arquivo original, quantidade de textos ``None``, valores distintos,
        exemplo e observação.
    """
    linhas: list[dict[str, object]] = []
    for coluna in df.columns:
        serie = df[coluna]
        n_distintos = int(serie.nunique(dropna=True))
        primeiro_valido = serie.dropna()
        linhas.append(
            {
                "coluna": coluna,
                "tipo": _tipo_simplificado(serie),
                "n_nulos": int(serie.isna().sum()),
                "pct_nulos": round(100 * float(serie.isna().mean()), 2),
                "n_interrogacoes_no_arquivo": int((df_texto[coluna] == cfg.VALOR_NULO_BRUTO).sum()),
                "n_textos_none": int((df_texto[coluna] == "None").sum()),
                "n_distintos": n_distintos,
                "exemplo": str(primeiro_valido.iloc[0]) if len(primeiro_valido) else "",
                "observacao": _observacao(coluna, n_distintos),
            }
        )
    return pd.DataFrame(linhas)


def resumo_geral(df: pd.DataFrame) -> dict[str, int]:
    """Calcula os números gerais da base (linhas, pacientes, encontros repetidos).

    Args:
        df: Base com ``patient_nbr`` e ``encounter_id``.

    Returns:
        Dicionário com contagens usadas na documentação.
    """
    n_pacientes = int(df["patient_nbr"].nunique())
    encontros_por_paciente = df.groupby("patient_nbr").size()
    return {
        "linhas": int(len(df)),
        "colunas": int(df.shape[1]),
        "pacientes_unicos": n_pacientes,
        # Encontros "a mais": linhas que não são o primeiro encontro do paciente.
        "encontros_repetidos": int(len(df) - n_pacientes),
        "pacientes_com_varios_encontros": int((encontros_por_paciente > 1).sum()),
        "linhas_duplicadas_inteiras": int(df.duplicated().sum()),
        "encounter_id_unico": int(df["encounter_id"].is_unique),
    }


def tabela_integracao(mapas: dict[str, pd.DataFrame], df: pd.DataFrame) -> pd.DataFrame:
    """Monta a tabela "código -> nome" com a contagem de linhas de cada código.

    Mostra o *antes* (código numérico) e o *depois* (nome legível), incluindo
    quais códigos foram unificados em ``Nao_informado`` (D04).

    Args:
        mapas: Saída de ``carregar.carregar_ids_mapping``.
        df: Base bruta (para contar quantas linhas usam cada código).

    Returns:
        Uma linha por código de cada uma das três tabelas.
    """
    partes: list[pd.DataFrame] = []
    for coluna_codigo, tabela in mapas.items():
        contagem = df[coluna_codigo].value_counts()
        parte = tabela.copy()
        parte.insert(0, "tabela", coluna_codigo)
        parte["n_linhas_no_bruto"] = parte["codigo"].map(contagem).fillna(0).astype(int)
        parte["unificado_em_nao_informado"] = (
            parte["descricao"] == cfg.ROTULO_NAO_INFORMADO
        ).astype(int)
        parte["tinha_espaco_ou_ponto_extra"] = (
            parte["descricao_original"] != parte["descricao_original"].str.strip(" .")
        ).astype(int)
        partes.append(parte)
    return pd.concat(partes, ignore_index=True)
