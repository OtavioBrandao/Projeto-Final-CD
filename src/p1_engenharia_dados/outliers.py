"""Detecção e **sinalização** de outliers (sem alterar nenhum valor).

Por que não removemos nem limitamos (*winsorizar*) os outliers aqui:

* eles são clinicamente plausíveis (um paciente grave pode ter 20 internações
  ou 70 medicamentos); removê-los apagaria justamente os casos de maior risco;
* limitar valores exige estimar limites e isso deve ser feito **só com o
  treino**, dentro do ``Pipeline`` do sklearn (Pessoa 3), para não vazar
  informação do teste.

O que fazemos: medir (regra do IQR de Tukey), documentar a cauda longa e criar
uma coluna ``<variavel>_outlier`` (0/1) para cada variável analisada.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence

import pandas as pd

from src.p1_engenharia_dados import configuracao as cfg

logger = logging.getLogger(__name__)


def calcular_limites_iqr(
    df: pd.DataFrame,
    colunas: Sequence[str] = cfg.COLUNAS_OUTLIER,
    fator: float = cfg.FATOR_IQR,
) -> pd.DataFrame:
    """Calcula os limites do IQR e a proporção de outliers por variável.

    Regra de Tukey: é outlier quem está fora de
    ``[Q1 - fator*IQR, Q3 + fator*IQR]``, onde ``IQR = Q3 - Q1``.

    Args:
        df: Base com as colunas numéricas.
        colunas: Variáveis a analisar.
        fator: Multiplicador do IQR (1,5 é o clássico).

    Returns:
        Uma linha por variável: quartis, limites, nº e % de outliers, mínimo,
        máximo e assimetria (``skew``; valores muito acima de 1 indicam cauda longa).
    """
    linhas: list[dict[str, object]] = []
    for coluna in colunas:
        serie = df[coluna]
        q1, q3 = float(serie.quantile(0.25)), float(serie.quantile(0.75))
        iqr = q3 - q1
        inferior, superior = q1 - fator * iqr, q3 + fator * iqr
        fora = (serie < inferior) | (serie > superior)
        linhas.append(
            {
                "variavel": coluna,
                "q1": q1,
                "q3": q3,
                "iqr": iqr,
                "limite_inferior": inferior,
                "limite_superior": superior,
                "n_outliers": int(fora.sum()),
                "pct_outliers": round(100 * float(fora.mean()), 2),
                "minimo": float(serie.min()),
                "maximo": float(serie.max()),
                "assimetria": round(float(serie.skew()), 2),
            }
        )
    return pd.DataFrame(linhas)


def sinalizar_outliers(df: pd.DataFrame, limites: pd.DataFrame) -> pd.DataFrame:
    """Cria as colunas ``<variavel>_outlier`` (1 = fora dos limites do IQR).

    Os valores originais **não são alterados**.

    Args:
        df: Base com as variáveis analisadas.
        limites: Saída de :func:`calcular_limites_iqr`.

    Returns:
        Cópia do DataFrame com uma coluna de sinalização por variável.
    """
    saida = df.copy()
    for linha in limites.itertuples(index=False):
        # Os limites guardados na tabela estão em precisão total (não arredondados),
        # então a sinalização é idêntica à contagem de outliers da tabela.
        fora = (saida[linha.variavel] < linha.limite_inferior) | (
            saida[linha.variavel] > linha.limite_superior
        )
        saida[f"{linha.variavel}_outlier"] = fora.astype("int8")
    return saida
