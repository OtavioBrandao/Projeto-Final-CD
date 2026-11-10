"""Auditoria de qualidade dos dados (perfil por coluna e resumos gerais).

A auditoria **não altera** nenhum dado. Ela mede o problema antes de resolvê-lo,
gerando as evidências (números) que depois aparecem na tabela
"o que estava errado -> o que fizemos -> por quê" da documentação.
"""

from __future__ import annotations

import logging
import re
import unicodedata

import pandas as pd
from pandas.api.types import is_numeric_dtype

from src.p1_engenharia_dados import configuracao as cfg

logger = logging.getLogger(__name__)

#: Observações fixas, por coluna, sobre o que a auditoria já sabe (ou verificou).
OBSERVACOES_FIXAS: dict[str, str] = {
    "id": "Identificador do acidente; liga ocorrências e pessoas.",
    "classificacao_acidente": "Consequência do acidente (vazamento se usada como atributo); contém 'NA' (D04).",
    "mortos": "Define o alvo acidente_fatal (F01); consequência, não pode ser atributo.",
    "pessoas": "Consequência (total de envolvidos); contagens nem sempre fecham (D03).",
    "tracado_via": "Valores compostos separados por ';' (F07).",
    "causa_acidente": "69 causas distintas; agrupadas em F06.",
    "condicao_metereologica": "Contém 'Ignorado' (D04).",
    "sentido_via": "Contém 'Não Informado' (D04).",
    "latitude": "Vírgula decimal; latitude positiva só vale no extremo norte (D07).",
    "longitude": "Vírgula decimal; conferida contra a caixa do Brasil (D07).",
    "idade": "Há valores impossíveis (ex.: 2024) (D05).",
    "ano_fabricacao_veiculo": "Contém 0 como 'não informado' (D06).",
    "sexo": "Atributo sensível; três formas de 'não sei' (D04).",
    "uop": "Coluna administrativa (quem atendeu); sai da modelagem.",
    "regional": "Coluna administrativa; sai da modelagem.",
    "delegacia": "Coluna administrativa; sai da modelagem.",
}


def _tipo_simplificado(serie: pd.Series) -> str:
    """Classifica uma coluna como ``numerica`` ou ``texto``.

    Args:
        serie: Coluna a classificar.

    Returns:
        ``"numerica"`` se o dtype for numérico, senão ``"texto"``.
    """
    return "numerica" if is_numeric_dtype(serie) else "texto"


def perfil_qualidade(df: pd.DataFrame, df_texto: pd.DataFrame, origem: str) -> pd.DataFrame:
    """Gera o perfil de qualidade por coluna.

    Args:
        df: Base lida com ``NA`` e vazio já convertidos em nulo.
        df_texto: Mesma base lida como texto puro (para contar ``NA`` e rótulos).
        origem: Nome do arquivo de origem (``ocorrencias`` ou ``pessoas``).

    Returns:
        Uma linha por coluna com: tipo, nulos (n e %), textos ``NA`` no arquivo,
        rótulos "sem informação" (``Ignorado``/``Não Informado``), % total sem
        informação, valores distintos, exemplo e observação.
    """
    linhas: list[dict[str, object]] = []
    n = len(df)
    for coluna in df.columns:
        if coluna == "ano_arquivo":  # coluna técnica criada na carga, não faz parte da base original
            continue
        serie = df[coluna]
        n_nulos = int(serie.isna().sum())
        n_rotulos = int(df_texto[coluna].isin(cfg.ROTULOS_SEM_INFORMACAO).sum()) if coluna in df_texto else 0
        validos = serie.dropna()
        linhas.append(
            {
                "origem": origem,
                "coluna": coluna,
                "tipo": _tipo_simplificado(serie),
                "n_nulos": n_nulos,
                "pct_nulos": round(100 * n_nulos / n, 2),
                "n_rotulos_sem_informacao": n_rotulos,
                "pct_sem_informacao": round(100 * (n_nulos + n_rotulos) / n, 2),
                "n_distintos": int(serie.nunique(dropna=True)),
                "exemplo": str(validos.iloc[0]) if len(validos) else "",
                "observacao": OBSERVACOES_FIXAS.get(coluna, ""),
            }
        )
    return pd.DataFrame(linhas)


def _normalizar(texto: str) -> str:
    """Tira acentos, caixa e espaços repetidos (para achar rótulos "iguais").

    Args:
        texto: Rótulo original.

    Returns:
        Rótulo em minúsculas, sem acento e sem espaços extras.
    """
    sem_acento = unicodedata.normalize("NFKD", texto).encode("ascii", "ignore").decode()
    return re.sub(r"\s+", " ", sem_acento).strip().lower()


def contar_rotulos_equivalentes(df: pd.DataFrame, colunas: tuple[str, ...] | list[str]) -> int:
    """Conta rótulos escritos de formas diferentes que significam a mesma coisa.

    Exemplo: ``"Céu Claro"`` e ``"CEU CLARO"``. Para cada coluna de texto, agrupa
    os valores distintos pelo texto normalizado (sem acento, caixa e espaços
    extras) e conta quantos grupos têm mais de uma grafia.

    Args:
        df: Base a inspecionar.
        colunas: Colunas de texto a verificar.

    Returns:
        Número de grupos com mais de uma grafia (0 = sem problema).
    """
    total = 0
    for coluna in colunas:
        if coluna not in df.columns:
            continue
        distintos = pd.Series(df[coluna].dropna().unique()).astype(str)
        total += int((distintos.groupby(distintos.map(_normalizar)).nunique() > 1).sum())
    return total


def resumo_geral(ocorrencias: pd.DataFrame, pessoas: pd.DataFrame) -> dict[str, int]:
    """Calcula os números gerais das duas bases brutas.

    Args:
        ocorrencias: Uma linha por acidente.
        pessoas: Uma linha por pessoa/veículo.

    Returns:
        Dicionário com contagens usadas na documentação.
    """
    sem_id = ocorrencias.drop(columns=[cfg.COLUNA_ID, "ano_arquivo"])
    return {
        "linhas_ocorrencias": int(len(ocorrencias)),
        "colunas_ocorrencias": int(ocorrencias.shape[1] - 1),  # sem a coluna ano_arquivo
        "linhas_pessoas": int(len(pessoas)),
        "colunas_pessoas": int(pessoas.shape[1] - 1),
        "id_unico": int(ocorrencias[cfg.COLUNA_ID].is_unique),
        "ocorrencias_sem_pessoas": int((~ocorrencias[cfg.COLUNA_ID].isin(pessoas[cfg.COLUNA_ID])).sum()),
        "pessoas_sem_ocorrencia": int((~pessoas[cfg.COLUNA_ID].isin(ocorrencias[cfg.COLUNA_ID])).sum()),
        "ocorrencias_duplicadas_sem_id": int(sem_id.duplicated().sum()),
    }
