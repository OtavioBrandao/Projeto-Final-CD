"""Carga, validação e seleção temporal sem ajuste, reamostragem ou escrita."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from pandas.api.types import is_integer_dtype, is_numeric_dtype

from src.p3_modelagem import configuracao as cfg


def validar_esquema(base: pd.DataFrame) -> None:
    """Valida os 42 nomes, tipos semânticos, nulos, domínios e índice único.

    Aceita um subconjunto de linhas/anos do contrato, inclusive só 2024.
    Não exige as contagens da versão auditada nem aprende categorias.
    """
    if base.columns.has_duplicates:
        raise ValueError("Colunas duplicadas no esquema da base.")
    faltantes = sorted(cfg.COLUNAS_ESPERADAS - set(base.columns))
    extras = sorted(set(base.columns) - cfg.COLUNAS_ESPERADAS)
    if faltantes or extras:
        raise ValueError(f"Esquema inesperado: colunas obrigatórias ausentes={faltantes}; extras={extras}.")
    if base.empty:
        raise ValueError("Base vazia: nenhum registro disponível.")
    if not base.index.is_unique or base.index.hasnans:
        raise ValueError("O índice dos registros deve ser único e sem nulos; preserve a referência original.")

    for coluna in base.columns:
        serie = base[coluna]
        if coluna not in cfg.COLUNAS_ACEITAM_NULOS and serie.isna().any():
            raise ValueError(f"Coluna '{coluna}' contém valores ausentes não permitidos.")
        if coluna in cfg.COLUNAS_CATEGORICAS:
            # BR pode chegar como código numérico; seleção a entrega como texto.
            if coluna == "br" and is_numeric_dtype(serie):
                if not np.isfinite(serie).all() or not serie.mod(1).eq(0).all():
                    raise ValueError("Coluna 'br' deve conter códigos categóricos inteiros ou texto.")
            elif not serie.map(lambda valor: isinstance(valor, str)).all():
                raise ValueError(f"Coluna '{coluna}' deve ser categórica textual.")
        elif not is_numeric_dtype(serie):
            raise ValueError(f"Coluna '{coluna}' deve ser numérica.")
        elif not np.isfinite(serie.dropna()).all():
            raise ValueError(f"Coluna '{coluna}' contém valores não finitos.")

    if not is_integer_dtype(base[cfg.COLUNA_ALVO]):
        raise ValueError(f"Alvo '{cfg.COLUNA_ALVO}' deve ter tipo inteiro binário (0/1).")
    for coluna in cfg.COLUNAS_BINARIAS:
        if not base[coluna].isin([0, 1]).all():
            raise ValueError(f"Coluna '{coluna}' deve conter somente valores binários 0/1.")
    for coluna, minimo, maximo in (("mes", 1, 12), ("hora", 0, 23)):
        serie = base[coluna]
        if not (serie.between(minimo, maximo) & serie.mod(1).eq(0)).all():
            raise ValueError(f"Coluna '{coluna}' deve conter inteiros entre {minimo} e {maximo}.")
    anos = {cfg.ANO_DESENVOLVIMENTO, cfg.ANO_TESTE_FINAL}
    if not base["ano"].isin(anos).all():
        raise ValueError(f"Anos inesperados na base: esperados somente {sorted(anos)}.")


def carregar_base(caminho: str | Path = cfg.CSV_MODELAGEM) -> pd.DataFrame:
    """Lê o CSV sem reordenar linhas; índice = posição original, começando em 0.

    ``linha_original`` é o nome do índice, nunca uma coluna preditora.
    BR como string é uma declaração de tipo, sem codificador ou vocabulário.
    """
    base = pd.read_csv(caminho, dtype={"br": "string"}, keep_default_na=False,
                       na_values=[""], low_memory=False)
    base.index.name = "linha_original"
    validar_esquema(base)
    return base


def selecionar_xy(base: pd.DataFrame, cenario: str) -> tuple[pd.DataFrame, pd.Series]:
    """Entrega cópias de X e y alinhadas pelo índice original, no cenário A/B."""
    if cenario not in cfg.FEATURES_POR_CENARIO:
        raise ValueError(f"Cenário desconhecido: {cenario!r}; use 'A' ou 'B'.")
    validar_esquema(base)
    features = cfg.FEATURES_POR_CENARIO[cenario]
    proibidas = set(features) & cfg.PREDITORES_PROIBIDOS
    if proibidas:
        raise ValueError(f"Preditores proibidos no cenário {cenario}: {sorted(proibidas)}.")
    x = base.loc[:, list(features)].copy()
    x["br"] = x["br"].astype("string")
    return x, base[cfg.COLUNA_ALVO].copy()


def separar_desenvolvimento_teste(base: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Retorna todos os registros de 2024 e todos de 2025, na ordem de entrada."""
    validar_esquema(base)
    desenvolvimento = base.loc[base["ano"].eq(cfg.ANO_DESENVOLVIMENTO)].copy()
    teste_final = base.loc[base["ano"].eq(cfg.ANO_TESTE_FINAL)].copy()
    if desenvolvimento.empty or teste_final.empty:
        raise ValueError("A separação exige registros de 2024 (desenvolvimento) e 2025 (teste final).")
    return desenvolvimento, teste_final


@dataclass(frozen=True)
class CorteTemporal:
    """Rótulos do índice original, para selecionar com .loc (nunca com .iloc)."""

    rodada: int
    treino: pd.Index
    validacao: pd.Index


def gerar_cortes_temporais(desenvolvimento: pd.DataFrame) -> tuple[CorteTemporal, ...]:
    """Gera as três janelas expansivas exclusivamente em 2024.

    As referências independem do cenário/algoritmo. Treinos de rodadas
    posteriores incorporam validações anteriores; dentro da rodada são disjuntos.
    Não ordena nem exige a data completa: os limites são mensais.
    """
    validar_esquema(desenvolvimento)
    if not desenvolvimento["ano"].eq(cfg.ANO_DESENVOLVIMENTO).all():
        raise ValueError("Os cortes internos aceitam exclusivamente desenvolvimento de 2024.")
    cortes = []
    mes = desenvolvimento["mes"]
    for rodada, fim_treino, inicio_validacao, fim_validacao in cfg.CORTES_TEMPORAIS:
        treino = desenvolvimento.index[mes.between(1, fim_treino)].copy()
        validacao = desenvolvimento.index[mes.between(inicio_validacao, fim_validacao)].copy()
        if treino.empty or validacao.empty:
            raise ValueError(f"Rodada {rodada}: treino ou validação sem registros no período definido.")
        cortes.append(CorteTemporal(rodada, treino, validacao))
    return tuple(cortes)
