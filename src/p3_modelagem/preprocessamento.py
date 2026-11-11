"""Transformações da Etapa 02, aprendidas somente no fit do treino recebido."""

import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

from src.p3_modelagem import configuracao as cfg


# Tipos semânticos do contrato da Etapa 01, sem inferência a partir dos dados.
COLUNAS_NUMERICAS = (
    "km", "veiculos", "latitude", "longitude", "idade_media_veiculos", "hora", "mes",
)


def identificar_tipos(cenario: str) -> dict[str, tuple[str, ...]]:
    """Particiona as features; rejeita proibições, duplicações e tipos ambíguos."""
    if cenario not in cfg.FEATURES_POR_CENARIO:
        raise ValueError(f"Cenário desconhecido: {cenario!r}; use 'A' ou 'B'.")
    features = cfg.FEATURES_POR_CENARIO[cenario]
    if len(features) != len(set(features)) or set(features) & cfg.PREDITORES_PROIBIDOS:
        raise ValueError("Contrato contém features duplicadas ou preditores proibidos.")
    tipos = {
        "numericas": tuple(c for c in features if c in COLUNAS_NUMERICAS),
        "categoricas": tuple(c for c in features if c in cfg.COLUNAS_CATEGORICAS),
        "binarias": tuple(c for c in features if c in cfg.COLUNAS_BINARIAS),
    }
    if any(sum(c in grupo for grupo in tipos.values()) != 1 for c in features):
        raise ValueError("Cada feature deve pertencer exatamente a uma categoria de pré-processamento.")
    return tipos


def _categorias_textuais(x):
    """Normaliza tipos/ausentes sem aprender vocabulário nem alterar a entrada.

    Inclusive BR numérica vira texto; pd.NA/None/NaN viram np.nan para o
    SimpleImputer, independentemente da versão/dtype textual do pandas.
    """
    texto = x.astype("string").astype(object)
    return texto.where(texto.notna(), np.nan)


def criar_preprocessamento(cenario: str, *, escalonar: bool = False) -> ColumnTransformer:
    """Cria um transformador novo, sem fit; recebe X de selecionar_xy().

    Binários 0/1 do contrato não têm nulos e passam intactos. Colunas numéricas
    inteiramente ausentes no treino permanecem na matriz (fallback zero do
    SimpleImputer). A concatenação conserva saída esparsa quando possível.
    """
    tipos = identificar_tipos(cenario)
    numericas = [("imputacao", SimpleImputer(strategy="median", keep_empty_features=True))]
    if escalonar:
        numericas.append(("escala", StandardScaler()))
    categoricas = Pipeline([
        ("texto", FunctionTransformer(_categorias_textuais, feature_names_out="one-to-one")),
        ("imputacao", SimpleImputer(
            strategy="constant", fill_value="__ausente__", keep_empty_features=True,
        )),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=True)),
    ])
    return ColumnTransformer(
        [
            ("numericas", Pipeline(numericas), list(tipos["numericas"])),
            ("categoricas", categoricas, list(tipos["categoricas"])),
            ("binarias", "passthrough", list(tipos["binarias"])),
        ],
        remainder="drop",
        sparse_threshold=1.0,
    )
