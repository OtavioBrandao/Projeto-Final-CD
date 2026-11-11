"""Fábricas iniciais; sem treinamento, busca ou avaliação na importação."""

from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from src.p3_modelagem.preprocessamento import criar_preprocessamento

SEMENTE = 42
ALGORITMOS = ("dummy", "regressao_logistica", "random_forest", "xgboost")


def criar_estimador(algoritmo: str):
    """Retorna um estimador novo com limites moderados e pesos padrão."""
    if algoritmo == "dummy":
        return DummyClassifier(strategy="prior")
    if algoritmo == "regressao_logistica":
        return LogisticRegression(max_iter=2000, random_state=SEMENTE)
    if algoritmo == "random_forest":
        return RandomForestClassifier(
            n_estimators=100, max_depth=12, min_samples_leaf=3,
            n_jobs=2, random_state=SEMENTE,
        )
    if algoritmo == "xgboost":
        # Importação adiada: a ausência da dependência não impede os outros três.
        try:
            from xgboost import XGBClassifier
        except ImportError as erro:
            raise ImportError(
                "XGBoost indisponível. Instale xgboost>=2.0 no ambiente Python do projeto."
            ) from erro
        return XGBClassifier(
            objective="binary:logistic", eval_metric="logloss",
            tree_method="hist", device="cpu", n_estimators=100,
            max_depth=4, learning_rate=0.1, n_jobs=2, random_state=SEMENTE,
        )
    raise ValueError(f"Algoritmo desconhecido: {algoritmo!r}; use um de {ALGORITMOS}.")


def criar_pipeline(cenario: str, algoritmo: str) -> Pipeline:
    """Pipeline A/B novo e não ajustado: fit somente no X/y de treino.

    Use predict_proba(X)[:, indice] com indice obtido de classes_ == 1 para
    P(acidente_fatal=1). Com ambas as classes no treino, classes_ é [0, 1].
    A fábrica não acessa dados, anos, cortes ou validações.
    """
    preprocessamento = criar_preprocessamento(
        cenario, escalonar=algoritmo == "regressao_logistica",
    )
    return Pipeline([
        ("preprocessamento", preprocessamento),
        ("modelo", criar_estimador(algoritmo)),
    ])
