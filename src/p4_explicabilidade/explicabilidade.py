"""Explicabilidade (SHAP) dos XGBoost da Pessoa 3.

Pontos de cuidado:

* Os modelos são ``Pipeline(preprocessamento, modelo)``. O SHAP é calculado sobre a matriz
  **já transformada** (One-Hot), mas reportado por **variável original**, somando as
  colunas One-Hot de cada variável. Sem isso, ``uf`` e ``br`` viram dezenas de barras.
* Os valores SHAP do XGBoost binário estão em **log-odds** (margem), não em probabilidade.
  Positivo = empurra para "fatal". Por isso a tabela de "sinais" traz também ``exp(Δ)``,
  o multiplicador de *odds*.
* Cenário A usa ``tipo_acidente`` e ``grupo_causa`` (definidos pela PRF após o atendimento):
  serve para **explicar**. Cenário B só usa local e calendário: serve para **priorizar**.
* SHAP descreve o que o **modelo** aprendeu, não causalidade.

Rodar na raiz:  ``python -m src.p4_explicabilidade.explicabilidade --cenarios A B``
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import sparse

from src.p3_modelagem import configuracao as cfg
from src.p3_modelagem.dados import carregar_base, selecionar_xy
from src.p3_modelagem.executar_avaliacao_final import carregar_pipeline

SAIDA_PADRAO = cfg.RAIZ_REPOSITORIO / "reports" / "p4"
SEMENTE = 42

# Famílias de negócio: agrupam variáveis para a leitura do gestor.
FAMILIAS = {
    "Tipo e causa do acidente": ("tipo_acidente", "grupo_causa"),
    "Horário": ("hora", "faixa_horaria", "fase_dia"),
    "Localização": ("uf", "br", "km", "latitude", "longitude", "regiao", "area_urbana"),
    "Via": ("tipo_pista", "sentido_via", "via_reta", "via_curva", "via_aclive", "via_declive",
            "via_intersecao", "via_rotatoria_retorno", "via_obras_desvio", "via_ponte_viaduto_tunel"),
    "Veículos e pessoas": ("veiculos", "envolve_moto", "envolve_caminhao", "envolve_onibus",
                           "envolve_bicicleta", "envolve_pedestre", "idade_media_veiculos"),
    "Calendário": ("dia_semana", "mes", "fim_de_semana", "eh_feriado"),
    "Clima": ("condicao_metereologica",),
}

# Sinais da EDA (Pessoa 2) a confirmar: (nome, variáveis cujo SHAP somar, condição, sinal esperado).
# Para moto e chuva a EDA encontrou RR < 1, então o SHAP deve vir NEGATIVO.
SINAIS = (
    ("Pedestre envolvido", ("envolve_pedestre",), lambda X: X["envolve_pedestre"].eq(1), +1),
    ("Madrugada ou noite", ("hora", "faixa_horaria", "fase_dia"),
     lambda X: X["faixa_horaria"].isin(["Madrugada", "Noite"]), +1),
    ("Pista simples", ("tipo_pista",), lambda X: X["tipo_pista"].eq("Simples"), +1),
    ("Norte ou Nordeste", ("uf", "br", "regiao", "latitude", "longitude"),
     lambda X: X["regiao"].isin(["Norte", "Nordeste"]), +1),
    ("Caminhão envolvido", ("envolve_caminhao",), lambda X: X["envolve_caminhao"].eq(1), +1),
    ("Trecho rural", ("area_urbana",), lambda X: X["area_urbana"].eq(0), +1),
    ("Colisão frontal", ("tipo_acidente",), lambda X: X["tipo_acidente"].eq("Colisão frontal"), +1),
    ("Motocicleta envolvida", ("envolve_moto",), lambda X: X["envolve_moto"].eq(1), -1),
    ("Chuva, garoa ou neblina", ("condicao_metereologica",),
     lambda X: X["condicao_metereologica"].isin(["Chuva", "Garoa/Chuvisco", "Nevoeiro/Neblina"]), -1),
)


# Agregação por variável original (testável sem shap/xgboost)
def mapa_colunas_originais(preprocessamento) -> list[str]:
    """Para cada coluna da matriz transformada, a variável original que a gerou.

    Segue a ordem de saída do ColumnTransformer (numéricas, categóricas, binárias) e
    confere com ``get_feature_names_out()``; falha alto se algo não bater.
    """
    mapa: list[str] = []
    for nome, transformador, colunas in preprocessamento.transformers_:
        if nome == "remainder":
            continue
        if nome == "categoricas":
            onehot = transformador.named_steps["onehot"]
            for coluna, categorias in zip(colunas, onehot.categories_):
                mapa += [coluna] * len(categorias)
        else:
            mapa += list(colunas)
    nomes = preprocessamento.get_feature_names_out()
    if len(mapa) != len(nomes):
        raise ValueError(f"Mapa com {len(mapa)} colunas, mas a matriz tem {len(nomes)}.")
    for original, nome in zip(mapa, nomes):
        if not nome.split("__", 1)[1].startswith(original):
            raise ValueError(f"Coluna transformada {nome!r} não corresponde a {original!r}.")
    return mapa


def agregar_por_variavel(shap_values: np.ndarray, mapa: list[str]) -> pd.DataFrame:
    """Soma os SHAP das colunas One-Hot de cada variável original (linhas = acidentes)."""
    valores = np.asarray(shap_values)
    if valores.shape[1] != len(mapa):
        raise ValueError("Número de colunas SHAP difere do mapa.")
    mapa_arr = np.asarray(mapa)
    variaveis = list(dict.fromkeys(mapa))  # preserva a ordem, sem repetir
    return pd.DataFrame({v: valores[:, mapa_arr == v].sum(axis=1) for v in variaveis})


def importancia(shap_var: pd.DataFrame) -> pd.DataFrame:
    """Importância global = média do |SHAP| por variável (em log-odds), ordenada."""
    imp = shap_var.abs().mean().sort_values(ascending=False).rename("shap_medio_abs").to_frame()
    imp["pct_do_total"] = imp["shap_medio_abs"] / imp["shap_medio_abs"].sum()
    return imp.rename_axis("variavel").reset_index()


def importancia_por_familia(imp: pd.DataFrame) -> pd.DataFrame:
    """Soma a importância por família de negócio (variáveis fora do mapa vão para 'Outras')."""
    var_para_fam = {v: f for f, vs in FAMILIAS.items() for v in vs}
    fam = imp.assign(familia=imp["variavel"].map(var_para_fam).fillna("Outras"))
    out = fam.groupby("familia")["shap_medio_abs"].sum().sort_values(ascending=False)
    return (out / out.sum()).rename("pct_do_total").rename_axis("familia").reset_index()


def confirmar_sinais(shap_var: pd.DataFrame, X: pd.DataFrame) -> pd.DataFrame:
    """Compara o SHAP médio com e sem cada condição e checa o sinal contra o esperado pela EDA."""
    linhas = []
    for nome, variaveis, condicao, esperado in SINAIS:
        usadas = [v for v in variaveis if v in shap_var.columns]
        try:
            mask = condicao(X).to_numpy()
        except KeyError:
            continue  # o cenário B não tem, p.ex., tipo_pista
        if not usadas:
            continue
        soma = shap_var[usadas].sum(axis=1).to_numpy()
        com, sem = soma[mask].mean(), soma[~mask].mean()
        delta = com - sem
        linhas.append({
            "sinal": nome, "variaveis_somadas": ", ".join(usadas), "n_com": int(mask.sum()),
            "shap_medio_com": com, "shap_medio_sem": sem, "diferenca_logodds": delta,
            "multiplicador_odds": float(np.exp(delta)),
            "esperado_pela_eda": "aumenta" if esperado > 0 else "reduz",
            "confirma": bool(np.sign(delta) == esperado),
        })
    return pd.DataFrame(linhas)


# Figuras
def grafico_importancia(imp: pd.DataFrame, titulo: str, caminho: Path, topo: int = 12) -> None:
    d = imp.head(topo).iloc[::-1]
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.barh(d["variavel"], d["shap_medio_abs"], color="#1f77b4")
    ax.set(xlabel="Média do |SHAP| (log-odds de acidente fatal)", title=titulo)
    ax.title.set_fontsize(10)
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(caminho, dpi=160)
    plt.close(fig)


def grafico_caso(shap_linha: pd.Series, x_linha: pd.Series, prob: float, caminho: Path, topo: int = 8) -> None:
    """Explicação local: por que ESTE acidente recebeu risco alto."""
    s = shap_linha.reindex(shap_linha.abs().sort_values(ascending=False).index).head(topo).iloc[::-1]
    rotulos = [f"{v} = {x_linha[v]}" for v in s.index]
    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    ax.barh(rotulos, s.values, color=["#d62728" if v > 0 else "#2ca02c" for v in s.values])
    ax.axvline(0, color="k", lw=0.8)
    ax.set(xlabel="Contribuição SHAP (log-odds)  →  vermelho: aumenta o risco; verde: reduz",
           title=f"Caso real de 2025: probabilidade prevista de morte = {prob:.0%}")
    ax.title.set_fontsize(10)
    fig.tight_layout()
    fig.savefig(caminho, dpi=160)
    plt.close(fig)


# Execução (precisa de shap + xgboost)
def explicar_cenario(cenario: str, base: pd.DataFrame, saida: Path, amostra: int) -> dict[str, pd.DataFrame]:
    import shap  # importação adiada: o resto do módulo funciona sem shap
    import xgboost

    pipeline = carregar_pipeline(cenario)
    pre, modelo = pipeline.named_steps["preprocessamento"], pipeline.named_steps["modelo"]
    X_all, y_all = selecionar_xy(base.loc[base["ano"].eq(cfg.ANO_TESTE_FINAL)], cenario)
    if 0 < amostra < len(X_all):
        X_all = X_all.sample(amostra, random_state=SEMENTE)
    y_all = y_all.loc[X_all.index]

    # ATENÇÃO: manter a matriz ESPARSA. O XGBoost foi treinado com CSR e, nesse formato,
    # os zeros não armazenados são tratados como "ausentes". Densificar mudaria o modelo.
    X_t = pre.transform(X_all)
    X_t = sparse.csr_matrix(X_t, dtype=np.float32)
    nomes = list(pre.get_feature_names_out())

    # Valores SHAP exatos (TreeSHAP) pelo próprio XGBoost: a última coluna é o viés.
    contribs = modelo.get_booster().predict(xgboost.DMatrix(X_t), pred_contribs=True)
    valores, vies = contribs[:, :-1], contribs[:, -1]

    # Checagem de sanidade: base + soma dos SHAP deve reproduzir a probabilidade do pipeline.
    prob = 1 / (1 + np.exp(-(valores.sum(axis=1) + vies)))
    prob_pipeline = pipeline.predict_proba(X_all)[:, list(modelo.classes_).index(1)]
    erro = float(np.abs(prob - prob_pipeline).max())
    if erro > 1e-3:
        raise RuntimeError(f"SHAP não reproduz o modelo (erro máx. {erro:.2e}); não use estes resultados.")
    print(f"Cenário {cenario}: checagem SHAP x modelo OK (erro máx. {erro:.1e}).")

    mapa = mapa_colunas_originais(pre)
    shap_var = agregar_por_variavel(valores, mapa)
    imp = importancia(shap_var)
    fam = importancia_por_familia(imp)
    sinais = confirmar_sinais(shap_var, X_all)

    tab, fig = saida / "tabelas", saida / "figuras"
    imp.to_csv(tab / f"p4_shap_importancia_{cenario}.csv", index=False)
    fam.to_csv(tab / f"p4_shap_familias_{cenario}.csv", index=False)
    sinais.to_csv(tab / f"p4_shap_sinais_{cenario}.csv", index=False)

    papel = "explicar (inclui tipo/causa)" if cenario == "A" else "priorizar (local + calendário)"
    grafico_importancia(imp, f"Cenário {cenario}: o que mais pesa na previsão — {papel}",
                        fig / f"p4_shap_importancia_{cenario}.png")

    plt.figure()
    shap.summary_plot(valores, X_t.toarray(), feature_names=nomes, max_display=15, show=False)
    plt.tight_layout()
    plt.savefig(fig / f"p4_shap_resumo_{cenario}.png", dpi=160, bbox_inches="tight")
    plt.close("all")

    # Caso local: acidente fatal real com a maior probabilidade prevista.
    fatais = np.flatnonzero(y_all.to_numpy() == 1)
    i = fatais[np.argmax(prob[fatais])]
    grafico_caso(shap_var.iloc[i], X_all.iloc[i], float(prob[i]), fig / f"p4_shap_caso_{cenario}.png")
    return {"importancia": imp, "familias": fam, "sinais": sinais}


def executar(cenarios=("A", "B"), saida: Path = SAIDA_PADRAO, amostra: int = 10000) -> None:
    (saida / "tabelas").mkdir(parents=True, exist_ok=True)
    (saida / "figuras").mkdir(parents=True, exist_ok=True)
    base = carregar_base()
    for c in cenarios:
        res = explicar_cenario(c, base, saida, amostra)
        print(f"\n=== Cenário {c} ===")
        print(res["importancia"].head(10).to_string(index=False))
        print(res["familias"].to_string(index=False))
        print(res["sinais"][["sinal", "multiplicador_odds", "esperado_pela_eda", "confirma"]].to_string(index=False))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--cenarios", nargs="+", default=["A", "B"], choices=["A", "B"])
    ap.add_argument("--amostra", type=int, default=10000, help="nº de acidentes de 2025 a explicar (0 = todos)")
    ap.add_argument("--saida", type=Path, default=SAIDA_PADRAO)
    a = ap.parse_args()
    executar(a.cenarios, a.saida, a.amostra)
