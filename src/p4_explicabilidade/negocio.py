"""Do modelo ao valor de negócio.

Duas saídas, ambas em ``reports/p4/``:

1. **Concentração de risco** (curva de ganho / lift) a partir das previsões de 2025
   da Pessoa 3: "se priorizarmos os X% de maior risco, que fatia das mortes cobrimos?"
2. **Segmentos dos insights**: tamanho, mortes, taxa fatal (IC de Wilson), risco relativo
   e réplica 2024 x 2025 de cada recorte usado no pitch.

Rodar na raiz do repositório:  ``python -m src.p4_explicabilidade.negocio``
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.p3_modelagem.configuracao import COLUNA_ALVO, RAIZ_REPOSITORIO
from src.p3_modelagem.dados import carregar_base

SAIDA_PADRAO = RAIZ_REPOSITORIO / "reports" / "p4"
PREVISOES_2025 = RAIZ_REPOSITORIO / "reports" / "p3_final" / "previsoes_2025.csv.gz"
FRACOES = (0.05, 0.10, 0.20, 0.30)
NOTURNO = ("Madrugada", "Noite")
NORTE_NORDESTE = ("Norte", "Nordeste")


# 1. Concentração de risco (lift / ganho acumulado)
def curva_ganho(y_real: pd.Series, probabilidade: pd.Series) -> pd.DataFrame:
    """Curva de ganho: fração dos acidentes (do mais ao menos arriscado) x fração das mortes."""
    d = pd.DataFrame({"y": np.asarray(y_real), "p": np.asarray(probabilidade)})
    d = d.sort_values("p", ascending=False, kind="mergesort").reset_index(drop=True)
    n, mortes = len(d), d["y"].sum()
    return pd.DataFrame({
        "fracao_acidentes": np.arange(1, n + 1) / n,
        "fracao_mortes": d["y"].cumsum().to_numpy() / mortes,
    })


def concentracao_por_fracao(previsoes: pd.DataFrame, fracoes=FRACOES) -> pd.DataFrame:
    """Para cada modelo e fração priorizada: taxa fatal no grupo, lift e % das mortes cobertas."""
    linhas = []
    for modelo, d in previsoes.groupby("modelo", sort=False):
        d = d.sort_values("probabilidade", ascending=False, kind="mergesort").reset_index(drop=True)
        n, mortes, taxa_geral = len(d), int(d["y_real"].sum()), d["y_real"].mean()
        for f in fracoes:
            k = int(n * f)
            topo = d.iloc[:k]
            linhas.append({
                "modelo": modelo, "fracao_priorizada": f, "n_acidentes": k,
                "taxa_fatal_no_grupo": topo["y_real"].mean(),
                "lift": topo["y_real"].mean() / taxa_geral,
                "mortes_cobertas": int(topo["y_real"].sum()),
                "pct_mortes_cobertas": topo["y_real"].sum() / mortes,
            })
    return pd.DataFrame(linhas)


def grafico_ganho(previsoes: pd.DataFrame, caminho: Path) -> None:
    """Figura do pitch: quanto das mortes fica no topo do ranking de risco."""
    rotulos = {"xgboost_A": "Modelo A (retrospectivo)", "xgboost_B": "Modelo B (local + calendário)"}
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot([0, 1], [0, 1], "--", color="gray", label="Sem modelo (aleatório)")
    for modelo, rotulo in rotulos.items():
        d = previsoes[previsoes["modelo"] == modelo]
        c = curva_ganho(d["y_real"], d["probabilidade"])
        ax.plot(c["fracao_acidentes"], c["fracao_mortes"], lw=2.2, label=rotulo)
    ax.axvline(0.10, color="k", lw=0.6, ls=":")
    ax.set(xlabel="% dos acidentes (do maior para o menor risco previsto)",
           ylabel="% das mortes capturadas", xlim=(0, 1), ylim=(0, 1),
           title="Priorizando os 10% de maior risco, o modelo A cobre metade das mortes")
    ax.title.set_fontsize(10)
    ax.legend(loc="lower right")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(caminho, dpi=160)
    plt.close(fig)


# 2. Segmentos dos insights
def intervalo_wilson(sucessos: float, n: float, z: float = 1.96) -> tuple[float, float]:
    """IC de Wilson para uma proporção (mesmo método adotado pela Pessoa 2)."""
    if n == 0:
        return (np.nan, np.nan)
    p = sucessos / n
    denom = 1 + z**2 / n
    centro = (p + z**2 / (2 * n)) / denom
    meia = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
    return (centro - meia, centro + meia)


def definir_segmentos(b: pd.DataFrame) -> dict[str, tuple[str, pd.Series]]:
    """Segmentos do pitch: nome -> (descrição curta, máscara booleana)."""
    noturno = b["faixa_horaria"].isin(NOTURNO)
    simples = b["tipo_pista"].eq("Simples")
    ped = b["envolve_pedestre"].eq(1)
    return {
        "noite_madrugada": ("Madrugada ou noite (18h–5h)", noturno),
        "pista_simples": ("Pista simples", simples),
        "frontal_pista_simples": ("Colisão frontal em pista simples",
                                  b["tipo_acidente"].eq("Colisão frontal") & simples),
        "pedestre": ("Pedestre envolvido", ped),
        "pedestre_rural": ("Pedestre em trecho rural", ped & b["area_urbana"].eq(0)),
        "pedestre_noturno": ("Pedestre à noite/madrugada", ped & noturno),
        "alvo_prioritario": ("Pista simples + noite/madrugada + Norte/Nordeste",
                             simples & noturno & b["regiao"].isin(NORTE_NORDESTE)),
        # Hipóteses derrubadas pela EDA: úteis para NÃO desperdiçar recurso.
        "moto": ("Motocicleta envolvida", b["envolve_moto"].eq(1)),
        "chuva_neblina": ("Chuva, garoa ou neblina",
                          b["condicao_metereologica"].isin(["Chuva", "Garoa/Chuvisco", "Nevoeiro/Neblina"])),
        "feriado": ("Feriado nacional", b["eh_feriado"].eq(1)),
    }


def tabela_segmentos(b: pd.DataFrame) -> pd.DataFrame:
    """Métricas de cada segmento: tamanho, mortes, taxa (IC 95%), RR vs. resto, réplica por ano."""
    y = b[COLUNA_ALVO]
    total, mortes_total = len(b), int(y.sum())
    linhas = []
    for chave, (descricao, mask) in definir_segmentos(b).items():
        n, m = int(mask.sum()), int(y[mask].sum())
        taxa = m / n
        taxa_resto = y[~mask].mean()
        lo, hi = intervalo_wilson(m, n)
        por_ano = {ano: y[mask & b["ano"].eq(ano)].mean() for ano in (2024, 2025)}
        linhas.append({
            "segmento": chave, "descricao": descricao, "n_acidentes": n,
            "pct_acidentes": n / total, "mortes": m, "pct_mortes": m / mortes_total,
            "taxa_fatal": taxa, "ic95_inf": lo, "ic95_sup": hi,
            "risco_relativo_vs_resto": taxa / taxa_resto,
            "taxa_2024": por_ano[2024], "taxa_2025": por_ano[2025],
        })
    return pd.DataFrame(linhas)


# Execução
def executar(saida: Path = SAIDA_PADRAO) -> dict[str, pd.DataFrame]:
    (saida / "tabelas").mkdir(parents=True, exist_ok=True)
    (saida / "figuras").mkdir(parents=True, exist_ok=True)

    previsoes = pd.read_csv(PREVISOES_2025)
    conc = concentracao_por_fracao(previsoes[previsoes["modelo"].isin(["xgboost_A", "xgboost_B"])])
    conc.to_csv(saida / "tabelas" / "p4_concentracao_risco_2025.csv", index=False)
    grafico_ganho(previsoes, saida / "figuras" / "p4_curva_ganho_2025.png")

    seg = tabela_segmentos(carregar_base())
    seg.to_csv(saida / "tabelas" / "p4_segmentos_insights.csv", index=False)
    return {"concentracao": conc, "segmentos": seg}


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--saida", type=Path, default=SAIDA_PADRAO)
    res = executar(ap.parse_args().saida)
    pd.set_option("display.width", 200, "display.float_format", "{:.3f}".format)
    print(res["concentracao"].to_string(index=False))
    print(res["segmentos"].drop(columns="descricao").to_string(index=False))
