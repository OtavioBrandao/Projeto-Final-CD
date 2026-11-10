"""Figuras da EDA (``reports/figuras/p2_*.png``).

Regras que todas as figuras seguem (para a banca reconhecer um padrão só):

* **O título é a conclusão**, montado com números das tabelas (nada digitado).
  O subtítulo diz o que está sendo medido e o tamanho da amostra.
* Mesma paleta da Pessoa 1: azul = normal/referência, laranja = risco/atenção,
  cinza = apoio. A rampa do mapa de calor é de um tom só (claro -> escuro).
* Toda taxa vem com **intervalo de confiança de 95%** (barra ou faixa). Sem IC,
  uma diferença de 0,5 ponto pode ser só ruído.
* Um eixo y por gráfico (nunca dois eixos com escalas diferentes).
* Poucas figuras, cada uma responde uma pergunta: quatro "principais" (artigo e
  slides) e quatro "de apoio" (README da EDA e eventual apêndice).
"""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # sem janela: roda em servidor e nos testes
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402

from src.p2_eda import configuracao as cfg  # noqa: E402
from src.p2_eda.estatistica import fmt_p, ic_wilson  # noqa: E402

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Formatação e estilo
# ---------------------------------------------------------------------------


def fmt_num(valor: float, casas: int = 1) -> str:
    return f"{float(valor):.{casas}f}".replace(".", ",")


def fmt_int(valor: float) -> str:
    return f"{int(round(float(valor))):,}".replace(",", ".")


def est_ic(taxa_pct: float, n: int) -> tuple[float, float]:
    inf, sup = ic_wilson(int(round(taxa_pct * n / 100)), int(n))
    return 100 * inf, 100 * sup


def _estilo() -> None:
    plt.rcParams.update({
        "figure.facecolor": "white", "axes.facecolor": "white", "savefig.facecolor": "white",
        "font.size": 10, "axes.titlesize": 12, "axes.labelsize": 10,
        "axes.edgecolor": cfg.CINZA, "axes.labelcolor": cfg.TEXTO_SECUNDARIO,
        "xtick.color": cfg.TEXTO_SECUNDARIO, "ytick.color": cfg.TEXTO_SECUNDARIO,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.color": cfg.CINZA_CLARO, "grid.linewidth": 0.8,
        "axes.axisbelow": True, "legend.frameon": False,
    })  # fmt: skip


def _titulos(fig: Figure, titulo: str, subtitulo: str) -> None:
    fig.text(0.01, 0.985, titulo, ha="left", va="top", fontsize=13, fontweight="bold", color=cfg.TEXTO)
    fig.text(0.01, 0.935, subtitulo, ha="left", va="top", fontsize=9.5, color=cfg.TEXTO_SECUNDARIO)


def _salvar(fig: Figure, nome: str, pasta: Path) -> Path:
    pasta.mkdir(parents=True, exist_ok=True)
    caminho = pasta / f"{nome}.png"
    fig.savefig(caminho, dpi=cfg.DPI_FIGURAS, bbox_inches="tight")
    plt.close(fig)
    logger.info("Figura gravada: %s", caminho)
    return caminho


def _fonte(fig: Figure, texto: str = "") -> None:
    base = "Fonte: PRF, Dados Abertos (acidentes 2024–2025). Elaboração própria."
    fig.text(0.01, -0.01, f"{base} {texto}".strip(), ha="left", va="top", fontsize=8, color=cfg.CINZA)


# ---------------------------------------------------------------------------
# Figura 1 — Fatores de risco (forest plot)
# ---------------------------------------------------------------------------


def fig_risco_relativo(fatores: pd.DataFrame, n_total: int, pasta: Path) -> Path:
    _estilo()
    t = fatores.sort_values("rr").reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(10, 6.2))
    fig.subplots_adjust(top=0.84, left=0.30, right=0.80)
    for i, r in t.iterrows():
        cor = cfg.CINZA if not r["significativo"] else (cfg.LARANJA if r["rr"] > 1 else cfg.AZUL)
        ax.plot([r["rr_ic_inf"], r["rr_ic_sup"]], [i, i], color=cor, lw=2, solid_capstyle="round")
        ax.plot(r["rr"], i, "o", color=cor, ms=8, mec="white", mew=1.5, zorder=3)
        txt = f"{fmt_num(r['rr'], 2)}×   ({fmt_num(r['taxa_expostos_pct'])}% vs {fmt_num(r['taxa_referencia_pct'])}%)"
        if not r["significativo"]:
            txt += "  n.s."
        ax.text(1.02, i, txt, transform=ax.get_yaxis_transform(), va="center", fontsize=9, color=cfg.TEXTO)
    ax.axvline(1, color=cfg.TEXTO_SECUNDARIO, lw=1, ls="--")
    ax.set_xscale("log")
    ticks = [0.5, 0.75, 1, 1.5, 2, 3, 4, 5]
    ax.set_xticks(ticks, [fmt_num(x, 2).rstrip("0").rstrip(",") for x in ticks])
    ax.minorticks_off()
    ax.set_xlim(0.6, 5.5)
    ax.set_yticks(range(len(t)), t["fator"])
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("Risco relativo de o acidente ser fatal (escala log; 1 = sem diferença)")
    top = t.iloc[-1]
    _titulos(
        fig,
        f"{top['fator']} multiplica o risco de morte por {fmt_num(top['rr'], 1)}; "
        "pista simples, caminhão e noite, por ~2",
        f"Risco relativo e IC 95% · taxa de fatais no grupo vs. referência · n = {fmt_int(n_total)} acidentes · "
        "n.s. = não significativo após Holm",
    )
    _fonte(fig, "Laranja: aumenta o risco; azul: reduz; cinza: sem evidência.")
    return _salvar(fig, "p2_fig1_risco_relativo", pasta)


# ---------------------------------------------------------------------------
# Figura 2 — Hora do dia (volume x letalidade)
# ---------------------------------------------------------------------------


def fig_hora_do_dia(hora: pd.DataFrame, taxa_geral: float, pasta: Path) -> Path:
    _estilo()
    tot = hora[hora["ano"] == "Total"].sort_values("hora")
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6.8), sharex=True, gridspec_kw={"height_ratios": [1, 1.4]})
    fig.subplots_adjust(top=0.86, hspace=0.12)
    x = tot["hora"].to_numpy()

    ax1.bar(x, tot["n"], color=cfg.CINZA_CLARO, width=0.8, label="Todos os acidentes")
    ax1.bar(x, tot["n_fatais"], color=cfg.LARANJA, width=0.8, label="Acidentes fatais")
    ax1.set_ylabel("Acidentes (2024+2025)")
    ax1.legend(loc="upper left", ncol=2, fontsize=9)
    ax1.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: fmt_int(v)))

    for madrugada in (ax1, ax2):
        madrugada.axvspan(-0.5, 5.5, color=cfg.CINZA_CLARO, alpha=0.45, lw=0, zorder=0)
    ax2.fill_between(x, tot["ic_inf_pct"], tot["ic_sup_pct"], color=cfg.AZUL, alpha=0.18, lw=0)
    ax2.plot(x, tot["taxa_pct"], color=cfg.AZUL, lw=2.2, marker="o", ms=4, label="2024–2025 (IC 95%)")
    for ano, estilo in (("2024", ":"), ("2025", "--")):
        a = hora[hora["ano"] == ano].sort_values("hora")
        ax2.plot(a["hora"], a["taxa_pct"], color=cfg.CINZA, lw=1.1, ls=estilo, label=ano)
    ax2.axhline(taxa_geral, color=cfg.TEXTO_SECUNDARIO, lw=1, ls="--")
    ax2.text(23.4, taxa_geral + 0.25, f"média {fmt_num(taxa_geral)}%", ha="right", fontsize=8.5,
             color=cfg.TEXTO_SECUNDARIO)  # fmt: skip
    ax2.set_ylabel("% de acidentes que foram fatais")
    ax2.set_xlabel("Hora do acidente")
    ax2.set_xticks(range(0, 24, 2))
    ax2.set_xlim(-0.5, 23.5)
    ax2.legend(loc="upper center", ncol=3, fontsize=9)
    ax2.text(2.5, ax2.get_ylim()[0] + 0.3, "madrugada", ha="center", fontsize=9, color=cfg.TEXTO_SECUNDARIO)

    pico_taxa = tot.loc[tot["taxa_pct"].idxmax()]
    pico_vol = tot.loc[tot["n"].idxmax()]
    vale = tot.loc[tot["taxa_pct"].idxmin()]
    _titulos(
        fig,
        f"Mais acidentes às {int(pico_vol['hora'])}h, mas a letalidade é máxima de madrugada "
        f"({fmt_num(pico_taxa['taxa_pct'])}% às {int(pico_taxa['hora'])}h vs {fmt_num(vale['taxa_pct'])}% às {int(vale['hora'])}h)",
        "Em cima: volume por hora. Embaixo: taxa de fatais com IC 95%; linhas cinza = cada ano separado "
        "(o padrão se repete).",
    )
    _fonte(fig)
    return _salvar(fig, "p2_fig2_hora_do_dia", pasta)


# ---------------------------------------------------------------------------
# Figura 3 — Causas: volume x gravidade
# ---------------------------------------------------------------------------

_NOMES_CAUSA: dict[str, str] = {
    "Alcool_e_substancias": "Álcool e substâncias", "Animais_na_pista": "Animais na pista",
    "Atencao_e_reacao_do_condutor": "Atenção/reação do condutor", "Clima_e_ambiente": "Clima e ambiente",
    "Condicao_da_via": "Condição da via", "Defeito_mecanico_ou_carga": "Defeito mecânico/carga",
    "Manobra_e_infracao": "Manobra e infração", "Pedestre": "Pedestre",
    "Velocidade_e_distancia": "Velocidade e distância",
}  # fmt: skip


#: Deslocamento (pontos) do rótulo de cada causa: as causas pequenas ficam todas
#: no canto inferior esquerdo e os rótulos se sobrepõem sem isso.
_POSICAO_ROTULO_CAUSA: dict[str, tuple[int, int]] = {
    "Animais_na_pista": (-10, 95), "Condicao_da_via": (30, 135), "Alcool_e_substancias": (75, 70),
    "Clima_e_ambiente": (15, -28), "Defeito_mecanico_ou_carga": (12, -4), "Velocidade_e_distancia": (14, 2),
    "Atencao_e_reacao_do_condutor": (-30, -40), "Manobra_e_infracao": (22, 22), "Pedestre": (14, 0),
}  # fmt: skip


def nome_causa(chave: str) -> str:
    return _NOMES_CAUSA.get(chave, chave.replace("_", " "))


def fig_volume_vs_gravidade(causas: pd.DataFrame, taxa_geral: float, pasta: Path) -> Path:
    _estilo()
    t = causas.copy()
    fig, ax = plt.subplots(figsize=(10, 6.2))
    fig.subplots_adjust(top=0.86)
    escala = 1600 / t["n_fatais"].max()
    for r in t.itertuples():
        cor = cfg.LARANJA if r.ic_inf_pct > taxa_geral else (cfg.AZUL if r.ic_sup_pct < taxa_geral else cfg.CINZA)
        ax.scatter(r.pct_dos_acidentes, r.taxa_pct, s=max(30, r.n_fatais * escala), color=cor, alpha=0.75,
                   edgecolor="white", linewidth=2, zorder=3)  # fmt: skip
        dx, dy = _POSICAO_ROTULO_CAUSA.get(r.grupo_causa, (10, 4))
        seta = {"arrowstyle": "-", "color": cfg.CINZA, "lw": 0.8} if abs(dy) > 15 else None
        ax.annotate(f"{nome_causa(r.grupo_causa)}\n{fmt_num(r.pct_dos_fatais)}% das mortes",
                    (r.pct_dos_acidentes, r.taxa_pct), xytext=(dx, dy), textcoords="offset points",
                    fontsize=8.5, color=cfg.TEXTO, va="center", arrowprops=seta)  # fmt: skip
    ax.axhline(taxa_geral, color=cfg.TEXTO_SECUNDARIO, lw=1, ls="--")
    ax.text(22, taxa_geral + 0.5, f"média {fmt_num(taxa_geral)}%", ha="right",
            fontsize=8.5, color=cfg.TEXTO_SECUNDARIO)  # fmt: skip
    ax.set_xlim(0, 40)
    ax.set_ylim(0, max(36, t["taxa_pct"].max() + 3))
    ax.set_xlabel("% de todos os acidentes (frequência da causa)")
    ax.set_ylabel("% dos acidentes da causa que foram fatais (letalidade)")
    letal = t.loc[t["taxa_pct"].idxmax()]
    volume = t.loc[t["n_fatais"].idxmax()]
    _titulos(
        fig,
        f"{nome_causa(letal['grupo_causa'])} é a causa mais letal ({fmt_num(letal['taxa_pct'])}%); "
        f"{nome_causa(volume['grupo_causa']).lower()} é a que mais mata em volume "
        f"({fmt_num(volume['pct_dos_fatais'])}% das mortes)",
        "Cada bolha é um grupo de causa; área proporcional ao nº de acidentes fatais. "
        "Laranja/azul: acima/abaixo da média com IC 95%.",
    )
    _fonte(fig)
    return _salvar(fig, "p2_fig3_volume_vs_gravidade_causas", pasta)


# ---------------------------------------------------------------------------
# Figura 4 — Estratificada: noite x dia dentro de cada tipo de pista
# ---------------------------------------------------------------------------


def fig_estratificada(detalhe: pd.DataFrame, resumo: pd.DataFrame, pasta: Path) -> Path:
    _estilo()
    analise = "Noite/madrugada, controlando o tipo de pista"
    d = detalhe[detalhe["analise"] == analise].copy()
    d["pista"] = d["estrato"].str.split("=").str[1]
    d = d.set_index("pista").loc[["Simples", "Dupla", "Múltipla"]].reset_index()
    r = resumo[resumo["analise"] == analise].iloc[0]

    fig, ax = plt.subplots(figsize=(10, 5.6))
    fig.subplots_adjust(top=0.84)
    x = np.arange(len(d))
    larg = 0.36
    ax.bar(x - larg / 2, d["taxa_referencia_pct"], larg, color=cfg.AZUL, label="Dia (6h–17h)")
    ax.bar(x + larg / 2, d["taxa_expostos_pct"], larg, color=cfg.LARANJA, label="Noite/madrugada (18h–5h)")
    for desloc, col in ((-larg / 2, "referencia"), (larg / 2, "expostos")):
        taxa = d[f"taxa_{col}_pct"].to_numpy()
        n = d[f"n_{col}"].to_numpy()
        ics = np.array([est_ic(t, m) for t, m in zip(taxa, n, strict=True)])
        ax.errorbar(x + desloc, taxa, yerr=[taxa - ics[:, 0], ics[:, 1] - taxa], fmt="none",
                    ecolor=cfg.TEXTO_SECUNDARIO, elinewidth=1, capsize=3)  # fmt: skip
    for i, linha in d.iterrows():
        ax.text(i - larg / 2 - 0.1, linha["taxa_referencia_pct"] + 0.3, f"{fmt_num(linha['taxa_referencia_pct'])}%",
                ha="center", fontsize=9)  # fmt: skip
        ax.text(i + larg / 2 + 0.1, linha["taxa_expostos_pct"] + 0.3, f"{fmt_num(linha['taxa_expostos_pct'])}%",
                ha="center", fontsize=9)  # fmt: skip
        ax.text(i, max(linha["taxa_expostos_pct"], linha["taxa_referencia_pct"]) + 1.6,
                f"RR = {fmt_num(linha['rr'], 2)}", ha="center", fontsize=9.5, fontweight="bold")  # fmt: skip
    ax.set_xticks(x, [f"Pista {p.lower()}" for p in d["pista"]])
    ax.set_ylabel("% de acidentes que foram fatais")
    ax.set_ylim(0, d["taxa_expostos_pct"].max() + 4)
    ax.grid(axis="x", visible=False)
    ax.legend(loc="upper right")
    _titulos(
        fig,
        "A noite é mais letal em todo tipo de pista: não é efeito de confusão",
        f"RR bruto {fmt_num(r['rr_bruto'], 2)} → RR ajustado por tipo de pista (Mantel-Haenszel) "
        f"{fmt_num(r['rr_ajustado_mh'], 2)} [IC 95% {fmt_num(r['rr_mh_ic_inf'], 2)}–{fmt_num(r['rr_mh_ic_sup'], 2)}]"
        f" · CMH p {fmt_p(r['p_valor'])}",
    )
    _fonte(fig)
    return _salvar(fig, "p2_fig4_estratificada_noite_pista", pasta)


# ---------------------------------------------------------------------------
# Figura 5 — Sazonalidade mensal
# ---------------------------------------------------------------------------


def fig_sazonalidade_mensal(mes: pd.DataFrame, v_cramer: float, taxa_geral: float, pasta: Path) -> Path:
    _estilo()
    tot = mes[mes["ano"] == "Total"].sort_values("mes")
    fig, ax = plt.subplots(figsize=(10, 5))
    fig.subplots_adjust(top=0.82)
    x = tot["mes"].astype(int).to_numpy()
    ax.fill_between(x, tot["ic_inf_pct"], tot["ic_sup_pct"], color=cfg.AZUL, alpha=0.18, lw=0)
    ax.plot(x, tot["taxa_pct"], color=cfg.AZUL, lw=2.2, marker="o", ms=5, label="2024–2025 (IC 95%)")
    for ano, estilo in (("2024", ":"), ("2025", "--")):
        a = mes[mes["ano"] == ano].sort_values("mes")
        ax.plot(a["mes"].astype(int), a["taxa_pct"], color=cfg.CINZA, lw=1.1, ls=estilo, label=ano)
    ax.axhline(taxa_geral, color=cfg.TEXTO_SECUNDARIO, lw=1, ls="--")
    ax.set_xticks(range(1, 13), cfg.ABREV_MES)
    ax.set_ylim(0, 12)
    ax.set_ylabel("% de acidentes que foram fatais")
    ax.legend(loc="upper left", ncol=3, fontsize=9)
    pico = tot.loc[tot["taxa_pct"].idxmax()]
    vale = tot.loc[tot["taxa_pct"].idxmin()]
    _titulos(
        fig,
        f"Sazonalidade mensal é fraca (V de Cramér = {fmt_num(v_cramer, 3)}): a taxa oscila só entre "
        f"{fmt_num(vale['taxa_pct'])}% ({cfg.ABREV_MES[int(vale['mes']) - 1]}) e "
        f"{fmt_num(pico['taxa_pct'])}% ({cfg.ABREV_MES[int(pico['mes']) - 1]})",
        "Taxa de fatais por mês com IC 95% (eixo começa em zero de propósito). O 'quando' que importa é a hora.",
    )
    _fonte(fig)
    return _salvar(fig, "p2_fig5_sazonalidade_mensal", pasta)


# ---------------------------------------------------------------------------
# Figura 6 — Mapa de calor dia da semana x faixa horária
# ---------------------------------------------------------------------------


def fig_semana_x_faixa(sxf: pd.DataFrame, pasta: Path) -> Path:
    _estilo()
    taxa = sxf.pivot(index="faixa_horaria", columns="dia_semana", values="taxa_pct")
    fatais = sxf.pivot(index="faixa_horaria", columns="dia_semana", values="n_fatais")
    ordem_f = list(cfg.ORDEM_FAIXA_HORARIA)
    ordem_d = list(cfg.ORDEM_DIA_SEMANA)
    taxa, fatais = taxa.loc[ordem_f, ordem_d], fatais.loc[ordem_f, ordem_d]
    cmap = LinearSegmentedColormap.from_list("p2_seq", list(cfg.RAMPA_SEQUENCIAL))
    fig, ax = plt.subplots(figsize=(10, 4.6))
    fig.subplots_adjust(top=0.80)
    im = ax.imshow(taxa.to_numpy(), cmap=cmap, aspect="auto", vmin=0, vmax=float(np.ceil(taxa.max().max())))
    for i in range(len(ordem_f)):
        for j in range(len(ordem_d)):
            v = taxa.iat[i, j]
            cor = "white" if v > 0.6 * im.get_clim()[1] else cfg.TEXTO
            ax.text(j, i, f"{fmt_num(v)}%\n{fmt_int(fatais.iat[i, j])} fatais", ha="center", va="center",
                    fontsize=8.5, color=cor)  # fmt: skip
    ax.set_xticks(range(len(ordem_d)), [cfg.ABREV_DIA_SEMANA[d] for d in ordem_d])
    rot = {"Madrugada": "Madrugada (0–5h)", "Manha": "Manhã (6–11h)", "Tarde": "Tarde (12–17h)", "Noite": "Noite (18–23h)"}
    ax.set_yticks(range(len(ordem_f)), [rot[f] for f in ordem_f])
    ax.grid(False)
    for lado in ("left", "bottom"):
        ax.spines[lado].set_visible(False)
    barra = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    barra.set_label("% fatais")
    barra.outline.set_visible(False)
    mais_letal = sxf.loc[sxf["taxa_pct"].idxmax()]
    mais_mortes = sxf.loc[sxf["n_fatais"].idxmax()]
    rotulo = {"Madrugada": "madrugada", "Manha": "manhã", "Tarde": "tarde", "Noite": "noite"}
    _titulos(
        fig,
        f"Turno mais letal: {mais_letal['dia_semana']} de {rotulo[mais_letal['faixa_horaria']]} "
        f"({fmt_num(mais_letal['taxa_pct'])}%); com mais mortes: {mais_mortes['dia_semana']} à "
        f"{rotulo[mais_mortes['faixa_horaria']]} ({fmt_int(mais_mortes['n_fatais'])})",
        "Cor = % de acidentes fatais no turno; texto = taxa e nº de acidentes fatais (2024+2025).",
    )
    _fonte(fig)
    return _salvar(fig, "p2_fig6_semana_x_faixa_horaria", pasta)


# ---------------------------------------------------------------------------
# Figura 7 — UF
# ---------------------------------------------------------------------------


def fig_uf(uf: pd.DataFrame, taxa_geral: float, pasta: Path) -> Path:
    _estilo()
    t = uf.sort_values("taxa_pct").reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(10, 7.6))
    fig.subplots_adjust(top=0.89)
    cores = [cfg.LARANJA if u == cfg.UF_DESTAQUE else (cfg.AZUL if lo > taxa_geral else cfg.CINZA)
             for u, lo in zip(t["uf"], t["ic_inf_pct"], strict=True)]  # fmt: skip
    ax.barh(t.index, t["taxa_pct"], color=cores, height=0.72)
    ax.errorbar(t["taxa_pct"], t.index, xerr=[t["taxa_pct"] - t["ic_inf_pct"], t["ic_sup_pct"] - t["taxa_pct"]],
                fmt="none", ecolor=cfg.TEXTO_SECUNDARIO, elinewidth=0.9, capsize=2)  # fmt: skip
    ax.axvline(taxa_geral, color=cfg.TEXTO_SECUNDARIO, lw=1, ls="--")
    ax.set_yticks(t.index, [f"{u}  (n={fmt_int(n)})" for u, n in zip(t["uf"], t["n"], strict=True)], fontsize=8.5)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("% de acidentes que foram fatais (IC 95%)")
    ax.text(taxa_geral + 0.2, len(t) - 0.4, f"média {fmt_num(taxa_geral)}%", fontsize=8.5, color=cfg.TEXTO_SECUNDARIO)
    ranking = t.sort_values("taxa_pct", ascending=False).reset_index(drop=True)
    pos_al = int(ranking.index[ranking["uf"] == cfg.UF_DESTAQUE][0]) + 1
    al = ranking[ranking["uf"] == cfg.UF_DESTAQUE].iloc[0]
    top3 = ", ".join(ranking["uf"].head(3))
    _titulos(
        fig,
        f"{top3} lideram a letalidade; Alagoas é a {pos_al}ª UF ({fmt_num(al['taxa_pct'])}%, "
        f"{fmt_num(al['taxa_pct'] / taxa_geral, 1)}× a média)",
        "Taxa de fatais por UF com IC 95%. Azul: acima da média com IC; laranja: Alagoas.",
    )
    _fonte(fig)
    return _salvar(fig, "p2_fig7_taxa_por_uf", pasta)


# ---------------------------------------------------------------------------
# Figura 8 — Boxplots das numéricas
# ---------------------------------------------------------------------------


def fig_boxplots(df: pd.DataFrame, mw: pd.DataFrame, pasta: Path) -> Path:
    _estilo()
    colunas = ["veiculos", "idade_media_condutores", "idade_media_veiculos"]
    fig, eixos = plt.subplots(1, 3, figsize=(11, 4.6))
    fig.subplots_adjust(top=0.78, wspace=0.35)
    alvo = df[cfg.COLUNA_ALVO]
    for ax, c in zip(eixos, colunas, strict=True):
        dados = [df.loc[alvo == 0, c].dropna(), df.loc[alvo == 1, c].dropna()]
        bp = ax.boxplot(dados, showfliers=False, widths=0.55, patch_artist=True,
                        medianprops={"color": cfg.TEXTO, "lw": 1.8})  # fmt: skip
        for caixa, cor in zip(bp["boxes"], (cfg.AZUL, cfg.LARANJA), strict=True):
            caixa.set_facecolor(cor)
            caixa.set_alpha(0.55)
            caixa.set_edgecolor(cor)
        ax.set_xticks([1, 2], ["Não fatal", "Fatal"])
        r = mw[mw["variavel"] == c].iloc[0]
        ax.set_title(f"{cfg.ROTULOS[c]}\nr = {fmt_num(r['r_rank_bisserial'], 2)} ({r['magnitude']})", fontsize=10)
        ax.grid(axis="x", visible=False)
    _titulos(
        fig,
        "As numéricas separam pouco fatais de não fatais; só o nº de veículos tem efeito pequeno",
        "Caixa = quartis, traço = mediana, sem pontos extremos. r = correlação rank-bisserial (Mann-Whitney).",
    )
    _fonte(fig)
    return _salvar(fig, "p2_fig8_boxplots_numericas", pasta)
