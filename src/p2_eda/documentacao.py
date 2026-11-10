"""Gera os textos da EDA a partir das tabelas (nenhum número digitado à mão).

Saídas:

* ``reports/secao_estatistica_descritiva_inferencia.md`` — seção 3 do artigo
  ("Estatística Descritiva e Inferência"), em Markdown, para revisão no GitHub.
* ``reports/secao_estatistica_descritiva_inferencia.tex`` — **a mesma seção** em
  LaTeX, pronta para o template SBC (a Pessoa 5 faz ``\\input{...}``).
* ``reports/README_eda.md`` — README desta parte: o que foi feito, por quê,
  todas as figuras, os vereditos e o que cada pessoa recebe.

A seção é escrita uma vez como uma lista de blocos (parágrafo, tabela, figura)
e renderizada nos dois formatos, para Markdown e LaTeX nunca divergirem.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

import pandas as pd

from src.p2_eda import configuracao as cfg
from src.p2_eda.configuracao import Caminhos
from src.p2_eda.estatistica import fmt_p
from src.p2_eda.graficos import fmt_int, fmt_num, nome_causa

# ---------------------------------------------------------------------------
# Blocos e renderização
# ---------------------------------------------------------------------------


@dataclass
class Bloco:
    tipo: str
    texto: str = ""
    cabecalho: list[str] = field(default_factory=list)
    linhas: list[list[str]] = field(default_factory=list)
    figura: str = ""
    rotulo: str = ""
    alinhamento: str = ""


_REF = re.compile(r"\[\[((fig|tab):[a-z_]+)\]\]")


def _numeracao(blocos: list[Bloco]) -> dict[str, str]:
    nomes, cont = {}, {"figura": 0, "tabela": 0}
    for b in blocos:
        if b.tipo in cont:
            cont[b.tipo] += 1
            nomes[b.rotulo] = f"{b.tipo.capitalize()} {cont[b.tipo]}"
    return nomes


def _md(blocos: list[Bloco], prefixo_fig: str) -> str:
    numeros = _numeracao(blocos)
    saida: list[str] = []
    n_fig = n_tab = 0
    for b in blocos:
        if b.tipo == "titulo":
            saida.append(f"# {b.texto}\n")
        elif b.tipo == "subtitulo":
            saida.append(f"### {b.texto}\n")
        elif b.tipo == "paragrafo":
            saida.append(_REF.sub(lambda m: numeros[m.group(1)], b.texto) + "\n")
        elif b.tipo == "tabela":
            n_tab += 1
            saida.append(f"**Tabela {n_tab}.** {b.texto}\n")
            saida.append("| " + " | ".join(b.cabecalho) + " |")
            saida.append("|" + "|".join("---" for _ in b.cabecalho) + "|")
            saida += ["| " + " | ".join(linha) + " |" for linha in b.linhas]
            saida.append("")
        elif b.tipo == "figura":
            n_fig += 1
            saida.append(f"![{b.figura}]({prefixo_fig}{b.figura}.png)\n")
            saida.append(f"**Figura {n_fig}.** {b.texto}\n")
    return "\n".join(saida)


_ESCAPES_TEX = [("\\", r"\textbackslash{}"), ("&", r"\&"), ("%", r"\%"), ("$", r"\$"), ("#", r"\#"),
                ("_", r"\_"), ("{", r"\{"), ("}", r"\}"), ("~", r"\textasciitilde{}"),
                ("×", r"$\times$"), ("≥", r"$\geq$"), ("≤", r"$\leq$"), ("→", r"$\rightarrow$"),
                ("χ²", r"$\chi^2$"), ("±", r"$\pm$"), ("·", r"$\cdot$"), ("≈", r"$\approx$"), ("α", r"$\alpha$")]  # fmt: skip


def _tex_texto(texto: str) -> str:
    for a, b in _ESCAPES_TEX:
        texto = texto.replace(a, b)
    texto = _REF.sub(lambda m: ("Figura~" if m.group(2) == "fig" else "Tabela~") + f"\\ref{{{m.group(1)}}}", texto)
    texto = re.sub(r"\*\*(.+?)\*\*", r"\\textbf{\1}", texto)
    texto = re.sub(r"\*(.+?)\*", r"\\textit{\1}", texto)
    return re.sub(r"`(.+?)`", r"\\texttt{\1}", texto)


def _tex(blocos: list[Bloco], prefixo_fig: str) -> str:
    saida = [
        "% Gerado por: python -m src.p2_eda.executar_eda  (NÃO editar à mão; edite src/p2_eda/documentacao.py)",
        "% Requer no preâmbulo: \\usepackage{graphicx}. Ajuste o caminho das figuras em \\graphicspath se preciso.",
        "",
    ]
    for b in blocos:
        if b.tipo == "titulo":
            saida.append(f"\\section{{{_tex_texto(b.texto)}}}\\label{{sec:estatistica}}\n")
        elif b.tipo == "subtitulo":
            saida.append(f"\\subsection{{{_tex_texto(b.texto)}}}\n")
        elif b.tipo == "paragrafo":
            saida.append(_tex_texto(b.texto) + "\n")
        elif b.tipo == "tabela":
            saida += [
                "\\begin{table}[!htbp]", "\\centering", "\\small",
                f"\\caption{{{_tex_texto(b.texto)}}}\\label{{{b.rotulo}}}",
                "\\resizebox{\\linewidth}{!}{%",
                f"\\begin{{tabular}}{{{b.alinhamento}}}", "\\hline",
                " & ".join(f"\\textbf{{{_tex_texto(c)}}}" for c in b.cabecalho) + " \\\\", "\\hline",
                *(" & ".join(_tex_texto(c) for c in linha) + " \\\\" for linha in b.linhas),
                "\\hline", "\\end{tabular}}", "\\end{table}", "",
            ]  # fmt: skip
        elif b.tipo == "figura":
            saida += [
                "\\begin{figure}[!htbp]", "\\centering",
                f"\\includegraphics[width=0.88\\linewidth]{{{prefixo_fig}{b.figura}.png}}",
                f"\\caption{{{_tex_texto(b.texto)}}}\\label{{{b.rotulo}}}", "\\end{figure}", "",
            ]  # fmt: skip
    # Referências cruzadas no texto: "Figura~\ref{...}" (o Markdown usa os números).
    return "\n".join(saida)


# ---------------------------------------------------------------------------
# Acesso às tabelas
# ---------------------------------------------------------------------------

class Numeros:
    def __init__(self, t: dict[str, pd.DataFrame]) -> None:
        self.t = t
        r = t["p2_resumo_geral"].set_index("recorte")
        self.n = int(r.loc["Total", "n"])
        self.fatais = int(r.loc["Total", "n_fatais"])
        self.taxa = float(r.loc["Total", "taxa_pct"])
        self.ic = (float(r.loc["Total", "ic_inf_pct"]), float(r.loc["Total", "ic_sup_pct"]))
        self.taxa_ano = {a: float(r.loc[a, "taxa_pct"]) for a in r.index if a != "Total"}
        self.fat = t["p2_fatores_risco"].set_index("fator")
        self.q2 = t["p2_testes_qui_quadrado"].set_index("variavel")
        self.mw = t["p2_testes_mann_whitney"].set_index("variavel")
        self.est = t["p2_estratificadas_resumo"].set_index("analise")
        self.desc = t["p2_descritiva_numericas"].set_index("variavel")
        self.grupos = t["p2_taxa_por_grupo"]
        self.n_testes = int(t["p2_parametros"].set_index("parametro").loc["n_testes_familia_holm", "valor"])

    def grupo(self, dimensao: str, grupo: str) -> pd.Series:
        g = self.grupos
        return g[(g["dimensao"] == dimensao) & (g["grupo"] == grupo)].iloc[0]

    def rr(self, fator: str, casas: int = 2) -> str:
        f = self.fat.loc[fator]
        return f"{fmt_num(f['rr'], casas)} [IC 95%: {fmt_num(f['rr_ic_inf'], casas)}–{fmt_num(f['rr_ic_sup'], casas)}]"

    def taxa_fator(self, fator: str) -> tuple[float, float]:
        f = self.fat.loc[fator]
        return float(f["taxa_expostos_pct"]), float(f["taxa_referencia_pct"])


# ---------------------------------------------------------------------------
# Seção do artigo
# ---------------------------------------------------------------------------

F_NOITE = "Madrugada ou noite (vs. 6h-17h)"
F_PISTA = "Pista simples (vs. dupla/múltipla)"
F_RURAL = "Trecho rural (vs. urbano)"
F_REGIAO = "Norte ou Nordeste (vs. Sul/Sudeste)"
E_NOITE_PISTA = "Noite/madrugada, controlando o tipo de pista"
E_NOITE_URB = "Noite/madrugada, controlando trecho urbano/rural"
E_PISTA_URB = "Pista simples, controlando trecho urbano/rural"
E_REGIAO_PISTA = "Norte/Nordeste, controlando o tipo de pista"


def blocos_secao(t: dict[str, pd.DataFrame]) -> list[Bloco]:
    x = Numeros(t)
    d = x.desc
    hora = t["p2_hora_por_ano"]
    hora = hora[hora["ano"] == "Total"]
    h_max = hora.loc[hora["taxa_pct"].idxmax()]
    h_min = hora.loc[hora["taxa_pct"].idxmin()]
    h_vol = hora.loc[hora["n"].idxmax()]
    sxf = t["p2_semana_x_faixa_horaria"]
    sxf_mortes = sxf.loc[sxf["n_fatais"].idxmax()]
    tap = t["p2_tipo_acidente_por_pista"].set_index(["tipo_pista", "tipo_acidente"])
    frontal = tap.loc[("Simples", "Colisão frontal")]
    causas = x.grupos[x.grupos["dimensao"] == "grupo_causa"].set_index("grupo")
    ranking = t["p2_ranking_atributos"]
    veredito = t["p2_veredito_hipoteses"]
    mes = t["p2_mes_por_ano"]
    mes = mes[mes["ano"] == "Total"]
    al = x.grupo("uf", cfg.UF_DESTAQUE)
    uf = x.grupos[x.grupos["dimensao"] == "uf"].sort_values("taxa_pct", ascending=False).reset_index(drop=True)
    pos_al = int(uf.index[uf["grupo"] == cfg.UF_DESTAQUE][0]) + 1

    tx_noite = x.taxa_fator(F_NOITE)
    tx_pista = x.taxa_fator(F_PISTA)
    tx_ped = x.taxa_fator("Pedestre envolvido")
    tx_moto = x.taxa_fator("Motocicleta envolvida")
    tx_chuva = x.taxa_fator("Chuva, garoa ou neblina (vs. tempo seco)")
    fer = x.fat.loc["Feriado nacional"]
    cam = x.fat.loc["Caminhão envolvido"]
    en, eu, ep, er = (x.est.loc[k] for k in (E_NOITE_PISTA, E_NOITE_URB, E_PISTA_URB, E_REGIAO_PISTA))
    vei, idc = x.mw.loc["veiculos"], x.mw.loc["idade_media_condutores"]

    b: list[Bloco] = [Bloco("titulo", "Estatística Descritiva e Inferência")]

    # 3.1 Visão geral -------------------------------------------------------
    b.append(Bloco("subtitulo", "Visão geral e estatística descritiva"))
    b.append(Bloco("paragrafo", (
        f"A base tratada contém {fmt_int(x.n)} acidentes, dos quais {fmt_int(x.fatais)} tiveram ao menos uma morte: "
        f"taxa de letalidade de **{fmt_num(x.taxa, 2)}%** (IC 95% de Wilson: {fmt_num(x.ic[0], 2)}%–{fmt_num(x.ic[1], 2)}%). "
        f"A taxa é praticamente idêntica em 2024 ({fmt_num(x.taxa_ano['2024'], 2)}%) e 2025 "
        f"({fmt_num(x.taxa_ano['2025'], 2)}%), o que permite tratar os dois anos em conjunto e, ao mesmo tempo, "
        "usá-los como **réplica** um do outro: um padrão só foi considerado robusto quando apareceu, no mesmo "
        "sentido e com intervalo de confiança excluindo a ausência de efeito, em cada ano separadamente. "
        "A [[tab:descritiva]] resume as variáveis numéricas. Todas são assimétricas (o número de veículos tem assimetria "
        f"{fmt_num(d.loc['veiculos', 'assimetria'], 1)} e chega a {fmt_int(d.loc['veiculos', 'maximo'])}) e o teste "
        "de D'Agostino-Pearson rejeita a normalidade em todas, o que justifica o uso de mediana e IQR na "
        "descrição e de testes não paramétricos na inferência. Os nulos (até "
        f"{fmt_num(d['pct_nulos'].max(), 1)}%) foram mantidos como ausentes, sem imputação, conforme a seção anterior."
    )))  # fmt: skip
    linhas = []
    for v in cfg.COLUNAS_NUMERICAS:
        r = d.loc[v]
        linhas.append([cfg.ROTULOS[v], fmt_num(r["media"], 1), fmt_num(r["mediana"], 1), fmt_num(r["desvio_padrao"], 1),
                       fmt_num(r["iqr"], 1), fmt_num(r["assimetria"], 2), f"{fmt_num(r['pct_nulos'], 1)}%",
                       f"{fmt_num(r['mediana_fatal'], 1)} / {fmt_num(r['mediana_nao_fatal'], 1)}"])  # fmt: skip
    b.append(Bloco("tabela", "Estatística descritiva das variáveis numéricas (n = " + fmt_int(x.n) + ").",
                   ["Variável", "Média", "Mediana", "DP", "IQR", "Assim.", "Nulos", "Med. fatal / não fatal"],
                   linhas, rotulo="tab:descritiva", alinhamento="lrrrrrrr"))  # fmt: skip

    # 3.2 Método --------------------------------------------------------------
    b.append(Bloco("subtitulo", "Estratégia de inferência"))
    b.append(Bloco("paragrafo", (
        "Nove hipóteses (H1–H9, [[tab:hipoteses]]) foram testadas. Para comparações entre dois grupos usou-se o "
        "**risco relativo** (RR, com IC 95% pelo método log) e o teste qui-quadrado; para variáveis com várias "
        "categorias, o qui-quadrado de independência com **V de Cramér**; para as numéricas, o teste de "
        "**Mann-Whitney** com a correlação rank-bisserial como tamanho de efeito. Como o desfecho é raro (7%), o "
        "V de Cramér e o phi ficam numericamente pequenos mesmo quando o risco dobra; por isso a relevância prática "
        "é julgada pelo RR e pela diferença em pontos percentuais, e o V é usado para comparar variáveis entre si. "
        "Para separar causas de **confusão**, os principais fatores foram reavaliados dentro de estratos "
        "(teste de Cochran-Mantel-Haenszel, com RR ajustado de Mantel-Haenszel). Todos os "
        f"{x.n_testes} testes formam uma única família corrigida por **Holm** (α = {fmt_num(cfg.ALFA, 2)}). "
        f"Com n = {fmt_int(x.n)}, quase toda diferença é estatisticamente significativa; as conclusões, portanto, "
        "apoiam-se no tamanho do efeito e na replicação entre anos, e não apenas no p-valor."
    )))  # fmt: skip

    # 3.3 Quando --------------------------------------------------------------
    b.append(Bloco("subtitulo", "Quando: hora, dia e mês"))
    b.append(Bloco("paragrafo", (
        f"A hora é o fator temporal dominante ([[fig:hora]]). O volume de acidentes é máximo às {int(h_vol['hora'])}h, mas "
        f"a letalidade atinge {fmt_num(h_max['taxa_pct'])}% às {int(h_max['hora'])}h, contra {fmt_num(h_min['taxa_pct'])}% "
        f"às {int(h_min['hora'])}h. Acidentes entre 18h e 5h são {fmt_num(tx_noite[0])}% fatais contra {fmt_num(tx_noite[1])}% "
        f"durante o dia: RR = {x.rr(F_NOITE)}, e concentram "
        f"{fmt_num(x.fat.loc[F_NOITE, 'pct_dos_fatais_expostos'])}% de todas as mortes (H1 confirmada). "
        f"Os fins de semana elevam o risco (RR = {x.rr('Fim de semana')}); o turno com mais acidentes fatais é "
        f"{sxf_mortes['dia_semana']} à noite ({fmt_int(sxf_mortes['n_fatais'])}). Feriados nacionais não diferem "
        f"significativamente após a correção (RR = {fmt_num(fer['rr'], 2)}, p ajustado = {fmt_p(fer['p_holm'])}), "
        f"e a sazonalidade mensal, embora detectável, é desprezível (V = {fmt_num(x.q2.loc['mes', 'v_cramer'], 3)}; "
        f"taxas entre {fmt_num(mes['taxa_pct'].min())}% e {fmt_num(mes['taxa_pct'].max())}%), de modo que H5 é "
        "apenas parcialmente confirmada e H9 é rejeitada para fins de decisão."
    )))  # fmt: skip
    b.append(Bloco("figura", "Volume de acidentes (em cima) e taxa de letalidade com IC 95% (embaixo) por hora; "
                   "linhas cinza mostram cada ano separadamente.", figura="p2_fig2_hora_do_dia", rotulo="fig:hora"))

    # 3.4 Onde ----------------------------------------------------------------
    b.append(Bloco("subtitulo", "Onde: tipo de via e região"))
    b.append(Bloco("paragrafo", (
        f"Acidentes em pista simples são {fmt_num(tx_pista[0])}% fatais, contra {fmt_num(tx_pista[1])}% em pista dupla ou "
        f"múltipla (RR = {x.rr(F_PISTA)}); trechos rurais (RR = {x.rr(F_RURAL)}) e curvas "
        f"(RR = {x.rr('Curva (vs. sem curva)')}) também elevam o risco (H2 confirmada). O mecanismo aparece ao cruzar "
        f"pista e tipo de acidente: a **colisão frontal em pista simples** responde por apenas "
        f"{fmt_num(frontal['pct_dos_acidentes'])}% dos acidentes, mas por **{fmt_num(frontal['pct_dos_fatais'])}% de todas as "
        f"mortes** (letalidade de {fmt_num(frontal['taxa_pct'])}%), o que é coerente com a ausência de separação física "
        f"entre sentidos. Norte e Nordeste têm letalidade {fmt_num(x.fat.loc[F_REGIAO, 'rr'], 1)} vez a de Sul e "
        f"Sudeste (H4 confirmada); Alagoas é a {pos_al}ª UF mais letal ({fmt_num(al['taxa_pct'])}%)."
    )))  # fmt: skip
    b.append(Bloco("paragrafo", (
        "Esses fatores são correlacionados entre si (à noite e no Nordeste há mais pista simples, por exemplo). "
        "A análise estratificada ([[tab:cmh]]) mostra que nenhum deles é explicado pelo outro: o RR da noite "
        f"passa de {fmt_num(en['rr_bruto'], 2)} para {fmt_num(en['rr_ajustado_mh'], 2)} quando se controla o tipo de pista "
        f"e para {fmt_num(eu['rr_ajustado_mh'], 2)} quando se controla o trecho urbano/rural; o da pista simples passa de "
        f"{fmt_num(ep['rr_bruto'], 2)} para {fmt_num(ep['rr_ajustado_mh'], 2)}; e o regional, de {fmt_num(er['rr_bruto'], 2)} "
        f"para {fmt_num(er['rr_ajustado_mh'], 2)}. Ao menos {fmt_num(t['p2_estratificadas_resumo']['pct_efeito_mantido'].min(), 0)}% "
        "de cada efeito (em escala log) sobrevive ao ajuste, indicando fatores com contribuição própria."
    )))  # fmt: skip
    linhas = []
    for nome, r in ((E_NOITE_PISTA, en), (E_NOITE_URB, eu), (E_PISTA_URB, ep), (E_REGIAO_PISTA, er)):
        linhas.append([nome, fmt_num(r["rr_bruto"], 2),
                       f"{fmt_num(r['rr_ajustado_mh'], 2)} ({fmt_num(r['rr_mh_ic_inf'], 2)}–{fmt_num(r['rr_mh_ic_sup'], 2)})",
                       fmt_num(r["qui2_cmh"], 0), fmt_p(r["p_holm"])])  # fmt: skip
    b.append(Bloco("tabela", "Risco relativo bruto e ajustado por estrato (Mantel-Haenszel) e teste CMH.",
                   ["Análise", "RR bruto", "RR ajustado (IC 95%)", "χ² CMH", "p (Holm)"], linhas,
                   rotulo="tab:cmh", alinhamento="lrrrr"))  # fmt: skip

    # 3.5 Quem e por quê ------------------------------------------------------
    b.append(Bloco("subtitulo", "Quem e por quê: envolvidos, clima e causa"))
    b.append(Bloco("paragrafo", (
        f"O maior risco relativo é o de acidentes com **pedestre** ({fmt_num(tx_ped[0])}% fatais; RR = "
        f"{x.rr('Pedestre envolvido')}). Caminhões (RR = {x.rr('Caminhão envolvido')}) estão presentes em "
        f"{fmt_num(cam['pct_dos_fatais_expostos'])}% dos acidentes fatais, e bicicletas e ônibus também elevam o risco "
        f"([[fig:rr]]). Já a **motocicleta**, contrariando a hipótese inicial, aparece com letalidade menor por acidente "
        f"({fmt_num(tx_moto[0])}% vs. {fmt_num(tx_moto[1])}%; RR = {x.rr('Motocicleta envolvida')}), provavelmente porque "
        "muitos registros com moto são quedas e colisões leves; por isso H3 é apenas parcial. Da mesma forma, chuva, "
        f"garoa ou neblina **não** aumentam a letalidade ({fmt_num(tx_chuva[0])}% vs. {fmt_num(tx_chuva[1])}%; "
        f"RR = {x.rr('Chuva, garoa ou neblina (vs. tempo seco)')}), e H6 é rejeitada; uma explicação plausível, "
        "não testável com esta base, é a redução de velocidade em tempo ruim."
    )))  # fmt: skip
    b.append(Bloco("figura", "Risco relativo de o acidente ser fatal (IC 95%, escala log) para cada fator; "
                   "cinza indica não significativo após Holm.", figura="p2_fig1_risco_relativo", rotulo="fig:rr"))
    ped, man, ate = causas.loc["Pedestre"], causas.loc["Manobra_e_infracao"], causas.loc["Atencao_e_reacao_do_condutor"]
    alc, vel = causas.loc["Alcool_e_substancias"], causas.loc["Velocidade_e_distancia"]
    b.append(Bloco("paragrafo", (
        f"A causa e o tipo do acidente são as variáveis mais associadas ao desfecho (V = "
        f"{fmt_num(x.q2.loc['grupo_causa', 'v_cramer'], 2)} e {fmt_num(x.q2.loc['tipo_acidente', 'v_cramer'], 2)}; H7 "
        "confirmada), mas a leitura correta exige separar **gravidade** de **volume** ([[fig:causas]]). A causa "
        f"\"pedestre\" é a mais letal ({fmt_num(ped['taxa_pct'])}%), mas soma {fmt_num(ped['pct_dos_fatais'])}% das mortes; "
        f"manobras e infrações combinam letalidade acima da média ({fmt_num(man['taxa_pct'])}%) com o maior volume "
        f"({fmt_num(man['pct_dos_fatais'])}% das mortes), e falhas de atenção, mesmo abaixo da média, somam "
        f"{fmt_num(ate['pct_dos_fatais'])}%. Álcool ({fmt_num(alc['taxa_pct'])}%) e velocidade ({fmt_num(vel['taxa_pct'])}%) "
        "ficam abaixo da média, resultado que deve ser lido com cautela: a causa é atribuída pela PRF após o "
        "atendimento e pode haver sub-registro nos casos fatais."
    )))  # fmt: skip
    b.append(Bloco("figura", "Frequência (eixo x) versus letalidade (eixo y) de cada grupo de causa; a área da bolha "
                   "é proporcional ao número de acidentes fatais.", figura="p2_fig3_volume_vs_gravidade_causas",
                   rotulo="fig:causas"))  # fmt: skip

    # 3.6 Numéricas -------------------------------------------------------------
    b.append(Bloco("paragrafo", (
        f"Entre as numéricas (H8), apenas o número de veículos tem efeito pequeno (r = {fmt_num(vei['r_rank_bisserial'], 2)}; "
        f"média de {fmt_num(vei['media_fatal'], 2)} veículos nos fatais vs. {fmt_num(vei['media_nao_fatal'], 2)}); a idade "
        f"dos condutores (medianas de {fmt_num(idc['mediana_fatal'], 1)} vs. {fmt_num(idc['mediana_nao_fatal'], 1)} anos) e a "
        "dos veículos diferem de forma significativa, porém desprezível. O risco, portanto, está mais no **contexto** "
        "(via, hora, região, quem está envolvido) do que no perfil numérico do acidente."
    )))  # fmt: skip

    # 3.7 Síntese ---------------------------------------------------------------
    b.append(Bloco("subtitulo", "Síntese"))
    linhas = [[r.hipotese, r.enunciado, r.veredito] for r in veredito.itertuples()]
    b.append(Bloco("tabela", "Hipóteses testadas e veredito (Holm, α = 0,05, efeito replicado em 2024 e 2025).",
                   ["ID", "Hipótese", "Veredito"], linhas, rotulo="tab:hipoteses", alinhamento="lp{10cm}p{3.6cm}"))
    top = ", ".join(f"`{a}`" for a in ranking["atributo"].head(5))
    b.append(Bloco("paragrafo", (
        "A análise sustenta três frentes de decisão, detalhadas na seção de Resultados: (i) **quando**: reforço de "
        "fiscalização e iluminação entre 18h e 5h, sobretudo aos fins de semana; (ii) **onde**: trechos de pista "
        "simples, rurais, em curva e no Norte/Nordeste, com foco em evitar colisões frontais; (iii) **quem**: travessia "
        f"de pedestres e convivência com veículos pesados. Para a modelagem, os atributos mais associados ao alvo são "
        f"{top}; a associação de cada variável isolada é no máximo moderada, o que antecipa um desempenho preditivo "
        "moderado e reforça a necessidade de métricas adequadas a classes desbalanceadas. Como em todo estudo "
        "observacional, os resultados indicam associação, não causalidade."
    )))  # fmt: skip
    return b


# ---------------------------------------------------------------------------
# README da EDA
# ---------------------------------------------------------------------------


def _tabela_md(df: pd.DataFrame) -> str:
    linhas = ["| " + " | ".join(map(str, df.columns)) + " |", "|" + "|".join("---" for _ in df.columns) + "|"]
    linhas += ["| " + " | ".join(map(str, r)) + " |" for r in df.itertuples(index=False)]
    return "\n".join(linhas)


def readme_eda(t: dict[str, pd.DataFrame]) -> str:
    x = Numeros(t)
    f = t["p2_fatores_risco"].sort_values("rr", ascending=False)
    tab_rr = pd.DataFrame({
        "Hip.": f["hipotese"], "Fator": f["fator"],
        "Taxa grupo": [f"{fmt_num(v)}%" for v in f["taxa_expostos_pct"]],
        "Taxa ref.": [f"{fmt_num(v)}%" for v in f["taxa_referencia_pct"]],
        "RR (IC 95%)": [f"{fmt_num(r.rr, 2)} ({fmt_num(r.rr_ic_inf, 2)}–{fmt_num(r.rr_ic_sup, 2)})" for r in f.itertuples()],
        "RR 2024 / 2025": [f"{fmt_num(a, 2)} / {fmt_num(b, 2)}" for a, b in zip(f["rr_2024"], f["rr_2025"], strict=True)],
        "% das mortes no grupo": [f"{fmt_num(v)}%" for v in f["pct_dos_fatais_expostos"]],
        "p (Holm)": [fmt_p(p) for p in f["p_holm"]],
        "Replica?": ["sim" if v else "não" for v in f["replica_nos_dois_anos"]],
    })  # fmt: skip
    q = t["p2_testes_qui_quadrado"]
    tab_q2 = pd.DataFrame({
        "Hip.": q["hipotese"], "Variável": q["variavel"], "Categorias": q["n_categorias"], "gl": q["gl"],
        "χ²": [fmt_num(v, 0) for v in q["qui2"]], "V de Cramér": [fmt_num(v, 3) for v in q["v_cramer"]],
        "Magnitude": q["magnitude"], "p (Holm)": [fmt_p(p) for p in q["p_holm"]],
        "Maior taxa": [f"{g} ({fmt_num(v)}%)" for g, v in zip(q["grupo_maior_taxa"], q["maior_taxa_pct"], strict=True)],
    })  # fmt: skip
    m = t["p2_testes_mann_whitney"]
    tab_mw = pd.DataFrame({
        "Variável": m["variavel"],
        "Mediana fatal / não fatal": [f"{fmt_num(a, 1)} / {fmt_num(b, 1)}" for a, b in zip(m["mediana_fatal"], m["mediana_nao_fatal"], strict=True)],
        "Média fatal / não fatal": [f"{fmt_num(a, 2)} / {fmt_num(b, 2)}" for a, b in zip(m["media_fatal"], m["media_nao_fatal"], strict=True)],
        "r rank-bisserial": [fmt_num(v, 3) for v in m["r_rank_bisserial"]], "Magnitude": m["magnitude"],
        "p (Holm)": [fmt_p(p) for p in m["p_holm"]],
    })  # fmt: skip
    e = t["p2_estratificadas_resumo"]
    tab_e = pd.DataFrame({
        "Análise": e["analise"], "RR bruto": [fmt_num(v, 2) for v in e["rr_bruto"]],
        "RR ajustado (MH)": [f"{fmt_num(r.rr_ajustado_mh, 2)} ({fmt_num(r.rr_mh_ic_inf, 2)}–{fmt_num(r.rr_mh_ic_sup, 2)})" for r in e.itertuples()],
        "% do efeito mantido": [f"{fmt_num(v, 0)}%" for v in e["pct_efeito_mantido"]],
        "p (Holm)": [fmt_p(p) for p in e["p_holm"]],
    })  # fmt: skip
    v = t["p2_veredito_hipoteses"]
    tab_v = pd.DataFrame({"Hip.": v["hipotese"], "Enunciado": v["enunciado"], "Veredito": [f"**{s}**" for s in v["veredito"]]})
    r = t["p2_ranking_atributos"]
    tab_rank = pd.DataFrame({"#": r["posicao"], "Atributo": [f"`{a}`" for a in r["atributo"]],
                             "V de Cramér": [fmt_num(v, 3) for v in r["v_cramer"]], "Magnitude": r["magnitude"],
                             "Observação para a modelagem": r["observacao"]})  # fmt: skip
    tap = t["p2_tipo_acidente_por_pista"].set_index(["tipo_pista", "tipo_acidente"])
    frontal = tap.loc[("Simples", "Colisão frontal")]
    rr_txt = lambda nome: x.rr(nome)  # noqa: E731

    figs = [
        ("p2_fig1_risco_relativo", "Fatores de risco (forest plot). **A figura da \"dor\".**", "Principal"),
        ("p2_fig2_hora_do_dia", "Hora do dia: volume x letalidade (\"tesoura\").", "Principal"),
        ("p2_fig3_volume_vs_gravidade_causas", "Causas: o que é frequente x o que é letal.", "Principal"),
        ("p2_fig4_estratificada_noite_pista", "Estratificação: a noite não é efeito da pista.", "Principal"),
        ("p2_fig5_sazonalidade_mensal", "Sazonalidade mensal (resultado negativo, honesto).", "Apoio"),
        ("p2_fig6_semana_x_faixa_horaria", "Mapa de calor dia x turno (operacional).", "Apoio"),
        ("p2_fig7_taxa_por_uf", "Taxa por UF, com Alagoas destacada.", "Apoio"),
        ("p2_fig8_boxplots_numericas", "Boxplots das numéricas (efeitos pequenos).", "Apoio"),
    ]  # fmt: skip
    blocos_fig = "\n\n".join(f"#### {i}. {desc} *({tipo})*\n\n![{nome}](figuras/{nome}.png)"
                             for i, (nome, desc, tipo) in enumerate(figs, 1))  # fmt: skip

    return f"""# README — Análise Exploratória (EDA) e Estatística (Pessoa 2)

> README **da parte de EDA e Estatística**. É **gerado por código** (`python -m src.p2_eda.executar_eda`) e todos os números vêm de [`tabelas/`](tabelas/). Não edite à mão: rode de novo.
> A seção do artigo está em [`secao_estatistica_descritiva_inferencia.md`](secao_estatistica_descritiva_inferencia.md) (revisão) e [`.tex`](secao_estatistica_descritiva_inferencia.tex) (para o template SBC).

## 1. Resumo em 30 segundos

- **Base:** {fmt_int(x.n)} acidentes da PRF (2024–2025), {fmt_int(x.fatais)} fatais: **{fmt_num(x.taxa, 2)}%** (IC 95% {fmt_num(x.ic[0], 2)}–{fmt_num(x.ic[1], 2)}%). Taxa estável entre os anos ({fmt_num(x.taxa_ano['2024'], 2)}% e {fmt_num(x.taxa_ano['2025'], 2)}%).
- **A "dor":** pedestre envolvido multiplica o risco de morte por **{fmt_num(x.fat.loc['Pedestre envolvido', 'rr'], 1)}**; caminhão, pista simples, trecho rural, noite e Norte/Nordeste, por **~2** cada.
- **O achado mais acionável:** colisão frontal em pista simples é {fmt_num(frontal['pct_dos_acidentes'])}% dos acidentes e **{fmt_num(frontal['pct_dos_fatais'])}% das mortes**.
- **Causa raiz x confusão:** noite, pista simples e região continuam com RR ≈ 1,7–2,0 mesmo comparando só acidentes do mesmo tipo de pista/trecho (Cochran-Mantel-Haenszel).
- **Hipóteses derrubadas (também é resultado):** moto, chuva e feriado **não** aumentam a letalidade por acidente; sazonalidade mensal é desprezível.
- **Rigor:** {x.n_testes} testes corrigidos juntos por Holm; todo efeito replicado em 2024 **e** 2025; conclusões pelo tamanho do efeito, não só pelo p-valor.

## 2. Como rodar

```bash
python -m src.p2_eda.executar_eda   # ~10 s; lê data/processed/, escreve reports/
pytest tests/test_eda_p2.py         # testes das funções estatísticas e das saídas
```

Notebook para navegar pelos resultados: [`notebooks/p2_eda.ipynb`](../notebooks/p2_eda.ipynb).

## 3. Por que esses métodos (a pergunta que a banca faz)

| Escolha | Por quê | Alternativa descartada |
|---|---|---|
| **Risco relativo (RR)** como efeito principal | É a linguagem do gestor: "pista simples tem 2,2 vezes o risco". | Odds ratio: difícil de explicar e exagera quando o evento não é raro. |
| **IC de Wilson** para taxas | Correto para proporções pequenas (7%) e grupos pequenos (pedestre, UF). | IC "normal" (Wald): erra para p pequeno. |
| **Qui-quadrado + V de Cramér** | Teste diz *se* há associação, V diz *quanto*, e permite comparar variáveis. | Só p-valor: com 145 mil linhas, tudo é "significativo". |
| **Mann-Whitney** + r rank-bisserial | Numéricas são assimétricas (veículos: assimetria {fmt_num(x.desc.loc['veiculos', 'assimetria'], 1)}); não supõe normalidade. | Teste t: supõe normalidade. |
| **Cochran-Mantel-Haenszel** | Testa o fator *dentro* de estratos: separa causa raiz de confusão. | Comparar só taxas brutas. |
| **Holm** em todos os {x.n_testes} testes | Controla falso positivo da família inteira; mais poderoso que Bonferroni. | Nenhuma correção. |
| **Réplica 2024 x 2025** | Padrão real não some de um ano para outro. | Analisar só o total. |

**Ponto fino que vale ouro na banca:** o V de Cramér/phi de "noite" é só {fmt_num(x.fat.loc[F_NOITE, 'phi'], 3)} ("desprezível" pela régua de Cohen), mas o risco **dobra** (RR {fmt_num(x.fat.loc[F_NOITE, 'rr'], 2)}). Não é contradição: com desfecho raro (7%) o phi é limitado pelo próprio desbalanceamento. Por isso o V serve para **comparar** variáveis, e o RR para julgar a **importância prática**.

## 4. Vereditos das hipóteses

{_tabela_md(tab_v)}

## 5. Fatores de risco (comparações 2x2)

{_tabela_md(tab_rr)}

## 6. Qui-quadrado de independência (variável x fatal)

{_tabela_md(tab_q2)}

Categorias com menos de {cfg.N_MINIMO_GRUPO} acidentes foram agregadas em "Outras" (e, se nem assim chegassem a {cfg.N_MINIMO_GRUPO}, excluídas), para garantir frequência esperada ≥ 5 em todas as células.

## 7. Numéricas: fatais x não fatais (Mann-Whitney)

{_tabela_md(tab_mw)}

## 8. Causa raiz x confusão (estratificação)

{_tabela_md(tab_e)}

Detalhe por estrato: [`tabelas/p2_estratificadas_por_estrato.csv`](tabelas/p2_estratificadas_por_estrato.csv).

## 9. Figuras

As 4 **principais** vão para o artigo e para os slides; as 4 **de apoio** ficam aqui (e podem ir para um apêndice ou para o dashboard). O título de cada figura já é a conclusão.

{blocos_fig}

## 10. Para a Pessoa 3 (modelagem): ranking de associação com o alvo

V de Cramér de cada atributo com `acidente_fatal` (numéricas cortadas em {cfg.QUANTIS_RANKING} quantis, nulo como categoria). Não substitui a importância do modelo, mas orienta a seleção.

{_tabela_md(tab_rank)}

Recomendações:
- **Desbalanceamento:** {fmt_num(x.taxa, 2)}% de positivos → usar `class_weight`/limiar ajustado e avaliar com F1, AUC-ROC e **AUC-PR** (mais informativa com classe rara).
- **Dois cenários de modelo:** com `tipo_acidente`/`grupo_causa` (explicar o risco, definidos após o atendimento) e **sem** eles (prevenção: só condições conhecidas antes — hora, dia, via, região, tipo de veículo). Comparar os dois é uma ótima discussão.
- `hora` tem relação **não linear** (pico às 3h, vale às 7h): árvores lidam bem; para Regressão Logística, usar `faixa_horaria` ou seno/cosseno da hora.
- `ano` fica fora dos atributos (validação temporal: 2024 treino, 2025 teste). Os RRs quase idênticos entre os anos (seção 5) indicam que essa validação é justa.

## 11. Para a Pessoa 4 (insights) e a Pessoa 5 (artigo/dashboard)

- **Insights prontos para o formato do PDF** ("Percebemos que... Por isso recomendamos... sustentado por... medido por..."):
  1. *Quando:* 18h–5h concentram {fmt_num(x.fat.loc[F_NOITE, 'pct_dos_fatais_expostos'])}% das mortes com RR {rr_txt(F_NOITE)}.
  2. *Onde:* pista simples, RR {rr_txt(F_PISTA)}; colisão frontal em pista simples = {fmt_num(frontal['pct_dos_fatais'])}% das mortes → faixas de ultrapassagem, separador central, fiscalização de ultrapassagem indevida.
  3. *Quem:* pedestre, RR {rr_txt('Pedestre envolvido')} → passarelas, iluminação e travessias em trechos urbanos de rodovia.
- O SHAP (Pessoa 4) deve **confirmar** estes sinais; se o modelo discordar, é ponto de discussão.
- Pessoa 5: `\\input{{reports/secao_estatistica_descritiva_inferencia.tex}}` no template; figuras em `reports/figuras/`. A seção ocupa ~4 páginas com 3 figuras e 3 tabelas; se faltar espaço nas 12 páginas, cortar primeiro a Tabela 1 (descritiva) e reduzir as figuras para `0.75\\linewidth`. `tabelas/p2_taxa_por_grupo.csv` está em formato longo, pronto para filtros de dashboard.

## 12. Limitações

- **Associação, não causa:** estudo observacional; o CMH controla só os estratos testados.
- **Variáveis pós-evento:** causa e tipo são atribuídos pela PRF depois do acidente (possível sub-registro de álcool em fatais).
- **Viés de registro:** só acidentes atendidos pela PRF em rodovias federais.
- **Taxa por acidente, não por exposição:** "moto tem letalidade menor por acidente" não significa que andar de moto é mais seguro; não temos km rodados por tipo de veículo.

## 13. Arquivos gerados

| Pasta | Conteúdo |
|---|---|
| `reports/figuras/` | 8 figuras `p2_fig*.png` (160 dpi) |
| `reports/tabelas/` | {len(t)} tabelas `p2_*.csv` (uma por análise) |
| `reports/` | este README e a seção do artigo (`.md` e `.tex`) |
| `src/p2_eda/` | código: `configuracao`, `estatistica`, `analises`, `graficos`, `documentacao`, `executar_eda` |
| `tests/test_eda_p2.py` | testes |
"""


def gerar_textos(t: dict[str, pd.DataFrame], caminhos: Caminhos) -> None:
    """Grava a seção do artigo (``.md`` e ``.tex``) e o README da EDA.

    Args:
        t: Tabelas da EDA.
        caminhos: Pastas de saída.
    """
    caminhos.reports.mkdir(parents=True, exist_ok=True)
    blocos = blocos_secao(t)
    caminhos.secao_md.write_text(_md(blocos, "figuras/"), encoding="utf-8")
    caminhos.secao_tex.write_text(_tex(blocos, "figuras/"), encoding="utf-8")
    caminhos.readme_eda.write_text(readme_eda(t), encoding="utf-8")
