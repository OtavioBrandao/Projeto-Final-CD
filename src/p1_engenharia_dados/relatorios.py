"""Tabelas e figuras de auditoria (``docs/tabelas`` e ``docs/figuras``).

Tudo que aparece na documentação nasce aqui: **nenhum número é digitado à mão**.
As tabelas são gravadas em CSV e as figuras em PNG (dpi 160, rótulos em
português). O título de cada figura já diz a conclusão que o leitor deve tirar,
e o texto do título também é montado com números calculados.

Convenção visual (uma só em todas as figuras): azul = dado/resultado normal,
laranja = problema, remoção ou atenção, cinza = apoio.
"""

from __future__ import annotations

import logging
import textwrap
from collections.abc import Mapping
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # backend sem janela: funciona em servidor e nos testes
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402

from src.p1_engenharia_dados import configuracao as cfg  # noqa: E402

logger = logging.getLogger(__name__)

# Paleta (azul/laranja/cinza do conjunto de referência de visualização).
AZUL = "#2a78d6"
LARANJA = "#eb6834"
CINZA = "#8a8985"
CINZA_CLARO = "#e5e4e0"
TEXTO = "#0b0b0b"
TEXTO_SECUNDARIO = "#52514e"

DPI_FIGURAS = 160


# ---------------------------------------------------------------------------
# Formatação pt-BR (usada também por ``documentacao.py``)
# ---------------------------------------------------------------------------


def fmt_int(valor: float) -> str:
    """Formata inteiro no padrão brasileiro (``101766`` -> ``101.766``).

    Args:
        valor: Número a formatar (é arredondado para inteiro).

    Returns:
        Texto com ponto como separador de milhar.
    """
    return f"{int(round(float(valor))):,}".replace(",", ".")


def fmt_num(valor: float, casas: int = 2) -> str:
    """Formata decimal no padrão brasileiro (``2.5`` -> ``2,50``).

    Args:
        valor: Número a formatar.
        casas: Casas decimais.

    Returns:
        Texto com vírgula decimal.
    """
    return f"{float(valor):.{casas}f}".replace(".", ",")


def fmt_pct(valor: float, casas: int = 2) -> str:
    """Formata percentual (``8.98`` -> ``8,98%``).

    Args:
        valor: Valor **já em porcentagem** (0 a 100).
        casas: Casas decimais.

    Returns:
        Texto com vírgula decimal e o símbolo ``%``.
    """
    return f"{fmt_num(valor, casas)}%"


# ---------------------------------------------------------------------------
# Tabelas
# ---------------------------------------------------------------------------


def salvar_tabela(df: pd.DataFrame, nome: str, pasta: Path) -> Path:
    """Grava uma tabela de auditoria em CSV (UTF-8, quebra de linha ``\\n``).

    Args:
        df: Tabela a gravar.
        nome: Nome do arquivo sem extensão.
        pasta: Pasta de destino (criada se não existir).

    Returns:
        Caminho do arquivo gravado.
    """
    pasta.mkdir(parents=True, exist_ok=True)
    caminho = pasta / f"{nome}.csv"
    df.to_csv(caminho, index=False, encoding="utf-8", lineterminator="\n")
    return caminho


def tabela_ausentes(auditoria: pd.DataFrame, final: pd.DataFrame) -> pd.DataFrame:
    """Compara ausentes **antes** (bruto) e **depois** (base tratada) por coluna.

    Args:
        auditoria: Saída de ``auditoria.perfil_qualidade`` (base bruta).
        final: Base tratada, com todas as colunas do interim.

    Returns:
        Uma linha para cada coluna que tinha nulo ou texto ``None``, com a
        decisão aplicada (D05–D09).
    """
    decisoes = {
        "weight": ("D05", "coluna removida"),
        "payer_code": ("D06", "nulo -> 'Desconhecido' + flag payer_code_ausente"),
        "medical_specialty": ("D06", "nulo -> 'Desconhecido' + flag medical_specialty_ausente"),
        "race": ("D07", "nulo -> 'Desconhecido'"),
        "diag_1": ("D08", "nulo -> 'Sem_diagnostico'"),
        "diag_2": ("D08", "nulo -> 'Sem_diagnostico'"),
        "diag_3": ("D08", "nulo -> 'Sem_diagnostico'"),
        "max_glu_serum": ("D09", "'None' -> 'Nao_medido' (não era nulo)"),
        "A1Cresult": ("D09", "'None' -> 'Nao_medido' (não era nulo)"),
    }
    linhas = []
    problema = auditoria[(auditoria["n_nulos"] > 0) | (auditoria["n_textos_none"] > 0)]
    for r in problema.itertuples():
        id_decisao, decisao = decisoes.get(r.coluna, ("", ""))
        existe = r.coluna in final.columns
        linhas.append(
            {
                "coluna": r.coluna,
                "id_decisao": id_decisao,
                "n_nulos_antes": r.n_nulos,
                "pct_nulos_antes": r.pct_nulos,
                "n_textos_none_antes": r.n_textos_none,
                "decisao": decisao,
                "n_nulos_depois": int(final[r.coluna].isna().sum()) if existe else "removida",
            }
        )
    return pd.DataFrame(linhas).sort_values("pct_nulos_antes", ascending=False, kind="stable")


def tabela_cardinalidade(antes: pd.DataFrame, depois: pd.DataFrame) -> pd.DataFrame:
    """Mostra quantos valores distintos cada coluna tinha antes e depois do agrupamento.

    Args:
        antes: Base antes do agrupamento (com ``diag_1``, ``medical_specialty``...).
        depois: Base depois (com ``grupo_diag_1``, ``especialidade_agrupada``...).

    Returns:
        Uma linha por par ``coluna original -> coluna agrupada``.
    """
    pares = [
        ("diag_1", "grupo_diag_1"),
        ("diag_2", "grupo_diag_2"),
        ("diag_3", "grupo_diag_3"),
        ("medical_specialty", "especialidade_agrupada"),
        ("destino_alta", "grupo_destino_alta"),
    ]
    return pd.DataFrame(
        [
            {
                "coluna_antes": a,
                "n_distintos_antes": int(antes[a].nunique(dropna=True)),
                "coluna_depois": d,
                "n_distintos_depois": int(depois[d].nunique(dropna=True)),
            }
            for a, d in pares
        ]
    )


def tabela_balanceamento(readmitted_bruto: pd.Series, alvo_final: pd.Series) -> pd.DataFrame:
    """Compara o alvo original (3 classes) com o alvo binário final.

    Args:
        readmitted_bruto: Coluna ``readmitted`` da base bruta já padronizada (D04).
        alvo_final: Coluna ``readmit_30d`` da base final.

    Returns:
        Tabela com ``momento``, ``classe``, ``n`` e ``pct``.
    """
    antes = readmitted_bruto.value_counts()
    depois = alvo_final.value_counts()
    linhas = [
        {"momento": "antes (bruto)", "classe": str(k), "n": int(v), "pct": round(100 * v / antes.sum(), 2)}
        for k, v in antes.items()
    ]
    linhas += [
        {"momento": "depois (final)", "classe": f"readmit_30d={k}", "n": int(v), "pct": round(100 * v / depois.sum(), 2)}
        for k, v in depois.sort_index().items()
    ]  # fmt: skip
    return pd.DataFrame(linhas)


def tabela_limiar_usuario(df: pd.DataFrame) -> pd.DataFrame:
    """Taxa de readmissão por nº de internações prévias (apoio ao limiar de F05).

    Esta tabela é **descritiva**: confirma que o limiar de ``usuario_frequente``
    faz sentido, mas o limiar em si vem da definição clínica (ver
    ``configuracao.LIMIAR_USUARIO_FREQUENTE``), não de uma busca no alvo.

    Args:
        df: Base final com ``number_inpatient`` e ``readmit_30d``.

    Returns:
        Uma linha por nº de internações prévias (0, 1, 2, 3, 4 e "5 ou mais").
    """
    teto = 5
    grupo = df["number_inpatient"].clip(upper=teto)
    tabela = (
        df.groupby(grupo)[cfg.COLUNA_ALVO].agg(n="size", n_readmitidos="sum").reset_index()
    )
    tabela["internacoes_previas"] = tabela["number_inpatient"].map(
        lambda v: f"{teto} ou mais" if v >= teto else str(int(v))
    )
    tabela["taxa_readmissao_pct"] = (100 * tabela["n_readmitidos"] / tabela["n"]).round(2)
    return tabela[["internacoes_previas", "n", "n_readmitidos", "taxa_readmissao_pct"]]


#: Metadados dos atributos criados: ``(id, colunas, como foi criado, intuição de negócio)``.
ATRIBUTOS_CRIADOS: list[tuple[str, tuple[str, ...], str, str]] = [
    ("F01", (cfg.COLUNA_ALVO,), "1 se readmitted == '<30', senão 0", "É o que queremos prever: voltou ao hospital em menos de 30 dias?"),
    ("F02", ("idade_media",), "Ponto médio da faixa etária", "Idade vira número: modelos entendem 'mais velho = maior risco' sem 10 categorias."),
    ("F03", ("grupo_diag_1", "grupo_diag_2", "grupo_diag_3"), "CID-9 agrupado em 9 grupos clínicos (+ 'Sem_diagnostico')", "Centenas de códigos viram grupos que um médico reconhece (circulatório, diabetes...)."),
    ("F04", ("total_visitas_previas",), "ambulatório + emergência + internações do ano anterior", "Resume o quanto o paciente já usa o sistema de saúde."),
    ("F05", ("usuario_frequente",), "1 se number_inpatient >= 1", "Candidato a insight: quem já internou antes é alvo de acompanhamento pós-alta."),
    ("F06", ("n_meds_alterados",), "Nº de remédios com dose aumentada ou reduzida", "Mede a intensidade da mudança de tratamento durante a internação."),
    ("F07", ("n_meds_em_uso",), "Nº de remédios diferentes de 'No'", "Mede a complexidade do tratamento."),
    ("F08", ("a1c_medido", "glicose_medida"), "1 se o exame foi feito", "Fazer (ou não) HbA1c e glicose diz algo sobre o cuidado; a HbA1c é o tema do artigo-fonte."),
    ("F09", ("grupo_destino_alta",), "Casa / Transferencia / Outros / Nao_informado", "O destino da alta importa para o risco, sem dezenas de categorias."),
    ("F10", ("exames_por_dia", "procedimentos_por_dia"), "exames (ou procedimentos) / dias de internação", "Intensidade do cuidado, justa para estadias curtas e longas."),
    ("F11", ("especialidade_agrupada",), "Top-N especialidades + 'Outras' (+ 'Desconhecido')", "As dezenas de especialidades raras viram um grupo só."),
]  # fmt: skip


def tabela_atributos_criados(final: pd.DataFrame) -> pd.DataFrame:
    """Descreve cada atributo criado (F01–F11) com um resumo numérico calculado.

    Args:
        final: Base com todas as colunas derivadas.

    Returns:
        Uma linha por coluna criada: ``id``, ``coluna``, ``como_foi_criado``,
        ``intuicao_de_negocio``, ``n_distintos`` e ``resumo``.
    """
    linhas = []
    for id_, colunas, como, intuicao in ATRIBUTOS_CRIADOS:
        for coluna in colunas:
            serie = final[coluna]
            if pd.api.types.is_numeric_dtype(serie) and serie.nunique() <= 2:
                resumo = f"{fmt_pct(100 * float(serie.mean()))} das linhas valem 1"
            elif pd.api.types.is_numeric_dtype(serie):
                resumo = (
                    f"média {fmt_num(serie.mean())} · mín {fmt_num(serie.min())} "
                    f"· máx {fmt_num(serie.max())}"
                )
            else:
                top = serie.value_counts()
                resumo = f"mais comum: {top.index[0]} ({fmt_pct(100 * top.iloc[0] / len(serie))})"
            linhas.append(
                {
                    "id": id_,
                    "coluna": coluna,
                    "como_foi_criado": como,
                    "intuicao_de_negocio": intuicao,
                    "n_distintos": int(serie.nunique()),
                    "resumo": resumo,
                }
            )
    return pd.DataFrame(linhas)


def montar_tabela_decisoes(m: Mapping[str, object]) -> pd.DataFrame:
    """Monta a tabela principal: o que estava errado -> o que fizemos -> por quê.

    Cada evidência é um número vindo de ``m`` (as métricas medidas pelo pipeline).

    Args:
        m: Métricas do pipeline (ver ``executar_pipeline``).

    Returns:
        Tabela com ``id``, ``problema``, ``evidencia``, ``decisao``, ``motivo``, ``impacto``.
    """
    i, p = fmt_int, fmt_pct
    linhas = [
        ("I01", "Códigos numéricos sem significado (tipo de admissão, destino de alta, origem)",
         f"3 tabelas dentro do IDS_mapping; {i(m['n_codigos_nao_informado'])} códigos significam 'sem informação'",
         "Join com o IDS_mapping criando tipo_admissao, destino_alta, origem_admissao; NULL/Not Available/Not Mapped viram 'Nao_informado'",
         "Nomes legíveis ajudam EDA, SHAP e a explicação ao hospital",
         f"+3 colunas legíveis; {i(m['n_linhas_codigo_nao_informado'])} linhas usam algum código 'sem informação'"),
        ("D01", "Sexo 'Unknown/Invalid'",
         f"{i(m['d01_removidas'])} registros",
         "Remover os registros",
         "Pouquíssimos casos e sem como recuperar o valor correto",
         f"-{i(m['d01_removidas'])} linhas"),
        ("D02", "Pacientes que faleceram ou foram para hospice na alta",
         f"{i(m['d02_removidas'])} encontros-índice nos códigos {m['codigos_obito_hospice']}",
         "Remover",
         "Não podem ser readmitidos: o alvo seria 0 por motivo alheio ao atendimento e a taxa de readmissão ficaria subestimada",
         f"-{i(m['d02_removidas'])} linhas"),
        ("D03", "Mesmo paciente com vários encontros",
         f"{i(m['encontros_repetidos_bruto'])} encontros além do primeiro; {i(m['pacientes_varios_encontros_bruto'])} pacientes com mais de um encontro",
         "Manter só o primeiro encontro (menor encounter_id) de cada paciente",
         "Linhas do mesmo paciente não são independentes; no treino e no teste ao mesmo tempo o modelo 'decora' o paciente (vazamento)",
         f"-{i(m['d03_removidas'])} linhas; 1 linha por paciente"),
        ("D04", "Mesma categoria escrita de formas diferentes",
         f"'NO' em {i(m['n_readmitted_no_maiusculo'])} linhas de readmitted (o resto da base usa 'No'); "
         f"{i(m['n_descricoes_com_espaco_ou_ponto'])} descrições do mapeamento com espaço/ponto sobrando",
         "Tirar espaços, padronizar a caixa e unificar NULL/Not Available/Not Mapped em 'Nao_informado'",
         "Grafias diferentes viram categorias falsas para o modelo",
         f"{i(m['celulas_alteradas_d04'])} células ajustadas nos dados"),
        ("D05", "Coluna weight quase toda ausente",
         f"{p(m['pct_nulo_weight'])} de nulos",
         "Remover a coluna",
         "Imputar o peso de quase todo mundo seria inventar dado",
         "-1 coluna"),
        ("D06", "payer_code e medical_specialty com muitos ausentes",
         f"{p(m['pct_nulo_payer_code'])} e {p(m['pct_nulo_medical_specialty'])} de nulos",
         "Nulo -> 'Desconhecido' e criar flag *_ausente (0/1)",
         "A ausência pode ser informativa (ex.: paciente sem convênio) e não perdemos linhas",
         "+2 colunas de flag; 0 nulos nessas colunas"),
        ("D07", "race com '?'",
         f"{p(m['pct_nulo_race'])} de nulos ({i(m['n_nulo_race'])} linhas)",
         "'?' -> 'Desconhecido'",
         "Evita perder linhas; é atributo sensível e não deve ser usado para discriminar",
         "0 nulos em race"),
        ("D08", "diag_1, diag_2 e diag_3 vazios",
         f"{p(m['pct_nulo_diag_1'])}, {p(m['pct_nulo_diag_2'])} e {p(m['pct_nulo_diag_3'])} de nulos",
         "Nulo -> 'Sem_diagnostico'",
         "Não ter 2º ou 3º diagnóstico é um fato clínico, não um erro",
         "0 nulos nos diagnósticos"),
        ("D09", "'None' em max_glu_serum e A1Cresult parece nulo, mas não é",
         f"'None' em {p(m['pct_none_max_glu_serum'])} (glicose) e {p(m['pct_none_a1cresult'])} (HbA1c) das linhas brutas; "
         "o pandas lê 'None' como nulo se não for avisado",
         "Ler com keep_default_na=False e renomear 'None' -> 'Nao_medido'",
         "Significa 'exame não feito', que é informação sobre o cuidado ao paciente",
         "Nenhum dado perdido; vira a categoria 'Nao_medido' e as flags F08"),
        ("O01", "Valores extremos (cauda longa) nas variáveis de contagem",
         f"{i(m['n_variaveis_com_outlier'])} de {i(m['n_variaveis_outlier'])} variáveis têm outliers pelo IQR (de {p(m['pct_outlier_min'])} a {p(m['pct_outlier_max'])} das linhas)",
         "Só sinalizar (colunas *_outlier); não remover nem limitar",
         "São clinicamente plausíveis (pacientes graves); limitar valores exige estimar limites só no treino (Pessoa 3)",
         f"+{i(m['n_variaveis_outlier'])} colunas de sinalização; valores intactos"),
        ("S01", "Colunas sem informação ou redundantes",
         f"{i(m['n_quase_constantes'])} colunas com um único valor em >= {p(100 * cfg.LIMIAR_DOMINANCIA, 1)} das linhas; {i(m['n_correlacoes_altas'])} pares com |r| > {fmt_num(cfg.LIMIAR_CORRELACAO, 1)}",
         "Remover identificadores, colunas substituídas, quase constantes e redundantes",
         "Menos ruído, nenhum vazamento (e o readmitted original sai da base de modelagem)",
         f"{i(m['colunas_interim'])} colunas no interim -> {i(m['colunas_modelagem'])} na base de modelagem"),
    ]  # fmt: skip
    return pd.DataFrame(
        linhas, columns=["id", "problema", "evidencia", "decisao", "motivo", "impacto"]
    )


# ---------------------------------------------------------------------------
# Figuras
# ---------------------------------------------------------------------------


def _titulo(fig: Figure, texto: str, largura: int = 95) -> None:
    """Escreve o título (a conclusão da figura) alinhado à esquerda, com quebra de linha.

    Args:
        fig: Figura a receber o título.
        texto: Frase com a conclusão.
        largura: Nº de caracteres por linha antes de quebrar.
    """
    fig.suptitle(
        textwrap.fill(texto, largura),
        x=0.01, ha="left", fontsize=12.5, fontweight="bold", color=TEXTO, y=0.995,
    )  # fmt: skip


def _estilizar(ax: plt.Axes) -> None:
    """Deixa o eixo discreto: sem bordas superior/direita, grade leve, texto cinza.

    Args:
        ax: Eixo a estilizar.
    """
    for lado in ("top", "right"):
        ax.spines[lado].set_visible(False)
    for lado in ("left", "bottom"):
        ax.spines[lado].set_color(CINZA_CLARO)
    ax.tick_params(colors=TEXTO_SECUNDARIO, labelsize=9)
    ax.xaxis.label.set_color(TEXTO_SECUNDARIO)
    ax.yaxis.label.set_color(TEXTO_SECUNDARIO)
    ax.set_axisbelow(True)


def _salvar(fig: Figure, caminho: Path) -> Path:
    """Grava a figura em PNG (dpi 160) e fecha.

    Args:
        fig: Figura pronta.
        caminho: Arquivo de destino.

    Returns:
        O mesmo ``caminho``.
    """
    caminho.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(caminho, dpi=DPI_FIGURAS, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return caminho


def figura_funil(funil: pd.DataFrame, caminho: Path) -> Path:
    """Gráfico de barras do funil de linhas.

    Args:
        funil: Tabela do funil (``etapa``, ``descricao``, ``linhas_depois``, ``removidas``).
        caminho: Arquivo PNG de saída.

    Returns:
        Caminho gravado.
    """
    fig, ax = plt.subplots(figsize=(10, 4.2))
    rotulos = [f"{r.etapa}\n{r.descricao}" for r in funil.itertuples()]
    y = np.arange(len(funil))[::-1]
    cores = [AZUL if r.removidas == 0 else LARANJA for r in funil.itertuples()]
    ax.barh(y, funil["linhas_depois"], color=cores, height=0.6)
    for yi, r in zip(y, funil.itertuples()):
        extra = f"  (-{fmt_int(r.removidas)})" if r.removidas else ""
        ax.text(r.linhas_depois, yi, f" {fmt_int(r.linhas_depois)}{extra}", va="center", fontsize=9, color=TEXTO)
    ax.set_yticks(y, rotulos)
    ax.set_xlim(0, funil["linhas_depois"].max() * 1.22)
    ax.xaxis.set_major_formatter(lambda v, _: fmt_int(v))
    ax.set_xlabel("Linhas (internações) após a etapa")
    _estilizar(ax)
    maior = funil.loc[funil["removidas"].idxmax()]
    inicio, fim = int(funil["linhas_depois"].iloc[0]), int(funil["linhas_depois"].iloc[-1])
    _titulo(
        fig,
        f"De {fmt_int(inicio)} internações ficaram {fmt_int(fim)} ({fmt_pct(100 * fim / inicio, 1)}): "
        f"o maior corte foi '{maior['descricao'].lower()}' (-{fmt_int(maior['removidas'])})",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    return _salvar(fig, caminho)


def figura_ausentes(auditoria: pd.DataFrame, n_linhas: int, caminho: Path) -> Path:
    """Barras horizontais do % de nulos por coluna (antes do tratamento).

    O texto ``None`` de ``max_glu_serum`` e ``A1Cresult`` aparece em cinza para
    deixar claro que **não é nulo** (exame não realizado, D09).

    Args:
        auditoria: Saída de ``auditoria.perfil_qualidade``.
        n_linhas: Nº de linhas da base bruta (denominador do %).
        caminho: Arquivo PNG de saída.

    Returns:
        Caminho gravado.
    """
    dados = auditoria[(auditoria["n_nulos"] > 0) | (auditoria["n_textos_none"] > 0)].copy()
    dados["pct_none"] = 100 * dados["n_textos_none"] / n_linhas
    dados["valor"] = np.where(dados["n_nulos"] > 0, dados["pct_nulos"], dados["pct_none"])
    dados["e_nulo"] = dados["n_nulos"] > 0
    dados = dados.sort_values("valor")
    fig, ax = plt.subplots(figsize=(9, 4.4))
    cores = [AZUL if e else CINZA for e in dados["e_nulo"]]
    ax.barh(dados["coluna"], dados["valor"], color=cores, height=0.6)
    for yi, r in enumerate(dados.itertuples()):
        sufixo = "" if r.e_nulo else "  ('None' = exame não feito, não é nulo)"
        ax.text(r.valor, yi, f" {fmt_pct(r.valor)}{sufixo}", va="center", fontsize=9, color=TEXTO)
    ax.set_xlim(0, 135)
    ax.set_xlabel("% das linhas da base bruta")
    _estilizar(ax)
    top = dados[dados["e_nulo"]].iloc[-1]
    _titulo(
        fig,
        f"{int((dados['e_nulo']).sum())} colunas têm dados ausentes; '{top['coluna']}' tem {fmt_pct(top['valor'], 1)}. "
        "Já o 'None' dos exames (cinza) não é ausência",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    return _salvar(fig, caminho)


def figura_outliers(df: pd.DataFrame, limites: pd.DataFrame, caminho: Path) -> Path:
    """Boxplots das variáveis de contagem, com os outliers (IQR) em laranja.

    Args:
        df: Base tratada (valores originais, sem alteração).
        limites: Saída de ``outliers.calcular_limites_iqr``.
        caminho: Arquivo PNG de saída.

    Returns:
        Caminho gravado.
    """
    n = len(limites)
    colunas = 4
    linhas = int(np.ceil(n / colunas))
    fig, eixos = plt.subplots(linhas, colunas, figsize=(11, 3.4 * linhas))
    for ax, r in zip(np.ravel(eixos), limites.itertuples()):
        valores = df[r.variavel]
        ax.boxplot(
            valores, widths=0.5, showfliers=False,
            boxprops={"color": AZUL}, whiskerprops={"color": AZUL}, capprops={"color": AZUL},
            medianprops={"color": TEXTO},
        )  # fmt: skip
        fora = valores[(valores < r.limite_inferior) | (valores > r.limite_superior)]
        rng = np.random.default_rng(cfg.SEMENTE_ALEATORIA)  # só o "jitter" visual
        ax.scatter(1 + rng.uniform(-0.2, 0.2, fora.nunique()), fora.drop_duplicates(), s=8, color=LARANJA, alpha=0.8)
        ax.set_title(f"{r.variavel}\n{fmt_pct(r.pct_outliers, 1)} outliers", fontsize=9.5, color=TEXTO)
        ax.set_xticks([])
        _estilizar(ax)
    for ax in np.ravel(eixos)[n:]:
        ax.axis("off")
    com_outlier = int((limites["n_outliers"] > 0).sum())
    _titulo(
        fig,
        f"{com_outlier} de {n} variáveis de contagem têm valores atípicos (laranja), mas são pacientes "
        "graves plausíveis: só sinalizamos, não removemos",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    return _salvar(fig, caminho)


def figura_alvo(balanceamento: pd.DataFrame, caminho: Path) -> Path:
    """Antes x depois do alvo: 3 classes originais x alvo binário final.

    Args:
        balanceamento: Saída de :func:`tabela_balanceamento`.
        caminho: Arquivo PNG de saída.

    Returns:
        Caminho gravado.
    """
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 3.8), gridspec_kw={"width_ratios": [3, 2]})
    for ax, momento, titulo in (
        (a1, "antes (bruto)", "Antes: readmitted (3 classes)"),
        (a2, "depois (final)", "Depois: readmit_30d (alvo)"),
    ):
        parte = balanceamento[balanceamento["momento"] == momento]
        cores = [LARANJA if ("<30" in c or c.endswith("=1")) else AZUL for c in parte["classe"]]
        ax.bar(parte["classe"], parte["pct"], color=cores, width=0.6)
        for x, r in enumerate(parte.itertuples()):
            ax.text(x, r.pct, f"{fmt_pct(r.pct)}\n({fmt_int(r.n)})", ha="center", va="bottom", fontsize=9, color=TEXTO)
        ax.set_ylim(0, parte["pct"].max() * 1.3)
        ax.set_title(titulo, fontsize=10, color=TEXTO_SECUNDARIO, loc="left")
        ax.set_ylabel("% das linhas")
        _estilizar(ax)
    positivo = balanceamento[balanceamento["classe"] == "readmit_30d=1"].iloc[0]
    _titulo(
        fig,
        f"Só {fmt_pct(positivo['pct'])} dos pacientes voltam em menos de 30 dias: a classe positiva é "
        "minoria, então acurácia sozinha engana (use F1 e AUC)",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.91))
    return _salvar(fig, caminho)


def figura_outlier_antes_depois(
    bruto: pd.DataFrame, final: pd.DataFrame, limites: pd.DataFrame, caminho: Path
) -> Path:
    """Histograma (escala log) da variável mais assimétrica: bruto x final.

    Mostra que a forma da distribuição **não muda** (só saem as linhas filtradas
    em D01–D03): os valores extremos foram mantidos e apenas sinalizados.

    Args:
        bruto: Base bruta.
        final: Base tratada.
        limites: Saída de ``outliers.calcular_limites_iqr`` (escolhe a variável).
        caminho: Arquivo PNG de saída.

    Returns:
        Caminho gravado.
    """
    r = limites.sort_values("assimetria", ascending=False).iloc[0]
    var = r["variavel"]
    maximo = int(bruto[var].max())
    classes = np.arange(0, maximo + 2)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 3.8), sharey=True)
    for ax, base, titulo, cor in (
        (a1, bruto, f"Antes ({fmt_int(len(bruto))} linhas)", CINZA),
        (a2, final, f"Depois ({fmt_int(len(final))} linhas)", AZUL),
    ):
        ax.hist(base[var], bins=classes, color=cor, align="left", rwidth=0.85)
        ax.axvline(r["limite_superior"], color=LARANJA, linestyle="--", linewidth=1.5)
        ax.set_yscale("log")
        ax.set_xlabel(var)
        ax.set_title(titulo, fontsize=10, color=TEXTO_SECUNDARIO, loc="left")
        _estilizar(ax)
    a1.set_ylabel("Nº de linhas (escala log)")
    a2.text(
        r["limite_superior"] + 0.04 * maximo + 0.3, a2.get_ylim()[1] * 0.35,
        "limite do IQR\n(acima = outlier)", color=LARANJA, fontsize=9, va="top",
    )  # fmt: skip
    _titulo(
        fig,
        f"'{var}' tem cauda longa (assimetria {fmt_num(r['assimetria'], 1)}, máximo {fmt_int(maximo)}): "
        "os valores extremos continuam na base, apenas sinalizados",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.91))
    return _salvar(fig, caminho)


def figura_diagnosticos(bruto: pd.DataFrame, final: pd.DataFrame, caminho: Path) -> Path:
    """Antes x depois de ``diag_1``: códigos CID-9 brutos x grupos clínicos.

    Args:
        bruto: Base bruta (com ``diag_1``).
        final: Base tratada (com ``grupo_diag_1``).
        caminho: Arquivo PNG de saída.

    Returns:
        Caminho gravado.
    """
    codigos = bruto["diag_1"].value_counts()
    grupos = final["grupo_diag_1"].value_counts()
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.2), gridspec_kw={"width_ratios": [1, 1]})
    topo = codigos.head(10)
    a1.barh(topo.index.astype(str)[::-1], topo.to_numpy()[::-1], color=CINZA, height=0.6)
    a1.set_title(f"Antes: {fmt_int(len(codigos))} códigos distintos (10 mais comuns)", fontsize=10, color=TEXTO_SECUNDARIO, loc="left")
    a2.barh(grupos.index.astype(str)[::-1], grupos.to_numpy()[::-1], color=AZUL, height=0.6)
    n_grupos = int((grupos.index != cfg.ROTULO_SEM_DIAGNOSTICO).sum())  # sem contar "Sem_diagnostico"
    a2.set_title(f"Depois: {fmt_int(n_grupos)} grupos clínicos (+ Sem_diagnostico)", fontsize=10, color=TEXTO_SECUNDARIO, loc="left")
    for ax in (a1, a2):
        ax.xaxis.set_major_formatter(lambda v, _: fmt_int(v))
        ax.set_xlabel("Nº de linhas")
        _estilizar(ax)
    maior = grupos.index[0]
    _titulo(
        fig,
        f"{fmt_int(len(codigos))} códigos CID-9 viraram {fmt_int(n_grupos)} grupos clínicos; "
        f"o diagnóstico principal mais comum é '{maior}' ({fmt_pct(100 * grupos.iloc[0] / grupos.sum(), 1)})",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    return _salvar(fig, caminho)


def figura_limiar_usuario(tabela: pd.DataFrame, caminho: Path) -> Path:
    """Taxa de readmissão por nº de internações prévias (apoio a F05).

    Args:
        tabela: Saída de :func:`tabela_limiar_usuario`.
        caminho: Arquivo PNG de saída.

    Returns:
        Caminho gravado.
    """
    fig, ax = plt.subplots(figsize=(8.5, 4))
    cores = [AZUL if r.internacoes_previas == "0" else LARANJA for r in tabela.itertuples()]
    ax.bar(tabela["internacoes_previas"], tabela["taxa_readmissao_pct"], color=cores, width=0.6)
    for x, r in enumerate(tabela.itertuples()):
        ax.text(x, r.taxa_readmissao_pct, f"{fmt_pct(r.taxa_readmissao_pct, 1)}\n(n={fmt_int(r.n)})", ha="center", va="bottom", fontsize=8.5, color=TEXTO)
    ax.set_ylim(0, tabela["taxa_readmissao_pct"].max() * 1.3)
    ax.set_xlabel("Internações no ano anterior (number_inpatient)")
    ax.set_ylabel("% readmitidos em < 30 dias")
    _estilizar(ax)
    zero = tabela.iloc[0]["taxa_readmissao_pct"]
    um = tabela.iloc[1]["taxa_readmissao_pct"]
    _titulo(
        fig,
        f"Quem já tinha 1 internação prévia volta mais ({fmt_pct(um, 1)}) do que quem não tinha "
        f"({fmt_pct(zero, 1)}): por isso usuario_frequente = 1 a partir de {cfg.LIMIAR_USUARIO_FREQUENTE}",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.91))
    return _salvar(fig, caminho)
