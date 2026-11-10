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
    """Formata inteiro no padrão brasileiro (``145731`` -> ``145.731``).

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
    """Formata percentual (``7.05`` -> ``7,05%``).

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


#: Decisão aplicada a cada coluna com "sem informação": ``{coluna: (ID, decisão, situação final)}``.
_DECISOES_SEM_INFORMACAO: dict[str, tuple[str, str, str]] = {
    "classificacao_acidente": ("D04", "'NA' -> 'Nao_informado'", "coluna sai da modelagem (consequência)"),
    "condicao_metereologica": ("D04", "'Ignorado' -> 'Nao_informado'", "categoria 'Nao_informado'"),
    "sentido_via": ("D04", "'Não Informado' -> 'Nao_informado'", "categoria 'Nao_informado'"),
    "tipo_veiculo": ("D04", "nulo -> 'Nao_informado'; veículo não entra nos tipos F11", "resumido por acidente (F11)"),
    "tipo_envolvido": ("D04", "nulo -> 'Nao_informado'", "resumido por acidente (F11)"),
    "estado_fisico": ("D04", "nulo -> 'Nao_informado'; não é usada (consequência)", "não usada"),
    "sexo": ("D04", "nulo/Ignorado/Não Informado -> 'Nao_informado'", "resumido por acidente (F11)"),
    "uop": ("", "coluna administrativa (quem atendeu)", "sai da modelagem"),
}


def tabela_ausentes(perfil: pd.DataFrame, final: pd.DataFrame, m: Mapping[str, object]) -> pd.DataFrame:
    """Lista tudo que estava "sem informação" ou impossível, e o que foi feito.

    Reúne três fontes: colunas com nulo ou rótulo "Ignorado/Não Informado" na
    auditoria, valores impossíveis anulados (idade, ano de fabricação,
    coordenadas) e os nulos que sobram nas colunas derivadas.

    Args:
        perfil: Auditoria inicial (ocorrências e pessoas juntas).
        final: Base completa (interim), para contar nulos restantes.
        m: Métricas do pipeline.

    Returns:
        Tabela com ``origem``, ``coluna``, ``id_decisao``, ``n_antes``,
        ``pct_antes``, ``decisao`` e ``situacao_final``, da maior para a menor
        porcentagem.
    """
    linhas: list[dict[str, object]] = []
    for r in perfil.itertuples():
        n = int(r.n_nulos + r.n_rotulos_sem_informacao)
        if n == 0:
            continue
        id_decisao, decisao, situacao = _DECISOES_SEM_INFORMACAO.get(r.coluna, ("", "", ""))
        linhas.append(
            {
                "origem": r.origem, "coluna": r.coluna, "id_decisao": id_decisao, "n_antes": n,
                "pct_antes": r.pct_sem_informacao, "decisao": decisao, "situacao_final": situacao,
            }
        )  # fmt: skip

    def _extra(origem: str, coluna: str, id_: str, n: int, total: int, decisao: str, situacao: str) -> None:
        """Acrescenta uma linha de valor impossível (que não aparece como nulo no bruto)."""
        linhas.append(
            {
                "origem": origem, "coluna": coluna, "id_decisao": id_, "n_antes": n,
                "pct_antes": round(100 * n / total, 2), "decisao": decisao, "situacao_final": situacao,
            }
        )  # fmt: skip

    n_pes, n_oc = int(m["linhas_pessoas_bruto"]), int(m["linhas_ocorrencias_bruto"])  # type: ignore[call-overload]
    _extra(
        "pessoas", "idade (impossível)", "D05",
        int(m["n_pessoas_idade_invalida"]) + int(m["n_condutores_idade_menor_10"]), n_pes,  # type: ignore[call-overload]
        "vira nulo (não é substituída)",
        f"{fmt_int(float(m['n_nulo_final_idade_media_condutores']))} acidentes sem idade média de condutores",  # type: ignore[arg-type]
    )  # fmt: skip
    _extra(
        "pessoas", "ano_fabricacao_veiculo (impossível)", "D06", int(m["n_pessoas_ano_fab_invalido"]), n_pes,  # type: ignore[call-overload]
        "vira nulo (não é substituído)",
        f"{fmt_int(float(m['n_nulo_final_idade_media_veiculos']))} acidentes sem idade média de veículos",  # type: ignore[arg-type]
    )  # fmt: skip
    _extra(
        "ocorrencias", "latitude/longitude (fora do Brasil)", "D07", int(m["n_coordenadas_invalidas"]), n_oc,  # type: ignore[call-overload]
        "vira nulo só se impossível", f"{fmt_int(float(m['n_nulo_final_latitude']))} nulos",  # type: ignore[arg-type]
    )  # fmt: skip
    tabela = pd.DataFrame(linhas).sort_values(["pct_antes", "coluna"], ascending=[False, True], kind="stable")
    return tabela.reset_index(drop=True)


def tabela_cardinalidade(antes: pd.DataFrame, depois: pd.DataFrame) -> pd.DataFrame:
    """Mostra quantos valores distintos cada coluna tinha antes e depois do agrupamento.

    Args:
        antes: Ocorrências brutas (``causa_acidente``, ``municipio``, ``tracado_via``).
        depois: Base final (``grupo_causa``, ``regiao``, colunas ``via_*``).

    Returns:
        Uma linha por transformação.
    """
    n_flags = len([c for c in depois.columns if c.startswith("via_")])
    return pd.DataFrame(
        [
            {
                "coluna_antes": "causa_acidente", "n_distintos_antes": int(antes["causa_acidente"].nunique()),
                "coluna_depois": "grupo_causa", "n_distintos_depois": int(depois["grupo_causa"].nunique()),
            },
            {
                "coluna_antes": "tracado_via", "n_distintos_antes": int(antes["tracado_via"].nunique()),
                "coluna_depois": "via_* (colunas 0/1)", "n_distintos_depois": n_flags,
            },
            {
                "coluna_antes": "municipio", "n_distintos_antes": int(antes["municipio"].nunique()),
                "coluna_depois": "regiao", "n_distintos_depois": int(depois["regiao"].nunique()),
            },
        ]
    )  # fmt: skip


def tabela_balanceamento(classificacao_bruta: pd.Series, alvo_final: pd.Series) -> pd.DataFrame:
    """Compara a classificação original (3 classes) com o alvo binário final.

    Args:
        classificacao_bruta: Coluna ``classificacao_acidente`` das ocorrências brutas.
        alvo_final: Coluna ``acidente_fatal`` da base final.

    Returns:
        Tabela com ``momento``, ``classe``, ``n`` e ``pct``.
    """
    antes = classificacao_bruta.fillna("NA").value_counts()
    depois = alvo_final.value_counts()
    linhas = [
        {"momento": "antes (bruto)", "classe": str(k), "n": int(v), "pct": round(100 * v / antes.sum(), 2)}
        for k, v in antes.items()
    ]
    linhas += [
        {"momento": "depois (final)", "classe": f"{cfg.COLUNA_ALVO}={k}", "n": int(v), "pct": round(100 * v / depois.sum(), 2)}
        for k, v in depois.sort_index().items()
    ]  # fmt: skip
    return pd.DataFrame(linhas)


#: Dimensões (e suas colunas) da taxa de acidentes fatais por grupo.
DIMENSOES_TAXA: tuple[str, ...] = (
    "faixa_horaria", "fase_dia", "tipo_pista", "grupo_causa", "regiao", "envolve_moto",
    "envolve_pedestre", "envolve_caminhao", "envolve_bicicleta", "fim_de_semana", "area_urbana",
    "eh_feriado",
)  # fmt: skip


def tabela_taxa_fatal(df: pd.DataFrame) -> pd.DataFrame:
    """Taxa de acidentes fatais por grupo (visão **descritiva**, não teste estatístico).

    Serve de "pré-visualização" dos padrões que a análise exploratória (Pessoa 2)
    vai testar. Não é inferência: não há teste de hipótese aqui.

    Args:
        df: Base com o alvo e as dimensões de ``DIMENSOES_TAXA``.

    Returns:
        Uma linha por grupo: ``dimensao``, ``grupo``, ``n``, ``n_fatais`` e
        ``taxa_fatal_pct`` (inclui a linha ``geral``/``todos``).
    """
    alvo = cfg.COLUNA_ALVO
    linhas = [
        {
            "dimensao": "geral", "grupo": "todos", "n": len(df), "n_fatais": int(df[alvo].sum()),
            "taxa_fatal_pct": round(100 * float(df[alvo].mean()), 2),
        }
    ]  # fmt: skip
    for dimensao in DIMENSOES_TAXA:
        agregado = df.groupby(dimensao)[alvo].agg(n="size", n_fatais="sum")
        for grupo, r in agregado.iterrows():
            linhas.append(
                {
                    "dimensao": dimensao, "grupo": str(grupo), "n": int(r["n"]), "n_fatais": int(r["n_fatais"]),
                    "taxa_fatal_pct": round(100 * r["n_fatais"] / r["n"], 2),
                }
            )  # fmt: skip
    return pd.DataFrame(linhas)


#: Metadados dos atributos criados: ``(id, colunas, como foi criado, intuição de negócio)``.
ATRIBUTOS_CRIADOS: list[tuple[str, tuple[str, ...], str, str]] = [
    ("F01", (cfg.COLUNA_ALVO,), "1 se mortos > 0, senão 0", "É o que queremos prever: o acidente teve morte?"),
    ("F02", ("hora", "faixa_horaria"), "Hora extraída do horário; faixa Madrugada/Manha/Tarde/Noite", "O risco muda ao longo do dia e a faixa deixa o padrão visível para o gestor."),
    ("F03", ("mes", "ano", "fim_de_semana"), "Extraídos da data (fim de semana = sábado ou domingo)", "Sazonalidade e rotina de viagem (fim de semana) mudam o tipo de acidente."),
    ("F04", ("eh_feriado",), "Data comparada ao calendário de feriados nacionais (I02)", "Feriados concentram viagens longas e mais risco."),
    ("F05", ("regiao",), "UF mapeada para região do Brasil (I03)", "Compara regiões sem as 27 UFs; mantém o sentido geográfico."),
    ("F06", ("grupo_causa",), "As 69 causas agrupadas em poucos grupos (atenção, velocidade, álcool, manobra, via...)", "Um gestor age sobre grupos de causa, não sobre 69 rótulos."),
    ("F07", ("via_reta", "via_curva", "via_aclive", "via_declive", "via_intersecao", "via_rotatoria_retorno", "via_obras_desvio", "via_ponte_viaduto_tunel"), "Traçado composto ('Reta;Declive') separado em colunas 0/1", "Curva, aclive e interseção podem ter risco diferente; juntos viravam centenas de combinações."),
    ("F08", ("area_urbana",), "1 quando uso_solo = 'Sim'", "Trecho urbano tem perfil de acidente diferente do rural."),
    ("F09", ("br",), "Código numérico vira categoria ('BR-116')", "BR é um nome de rodovia, não uma quantidade."),
    ("F10", ("trecho",), "Rodovia + faixa de 10 km (ex.: BR-116_km230)", "Permite achar trechos críticos (só no interim, por ter muitos níveis)."),
    ("F11", ("envolve_moto", "envolve_caminhao", "envolve_onibus", "envolve_bicicleta", "envolve_pedestre", "idade_media_condutores", "pct_condutores_homens", "idade_media_veiculos"), "Arquivo de pessoas resumido por acidente (I01)", "Quem e o quê se envolveu: moto, caminhão, pedestre, idade do condutor e do veículo."),
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
            if pd.api.types.is_numeric_dtype(serie) and set(serie.dropna().unique()) <= {0, 1}:
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
    f = lambda chave: float(m[chave])  # type: ignore[arg-type]  # noqa: E731
    linhas = [
        ("I01", "Idade, sexo e tipo de veículo estão em outro arquivo (um por pessoa)",
         f"{i(f('linhas_pessoas_bruto'))} linhas de pessoas para {i(f('linhas_ocorrencias_bruto'))} acidentes; {i(f('ocorrencias_sem_pessoas'))} acidentes sem pessoas",
         "Resumir as pessoas por acidente (F11) e juntar pelo `id` (join à esquerda)",
         "Idade do condutor, moto, caminhão e pedestre ajudam a explicar o risco e só existem nesse arquivo",
         f"+{i(f('n_colunas_resumo_pessoas'))} colunas; nenhum acidente perdido"),
        ("I02", "Os dados não dizem se o dia era feriado",
         f"{i(f('n_feriados_nos_dados'))} acidentes ({p(f('pct_feriados_nos_dados'))}) caíram em feriado nacional",
         "Calcular feriados nacionais (fixos + Páscoa) e criar `eh_feriado`",
         "Feriados concentram viagens longas e mudam o perfil do tráfego",
         "+1 coluna"),
        ("I03", "27 UFs e quase 2 mil municípios para descrever o local",
         f"{i(f('n_ufs'))} UFs e {i(f('n_municipios'))} municípios",
         "Mapear UF -> região do Brasil (IBGE) e criar `regiao`",
         "Reduz a cardinalidade mantendo o sentido geográfico",
         "+1 coluna"),
        ("D01", "Acidentes que podem ter sido lançados duas vezes",
         f"{i(f('ocorrencias_duplicadas_sem_id'))} registro(s) igual(is) ao anterior em todos os campos do arquivo de ocorrências (exceto o `id`); depois de juntar as pessoas, {i(f('d01_removidas'))} continuam idênticos",
         "Checar a duplicidade **depois** de juntar as pessoas e remover só os idênticos também nelas",
         "Dois registros iguais só no cabeçalho podem ser acidentes distintos no mesmo minuto; remover seria apagar um acidente real",
         f"-{i(f('d01_removidas'))} linha(s)"),
        ("D02", "Suicídios presumidos e sinistros pessoais não são acidentes de trânsito",
         f"{i(f('d02_removidas'))} acidentes: causa 'Suicídio (presumido)' ({i(f('n_causa_intencional'))}) ou tipo 'Sinistro pessoal de trânsito' ({i(f('n_tipo_pessoal'))})",
         "Remover",
         "Nenhuma ação de segurança viária (fiscalização, sinalização, obra) evitaria esses eventos; eles inflariam a taxa de fatais",
         f"-{i(f('d02_removidas'))} linhas ({i(f('n_fatais_removidos_d02'))} fatais)"),
        ("D03", "As contagens de vítimas não fecham",
         f"{i(f('n_contagens_inconsistentes'))} acidentes ({p(f('pct_contagens_inconsistentes'))}) em que mortos + feridos + ilesos + ignorados != pessoas",
         "Sinalizar em `contagens_inconsistentes` (só no interim); não corrigir",
         "Não sabemos qual campo está errado; o alvo usa só `mortos`",
         "0 linhas removidas"),
        ("D04", "A PRF escreve \"não sei\" de várias formas",
         f"'Ignorado' em {i(f('n_clima_ignorado'))} acidentes (clima), 'Não Informado' em {i(f('n_sentido_nao_informado'))} (sentido da via), 'NA' em {i(f('n_na_classificacao'))} (classificação); sexo sem informação em {p(f('pct_pessoas_sexo_sem_info'))} das linhas de pessoas",
         "Unificar em `Nao_informado`",
         "Palavras diferentes para a mesma coisa viram categorias falsas para o modelo",
         f"{i(f('celulas_alteradas_d04'))} células ajustadas"),
        ("D05", "Idades impossíveis",
         f"Idade máxima de {int(f('idade_maxima_bruta'))} anos; {i(f('n_pessoas_idade_invalida'))} idades fora de 0 a 110; {i(f('n_condutores_idade_menor_10'))} condutores ({p(f('pct_condutores_idade_menor_10'))}) com menos de 10 anos (a PRF usa 0 como 'não informado')",
         "Idade impossível vira nulo (não é substituída por média)",
         "Não dá para saber a idade real; inventar uma distorceria a média de idade dos condutores",
         f"{i(f('n_nulo_final_idade_media_condutores'))} acidentes ({p(f('pct_nulo_final_idade_media_condutores'))}) ficam sem idade média"),
        ("D06", "Ano de fabricação impossível",
         f"{i(f('n_pessoas_ano_fab_invalido'))} linhas de pessoas ({p(f('pct_pessoas_ano_fab_invalido'))}) com ano 0, 1900 ou futuro, sendo {i(f('n_pessoas_ano_fab_zero'))} com 0",
         "Vira nulo; a idade do veículo só usa anos válidos",
         "O valor 0 viraria veículos de 2.025 anos",
         f"{i(f('n_nulo_final_idade_media_veiculos'))} acidentes ({p(f('pct_nulo_final_idade_media_veiculos'))}) ficam sem idade média de veículos"),
        ("D07", "Latitude positiva parece erro, mas não é",
         f"{i(f('n_latitude_positiva'))} acidentes com latitude >= 0, todos em UFs ao norte do Equador (válidos: {i(f('n_latitude_positiva_valida'))}); {i(f('n_coordenadas_invalidas'))} coordenadas fora do Brasil",
         "Conferir contra a caixa do Brasil e a UF; anular só as impossíveis",
         "Latitude positiva é correta em AP e RR; removê-las seria um erro",
         f"{i(f('n_coordenadas_invalidas'))} coordenadas anuladas"),
        ("O01", "Valores extremos (cauda longa)",
         f"{i(f('n_variaveis_com_outlier'))} de {i(f('n_variaveis_outlier'))} variáveis têm outliers pelo IQR (de {p(f('pct_outlier_min'))} a {p(f('pct_outlier_max'))} das linhas); um acidente chega a {i(f('veiculos_maximo_bruto'))} veículos",
         "Só sinalizar (colunas `*_outlier`); não remover nem limitar",
         "Acidentes com muitos veículos existem (engavetamentos) e estão entre os mais graves; limitar valores exige estimar limites só no treino (Pessoa 3)",
         f"+{i(f('n_variaveis_outlier'))} colunas de sinalização; valores intactos"),
        ("S01", "Colunas que vazam o alvo, são administrativas, redundantes ou sem informação",
         f"{i(f('n_colunas_resultado_removidas'))} colunas de consequência do acidente; {i(f('n_quase_constantes'))} quase constantes; {i(f('n_correlacoes_altas'))} pares com |r| > {fmt_num(cfg.LIMIAR_CORRELACAO, 1)}",
         "Remover identificador, consequências, administrativas, substituídas, quase constantes e redundantes",
         "As consequências (mortos, feridos, classificação) definem o alvo: usá-las seria vazamento",
         f"{i(f('colunas_interim'))} colunas no interim -> {i(f('colunas_modelagem'))} na base de modelagem"),
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
    """Deixa o eixo discreto: sem bordas superior/direita, texto cinza.

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
    fig, ax = plt.subplots(figsize=(10, 3.6))
    rotulos = [f"{r.etapa}\n{r.descricao}" for r in funil.itertuples()]
    y = np.arange(len(funil))[::-1]
    cores = [AZUL if r.removidas == 0 else LARANJA for r in funil.itertuples()]
    ax.barh(y, funil["linhas_depois"], color=cores, height=0.6)
    for yi, r in zip(y, funil.itertuples()):
        extra = f"  (-{fmt_int(r.removidas)})" if r.removidas else ""
        ax.text(r.linhas_depois, yi, f" {fmt_int(r.linhas_depois)}{extra}", va="center", fontsize=9, color=TEXTO)
    ax.set_yticks(y, rotulos)
    ax.set_xlim(0, funil["linhas_depois"].max() * 1.25)
    ax.xaxis.set_major_formatter(lambda v, _: fmt_int(v))
    ax.set_xlabel("Acidentes (linhas) após a etapa")
    _estilizar(ax)
    maior = funil.loc[funil["removidas"].idxmax()]
    inicio, fim = int(funil["linhas_depois"].iloc[0]), int(funil["linhas_depois"].iloc[-1])
    _titulo(
        fig,
        f"De {fmt_int(inicio)} acidentes ficaram {fmt_int(fim)} ({fmt_pct(100 * fim / inicio, 1)}): "
        f"a base já vem organizada e o maior corte foi '{maior['descricao'].lower()}' (-{fmt_int(maior['removidas'])})",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    return _salvar(fig, caminho)


def figura_ausentes(ausentes: pd.DataFrame, caminho: Path) -> Path:
    """Barras horizontais do % "sem informação" por coluna (antes do tratamento).

    Mostra rótulos "Ignorado/Não Informado/NA" e valores impossíveis (idade,
    ano de fabricação). Azul = arquivo de acidentes, cinza = arquivo de pessoas.

    Args:
        ausentes: Saída de :func:`tabela_ausentes`.
        caminho: Arquivo PNG de saída.

    Returns:
        Caminho gravado.
    """
    dados = ausentes[ausentes["pct_antes"] > 0].sort_values("pct_antes").copy()
    fig, ax = plt.subplots(figsize=(9.5, 0.45 * len(dados) + 1.8))
    rotulos = [f"{r.coluna}\n({r.origem})" for r in dados.itertuples()]
    cores = [AZUL if o == "ocorrencias" else CINZA for o in dados["origem"]]
    ax.barh(rotulos, dados["pct_antes"], color=cores, height=0.6)
    for yi, r in enumerate(dados.itertuples()):
        ax.text(r.pct_antes, yi, f" {fmt_pct(r.pct_antes)}", va="center", fontsize=9, color=TEXTO)
    ax.set_xlim(0, dados["pct_antes"].max() * 1.2)
    ax.set_xlabel("% das linhas do arquivo bruto")
    _estilizar(ax)
    topo = dados.iloc[-1]
    _titulo(
        fig,
        f"A base é bem preenchida: o pior caso é '{topo['coluna']}' com {fmt_pct(topo['pct_antes'], 1)}. "
        "O problema são valores impossíveis e rótulos 'não sei', não colunas vazias",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    return _salvar(fig, caminho)


def figura_outliers(df: pd.DataFrame, limites: pd.DataFrame, caminho: Path) -> Path:
    """Boxplots das variáveis numéricas, com os outliers (IQR) em laranja.

    Args:
        df: Base tratada (valores originais, sem alteração).
        limites: Saída de ``outliers.calcular_limites_iqr``.
        caminho: Arquivo PNG de saída.

    Returns:
        Caminho gravado.
    """
    n = len(limites)
    colunas = min(n, 4)
    linhas = int(np.ceil(n / colunas))
    fig, eixos = plt.subplots(linhas, colunas, figsize=(11, 3.6 * linhas))
    for ax, r in zip(np.ravel(eixos), limites.itertuples()):
        valores = df[r.variavel].dropna()
        ax.boxplot(
            valores, widths=0.5, showfliers=False,
            boxprops={"color": AZUL}, whiskerprops={"color": AZUL}, capprops={"color": AZUL},
            medianprops={"color": TEXTO},
        )  # fmt: skip
        fora = valores[(valores < r.limite_inferior) | (valores > r.limite_superior)]
        rng = np.random.default_rng(cfg.SEMENTE_ALEATORIA)  # só o "jitter" visual
        unicos = fora.drop_duplicates()
        ax.scatter(1 + rng.uniform(-0.2, 0.2, len(unicos)), unicos, s=8, color=LARANJA, alpha=0.8)
        ax.set_title(f"{r.variavel}\n{fmt_pct(r.pct_outliers, 1)} outliers", fontsize=9.5, color=TEXTO)
        ax.set_xticks([])
        _estilizar(ax)
    for ax in np.ravel(eixos)[n:]:
        ax.axis("off")
    com_outlier = int((limites["n_outliers"] > 0).sum())
    _titulo(
        fig,
        f"{com_outlier} de {n} variáveis têm valores atípicos (laranja), mas são acidentes e veículos reais: "
        "só sinalizamos, não removemos",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.88))
    return _salvar(fig, caminho)


def figura_alvo(balanceamento: pd.DataFrame, caminho: Path) -> Path:
    """Antes x depois do alvo: 3 classes originais x alvo binário final.

    Args:
        balanceamento: Saída de :func:`tabela_balanceamento`.
        caminho: Arquivo PNG de saída.

    Returns:
        Caminho gravado.
    """
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10.5, 4), gridspec_kw={"width_ratios": [3, 2]})
    for ax, momento, titulo in (
        (a1, "antes (bruto)", "Antes: classificacao_acidente"),
        (a2, "depois (final)", f"Depois: {cfg.COLUNA_ALVO} (alvo)"),
    ):
        parte = balanceamento[balanceamento["momento"] == momento]
        rotulos = [c.replace("Com Vítimas ", "").replace("Sem Vítimas", "Sem vítimas") for c in parte["classe"]]
        cores = [LARANJA if ("Fatais" in c or c.endswith("=1")) else AZUL for c in parte["classe"]]
        ax.bar(rotulos, parte["pct"], color=cores, width=0.6)
        for x, r in enumerate(parte.itertuples()):
            ax.text(x, r.pct, f"{fmt_pct(r.pct)}\n({fmt_int(r.n)})", ha="center", va="bottom", fontsize=9, color=TEXTO)
        ax.set_ylim(0, min(parte["pct"].max() * 1.3, 108))  # nunca passa muito de 100%
        ax.set_title(titulo, fontsize=10, color=TEXTO_SECUNDARIO, loc="left")
        ax.set_ylabel("% dos acidentes")
        _estilizar(ax)
    positivo = balanceamento[balanceamento["classe"] == f"{cfg.COLUNA_ALVO}=1"].iloc[0]
    _titulo(
        fig,
        f"Só {fmt_pct(positivo['pct'])} dos acidentes têm morte: a classe positiva é minoria, "
        "então acurácia sozinha engana (use F1 e AUC)",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    return _salvar(fig, caminho)


def figura_outlier_antes_depois(
    bruto: pd.DataFrame, final: pd.DataFrame, limites: pd.DataFrame, caminho: Path
) -> Path:
    """Histograma (escala log) da variável mais assimétrica: bruto x final.

    Escolhe, entre as variáveis que existem na base bruta, a de maior
    assimetria. Mostra que a forma da distribuição **não muda** (só saem as
    linhas filtradas em D01 e D02): os valores extremos foram mantidos e apenas sinalizados.

    Args:
        bruto: Ocorrências brutas.
        final: Base tratada.
        limites: Saída de ``outliers.calcular_limites_iqr``.
        caminho: Arquivo PNG de saída.

    Returns:
        Caminho gravado.
    """
    candidatas = limites[limites["variavel"].isin(bruto.columns)]
    r = candidatas.sort_values("assimetria", ascending=False).iloc[0]
    var = r["variavel"]
    maximo = float(bruto[var].max())
    bins = np.linspace(0, maximo, 41) if maximo > 60 else np.arange(0, maximo + 2)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10.5, 4), sharey=True)
    for ax, base, titulo, cor in (
        (a1, bruto, f"Antes ({fmt_int(len(bruto))} linhas)", CINZA),
        (a2, final, f"Depois ({fmt_int(len(final))} linhas)", AZUL),
    ):
        ax.hist(base[var], bins=bins, color=cor, align="mid" if maximo > 60 else "left", rwidth=0.85)
        ax.axvline(r["limite_superior"], color=LARANJA, linestyle="--", linewidth=1.5)
        ax.set_yscale("log")
        ax.set_xlabel(var)
        ax.set_title(titulo, fontsize=10, color=TEXTO_SECUNDARIO, loc="left")
        _estilizar(ax)
    a1.set_ylabel("Nº de acidentes (escala log)")
    a2.text(
        r["limite_superior"] + 0.04 * maximo, a2.get_ylim()[1] * 0.35,
        "limite do IQR\n(acima = outlier)", color=LARANJA, fontsize=9, va="top",
    )  # fmt: skip
    _titulo(
        fig,
        f"'{var}' tem cauda longa (assimetria {fmt_num(r['assimetria'], 1)}, máximo {fmt_int(maximo)}): "
        "os valores extremos continuam na base, apenas sinalizados",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    return _salvar(fig, caminho)


def figura_causas(bruto: pd.DataFrame, final: pd.DataFrame, caminho: Path) -> Path:
    """Antes x depois da causa do acidente: 69 causas x grupos de causa.

    Args:
        bruto: Ocorrências brutas (com ``causa_acidente``).
        final: Base tratada (com ``grupo_causa``).
        caminho: Arquivo PNG de saída.

    Returns:
        Caminho gravado.
    """
    causas = bruto["causa_acidente"].value_counts()
    grupos = final["grupo_causa"].value_counts()
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4.6), gridspec_kw={"width_ratios": [1.2, 1]})
    topo = causas.head(10)
    rotulos = [textwrap.shorten(str(c), 42, placeholder="...") for c in topo.index][::-1]
    a1.barh(rotulos, topo.to_numpy()[::-1], color=CINZA, height=0.6)
    a1.set_title(f"Antes: {fmt_int(len(causas))} causas distintas (10 mais comuns)", fontsize=10, color=TEXTO_SECUNDARIO, loc="left")
    a2.barh(grupos.index.astype(str)[::-1], grupos.to_numpy()[::-1], color=AZUL, height=0.6)
    a2.set_title(f"Depois: {fmt_int(len(grupos))} grupos de causa", fontsize=10, color=TEXTO_SECUNDARIO, loc="left")
    for ax in (a1, a2):
        ax.xaxis.set_major_formatter(lambda v, _: fmt_int(v))
        ax.set_xlabel("Nº de acidentes")
        _estilizar(ax)
    maior = grupos.index[0]
    _titulo(
        fig,
        f"{fmt_int(len(causas))} causas viraram {fmt_int(len(grupos))} grupos; "
        f"o mais comum é '{maior}' ({fmt_pct(100 * grupos.iloc[0] / grupos.sum(), 1)} dos acidentes)",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    return _salvar(fig, caminho)


def figura_taxa_fatal(taxas: pd.DataFrame, caminho: Path) -> Path:
    """Taxa de acidentes fatais por faixa horária e por grupo de causa (descritivo).

    É uma **pré-visualização** dos padrões que a análise exploratória vai
    testar, e não um teste estatístico. Serve para justificar os atributos F02 e F06.

    Args:
        taxas: Saída de :func:`tabela_taxa_fatal`.
        caminho: Arquivo PNG de saída.

    Returns:
        Caminho gravado.
    """
    geral = float(taxas.loc[taxas["dimensao"] == "geral", "taxa_fatal_pct"].iloc[0])
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4.4), gridspec_kw={"width_ratios": [1, 1.5]})
    for ax, dimensao, titulo in (
        (a1, "faixa_horaria", "Por faixa do dia"),
        (a2, "grupo_causa", "Por grupo de causa"),
    ):
        parte = taxas[taxas["dimensao"] == dimensao].sort_values("taxa_fatal_pct")
        cores = [LARANJA if v > geral else AZUL for v in parte["taxa_fatal_pct"]]
        ax.barh(parte["grupo"], parte["taxa_fatal_pct"], color=cores, height=0.6)
        for yi, r in enumerate(parte.itertuples()):
            ax.text(r.taxa_fatal_pct, yi, f" {fmt_pct(r.taxa_fatal_pct, 1)}", va="center", fontsize=9, color=TEXTO)
        ax.axvline(geral, color=CINZA, linestyle="--", linewidth=1.2)
        ax.set_xlim(0, parte["taxa_fatal_pct"].max() * 1.25)
        ax.set_xlabel("% de acidentes com morte")
        ax.set_title(titulo, fontsize=10, color=TEXTO_SECUNDARIO, loc="left")
        _estilizar(ax)
    faixas = taxas[taxas["dimensao"] == "faixa_horaria"].sort_values("taxa_fatal_pct")
    pior, melhor = faixas.iloc[-1], faixas.iloc[0]
    _titulo(
        fig,
        f"A taxa geral de acidentes fatais é {fmt_pct(geral, 1)} (linha cinza), mas varia muito: "
        f"{fmt_pct(pior['taxa_fatal_pct'], 1)} na faixa '{pior['grupo']}' contra {fmt_pct(melhor['taxa_fatal_pct'], 1)} na '{melhor['grupo']}'",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    return _salvar(fig, caminho)
