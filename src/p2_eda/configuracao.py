"""Configuração central da Análise Exploratória (EDA) e Estatística (Pessoa 2).

Mesma filosofia da Engenharia de Dados (``src/p1_engenharia_dados/configuracao.py``):
**todo número que muda uma conclusão mora aqui**, com o motivo ao lado. Assim a
banca pode perguntar "por que 5%?" ou "por que essas faixas?" e a resposta está
num lugar só.

Entrada: ``data/processed/acidentes_prf_modelagem.csv`` (versionado pela Pessoa 1).
Saídas: ``reports/figuras/p2_*.png``, ``reports/tabelas/p2_*.csv`` e os textos
gerados em ``reports/`` (README da EDA e seção do artigo em Markdown e LaTeX).

Os IDs ``H1..H9`` são as hipóteses testadas (ver ``HIPOTESES`` abaixo); eles
aparecem nas tabelas, nas figuras e no texto do artigo com o mesmo código.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

# ---------------------------------------------------------------------------
# Caminhos
# ---------------------------------------------------------------------------

#: Raiz do repositório (duas pastas acima de ``src/p2_eda``).
RAIZ_REPOSITORIO: Path = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class Caminhos:

    raiz: Path = RAIZ_REPOSITORIO
    entrada: Path | None = None

    @property
    def csv_modelagem(self) -> Path:
        if self.entrada is not None:
            return self.entrada
        return RAIZ_REPOSITORIO / "data" / "processed" / "acidentes_prf_modelagem.csv"

    @property
    def reports(self) -> Path:
        return self.raiz / "reports"

    @property
    def figuras(self) -> Path:
        return self.reports / "figuras"

    @property
    def tabelas(self) -> Path:
        return self.reports / "tabelas"

    @property
    def readme_eda(self) -> Path:
        return self.reports / "README_eda.md"

    @property
    def secao_md(self) -> Path:
        return self.reports / "secao_estatistica_descritiva_inferencia.md"

    @property
    def secao_tex(self) -> Path:
        return self.reports / "secao_estatistica_descritiva_inferencia.tex"


# ---------------------------------------------------------------------------
# Colunas
# ---------------------------------------------------------------------------

COLUNA_ALVO: str = "acidente_fatal"
COLUNA_ANO: str = "ano"

#: Numéricas contínuas/contagens descritas com média, mediana, dispersão e forma.
#: ``mes`` e ``hora`` são numéricas no arquivo, mas aqui são tratadas como tempo
#: (linha e mapa de calor), não como medida.
COLUNAS_NUMERICAS: tuple[str, ...] = (
    "km", "veiculos", "idade_media_condutores", "pct_condutores_homens", "idade_media_veiculos",
)  # fmt: skip

#: Numéricas comparadas entre fatais e não fatais (Mann-Whitney, H8). ``km`` fica
#: de fora: é posição na rodovia, e km 100 da BR-101 nada tem a ver com km 100 da BR-116.
COLUNAS_NUMERICAS_TESTE: tuple[str, ...] = (
    "veiculos", "idade_media_condutores", "idade_media_veiculos", "pct_condutores_homens",
)  # fmt: skip

#: Nome legível de cada coluna para figuras e textos.
ROTULOS: dict[str, str] = {
    "km": "Quilômetro da rodovia",
    "veiculos": "Veículos envolvidos",
    "idade_media_condutores": "Idade média dos condutores (anos)",
    "pct_condutores_homens": "% de condutores homens",
    "idade_media_veiculos": "Idade média dos veículos (anos)",
    "hora": "Hora",
    "mes": "Mês",
    "faixa_horaria": "Faixa horária",
    "fase_dia": "Fase do dia",
    "tipo_pista": "Tipo de pista",
    "regiao": "Região",
    "uf": "UF",
    "grupo_causa": "Grupo de causa",
    "tipo_acidente": "Tipo de acidente",
    "condicao_metereologica": "Condição do tempo",
    "dia_semana": "Dia da semana",
}

# ---------------------------------------------------------------------------
# Estatística
# ---------------------------------------------------------------------------

#: Nível de significância. 5% é o padrão da área; com a correção de Holm abaixo,
#: ele vale para a **família** de testes, não para cada teste isolado.
ALFA: float = 0.05

#: Nível de confiança dos intervalos (IC 95%), coerente com ``ALFA``.
CONFIANCA: float = 0.95

#: Faixas de Cohen (1988) para interpretar tamanho de efeito de V de Cramér/phi
#: (gl = 1) e da correlação ponto-bisserial/rank-bisserial: pequeno, médio, grande.
#: Com 145 mil linhas **quase tudo dá p < 0,05**; por isso toda conclusão é
#: tirada do tamanho do efeito, e não só do p-valor.
#: Como o alvo é binário, min(linhas-1, colunas-1) = 1 em toda tabela e os mesmos
#: limiares valem para qualquer variável, tenha ela 2 ou 27 categorias.
LIMIARES_EFEITO: tuple[float, float, float] = (0.10, 0.30, 0.50)

#: Mínimo de acidentes para um grupo aparecer em ranking (evita taxa instável:
#: com 2 acidentes de neve, 0% não quer dizer nada).
N_MINIMO_GRUPO: int = 200

#: Semente fixa. Só é usada na amostra do teste de normalidade (D'Agostino), que
#: com 145 mil linhas rejeitaria qualquer coisa; ver ``estatistica.normalidade``.
SEMENTE_ALEATORIA: int = 42
TAMANHO_AMOSTRA_NORMALIDADE: int = 5000

#: Número de quantis usados para discretizar numéricas no ranking de associação
#: com o alvo (permite comparar numéricas e categóricas na mesma régua: V de Cramér).
QUANTIS_RANKING: int = 5

# ---------------------------------------------------------------------------
# Fatores de risco (comparações 2x2 com risco relativo)
# ---------------------------------------------------------------------------

#: Cada fator é ``(id_hipotese, nome legível, coluna, valores "expostos", valores "referência")``.
#: ``None`` na referência = "todo o resto". Ordem = ordem nas tabelas.
FATORES_RISCO: tuple[tuple[str, str, str, tuple[object, ...], tuple[object, ...] | None], ...] = (
    ("H3", "Pedestre envolvido", "envolve_pedestre", (1,), (0,)),
    ("H3", "Caminhão envolvido", "envolve_caminhao", (1,), (0,)),
    ("H3", "Bicicleta envolvida", "envolve_bicicleta", (1,), (0,)),
    ("H3", "Ônibus envolvido", "envolve_onibus", (1,), (0,)),
    ("H3", "Motocicleta envolvida", "envolve_moto", (1,), (0,)),
    ("H2", "Pista simples (vs. dupla/múltipla)", "tipo_pista", ("Simples",), ("Dupla", "Múltipla")),
    ("H2", "Trecho rural (vs. urbano)", "area_urbana", (0,), (1,)),
    ("H1", "Madrugada ou noite (vs. 6h-17h)", "faixa_horaria", ("Madrugada", "Noite"), ("Manha", "Tarde")),
    ("H4", "Norte ou Nordeste (vs. Sul/Sudeste)", "regiao", ("Norte", "Nordeste"), ("Sul", "Sudeste")),
    ("H5", "Fim de semana", "fim_de_semana", (1,), (0,)),
    ("H5", "Feriado nacional", "eh_feriado", (1,), (0,)),
    ("H2", "Curva (vs. sem curva)", "via_curva", (1,), (0,)),
    ("H6", "Chuva, garoa ou neblina (vs. tempo seco)", "condicao_metereologica",
     ("Chuva", "Garoa/Chuvisco", "Nevoeiro/Neblina"), ("Céu Claro", "Sol", "Nublado")),
)  # fmt: skip

#: Variáveis categóricas testadas com qui-quadrado de independência (variável x alvo).
#: ``(id_hipotese, coluna)``.
TESTES_QUI_QUADRADO: tuple[tuple[str, str], ...] = (
    ("H1", "faixa_horaria"),
    ("H1", "dia_semana"),
    ("H2", "tipo_pista"),
    ("H4", "regiao"),
    ("H4", "uf"),
    ("H6", "condicao_metereologica"),
    ("H7", "grupo_causa"),
    ("H7", "tipo_acidente"),
    ("H9", "mes"),
)  # fmt: skip

#: Análises estratificadas (Cochran-Mantel-Haenszel): o efeito do fator continua
#: quando comparamos só acidentes do mesmo estrato? Responde "é causa raiz ou é
#: confusão?". ``(id, nome, coluna fator, expostos, referência, coluna estrato)``.
ANALISES_ESTRATIFICADAS: tuple[tuple[str, str, str, tuple[object, ...], tuple[object, ...], str], ...] = (
    ("H1", "Noite/madrugada, controlando o tipo de pista", "faixa_horaria",
     ("Madrugada", "Noite"), ("Manha", "Tarde"), "tipo_pista"),
    ("H1", "Noite/madrugada, controlando trecho urbano/rural", "faixa_horaria",
     ("Madrugada", "Noite"), ("Manha", "Tarde"), "area_urbana"),
    ("H2", "Pista simples, controlando trecho urbano/rural", "tipo_pista",
     ("Simples",), ("Dupla", "Múltipla"), "area_urbana"),
    ("H4", "Norte/Nordeste, controlando o tipo de pista", "regiao",
     ("Norte", "Nordeste"), ("Sul", "Sudeste"), "tipo_pista"),
)  # fmt: skip

#: Texto curto de cada hipótese (vai para as tabelas e para o artigo).
HIPOTESES: dict[str, str] = {
    "H1": "Acidentes de madrugada e à noite são mais letais do que os de dia.",
    "H2": "Pista simples, trecho rural e curva elevam a letalidade.",
    "H3": "Pedestres, ciclistas, motociclistas e veículos pesados elevam a letalidade.",
    "H4": "Norte e Nordeste têm letalidade maior do que Sul e Sudeste.",
    "H5": "Fins de semana e feriados são mais letais.",
    "H6": "Tempo ruim (chuva, neblina) eleva a letalidade.",
    "H7": "A causa e o tipo do acidente mudam muito a letalidade.",
    "H8": "Idade dos condutores, idade dos veículos e nº de veículos diferem entre fatais e não fatais.",
    "H9": "Há sazonalidade mensal na letalidade.",
}

# ---------------------------------------------------------------------------
# Tempo
# ---------------------------------------------------------------------------

ORDEM_DIA_SEMANA: tuple[str, ...] = (
    "segunda-feira", "terça-feira", "quarta-feira", "quinta-feira", "sexta-feira", "sábado", "domingo",
)  # fmt: skip
ABREV_DIA_SEMANA: dict[str, str] = {
    "segunda-feira": "Seg", "terça-feira": "Ter", "quarta-feira": "Qua", "quinta-feira": "Qui",
    "sexta-feira": "Sex", "sábado": "Sáb", "domingo": "Dom",
}  # fmt: skip
ORDEM_FAIXA_HORARIA: tuple[str, ...] = ("Madrugada", "Manha", "Tarde", "Noite")
ABREV_MES: tuple[str, ...] = ("Jan", "Fev", "Mar", "Abr", "Mai", "Jun", "Jul", "Ago", "Set", "Out", "Nov", "Dez")

#: UF destacada nas figuras (a disciplina é na UFAL; a banca vai perguntar de Alagoas).
UF_DESTAQUE: str = "AL"

# ---------------------------------------------------------------------------
# Figuras
# ---------------------------------------------------------------------------

#: Mesma paleta e convenção da Pessoa 1 (azul = normal, laranja = atenção/risco,
#: cinza = apoio). As duas cores passaram no validador de paleta (separação
#: para daltonismo ΔE 24,7 e contraste >= 3:1 sobre fundo claro).
AZUL: str = "#2a78d6"
LARANJA: str = "#eb6834"
CINZA: str = "#8a8985"
CINZA_CLARO: str = "#e5e4e0"
TEXTO: str = "#0b0b0b"
TEXTO_SECUNDARIO: str = "#52514e"
#: Rampa sequencial de um só tom (laranja claro -> escuro) para o mapa de calor.
RAMPA_SEQUENCIAL: tuple[str, ...] = ("#fdf0ea", "#f7c3a9", "#f09367", "#eb6834", "#b8461c", "#7a2a0d")
DPI_FIGURAS: int = 160

#: Figuras "principais" (artigo e slides) e "de apoio" (README da EDA / apêndice).
#: O artigo tem 12 páginas no total: a seção de estatística deve caber em ~2.
FIGURAS_PRINCIPAIS: tuple[str, ...] = (
    "p2_fig1_risco_relativo", "p2_fig2_hora_do_dia", "p2_fig3_volume_vs_gravidade_causas",
    "p2_fig4_estratificada_noite_pista",
)  # fmt: skip
