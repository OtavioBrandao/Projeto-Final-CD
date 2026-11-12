"""Features, regras de dados e decisões da modelagem selecionadas em 2024."""

from pathlib import Path
from types import MappingProxyType

RAIZ_REPOSITORIO = Path(__file__).resolve().parents[2]
CSV_MODELAGEM = RAIZ_REPOSITORIO / "data/processed/acidentes_prf_modelagem.csv"
COLUNA_ALVO = "acidente_fatal"
ANO_DESENVOLVIMENTO = 2024
ANO_TESTE_FINAL = 2025
LIMIARES = MappingProxyType({"A": 0.2066718339920044, "B": 0.101788230240345})

# Listas deliberadamente explícitas. B não utiliza estados da via apenas
# observados no boletim. Idade/sexo dos condutores ficam para análise de equidade.
FEATURES_A = (
    "dia_semana", "uf", "br", "km", "tipo_acidente", "fase_dia",
    "sentido_via", "condicao_metereologica", "tipo_pista", "veiculos",
    "latitude", "longitude", "envolve_moto", "envolve_caminhao",
    "envolve_onibus", "envolve_bicicleta", "envolve_pedestre",
    "idade_media_veiculos", "hora", "faixa_horaria", "mes", "fim_de_semana",
    "eh_feriado", "regiao", "grupo_causa", "via_reta", "via_curva",
    "via_aclive", "via_declive", "via_intersecao", "via_rotatoria_retorno",
    "via_obras_desvio", "via_ponte_viaduto_tunel", "area_urbana",
)
FEATURES_B = (
    "dia_semana", "uf", "br", "km", "latitude", "longitude", "hora",
    "faixa_horaria", "mes", "fim_de_semana", "eh_feriado", "regiao",
)
FEATURES_POR_CENARIO = {"A": FEATURES_A, "B": FEATURES_B}

OUTLIERS_GLOBAIS = (
    "veiculos_outlier", "km_outlier", "idade_media_condutores_outlier",
    "idade_media_veiculos_outlier",
)
COLUNAS_SENSIVEIS = ("idade_media_condutores", "pct_condutores_homens")
CONSEQUENCIAS_PROIBIDAS = (
    "mortos", "feridos", "feridos_leves", "feridos_graves", "ilesos",
    "ignorados", "pessoas", "classificacao_acidente", "estado_fisico",
    "contagens_inconsistentes",
)
IDENTIFICADORES_PROIBIDOS = ("id", "id_veiculo", "ano_arquivo", "regional", "delegacia", "uop")
PREDITORES_PROIBIDOS = frozenset((
    COLUNA_ALVO, "ano", *OUTLIERS_GLOBAIS, *COLUNAS_SENSIVEIS,
    *CONSEQUENCIAS_PROIBIDAS, *IDENTIFICADORES_PROIBIDOS,
))

# O contrato cobre também colunas excluídas: uma adição inesperada nunca é
# silenciosamente aceita ou automaticamente promovida a preditor.
COLUNAS_ESPERADAS = frozenset((
    *FEATURES_A, *COLUNAS_SENSIVEIS, *OUTLIERS_GLOBAIS, "ano", COLUNA_ALVO,
))
COLUNAS_CATEGORICAS = (
    "dia_semana", "uf", "br", "tipo_acidente", "fase_dia", "sentido_via",
    "condicao_metereologica", "tipo_pista", "faixa_horaria", "regiao", "grupo_causa",
)
COLUNAS_BINARIAS = (
    "envolve_moto", "envolve_caminhao", "envolve_onibus", "envolve_bicicleta",
    "envolve_pedestre", "fim_de_semana", "eh_feriado", "via_reta", "via_curva",
    "via_aclive", "via_declive", "via_intersecao", "via_rotatoria_retorno",
    "via_obras_desvio", "via_ponte_viaduto_tunel", "area_urbana",
    *OUTLIERS_GLOBAIS, COLUNA_ALVO,
)
COLUNAS_ACEITAM_NULOS = frozenset((
    "latitude", "longitude", "idade_media_condutores", "pct_condutores_homens",
    "idade_media_veiculos",
))

# (rodada, último mês de treino, primeiro e último mês de validação).
# Todos os limites são inclusivos. Treino sempre começa em janeiro de 2024.
CORTES_TEMPORAIS = ((1, 6, 7, 8), (2, 8, 9, 10), (3, 10, 11, 12))
