"""Configuração central do pipeline da Pessoa 1.

Aqui ficam **todas** as constantes do tratamento: caminhos, listas de colunas,
códigos de alta, limiares e rótulos. Nenhuma regra de limpeza "esconde" um
número solto no meio do código: se uma decisão depende de um valor, ele mora
neste arquivo, com um comentário dizendo de onde veio.

Os IDs (D01..D09 e F01..F11) citados nos comentários são os mesmos de
``docs/PLANO_IMPLEMENTACAO_PESSOA1.md`` e ``docs/README.md``.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

# ---------------------------------------------------------------------------
# Caminhos
# ---------------------------------------------------------------------------

#: Raiz do repositório (duas pastas acima de ``src/p1_engenharia_dados``).
RAIZ_REPOSITORIO: Path = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class Caminhos:
    """Agrupa todos os caminhos de entrada e saída do pipeline.

    Existe como classe (e não como constantes globais) para que os testes
    consigam rodar o pipeline em uma pasta temporária, sem sobrescrever os
    arquivos versionados do repositório.

    Attributes:
        raiz: Pasta-raiz onde as saídas (``data/``, ``docs/``) serão escritas.
        raw: Pasta com ``diabetic_data.csv`` e ``IDS_mapping.csv``. Se ``None``,
            usa ``raiz/data/raw``.
    """

    raiz: Path = RAIZ_REPOSITORIO
    raw: Path | None = None

    @property
    def pasta_raw(self) -> Path:
        """Pasta dos dados brutos (não versionada)."""
        return self.raw if self.raw is not None else self.raiz / "data" / "raw"

    @property
    def csv_bruto(self) -> Path:
        """Arquivo ``diabetic_data.csv``."""
        return self.pasta_raw / "diabetic_data.csv"

    @property
    def csv_mapeamento(self) -> Path:
        """Arquivo ``IDS_mapping.csv`` (três tabelas de códigos)."""
        return self.pasta_raw / "IDS_mapping.csv"

    @property
    def interim(self) -> Path:
        """Pasta de dados intermediários (base legível)."""
        return self.raiz / "data" / "interim"

    @property
    def processed(self) -> Path:
        """Pasta da base final para modelagem."""
        return self.raiz / "data" / "processed"

    @property
    def docs(self) -> Path:
        """Pasta da documentação."""
        return self.raiz / "docs"

    @property
    def tabelas(self) -> Path:
        """Tabelas de auditoria geradas pelo código (CSV)."""
        return self.docs / "tabelas"

    @property
    def figuras(self) -> Path:
        """Figuras geradas pelo código (PNG)."""
        return self.docs / "figuras"

    @property
    def csv_limpo(self) -> Path:
        """Saída: base legível, com IDs (``data/interim``)."""
        return self.interim / "diabetes_limpo.csv"

    @property
    def csv_modelagem(self) -> Path:
        """Saída: base final em CSV (``data/processed``)."""
        return self.processed / "diabetes_modelagem.csv"

    @property
    def parquet_modelagem(self) -> Path:
        """Saída: base final em Parquet (``data/processed``)."""
        return self.processed / "diabetes_modelagem.parquet"

    @property
    def csv_dicionario(self) -> Path:
        """Saída: dicionário de dados (``data/processed``)."""
        return self.processed / "dicionario_de_dados.csv"


# ---------------------------------------------------------------------------
# Reprodutibilidade
# ---------------------------------------------------------------------------

#: Semente fixa. O pipeline é determinístico (não sorteia nada), mas fixamos
#: a semente por segurança caso alguma etapa futura use aleatoriedade.
SEMENTE_ALEATORIA: int = 42

# ---------------------------------------------------------------------------
# Leitura do CSV bruto
# ---------------------------------------------------------------------------

#: A base usa ``?`` para dado ausente (descrito na página do UCI).
VALOR_NULO_BRUTO: str = "?"

# ARMADILHA IMPORTANTE: o pandas, por padrão, trata o texto "None" como nulo.
# Em ``max_glu_serum`` e ``A1Cresult`` "None" significa "exame NÃO realizado",
# que é uma informação (D09), não um dado faltando. Por isso a leitura usa
# ``keep_default_na=False`` e só ``?`` vira nulo (ver ``carregar.py``).

# ---------------------------------------------------------------------------
# Colunas
# ---------------------------------------------------------------------------

#: Identificadores: ficam no interim (rastreio) e saem da base de modelagem.
COLUNAS_ID: tuple[str, str] = ("encounter_id", "patient_nbr")

#: As 23 colunas de medicamentos. Valores: No / Steady / Up / Down.
COLUNAS_MEDICAMENTOS: tuple[str, ...] = (
    "metformin", "repaglinide", "nateglinide", "chlorpropamide", "glimepiride",
    "acetohexamide", "glipizide", "glyburide", "tolbutamide", "pioglitazone",
    "rosiglitazone", "acarbose", "miglitol", "troglitazone", "tolazamide",
    "examide", "citoglipton", "insulin", "glyburide-metformin",
    "glipizide-metformin", "glimepiride-pioglitazone", "metformin-rosiglitazone",
    "metformin-pioglitazone",
)  # fmt: skip

#: Colunas de contagem analisadas quanto a outliers (plano, seção 4.6).
COLUNAS_OUTLIER: tuple[str, ...] = (
    "time_in_hospital", "num_lab_procedures", "num_procedures", "num_medications",
    "number_outpatient", "number_emergency", "number_inpatient", "number_diagnoses",
)  # fmt: skip

#: Colunas de diagnóstico (códigos CID-9).
COLUNAS_DIAGNOSTICO: tuple[str, str, str] = ("diag_1", "diag_2", "diag_3")

# ---------------------------------------------------------------------------
# Rótulos padronizados (todos ASCII, sem espaço, para evitar problemas de encoding)
# ---------------------------------------------------------------------------

ROTULO_NAO_INFORMADO: str = "Nao_informado"  # D04: NULL / Not Available / Not Mapped
ROTULO_DESCONHECIDO: str = "Desconhecido"  # D06, D07: valor ausente em colunas categóricas
ROTULO_SEM_DIAGNOSTICO: str = "Sem_diagnostico"  # D08: diagnóstico secundário vazio
ROTULO_NAO_MEDIDO: str = "Nao_medido"  # D09: exame não realizado
ROTULO_OUTRAS: str = "Outras"  # F11: especialidades raras

#: Descrições do IDS_mapping que significam "não sabemos" (comparação em minúsculas).
DESCRICOES_NAO_INFORMADO: frozenset[str] = frozenset(
    {"null", "not available", "not mapped", "unknown/invalid"}
)

#: Erros de digitação encontrados no IDS_mapping.csv (D04). Chave = erro, valor = correção.
CORRECOES_TEXTO: dict[str, str] = {"critial": "critical"}

#: D04: categorias iguais escritas de formas diferentes dentro dos dados.
#: ``readmitted`` usa "NO" (maiúsculo) enquanto todo o resto da base usa "No".
PADRONIZACAO_CATEGORIAS: dict[str, dict[str, str]] = {"readmitted": {"NO": "No"}}

# ---------------------------------------------------------------------------
# D01 / D02 — filtros de linhas
# ---------------------------------------------------------------------------

#: D01: valor de ``gender`` a remover.
GENERO_INVALIDO: str = "Unknown/Invalid"

#: D02: códigos de alta de óbito/hospice **segundo o plano**. O pipeline
#: identifica os códigos lendo o IDS_mapping (``PADRAO_OBITO_HOSPICE``) e
#: compara com esta lista; se divergirem, vale o que está nos dados e a
#: diferença é registrada nas métricas.
CODIGOS_ALTA_OBITO_HOSPICE_PLANO: frozenset[int] = frozenset({11, 13, 14, 19, 20, 21})

#: Expressão que identifica, na descrição do IDS_mapping, alta por óbito ou hospice.
PADRAO_OBITO_HOSPICE: str = r"expired|hospice"

# ---------------------------------------------------------------------------
# F01–F11 — engenharia de atributos
# ---------------------------------------------------------------------------

#: F01: valor de ``readmitted`` que define a classe positiva do alvo.
CLASSE_POSITIVA_READMITIDO: str = "<30"

#: F05: ``usuario_frequente`` = 1 se ``number_inpatient`` >= este valor.
#: Definição clínica ("já teve ao menos uma internação no ano anterior"),
#: confirmada nos dados: mediana = 0 e 3º quartil = 1; a taxa de readmissão
#: cresce a partir de 1 (ver ``docs/tabelas/limiar_usuario_frequente.csv``).
#: NÃO foi escolhido olhando o alvo, para não vazar informação.
LIMIAR_USUARIO_FREQUENTE: int = 1

#: F09: agrupamento do destino de alta (códigos do IDS_mapping).
CODIGOS_ALTA_CASA: frozenset[int] = frozenset({1, 6, 8})
CODIGOS_ALTA_TRANSFERENCIA: frozenset[int] = frozenset(
    {2, 3, 4, 5, 9, 10, 15, 16, 17, 22, 23, 24, 27, 28, 29, 30}
)
# Qualquer outro código bem mapeado (ex.: 7 = saiu contra orientação médica,
# 12 = ainda paciente) vai para "Outros"; códigos NULL/Not Mapped viram "Nao_informado".

#: F11: quantas especialidades mais frequentes manter (o resto vira "Outras").
TOP_N_ESPECIALIDADES: int = 10

#: F03: nomes dos grupos de diagnóstico (agrupamento de Strack et al., 2014).
GRUPO_CIRCULATORIO: str = "Circulatorio"
GRUPO_RESPIRATORIO: str = "Respiratorio"
GRUPO_DIGESTIVO: str = "Digestivo"
GRUPO_DIABETES: str = "Diabetes"
GRUPO_LESOES: str = "Lesoes"
GRUPO_MUSCULOESQUELETICO: str = "Musculoesqueletico"
GRUPO_GENITURINARIO: str = "Geniturinario"
GRUPO_NEOPLASIAS: str = "Neoplasias"
GRUPO_OUTROS: str = "Outros"

# ---------------------------------------------------------------------------
# Outliers e seleção
# ---------------------------------------------------------------------------

#: Fator do IQR (regra clássica de Tukey): fora de [Q1 - 1,5*IQR, Q3 + 1,5*IQR].
FATOR_IQR: float = 1.5

#: Coluna é "quase constante" se um único valor aparece em pelo menos esta fração.
LIMIAR_DOMINANCIA: float = 0.999

#: Duas numéricas são redundantes se |correlação de Pearson| passa deste valor.
LIMIAR_CORRELACAO: float = 0.9

#: Nome do alvo binário (F01).
COLUNA_ALVO: str = "readmit_30d"

#: Casas decimais das razões criadas (F10); fixas para a saída ser idêntica sempre.
CASAS_DECIMAIS_RAZOES: int = 4
