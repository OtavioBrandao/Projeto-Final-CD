"""Configuração central do tratamento de dados (Engenharia de Dados e Pré-processamento).

Aqui ficam **todas** as constantes do tratamento: caminhos, listas de colunas,
agrupamentos, limiares e rótulos. Nenhuma regra de limpeza "esconde" um número
solto no meio do código: se uma decisão depende de um valor, ele mora neste
arquivo, com um comentário dizendo de onde veio.

Base: acidentes em rodovias federais registrados pela PRF (Polícia Rodoviária
Federal), anos de 2024 e 2025. Os IDs (I01.., D01.., F01.., O01, S01) citados
nos comentários são os mesmos de ``docs/PLANO_IMPLEMENTACAO_ENGENHARIA_DE_DADOS.md``
e ``docs/README_pre_processamento.md``.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

# ---------------------------------------------------------------------------
# Caminhos
# ---------------------------------------------------------------------------

#: Raiz do repositório (duas pastas acima de ``src/p1_engenharia_dados``).
RAIZ_REPOSITORIO: Path = Path(__file__).resolve().parents[2]

#: Anos usados. 2024 e 2025 são anos completos; 2026 é parcial e a PRF atualiza
#: o arquivo todo mês, o que quebraria a reprodutibilidade.
ANOS: tuple[int, ...] = (2024, 2025)


@dataclass(frozen=True)
class Caminhos:
    """Agrupa todos os caminhos de entrada e saída do pipeline.

    Existe como classe (e não como constantes globais) para que os testes
    consigam rodar o pipeline em uma pasta temporária, sem sobrescrever os
    arquivos versionados do repositório.

    Attributes:
        raiz: Pasta-raiz onde as saídas (``data/``, ``docs/``) serão escritas.
        raw: Pasta com os CSVs brutos. Se ``None``, usa ``raiz/data/raw``.
    """

    raiz: Path = RAIZ_REPOSITORIO
    raw: Path | None = None

    @property
    def pasta_raw(self) -> Path:
        """Pasta dos dados brutos (não versionada)."""
        return self.raw if self.raw is not None else self.raiz / "data" / "raw"

    def csv_ocorrencias(self, ano: int) -> Path:
        """Arquivo de ocorrências (um acidente por linha) de um ano."""
        return self.pasta_raw / f"datatran{ano}.csv"

    def csv_pessoas(self, ano: int) -> Path:
        """Arquivo "agrupado por pessoa" (uma pessoa/veículo por linha) de um ano."""
        return self.pasta_raw / f"pessoas{ano}.csv"

    @property
    def interim(self) -> Path:
        """Pasta de dados intermediários (base legível, não versionada)."""
        return self.raiz / "data" / "interim"

    @property
    def processed(self) -> Path:
        """Pasta da base final para modelagem (versionada)."""
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
        """Saída: base legível, com ID e colunas de resultado (``data/interim``)."""
        return self.interim / "acidentes_prf_limpo.csv"

    @property
    def csv_modelagem(self) -> Path:
        """Saída: base final em CSV (``data/processed``)."""
        return self.processed / "acidentes_prf_modelagem.csv"

    @property
    def parquet_modelagem(self) -> Path:
        """Saída: base final em Parquet (``data/processed``)."""
        return self.processed / "acidentes_prf_modelagem.parquet"

    @property
    def csv_dicionario(self) -> Path:
        """Saída: dicionário de dados (``data/processed``)."""
        return self.processed / "dicionario_de_dados.csv"


# ---------------------------------------------------------------------------
# Reprodutibilidade
# ---------------------------------------------------------------------------

#: Semente fixa. O pipeline é determinístico (não sorteia nada), mas fixamos a
#: semente por segurança caso alguma etapa futura use aleatoriedade.
SEMENTE_ALEATORIA: int = 42

# ---------------------------------------------------------------------------
# Leitura dos CSVs da PRF
# ---------------------------------------------------------------------------

SEPARADOR_CSV: str = ";"  # os arquivos da PRF usam ponto e vírgula
CODIFICACAO_CSV: str = "latin1"  # e não vêm em UTF-8 (acentos quebram se lidos como UTF-8)
DECIMAL_CSV: str = ","  # latitude/longitude/km usam vírgula decimal

#: Textos que a PRF usa para "dado ausente" no arquivo. Só estes viram nulo.
#: ``keep_default_na=False`` evita que o pandas trate outros textos (como "None"
#: ou "NULL") como nulo sem querermos.
TOKENS_NULO_BRUTO: tuple[str, ...] = ("NA", "")

# ---------------------------------------------------------------------------
# Rótulos padronizados (ASCII, sem espaço, para evitar problemas de encoding)
# ---------------------------------------------------------------------------

ROTULO_NAO_INFORMADO: str = "Nao_informado"  # D04: Ignorado / Não Informado / NA / vazio
ROTULO_OUTRAS: str = "Outras"  # causa não mapeada em nenhum grupo

#: Rótulos que a PRF usa para "não sabemos". D04 unifica todos em ``Nao_informado``.
ROTULOS_SEM_INFORMACAO: frozenset[str] = frozenset({"Ignorado", "Não Informado", "Nao Informado"})

#: Colunas de texto que recebem a padronização D04 (aparece em qualquer uma).
COLUNAS_CATEGORICAS_PADRONIZAR: tuple[str, ...] = (
    "dia_semana", "causa_acidente", "tipo_acidente", "classificacao_acidente", "fase_dia",
    "sentido_via", "condicao_metereologica", "tipo_pista", "tracado_via", "uso_solo",
)  # fmt: skip

# ---------------------------------------------------------------------------
# Colunas
# ---------------------------------------------------------------------------

COLUNA_ID: str = "id"  # identificador do acidente (liga ocorrências e pessoas)

#: Colunas que descrevem a **consequência** do acidente. ``mortos`` define o
#: alvo e as demais só existem depois que o acidente aconteceu; usá-las como
#: atributo seria vazamento. Ficam no interim (rastreio) e saem da modelagem.
COLUNAS_RESULTADO: tuple[str, ...] = (
    "classificacao_acidente", "pessoas", "mortos", "feridos_leves", "feridos_graves",
    "ilesos", "ignorados", "feridos",
)  # fmt: skip

#: Colunas administrativas da PRF (quem atendeu); redundantes com ``uf`` e sem
#: relação com a causa do acidente.
COLUNAS_ADMINISTRATIVAS: tuple[str, ...] = ("regional", "delegacia", "uop")

#: Nome do alvo binário (F01).
COLUNA_ALVO: str = "acidente_fatal"

#: Colunas de contagem/medida analisadas quanto a outliers (O01).
COLUNAS_OUTLIER: tuple[str, ...] = (
    "veiculos", "km", "idade_media_condutores", "idade_media_veiculos",
)  # fmt: skip

# ---------------------------------------------------------------------------
# D02 — eventos fora do escopo
# ---------------------------------------------------------------------------

#: Causa/tipo que indicam evento intencional ou pessoal, não um acidente de
#: trânsito. Estão fora do escopo da decisão (segurança viária).
CAUSA_INTENCIONAL: str = "Suicídio (presumido)"
TIPO_PESSOAL: str = "Sinistro pessoal de trânsito"

# ---------------------------------------------------------------------------
# D05, D06, D07 — limites de plausibilidade
# ---------------------------------------------------------------------------

IDADE_MINIMA: int = 0
IDADE_MAXIMA: int = 110  # acima disso é erro de digitação (há valores como 2024)
#: Condutor com menos de 10 anos não existe; nesses registros a PRF usa ``0`` como "não informado".
IDADE_MINIMA_CONDUTOR: int = 10
#: Anos abaixo disso são "não informado": ``0`` e também ``1900``, que aparece como valor padrão.
ANO_FABRICACAO_MINIMO: int = 1950

#: Caixa que contém o território brasileiro (inclui ilhas oceânicas).
LATITUDE_MINIMA: float = -33.8
LATITUDE_MAXIMA: float = 5.3
LONGITUDE_MINIMA: float = -74.0
LONGITUDE_MAXIMA: float = -28.8

#: Latitude positiva só é plausível nestas UFs (parte do território fica ao norte da linha do Equador).
UFS_NORTE_DO_EQUADOR: frozenset[str] = frozenset({"AP", "RR", "AM", "PA"})

# ---------------------------------------------------------------------------
# F — engenharia de atributos
# ---------------------------------------------------------------------------

#: F02: faixas do dia (limite inferior inclusivo da hora, nome).
FAIXAS_HORARIAS: tuple[tuple[int, str], ...] = (
    (0, "Madrugada"), (6, "Manha"), (12, "Tarde"), (18, "Noite"),
)  # fmt: skip

#: F04: feriados nacionais fixos (mês, dia). Móveis são calculados pela Páscoa.
FERIADOS_FIXOS: tuple[tuple[int, int], ...] = (
    (1, 1), (4, 21), (5, 1), (9, 7), (10, 12), (11, 2), (11, 15), (12, 25),
)  # fmt: skip
#: 20/11 (Consciência Negra) é feriado nacional a partir de 2024 (Lei 14.759/2023).
FERIADO_CONSCIENCIA_NEGRA: tuple[int, int] = (11, 20)
#: Dias em relação à Páscoa: Carnaval (segunda e terça), Sexta-feira Santa, Corpus Christi.
FERIADOS_MOVEIS_DIAS_APOS_PASCOA: tuple[int, ...] = (-48, -47, -2, 60)

#: F05: região do Brasil de cada UF (IBGE).
REGIAO_POR_UF: dict[str, str] = {
    **dict.fromkeys(["AC", "AM", "AP", "PA", "RO", "RR", "TO"], "Norte"),
    **dict.fromkeys(["AL", "BA", "CE", "MA", "PB", "PE", "PI", "RN", "SE"], "Nordeste"),
    **dict.fromkeys(["DF", "GO", "MT", "MS"], "Centro-Oeste"),
    **dict.fromkeys(["ES", "MG", "RJ", "SP"], "Sudeste"),
    **dict.fromkeys(["PR", "RS", "SC"], "Sul"),
}

#: F07: traçado da via. Cada valor de ``tracado_via`` (separados por ";") vira
#: uma coluna 0/1. ``{texto na base: nome da coluna}``.
FLAGS_TRACADO: dict[str, str] = {
    "Reta": "via_reta",
    "Curva": "via_curva",
    "Aclive": "via_aclive",
    "Declive": "via_declive",
    "Interseção de Vias": "via_intersecao",
    "Rotatória": "via_rotatoria_retorno",
    "Retorno Regulamentado": "via_rotatoria_retorno",
    "Em Obras": "via_obras_desvio",
    "Desvio Temporário": "via_obras_desvio",
    "Ponte": "via_ponte_viaduto_tunel",
    "Viaduto": "via_ponte_viaduto_tunel",
    "Túnel": "via_ponte_viaduto_tunel",
}

#: F10: tamanho (km) da faixa que define um "trecho" (BR + faixa de km).
KM_POR_TRECHO: int = 10

#: F11: grupos de tipo de veículo (``tipo_veiculo`` do arquivo por pessoa).
VEICULOS_MOTO: frozenset[str] = frozenset(
    {"Motocicleta", "Motoneta", "Ciclomotor", "Triciclo", "Quadriciclo"}
)
VEICULOS_CAMINHAO: frozenset[str] = frozenset({"Caminhão", "Caminhão-trator"})
VEICULOS_ONIBUS: frozenset[str] = frozenset({"Ônibus", "Micro-ônibus"})
VEICULOS_BICICLETA: frozenset[str] = frozenset({"Bicicleta"})

#: F06: agrupamento das 69 causas em grupos que um gestor entende.
#: Causa que não estiver aqui vai para ``Outras`` (e o pipeline avisa).
GRUPOS_CAUSA: dict[str, list[str]] = {
    "Atencao_e_reacao_do_condutor": [
        "Ausência de reação do condutor", "Reação tardia ou ineficiente do condutor",
        "Condutor Dormindo", "Mal súbito do condutor", "Condutor usando celular",
        "Transtornos Mentais (exceto suicidio)",
    ],
    "Velocidade_e_distancia": [
        "Velocidade Incompatível", "Condutor deixou de manter distância do veículo da frente",
        "Participar de racha",
    ],
    "Alcool_e_substancias": [
        "Ingestão de álcool pelo condutor", "Ingestão de substâncias psicoativas pelo condutor",
        "Pedestre - Ingestão de álcool/ substâncias psicoativas",
    ],
    "Manobra_e_infracao": [
        "Acessar a via sem observar a presença dos outros veículos", "Manobra de mudança de faixa",
        "Transitar na contramão", "Ultrapassagem Indevida",
        "Trafegar com motocicleta (ou similar) entre as faixas",
        "Desrespeitar a preferência no cruzamento", "Conversão proibida", "Acesso irregular",
        "Transitar no Acostamento", "Frear bruscamente", "Retorno proibido",
        "Condutor desrespeitou a iluminação vermelha do semáforo",
        "Estacionar ou parar em local proibido",
        "Deixar de acionar o farol da motocicleta (ou similar)", "Modificação proibida",
        "Transitar na calçada",
    ],
    "Pedestre": [
        "Entrada inopinada do pedestre", "Pedestre andava na pista",
        "Pedestre cruzava a pista fora da faixa",
        "Área urbana sem a presença de local apropriado para a travessia de pedestres",
    ],
    "Defeito_mecanico_ou_carga": [
        "Demais falhas mecânicas ou elétricas", "Avarias e/ou desgaste excessivo no pneu",
        "Problema com o freio", "Problema na suspensão", "Faróis desregulados",
        "Carga excessiva e/ou mal acondicionada",
    ],
    "Condicao_da_via": [
        "Pista Escorregadia", "Acumulo de água sobre o pavimento",
        "Acumulo de areia ou detritos sobre o pavimento", "Acumulo de óleo sobre o pavimento",
        "Pista esburacada", "Curva acentuada", "Ausência de sinalização",
        "Afundamento ou ondulação no pavimento",
        "Deficiência do Sistema de Iluminação/Sinalização", "Acostamento em desnível",
        "Iluminação deficiente", "Demais falhas na via",
        "Falta de elemento de contenção que evite a saída do leito carroçável",
        "Falta de acostamento", "Sinalização mal posicionada",
        "Restrição de visibilidade em curvas horizontais", "Restrição de visibilidade em curvas verticais",
        "Declive acentuado", "Faixas de trânsito com largura insuficiente", "Semáforo com defeito",
        "Redutor de velocidade em desacordo", "Sistema de drenagem ineficiente",
        "Sinalização encoberta", "Objeto estático sobre o leito carroçável", "Desvio temporário",
    ],
    "Clima_e_ambiente": ["Chuva", "Neblina", "Fumaça", "Demais Fenômenos da natureza"],
    "Animais_na_pista": ["Animais na Pista"],
}  # fmt: skip

# ---------------------------------------------------------------------------
# Seleção de atributos
# ---------------------------------------------------------------------------

#: Fator do IQR (regra clássica de Tukey): fora de [Q1 - 1,5*IQR, Q3 + 1,5*IQR].
FATOR_IQR: float = 1.5

#: Coluna é "quase constante" se um único valor aparece em pelo menos esta fração.
LIMIAR_DOMINANCIA: float = 0.999

#: Duas numéricas são redundantes se |correlação de Pearson| passa deste valor.
LIMIAR_CORRELACAO: float = 0.9

#: Casas decimais das médias/proporções criadas; fixas para a saída ser idêntica sempre.
CASAS_DECIMAIS: int = 4
