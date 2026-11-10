"""Testes do pipeline da Pessoa 1 (``pytest``).

Verificam as regras do plano (seção 5): sem ``?``, filtros D01–D03, alvo
correto, dicionário igual à base, funil de linhas fechando, reprodutibilidade
(mesmo hash ao rodar duas vezes) e que todo número do ``docs/README.md`` vem das
tabelas de ``docs/tabelas``.

Os testes rodam o pipeline em **pastas temporárias**, sem tocar nos arquivos
versionados. Precisam de ``data/raw`` (``python -m src.p1_engenharia_dados.baixar_base``).
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import pytest

from src.p1_engenharia_dados import carregar, configuracao as cfg, engenharia_atributos, executar_pipeline
from src.p1_engenharia_dados.configuracao import Caminhos
from src.p1_engenharia_dados.relatorios import fmt_int, fmt_num

RAIZ = Path(__file__).resolve().parents[1]
CSV_BRUTO = RAIZ / "data" / "raw" / "diabetic_data.csv"

pytestmark = pytest.mark.skipif(
    not CSV_BRUTO.exists(),
    reason="dados brutos ausentes: rode python -m src.p1_engenharia_dados.baixar_base",
)


def _sha256(caminho: Path) -> str:
    """Calcula o hash SHA-256 de um arquivo.

    Args:
        caminho: Arquivo a ler.

    Returns:
        Hash em hexadecimal.
    """
    return hashlib.sha256(caminho.read_bytes()).hexdigest()


@dataclass
class Execucao:
    """Resultado de uma execução do pipeline em pasta temporária.

    Attributes:
        caminhos: Caminhos usados.
        metricas: Métricas devolvidas pelo pipeline.
        limpo: Base legível (interim).
        modelagem: Base final.
        dicionario: Dicionário de dados.
    """

    caminhos: Caminhos
    metricas: dict[str, object]
    limpo: pd.DataFrame
    modelagem: pd.DataFrame
    dicionario: pd.DataFrame


def _executar(pasta: Path, documentacao: bool) -> Execucao:
    """Roda o pipeline escrevendo em ``pasta``.

    Args:
        pasta: Pasta temporária de saída.
        documentacao: Se ``True``, gera também tabelas, figuras e textos.

    Returns:
        Objeto com as saídas já carregadas.
    """
    caminhos = Caminhos(raiz=pasta, raw=RAIZ / "data" / "raw")
    metricas = executar_pipeline.executar(caminhos, gerar_documentacao=documentacao)
    return Execucao(
        caminhos=caminhos,
        metricas=metricas,
        limpo=pd.read_csv(caminhos.csv_limpo, keep_default_na=False, na_values=[""]),
        modelagem=pd.read_csv(caminhos.csv_modelagem, keep_default_na=False, na_values=[""]),
        dicionario=pd.read_csv(caminhos.csv_dicionario, keep_default_na=False),
    )


@pytest.fixture(scope="module")
def exec_a(tmp_path_factory: pytest.TempPathFactory) -> Execucao:
    """Primeira execução (com documentação)."""
    return _executar(tmp_path_factory.mktemp("saida_a"), documentacao=True)


@pytest.fixture(scope="module")
def exec_b(tmp_path_factory: pytest.TempPathFactory) -> Execucao:
    """Segunda execução, independente (só dados), para testar reprodutibilidade."""
    return _executar(tmp_path_factory.mktemp("saida_b"), documentacao=False)


@pytest.fixture(scope="module")
def bruto() -> pd.DataFrame:
    """Base bruta lida com a regra correta de nulos."""
    return carregar.carregar_dados_brutos(CSV_BRUTO)


# ---------------------------------------------------------------------------
# Carga
# ---------------------------------------------------------------------------


def test_bruto_tem_101766_linhas(bruto: pd.DataFrame) -> None:
    """A base baixada tem o tamanho documentado no UCI."""
    assert len(bruto) == 101_766


def test_none_dos_exames_nao_vira_nulo(bruto: pd.DataFrame) -> None:
    """``None`` em max_glu_serum/A1Cresult é 'exame não feito', não nulo (D09)."""
    assert bruto["max_glu_serum"].isna().sum() == 0
    assert (bruto["max_glu_serum"] == "None").sum() > 90_000
    assert bruto["A1Cresult"].isna().sum() == 0


def test_ids_mapping_tem_tres_tabelas_e_codigos_de_obito() -> None:
    """O IDS_mapping é lido como 3 tabelas e os códigos de óbito/hospice conferem com o plano."""
    from src.p1_engenharia_dados import limpeza

    mapas = carregar.carregar_ids_mapping(RAIZ / "data" / "raw" / "IDS_mapping.csv")
    assert set(mapas) == set(carregar.MAPEAMENTOS)
    codigos = limpeza.identificar_codigos_obito_hospice(mapas["discharge_disposition_id"])
    assert codigos == cfg.CODIGOS_ALTA_OBITO_HOSPICE_PLANO
    # NULL / Not Available / Not Mapped viram uma única categoria (D04).
    descricoes = set(mapas["admission_type_id"]["descricao"])
    assert cfg.ROTULO_NAO_INFORMADO in descricoes
    assert not descricoes & {"NULL", "Not Available", "Not Mapped"}


# ---------------------------------------------------------------------------
# Regras de limpeza (plano, seção 5)
# ---------------------------------------------------------------------------


def test_nenhum_interrogacao_restante(exec_a: Execucao) -> None:
    """Nenhum ``?`` sobra em nenhuma das bases."""
    for base in (exec_a.limpo, exec_a.modelagem):
        texto = base.select_dtypes(exclude="number")
        assert not (texto == "?").any().any()


def test_sem_nulos_onde_a_regra_manda_preencher(exec_a: Execucao) -> None:
    """Colunas D06–D09 não têm nulos; weight foi removida (D05)."""
    regras = ["payer_code", "medical_specialty", "race", "diag_1", "diag_2", "diag_3",
              "max_glu_serum", "A1Cresult"]  # fmt: skip
    assert exec_a.limpo[regras].isna().sum().sum() == 0
    assert "weight" not in exec_a.limpo.columns
    assert exec_a.modelagem.isna().sum().sum() == 0


def test_d01_sem_genero_invalido(exec_a: Execucao) -> None:
    """D01: nenhum ``Unknown/Invalid`` em gender."""
    assert not (exec_a.limpo["gender"] == cfg.GENERO_INVALIDO).any()


def test_d02_sem_obito_hospice(exec_a: Execucao) -> None:
    """D02: nenhum código de alta de óbito/hospice."""
    assert not exec_a.limpo["discharge_disposition_id"].isin(cfg.CODIGOS_ALTA_OBITO_HOSPICE_PLANO).any()


def test_d03_paciente_unico(exec_a: Execucao) -> None:
    """D03: ``patient_nbr`` único na saída."""
    assert exec_a.limpo["patient_nbr"].is_unique


def test_alvo_binario_e_proporcao_bate_com_o_bruto(exec_a: Execucao, bruto: pd.DataFrame) -> None:
    """``readmit_30d`` é 0/1 e a proporção de 1 bate com o cálculo direto no CSV bruto."""
    alvo = exec_a.modelagem[cfg.COLUNA_ALVO]
    assert set(alvo.unique()) == {0, 1}

    # Recalcula "na mão", só com pandas, aplicando as mesmas regras ao bruto.
    ref = bruto[bruto["gender"] != "Unknown/Invalid"]
    ref = ref.sort_values("encounter_id").drop_duplicates("patient_nbr", keep="first")
    ref = ref[~ref["discharge_disposition_id"].isin({11, 13, 14, 19, 20, 21})]
    assert len(ref) == len(exec_a.modelagem)
    assert (ref["readmitted"] == "<30").mean() == pytest.approx(alvo.mean(), abs=1e-12)
    assert int((ref["readmitted"] == "<30").sum()) == int(alvo.sum())


def test_sem_vazamento_de_alvo_e_ids(exec_a: Execucao) -> None:
    """A base de modelagem não tem ``readmitted``, IDs e o alvo é a última coluna."""
    colunas = list(exec_a.modelagem.columns)
    for proibida in ("readmitted", "encounter_id", "patient_nbr"):
        assert proibida not in colunas
    assert colunas[-1] == cfg.COLUNA_ALVO
    # O interim guarda tudo para rastreio.
    assert {"readmitted", "encounter_id", "patient_nbr"} <= set(exec_a.limpo.columns)


def test_nao_escalonou_nem_fez_onehot(exec_a: Execucao, bruto: pd.DataFrame) -> None:
    """Valores numéricos originais intactos e categorias ainda em texto (isso é da Pessoa 3)."""
    m = exec_a.modelagem
    assert m["time_in_hospital"].between(1, 14).all()
    assert pd.api.types.is_integer_dtype(m["time_in_hospital"])
    assert not pd.api.types.is_numeric_dtype(m["race"])
    # Nenhuma coluna one-hot (nome com "_" + categoria) foi criada.
    assert not any(c.startswith(("race_", "gender_")) for c in m.columns)


def test_outliers_so_sinalizados_valores_intactos(exec_a: Execucao, bruto: pd.DataFrame) -> None:
    """As colunas de contagem têm exatamente os valores do bruto; só ganham ``*_outlier``."""
    original = bruto.set_index("encounter_id")
    limpo = exec_a.limpo.set_index("encounter_id")
    for coluna in cfg.COLUNAS_OUTLIER:
        assert (limpo[coluna] == original.loc[limpo.index, coluna]).all()
        assert set(limpo[f"{coluna}_outlier"].unique()) <= {0, 1}


# ---------------------------------------------------------------------------
# Dicionário, funil e reprodutibilidade
# ---------------------------------------------------------------------------


def test_dicionario_igual_as_colunas_da_modelagem(exec_a: Execucao) -> None:
    """Colunas do dicionário = colunas do arquivo de modelagem (mesma ordem)."""
    assert list(exec_a.dicionario["coluna"]) == list(exec_a.modelagem.columns)
    assert (exec_a.dicionario["descricao_negocio"].str.len() > 10).all()
    assert set(exec_a.dicionario["tipo"]) <= {"numerica", "categorica", "binaria", "alvo"}
    sensiveis = set(exec_a.dicionario.loc[exec_a.dicionario["sensivel"] == "sim", "coluna"])
    assert {"race", "gender"} <= sensiveis


def test_parquet_igual_ao_csv(exec_a: Execucao) -> None:
    """O Parquet tem os mesmos dados do CSV de modelagem."""
    parquet = pd.read_parquet(exec_a.caminhos.parquet_modelagem)
    assert list(parquet.columns) == list(exec_a.modelagem.columns)
    assert len(parquet) == len(exec_a.modelagem)
    assert parquet[cfg.COLUNA_ALVO].sum() == exec_a.modelagem[cfg.COLUNA_ALVO].sum()


def test_funil_fecha(exec_a: Execucao) -> None:
    """Funil: ``linhas_entrada - removidas = linhas_saida`` e bate com a base final."""
    funil = pd.read_csv(exec_a.caminhos.tabelas / "funil_linhas.csv")
    entrada = int(funil["linhas_antes"].iloc[0])
    saida = int(funil["linhas_depois"].iloc[-1])
    assert entrada - int(funil["removidas"].sum()) == saida == len(exec_a.modelagem)
    # Cada etapa encadeia na seguinte.
    assert (funil["linhas_antes"] - funil["removidas"] == funil["linhas_depois"]).all()
    assert (funil["linhas_depois"].iloc[:-1].to_numpy() == funil["linhas_antes"].iloc[1:].to_numpy()).all()


def test_reprodutivel_mesmo_hash_duas_execucoes(exec_a: Execucao, exec_b: Execucao) -> None:
    """Rodar o pipeline duas vezes gera arquivos byte a byte idênticos."""
    for nome in ("csv_limpo", "csv_modelagem", "parquet_modelagem", "csv_dicionario"):
        a, b = getattr(exec_a.caminhos, nome), getattr(exec_b.caminhos, nome)
        assert _sha256(a) == _sha256(b), f"{nome} difere entre as execuções"


def test_agrupamento_cid9() -> None:
    """Exemplos conhecidos do agrupamento CID-9 (F03)."""
    codigos = pd.Series(["250.83", "250", "428", "414", "786", "788", "715", "800", "150",
                         "V57", "E909", "585", "530.81", "Sem_diagnostico", None])  # fmt: skip
    grupos = engenharia_atributos.agrupar_cid9(codigos).tolist()
    assert grupos == [
        "Diabetes", "Diabetes", "Circulatorio", "Circulatorio", "Respiratorio", "Geniturinario",
        "Musculoesqueletico", "Lesoes", "Neoplasias", "Outros", "Outros", "Geniturinario",
        "Digestivo", "Sem_diagnostico", "Sem_diagnostico",
    ]  # fmt: skip


# ---------------------------------------------------------------------------
# Documentação: todo número vem das tabelas
# ---------------------------------------------------------------------------

#: Números que fazem parte do texto fixo (não são resultado de cálculo).
NUMEROS_LIVRES = {"130", "1999", "2008", "2014", "70,000", "10.1155", "10.24432"}  # hospitais, período, citação (ano e título), DOIs
RE_NUMERO = re.compile(r"(?<![\w.,/\-])\d[\d.]*(?:,\d+)?(?![\w])")


def _numeros_do_texto(texto: str) -> set[str]:
    """Extrai os números de um texto Markdown, ignorando blocos e trechos de código.

    Args:
        texto: Conteúdo Markdown.

    Returns:
        Conjunto de números como aparecem no texto (padrão brasileiro).
    """
    texto = re.sub(r"```.*?```", " ", texto, flags=re.S)
    texto = re.sub(r"^#+ .*$", " ", texto, flags=re.M)  # títulos (ex.: "6.1") não são resultados
    texto = re.sub(r"`[^`]*`", " ", texto)
    texto = re.sub(r"!?\[[^\]]*\]\([^)]*\)", " ", texto)
    return {m.rstrip(".") for m in RE_NUMERO.findall(texto)}


def _universo_das_tabelas(pasta: Path) -> set[str]:
    """Todos os números que as tabelas CSV permitem escrever (com arredondamentos).

    Args:
        pasta: ``docs/tabelas``.

    Returns:
        Conjunto de textos numéricos aceitos (inteiros, 0–3 casas, com vírgula).
    """
    aceitos: set[str] = set()
    for csv in pasta.glob("*.csv"):
        conteudo = csv.read_text(encoding="utf-8")
        # Números já formatados em português dentro das células (ex.: "96,86%", "1.545").
        aceitos |= _numeros_do_texto(conteudo)
        # Números "crus" (ponto decimal): geram as formas formatadas e arredondadas.
        for cru in re.findall(r"(?<![\w.])\d+(?:\.\d+)?(?![\w])", conteudo):
            valor = float(cru)
            aceitos.add(cru)
            aceitos.add(fmt_int(valor))
            aceitos.update(fmt_num(valor, casas) for casas in (0, 1, 2, 3))
    return aceitos


def _checar_numeros(texto: str, pasta_tabelas: Path) -> set[str]:
    """Devolve os números do texto que NÃO aparecem nas tabelas.

    Inteiros pequenos (0 a 31: minutos, nº de etapas, nº de perguntas) e os
    números de ``NUMEROS_LIVRES`` não precisam estar nas tabelas.

    Args:
        texto: Markdown a verificar.
        pasta_tabelas: ``docs/tabelas``.

    Returns:
        Números sem origem nas tabelas (vazio = tudo certo).
    """
    universo = _universo_das_tabelas(pasta_tabelas)
    universo |= NUMEROS_LIVRES | {fmt_num(100 * cfg.LIMIAR_DOMINANCIA, 1), fmt_num(cfg.LIMIAR_CORRELACAO, 1)}
    orfaos = set()
    for numero in _numeros_do_texto(texto):
        eh_inteiro_pequeno = numero.isdigit() and int(numero) <= 31
        if numero not in universo and not eh_inteiro_pequeno:
            orfaos.add(numero)
    return orfaos


def test_todo_numero_do_readme_vem_das_tabelas(exec_a: Execucao) -> None:
    """Cada número de ``docs/README.md`` e de ``secao_base_de_dados.md`` existe em ``docs/tabelas``."""
    for nome in ("README.md", "secao_base_de_dados.md"):
        texto = (exec_a.caminhos.docs / nome).read_text(encoding="utf-8")
        orfaos = _checar_numeros(texto, exec_a.caminhos.tabelas)
        assert not orfaos, f"{nome}: números sem origem nas tabelas: {sorted(orfaos)}"


def test_readme_tem_as_secoes_obrigatorias(exec_a: Execucao) -> None:
    """O README segue a estrutura obrigatória (Mermaid, tabela de decisões, 10 perguntas...)."""
    texto = (exec_a.caminhos.docs / "README.md").read_text(encoding="utf-8")
    assert "```mermaid" in texto
    for trecho in ("Resumo em 30 segundos", "Funil de linhas", "Mapa de ausentes",
                   "O que estava errado", "Antes × depois", "Atributos criados", "Cuidados",
                   "O que entregamos", "Roteiro de fala", "Como reproduzir"):  # fmt: skip
        assert trecho in texto, trecho
    perguntas = re.findall(r"^\d+\. \*\*.+\?\*\*", texto, flags=re.M)
    assert len(perguntas) == 10


def test_figuras_existem_com_dpi_adequado(exec_a: Execucao) -> None:
    """As figuras citadas no README existem e têm dpi >= 150."""
    from PIL import Image

    texto = (exec_a.caminhos.docs / "README.md").read_text(encoding="utf-8")
    figuras = re.findall(r"\]\((figuras/[^)]+\.png)\)", texto)
    assert len(figuras) >= 7
    for figura in figuras:
        caminho = exec_a.caminhos.docs / figura
        assert caminho.exists(), figura
        with Image.open(caminho) as img:
            assert img.info["dpi"][0] >= 150, figura
