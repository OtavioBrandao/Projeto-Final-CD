"""Testes do tratamento de dados da PRF (``pytest``).

Verificam as regras do plano: rótulos "não sei" unificados, filtros D01 e D02,
alvo correto e sem vazamento, valores impossíveis anulados, dicionário igual à
base, funil de linhas fechando, reprodutibilidade (mesmo hash ao rodar duas
vezes) e que todo número do ``docs/README_pre_processamento.md`` vem das tabelas de ``docs/tabelas``.

Os testes rodam o pipeline em **pastas temporárias**, sem tocar nos arquivos
versionados. Precisam de ``data/raw`` (``python -m src.p1_engenharia_dados.baixar_base``).
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import pandas as pd
import pytest

from src.p1_engenharia_dados import carregar, engenharia_atributos, executar_pipeline, limpeza
from src.p1_engenharia_dados import configuracao as cfg
from src.p1_engenharia_dados.configuracao import Caminhos
from src.p1_engenharia_dados.relatorios import fmt_int, fmt_num

RAIZ = Path(__file__).resolve().parents[1]
CAMINHOS_REAIS = Caminhos(raiz=RAIZ)

pytestmark = pytest.mark.skipif(
    not all(CAMINHOS_REAIS.csv_ocorrencias(a).exists() for a in cfg.ANOS),
    reason="dados brutos ausentes: rode python -m src.p1_engenharia_dados.baixar_base",
)

#: Colunas de texto onde os rótulos "não sei" devem ter sido unificados (D04).
ROTULOS_PROIBIDOS = {"Ignorado", "Não Informado", "NA"}


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
    caminhos = Caminhos(raiz=pasta, raw=CAMINHOS_REAIS.pasta_raw)
    metricas = executar_pipeline.executar(caminhos, gerar_documentacao=documentacao)
    ler = lambda c: pd.read_csv(c, keep_default_na=False, na_values=[""], low_memory=False)  # noqa: E731
    return Execucao(
        caminhos=caminhos,
        metricas=metricas,
        limpo=ler(caminhos.csv_limpo),
        modelagem=ler(caminhos.csv_modelagem),
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
    """Ocorrências brutas lidas com a regra correta de nulos."""
    return carregar.carregar_ocorrencias(CAMINHOS_REAIS)


# ---------------------------------------------------------------------------
# Carga
# ---------------------------------------------------------------------------


def test_bruto_tem_o_tamanho_documentado(bruto: pd.DataFrame) -> None:
    """Os dois anos de ocorrências somam 145.731 acidentes, com ``id`` único."""
    assert len(bruto) == 73_202 + 72_529
    assert bruto["id"].is_unique


def test_leitura_usa_latin1_e_virgula_decimal(bruto: pd.DataFrame) -> None:
    """Acentos saem corretos e latitude/km são numéricos (vírgula decimal)."""
    assert "Céu Claro" in set(bruto["condicao_metereologica"])
    assert pd.api.types.is_float_dtype(bruto["latitude"])
    assert pd.api.types.is_float_dtype(bruto["km"])


def test_na_do_arquivo_vira_nulo_e_ignorado_continua_texto(bruto: pd.DataFrame) -> None:
    """Só ``NA`` e vazio viram nulo; ``Ignorado`` continua texto até a etapa D04."""
    assert bruto["classificacao_acidente"].isna().sum() == 2
    assert (bruto["condicao_metereologica"] == "Ignorado").sum() > 1_000


def test_pessoas_casam_com_ocorrencias(bruto: pd.DataFrame) -> None:
    """Todo acidente tem pessoas e toda pessoa tem acidente (join 100%)."""
    pessoas = carregar.carregar_pessoas(CAMINHOS_REAIS)
    assert len(pessoas) == 196_448 + 194_629
    assert set(pessoas["id"]) == set(bruto["id"])


# ---------------------------------------------------------------------------
# Regras de limpeza (testes pequenos, sem rodar o pipeline)
# ---------------------------------------------------------------------------


def test_padronizar_rotulos_unifica_nao_sei() -> None:
    """D04: Ignorado, Não Informado e nulo viram ``Nao_informado``."""
    df = pd.DataFrame({"sexo": ["Masculino", "Ignorado", "Não Informado", None, " Feminino "]})
    saida = limpeza.padronizar_rotulos(df, ["sexo"])
    assert saida["sexo"].tolist() == ["Masculino", "Nao_informado", "Nao_informado", "Nao_informado", "Feminino"]


def test_invalidar_idade() -> None:
    """D05: idade > 110 e condutor com menos de 10 anos viram nulo; passageiro de 0 ano fica."""
    df = pd.DataFrame(
        {
            "idade": [30.0, 2024.0, 0.0, 0.0, 111.0, 9.0],
            "tipo_envolvido": ["Condutor", "Condutor", "Condutor", "Passageiro", "Passageiro", "Condutor"],
        }
    )
    saida = limpeza.invalidar_idade(df)
    assert saida["idade"].isna().tolist() == [False, True, True, False, True, True]


def test_invalidar_ano_fabricacao() -> None:
    """D06: ano 0, 1900 e futuro viram nulo; 1950 e o ano seguinte ao acidente são válidos."""
    df = pd.DataFrame(
        {
            "ano_fabricacao_veiculo": [0.0, 1900.0, 1950.0, 2010.0, 2026.0, 2027.0],
            "data_inversa": ["2025-03-01"] * 6,
        }
    )
    saida = limpeza.invalidar_ano_fabricacao(df)
    assert saida["ano_fabricacao_veiculo"].isna().tolist() == [True, True, False, False, False, True]


def test_invalidar_coordenadas_respeita_norte_do_equador() -> None:
    """D07: latitude positiva vale em AP/RR; fora do Brasil ou positiva em SP vira nulo."""
    df = pd.DataFrame(
        {
            "uf": ["AP", "RR", "SP", "SP", "MT"],
            "latitude": [0.5, 2.8, 10.0, -23.0, -50.0],
            "longitude": [-51.0, -60.0, -46.0, -46.0, -55.0],
        }
    )
    saida = limpeza.invalidar_coordenadas(df)
    assert saida["latitude"].isna().tolist() == [False, False, True, False, True]
    assert saida["longitude"].isna().tolist() == [False, False, True, False, True]


def test_remover_eventos_intencionais() -> None:
    """D02: suicídio presumido e sinistro pessoal saem; o resto fica."""
    df = pd.DataFrame(
        {
            "causa_acidente": ["Chuva", cfg.CAUSA_INTENCIONAL, "Chuva"],
            "tipo_acidente": ["Tombamento", "Tombamento", cfg.TIPO_PESSOAL],
        }
    )
    assert len(limpeza.remover_eventos_intencionais(df)) == 1


# ---------------------------------------------------------------------------
# Atributos (testes pequenos)
# ---------------------------------------------------------------------------


def test_feriados_nacionais_conhecidos() -> None:
    """F04: fixos, Carnaval, Sexta-feira Santa, Corpus Christi e Consciência Negra (2024+)."""
    feriados = engenharia_atributos.feriados_nacionais([2024, 2025])
    esperados = {
        date(2024, 1, 1), date(2024, 2, 12), date(2024, 2, 13), date(2024, 3, 29),
        date(2024, 5, 30), date(2024, 11, 20), date(2025, 3, 3), date(2025, 3, 4),
        date(2025, 4, 18), date(2025, 6, 19), date(2025, 11, 20), date(2025, 12, 25),
    }  # fmt: skip
    assert esperados <= feriados
    assert date(2024, 7, 10) not in feriados


def test_faixas_do_dia() -> None:
    """F02: limites das faixas horárias."""
    nomes = [engenharia_atributos._nome_da_faixa(h) for h in (0, 5, 6, 11, 12, 17, 18, 23)]
    assert nomes == ["Madrugada", "Madrugada", "Manha", "Manha", "Tarde", "Tarde", "Noite", "Noite"]


def test_flags_de_tracado() -> None:
    """F07: valores compostos viram colunas 0/1."""
    df = pd.DataFrame({"tracado_via": ["Reta;Declive", "Curva", "Interseção de Vias;Ponte", "Rotatória"]})
    saida = engenharia_atributos.criar_flags_tracado(df)
    assert saida["via_reta"].tolist() == [1, 0, 0, 0]
    assert saida["via_declive"].tolist() == [1, 0, 0, 0]
    assert saida["via_curva"].tolist() == [0, 1, 0, 0]
    assert saida["via_ponte_viaduto_tunel"].tolist() == [0, 0, 1, 0]
    assert saida["via_rotatoria_retorno"].tolist() == [0, 0, 0, 1]


def test_resumir_pessoas_em_exemplo_pequeno() -> None:
    """F11: dois acidentes com moto, pedestre, idades e sexo conhecidos."""
    pessoas = pd.DataFrame(
        {
            "id": [1, 1, 1, 2, 2],
            "id_veiculo": [10.0, 10.0, 11.0, 20.0, None],
            "tipo_veiculo": ["Motocicleta", "Motocicleta", "Caminhão", "Automóvel", "Nao_informado"],
            "tipo_envolvido": ["Condutor", "Passageiro", "Condutor", "Condutor", "Pedestre"],
            "idade": [30.0, 10.0, 50.0, None, 40.0],
            "sexo": ["Masculino", "Feminino", "Feminino", "Masculino", "Feminino"],
            "ano_fabricacao_veiculo": [2020.0, 2020.0, 2010.0, None, None],
            "data_inversa": ["2024-05-01"] * 5,
        }
    )
    r = engenharia_atributos.resumir_pessoas(pessoas).set_index("id")
    assert r.loc[1, "envolve_moto"] == 1 and r.loc[1, "envolve_caminhao"] == 1 and r.loc[1, "envolve_pedestre"] == 0
    assert r.loc[2, "envolve_pedestre"] == 1 and r.loc[2, "envolve_moto"] == 0
    assert r.loc[1, "idade_media_condutores"] == 40.0  # (30 + 50) / 2; o passageiro não conta
    assert pd.isna(r.loc[2, "idade_media_condutores"])  # único condutor sem idade válida
    assert r.loc[1, "pct_condutores_homens"] == 50.0
    assert r.loc[1, "idade_media_veiculos"] == 9.0  # veículos de 4 e 14 anos, contados uma vez cada


def test_todas_as_causas_da_base_tem_grupo(bruto: pd.DataFrame) -> None:
    """F06: nenhuma causa que chega à etapa de atributos cai em ``Outras``.

    A causa intencional (suicídio presumido) já saiu em D02, por isso não entra aqui.
    """
    restantes = bruto.loc[bruto["causa_acidente"] != cfg.CAUSA_INTENCIONAL, ["causa_acidente"]]
    saida = engenharia_atributos.agrupar_causas(restantes)
    assert (saida["grupo_causa"] != cfg.ROTULO_OUTRAS).all()


# ---------------------------------------------------------------------------
# Pipeline completo
# ---------------------------------------------------------------------------


def test_rotulos_nao_sei_unificados(exec_a: Execucao) -> None:
    """D04: nenhum ``Ignorado``/``Não Informado``/``NA`` sobra nas bases."""
    for base in (exec_a.limpo, exec_a.modelagem):
        texto = base.select_dtypes(exclude="number")
        assert not texto.isin(ROTULOS_PROIBIDOS).any().any()


def test_d02_sem_eventos_intencionais(exec_a: Execucao) -> None:
    """D02: nenhum suicídio presumido nem sinistro pessoal no interim."""
    assert not (exec_a.limpo["causa_acidente"] == cfg.CAUSA_INTENCIONAL).any()
    assert not (exec_a.limpo["tipo_acidente"] == cfg.TIPO_PESSOAL).any()


def test_id_unico_e_anos(exec_a: Execucao) -> None:
    """``id`` único no interim; só 2024 e 2025."""
    assert exec_a.limpo["id"].is_unique
    assert set(exec_a.modelagem["ano"].unique()) == {2024, 2025}


def test_alvo_binario_e_proporcao_bate_com_o_bruto(exec_a: Execucao, bruto: pd.DataFrame) -> None:
    """``acidente_fatal`` é 0/1 e bate com o cálculo direto (mortos > 0) no bruto filtrado."""
    alvo = exec_a.modelagem[cfg.COLUNA_ALVO]
    assert set(alvo.unique()) == {0, 1}
    # Recalcula "na mão", só com pandas, aplicando a regra D02 ao bruto.
    ref = bruto[(bruto["causa_acidente"] != "Suicídio (presumido)") & (bruto["tipo_acidente"] != "Sinistro pessoal de trânsito")]
    assert len(ref) == len(exec_a.modelagem)
    assert (ref["mortos"] > 0).mean() == pytest.approx(alvo.mean(), abs=1e-12)
    assert int((ref["mortos"] > 0).sum()) == int(alvo.sum())


def test_sem_vazamento_de_alvo(exec_a: Execucao) -> None:
    """A base de modelagem não tem consequências, ``id``, e o alvo é a última coluna."""
    colunas = list(exec_a.modelagem.columns)
    for proibida in (*cfg.COLUNAS_RESULTADO, "id", *cfg.COLUNAS_ADMINISTRATIVAS):
        assert proibida not in colunas, proibida
    assert colunas[-1] == cfg.COLUNA_ALVO
    # O interim guarda tudo para rastreio.
    assert set(cfg.COLUNAS_RESULTADO) <= set(exec_a.limpo.columns)


def test_nulos_so_onde_documentado(exec_a: Execucao) -> None:
    """Só as 3 colunas de resumo de pessoas têm nulo (imputação é da Pessoa 3)."""
    nulos = set(exec_a.modelagem.columns[exec_a.modelagem.isna().any()])
    assert nulos == {"idade_media_condutores", "pct_condutores_homens", "idade_media_veiculos"}


def test_valores_plausiveis(exec_a: Execucao) -> None:
    """Idades e coordenadas dentro dos limites; hora de 0 a 23; rodovia como texto."""
    m = exec_a.modelagem
    assert m["idade_media_condutores"].dropna().between(cfg.IDADE_MINIMA_CONDUTOR, cfg.IDADE_MAXIMA).all()
    assert m["idade_media_veiculos"].dropna().between(0, 75).all()
    assert m["latitude"].dropna().between(cfg.LATITUDE_MINIMA, cfg.LATITUDE_MAXIMA).all()
    assert m["longitude"].dropna().between(cfg.LONGITUDE_MINIMA, cfg.LONGITUDE_MAXIMA).all()
    assert m["hora"].between(0, 23).all()
    assert m["br"].str.startswith("BR-").all()


def test_nao_escalonou_nem_fez_onehot(exec_a: Execucao) -> None:
    """Valores originais intactos e categorias ainda em texto (isso é da Pessoa 3)."""
    m = exec_a.modelagem
    assert pd.api.types.is_integer_dtype(m["veiculos"]) and m["veiculos"].min() >= 1
    assert not pd.api.types.is_numeric_dtype(m["uf"])
    assert not any(c.startswith(("uf_", "regiao_")) for c in m.columns)


def test_outliers_so_sinalizados_valores_intactos(exec_a: Execucao, bruto: pd.DataFrame) -> None:
    """``km`` e ``veiculos`` têm exatamente os valores do bruto; só ganham ``*_outlier``."""
    original = bruto.set_index("id")
    limpo = exec_a.limpo.set_index("id")
    for coluna in ("km", "veiculos"):
        assert (limpo[coluna] == original.loc[limpo.index, coluna]).all()
    for coluna in cfg.COLUNAS_OUTLIER:
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
    assert sensiveis == {"idade_media_condutores", "pct_condutores_homens"}


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
    assert (funil["linhas_antes"] - funil["removidas"] == funil["linhas_depois"]).all()
    assert (funil["linhas_depois"].iloc[:-1].to_numpy() == funil["linhas_antes"].iloc[1:].to_numpy()).all()


def test_reprodutivel_mesmo_hash_duas_execucoes(exec_a: Execucao, exec_b: Execucao) -> None:
    """Rodar o pipeline duas vezes gera arquivos byte a byte idênticos."""
    for nome in ("csv_limpo", "csv_modelagem", "parquet_modelagem", "csv_dicionario"):
        a, b = getattr(exec_a.caminhos, nome), getattr(exec_b.caminhos, nome)
        assert _sha256(a) == _sha256(b), f"{nome} difere entre as execuções"


# ---------------------------------------------------------------------------
# Documentação: todo número vem das tabelas
# ---------------------------------------------------------------------------

#: Números que fazem parte do texto fixo (anos dos dados e da consulta).
NUMEROS_LIVRES = {"2024", "2025", "2026"}
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
    texto = re.sub(r"https?://\S+", " ", texto)
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
        # Números já formatados em português dentro das células (ex.: "7,05%", "145.342").
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
    """Cada número de ``docs/README_pre_processamento.md`` e de ``secao_base_de_dados.md`` existe em ``docs/tabelas``."""
    for nome in ("README_pre_processamento.md", "secao_base_de_dados.md"):
        texto = (exec_a.caminhos.docs / nome).read_text(encoding="utf-8")
        orfaos = _checar_numeros(texto, exec_a.caminhos.tabelas)
        assert not orfaos, f"{nome}: números sem origem nas tabelas: {sorted(orfaos)}"


def test_verificador_de_numeros_acusa_numero_inventado(exec_a: Execucao) -> None:
    """O próprio verificador funciona: um número adulterado é pego."""
    texto = (exec_a.caminhos.docs / "README_pre_processamento.md").read_text(encoding="utf-8")
    adulterado = texto.replace("145.342 acidentes", "145.399 acidentes", 1)
    assert "145.399" in _checar_numeros(adulterado, exec_a.caminhos.tabelas)


def test_readme_tem_as_secoes_obrigatorias(exec_a: Execucao) -> None:
    """O README segue a estrutura obrigatória (Mermaid, tabela de decisões, 10 perguntas...)."""
    texto = (exec_a.caminhos.docs / "README_pre_processamento.md").read_text(encoding="utf-8")
    assert texto.startswith("# README — Engenharia de Dados e Pré-processamento")
    assert "```mermaid" in texto
    for trecho in ("Resumo em 30 segundos", "Funil de linhas", "Mapa de ausentes",
                   "O que estava errado", "Antes × depois", "Atributos criados", "Cuidados",
                   "O que entregamos", "Roteiro de fala", "Como reproduzir"):  # fmt: skip
        assert trecho in texto, trecho
    perguntas = re.findall(r"^\d+\. \*\*.+\?\*\*", texto, flags=re.M)
    assert len(perguntas) == 10
    # Nada do tema antigo (hospital) pode sobrar.
    for antigo in ("readmi", "diabet", "hospital", "readmit"):
        assert antigo not in texto.lower(), antigo


def test_figuras_existem_com_dpi_adequado(exec_a: Execucao) -> None:
    """As figuras citadas no README existem e têm dpi >= 150."""
    from PIL import Image

    texto = (exec_a.caminhos.docs / "README_pre_processamento.md").read_text(encoding="utf-8")
    figuras = re.findall(r"\]\((figuras/[^)]+\.png)\)", texto)
    assert len(figuras) >= 7
    for figura in figuras:
        caminho = exec_a.caminhos.docs / figura
        assert caminho.exists(), figura
        with Image.open(caminho) as img:
            assert img.info["dpi"][0] >= 150, figura
