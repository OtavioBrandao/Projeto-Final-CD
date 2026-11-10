"""Limpeza determinística: inconsistências (D01–D07) e valores ausentes/inválidos.

Todas as funções são **puras**: recebem um DataFrame e devolvem um DataFrame
novo, sem alterar o original. Elas só aplicam regras fixas, baseadas no
conhecimento do domínio. Nada aqui é "aprendido" dos dados (nada de média,
mediana ou modelo para imputar), para não haver vazamento: essas estimativas
pertencem ao ``Pipeline`` do sklearn da Pessoa 3, depois da divisão treino/teste.

Quando um valor é impossível (idade 2024, ano de fabricação 0) ele vira nulo
(``NaN``): não inventamos um valor no lugar. O nulo é tratado na modelagem.
"""

from __future__ import annotations

import logging

import pandas as pd
from pandas.api.types import is_numeric_dtype

from src.p1_engenharia_dados import configuracao as cfg

logger = logging.getLogger(__name__)


def colunas_de_texto(df: pd.DataFrame) -> list[str]:
    """Lista as colunas não numéricas (texto) de um DataFrame.

    Usamos ``is_numeric_dtype`` em vez de ``select_dtypes("object")`` porque, no
    pandas 3, texto passa a ter dtype próprio (``str``) e o filtro por
    ``object`` deixaria de encontrá-lo.

    Args:
        df: DataFrame a inspecionar.

    Returns:
        Nomes das colunas de texto, na ordem original.
    """
    return [c for c in df.columns if not is_numeric_dtype(df[c])]


# ---------------------------------------------------------------------------
# D04 — rótulos "sem informação"
# ---------------------------------------------------------------------------


def padronizar_rotulos(df: pd.DataFrame, colunas: tuple[str, ...] | list[str]) -> pd.DataFrame:
    """D04 — unifica todas as formas de "não sabemos" em ``Nao_informado``.

    A PRF usa várias palavras para a mesma coisa: ``Ignorado`` (clima),
    ``Não Informado`` (sentido da via, sexo), ``NA`` e campo vazio. Para o
    modelo, são todas "sem informação". Também tira espaços das pontas e
    espaços repetidos.

    Args:
        df: Base com as colunas de texto.
        colunas: Colunas a padronizar (as que não existirem são ignoradas).

    Returns:
        Cópia do DataFrame com os rótulos padronizados.
    """
    saida = df.copy()
    for coluna in colunas:
        if coluna not in saida.columns:
            continue
        texto = saida[coluna].str.strip().str.replace(r"\s+", " ", regex=True)
        sem_info = texto.isna() | texto.isin(cfg.ROTULOS_SEM_INFORMACAO)
        saida[coluna] = texto.mask(sem_info, cfg.ROTULO_NAO_INFORMADO)
    return saida


def contar_celulas_alteradas(antes: pd.DataFrame, depois: pd.DataFrame) -> int:
    """Conta quantas células de texto mudaram entre duas versões da base.

    Args:
        antes: Base antes da padronização.
        depois: Base depois da padronização (mesmas linhas e colunas).

    Returns:
        Número de células de texto com valor diferente (nulo contra nulo conta
        como igual).
    """
    total = 0
    for coluna in colunas_de_texto(antes):
        a, d = antes[coluna], depois[coluna]
        total += int((~((a == d) | (a.isna() & d.isna()))).sum())
    return total


# ---------------------------------------------------------------------------
# D05, D06 — valores impossíveis nos registros de pessoas
# ---------------------------------------------------------------------------


def invalidar_idade(pessoas: pd.DataFrame) -> pd.DataFrame:
    """D05 — idade impossível vira nulo.

    Dois casos:

    * idade fora de [0, 110]: no arquivo há idades como 2024 (provável ano
      digitado no campo errado);
    * **condutor** com menos de 10 anos: a PRF usa ``0`` como "não informado" e
      há milhares de condutores "com 0 anos", que derrubariam a média de idade.

    Não dá para saber a idade verdadeira, então o valor é descartado em vez de
    entrar na média de idade dos condutores.

    Args:
        pessoas: Base com ``idade`` e ``tipo_envolvido`` (já padronizado em D04).

    Returns:
        Cópia com ``idade`` nula quando impossível.
    """
    saida = pessoas.copy()
    idade = saida["idade"]
    fora_do_intervalo = ~idade.between(cfg.IDADE_MINIMA, cfg.IDADE_MAXIMA) & idade.notna()
    condutor_crianca = (saida["tipo_envolvido"] == "Condutor") & (idade < cfg.IDADE_MINIMA_CONDUTOR)
    saida["idade"] = idade.mask(fora_do_intervalo | condutor_crianca)
    return saida


def invalidar_ano_fabricacao(pessoas: pd.DataFrame) -> pd.DataFrame:
    """D06 — ano de fabricação impossível vira nulo.

    Válido: de 1950 até o ano do acidente + 1 (modelo do ano seguinte existe).
    O valor ``0`` é a forma da PRF dizer "não informado" e distorceria a idade
    do veículo (viraria "2.025 anos"); ``1900`` aparece como valor padrão.

    Args:
        pessoas: Base com ``ano_fabricacao_veiculo`` e ``data_inversa`` (``AAAA-MM-DD``).

    Returns:
        Cópia com o ano nulo quando impossível.
    """
    saida = pessoas.copy()
    ano_acidente = saida["data_inversa"].str[:4].astype(int)
    ano = saida["ano_fabricacao_veiculo"]
    invalido = ano.notna() & ((ano < cfg.ANO_FABRICACAO_MINIMO) | (ano > ano_acidente + 1))
    saida["ano_fabricacao_veiculo"] = ano.mask(invalido)
    return saida


# ---------------------------------------------------------------------------
# D01, D02 — filtros de linhas (cada um vira uma linha do funil)
# ---------------------------------------------------------------------------


def remover_duplicados(ocorrencias: pd.DataFrame) -> pd.DataFrame:
    """D01 — remove acidentes idênticos em todos os campos (exceto ``id``).

    Dois registros com data, hora, local, causa, vítimas e demais campos
    iguais são o mesmo acidente lançado duas vezes; ficamos com o primeiro.

    Args:
        ocorrencias: Uma linha por acidente.

    Returns:
        DataFrame sem as duplicatas (índice refeito).
    """
    campos = [c for c in ocorrencias.columns if c not in (cfg.COLUNA_ID, "ano_arquivo")]
    return ocorrencias.drop_duplicates(subset=campos, keep="first").reset_index(drop=True)


def remover_eventos_intencionais(ocorrencias: pd.DataFrame) -> pd.DataFrame:
    """D02 — remove suicídios presumidos e "sinistros pessoais de trânsito".

    São eventos intencionais ou pessoais, não acidentes de trânsito. Mantê-los
    infla a taxa de acidentes fatais por motivos que nenhuma ação de segurança
    viária (fiscalização, sinalização, obra) poderia evitar.

    Args:
        ocorrencias: Base com ``causa_acidente`` e ``tipo_acidente``.

    Returns:
        DataFrame sem esses eventos (índice refeito).
    """
    fora_do_escopo = (ocorrencias["causa_acidente"] == cfg.CAUSA_INTENCIONAL) | (
        ocorrencias["tipo_acidente"] == cfg.TIPO_PESSOAL
    )
    return ocorrencias.loc[~fora_do_escopo].reset_index(drop=True)


# ---------------------------------------------------------------------------
# D03, D07 — inconsistências nas ocorrências
# ---------------------------------------------------------------------------


def sinalizar_contagens_inconsistentes(ocorrencias: pd.DataFrame) -> pd.DataFrame:
    """D03 — marca acidentes em que as contagens de vítimas não fecham.

    Regra: ``mortos + feridos_leves + feridos_graves + ilesos + ignorados`` deve
    ser igual a ``pessoas``. Não removemos nem corrigimos (não sabemos qual
    campo está errado); só sinalizamos. O alvo usa apenas ``mortos``.

    Args:
        ocorrencias: Base com as colunas de contagem.

    Returns:
        Cópia com ``contagens_inconsistentes`` (0/1).
    """
    saida = ocorrencias.copy()
    soma = saida[["mortos", "feridos_leves", "feridos_graves", "ilesos", "ignorados"]].sum(axis=1)
    saida["contagens_inconsistentes"] = (soma != saida["pessoas"]).astype("int8")
    return saida


def invalidar_coordenadas(ocorrencias: pd.DataFrame) -> pd.DataFrame:
    """D07 — coordenada fora do Brasil vira nula (latitude e longitude juntas).

    Fora do Brasil = fora da caixa de ``cfg.LATITUDE_*``/``cfg.LONGITUDE_*`` ou
    latitude positiva em UF que não fica ao norte do Equador. Latitude
    positiva em AP e RR é **correta** (ficam ao norte) e não é tocada.

    Args:
        ocorrencias: Base com ``latitude``, ``longitude`` e ``uf``.

    Returns:
        Cópia com as coordenadas nulas quando implausíveis.
    """
    saida = ocorrencias.copy()
    lat, lon = saida["latitude"], saida["longitude"]
    fora_da_caixa = (
        (lat < cfg.LATITUDE_MINIMA) | (lat > cfg.LATITUDE_MAXIMA)
        | (lon < cfg.LONGITUDE_MINIMA) | (lon > cfg.LONGITUDE_MAXIMA)
    )  # fmt: skip
    norte_errado = (lat > 0) & ~saida["uf"].isin(cfg.UFS_NORTE_DO_EQUADOR)
    invalida = fora_da_caixa | norte_errado
    saida["latitude"] = lat.mask(invalida)
    saida["longitude"] = lon.mask(invalida)
    return saida


def nulos_restantes(df: pd.DataFrame) -> pd.Series:
    """Lista as colunas que ainda têm nulos depois do tratamento.

    Args:
        df: Base tratada.

    Returns:
        Série com a contagem de nulos, apenas das colunas com pelo menos um.
    """
    contagem = df.isna().sum()
    return contagem[contagem > 0]


def contar_coordenadas_positivas_validas(ocorrencias: pd.DataFrame) -> int:
    """Conta latitudes positivas que são **legítimas** (UF ao norte do Equador).

    Serve de evidência de que latitude ≥ 0 não é erro por si só.

    Args:
        ocorrencias: Base com ``latitude`` e ``uf``.

    Returns:
        Número de acidentes com latitude positiva em AP, RR, AM ou PA.
    """
    return int(((ocorrencias["latitude"] > 0) & ocorrencias["uf"].isin(cfg.UFS_NORTE_DO_EQUADOR)).sum())

