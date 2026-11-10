"""Leitura dos arquivos brutos da PRF e integração ocorrências x pessoas.

A PRF publica cada ano em dois arquivos:

* **ocorrências** (``datatran``): uma linha por acidente;
* **pessoas**: uma linha por pessoa/veículo envolvido, com idade, sexo, tipo e
  ano de fabricação do veículo (informações que não existem no arquivo de
  ocorrências).

Este módulo lê os dois e faz a integração (decisão I01): os dados de pessoas
são resumidos por acidente (em ``engenharia_atributos``) e juntados ao arquivo
de ocorrências pelo ``id``.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from src.p1_engenharia_dados import configuracao as cfg
from src.p1_engenharia_dados.configuracao import Caminhos

logger = logging.getLogger(__name__)


def ler_csv_prf(caminho: Path, como_texto: bool = False) -> pd.DataFrame:
    """Lê um CSV da PRF com a codificação e os separadores corretos.

    Por que cada opção:

    * ``sep=";"`` e ``encoding="latin1"``: é como a PRF grava (em UTF-8 os
      acentos quebram);
    * ``decimal=","``: latitude, longitude e km usam vírgula decimal;
    * ``keep_default_na=False`` + ``na_values=cfg.TOKENS_NULO_BRUTO``: só ``NA`` e
      campo vazio viram nulo; o pandas não decide sozinho o que é nulo.

    Args:
        caminho: Caminho do CSV.
        como_texto: Se ``True``, lê tudo como texto, sem converter nada (usado
            pela auditoria para contar os textos ``NA`` do arquivo original).

    Returns:
        DataFrame com os dados do arquivo.

    Raises:
        FileNotFoundError: Se o arquivo não existir (rode ``baixar_base`` antes).
    """
    if not caminho.exists():
        raise FileNotFoundError(
            f"{caminho} não existe. Rode: python -m src.p1_engenharia_dados.baixar_base"
        )
    if como_texto:
        return pd.read_csv(
            caminho, sep=cfg.SEPARADOR_CSV, encoding=cfg.CODIFICACAO_CSV, dtype=str,
            keep_default_na=False, na_filter=False,
        )  # fmt: skip
    return pd.read_csv(
        caminho,
        sep=cfg.SEPARADOR_CSV,
        encoding=cfg.CODIFICACAO_CSV,
        decimal=cfg.DECIMAL_CSV,
        keep_default_na=False,
        na_values=list(cfg.TOKENS_NULO_BRUTO),
        low_memory=False,
    )


def carregar_ocorrencias(caminhos: Caminhos, como_texto: bool = False) -> pd.DataFrame:
    """Empilha os arquivos de ocorrências de todos os anos.

    Args:
        caminhos: Caminhos do projeto.
        como_texto: Ver :func:`ler_csv_prf`.

    Returns:
        Uma linha por acidente, com a coluna extra ``ano_arquivo``.
    """
    partes = [ler_csv_prf(caminhos.csv_ocorrencias(a), como_texto).assign(ano_arquivo=a) for a in cfg.ANOS]
    df = pd.concat(partes, ignore_index=True)
    logger.info("Ocorrências: %d linhas x %d colunas", *df.shape)
    return df


def carregar_pessoas(caminhos: Caminhos, como_texto: bool = False) -> pd.DataFrame:
    """Empilha os arquivos "por pessoa" de todos os anos.

    Args:
        caminhos: Caminhos do projeto.
        como_texto: Ver :func:`ler_csv_prf`.

    Returns:
        Uma linha por pessoa/veículo envolvido, com a coluna extra ``ano_arquivo``.
    """
    partes = [ler_csv_prf(caminhos.csv_pessoas(a), como_texto).assign(ano_arquivo=a) for a in cfg.ANOS]
    df = pd.concat(partes, ignore_index=True)
    logger.info("Pessoas: %d linhas x %d colunas", *df.shape)
    return df


def integrar_pessoas(ocorrencias: pd.DataFrame, resumo_pessoas: pd.DataFrame) -> pd.DataFrame:
    """I01 — junta o resumo de pessoas/veículos ao arquivo de ocorrências.

    Join à esquerda pelo ``id``: nenhuma ocorrência é perdida, mesmo que não
    tenha registros de pessoas (nesse caso as colunas novas ficam nulas).

    Args:
        ocorrencias: Uma linha por acidente (coluna ``id``).
        resumo_pessoas: Uma linha por acidente (``id``), saída de
            ``engenharia_atributos.resumir_pessoas``.

    Returns:
        Novo DataFrame com as mesmas linhas e ordem de ``ocorrencias`` e as
        colunas do resumo.

    Raises:
        RuntimeError: Se o join alterar o número de linhas.
    """
    # validate="one_to_one": cada acidente tem exatamente um resumo; o join não pode duplicar linhas.
    juntado = ocorrencias.merge(resumo_pessoas, on=cfg.COLUNA_ID, how="left", validate="one_to_one")
    if len(juntado) != len(ocorrencias):  # pragma: no cover - proteção contra join mal feito
        raise RuntimeError("O join com o resumo de pessoas alterou o número de linhas.")
    return juntado
