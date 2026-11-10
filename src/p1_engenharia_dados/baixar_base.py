"""Download dos acidentes em rodovias federais (PRF, Dados Abertos), 2024 e 2025.

Uso (a partir da raiz do repositório)::

    python -m src.p1_engenharia_dados.baixar_base            # baixa se faltar
    python -m src.p1_engenharia_dados.baixar_base --forcar   # baixa de novo

Baixa, para ``data/raw/``, quatro arquivos da página oficial
https://www.gov.br/prf/pt-br/acesso-a-informacao/dados-abertos/dados-abertos-da-prf

========================  ====================================================
``datatran2024.csv``      acidentes de 2024, **uma linha por ocorrência**
``datatran2025.csv``      acidentes de 2025, uma linha por ocorrência
``pessoas2024.csv``       2024, **uma linha por pessoa/veículo** envolvido
``pessoas2025.csv``       2025, uma linha por pessoa/veículo
========================  ====================================================

A PRF publica os arquivos como ZIP em links do Google Drive. Se algum dia o
download falhar, abra a página acima, baixe os arquivos "Acidentes 2024/2025
(Agrupados por ocorrência)" e "(Agrupados por pessoa)", extraia o CSV e salve
com os nomes da tabela em ``data/raw/``. Dados brutos não são versionados.

Os dados são atualizados todo mês pela PRF; guardamos a contagem de linhas
esperada para detectar se o arquivo foi revisado.
"""

from __future__ import annotations

import argparse
import csv
import io
import logging
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)

#: Endereço de download direto de um arquivo do Google Drive (testado).
URL_DRIVE = "https://drive.usercontent.google.com/download?id={id}&export=download&confirm=t"

# Pasta de destino, relativa à raiz do repositório.
RAIZ = Path(__file__).resolve().parents[2]
PASTA_RAW = RAIZ / "data" / "raw"

# Alguns servidores recusam o User-Agent padrão do urllib; usamos um nome genérico.
CABECALHOS = {"User-Agent": "projeto-final-cd/1.0 (trabalho academico)"}


@dataclass(frozen=True)
class ArquivoPRF:
    """Descreve um arquivo da PRF.

    Attributes:
        destino: Nome do CSV salvo em ``data/raw``.
        drive_id: Identificador do arquivo no Google Drive da PRF.
        linhas: Número de linhas de dados esperado (sem o cabeçalho).
    """

    destino: str
    drive_id: str
    linhas: int


#: Arquivos usados. As linhas esperadas foram contadas no download de 10/11/2026.
ARQUIVOS: tuple[ArquivoPRF, ...] = (
    ArquivoPRF("datatran2024.csv", "14lB0vqMFkaZj8HZ44b0njYgxs9nAN8KO", 73_202),
    ArquivoPRF("datatran2025.csv", "1-G3MdmHBt6CprDwcW99xxC4BZ2DU5ryR", 72_529),
    ArquivoPRF("pessoas2024.csv", "14lVfqdoE2gxDliaKZu7K9Mx6847maPtl", 196_448),
    ArquivoPRF("pessoas2025.csv", "1-Gp9S-ALO0D1nT8S_OKoC8xlW7BY8F82", 194_629),
)


def baixar_arquivo(arquivo: ArquivoPRF, destino: Path = PASTA_RAW) -> Path:
    """Baixa um ZIP da PRF e extrai o CSV com o nome de destino.

    Args:
        arquivo: Arquivo a baixar.
        destino: Pasta onde o CSV será salvo (criada se não existir).

    Returns:
        Caminho do CSV salvo.

    Raises:
        ValueError: Se o conteúdo baixado não for um ZIP com exatamente um CSV.
        urllib.error.URLError: Se o download falhar (sem internet, link mudou...).
    """
    destino.mkdir(parents=True, exist_ok=True)
    url = URL_DRIVE.format(id=arquivo.drive_id)
    logger.info("Baixando %s ...", arquivo.destino)
    requisicao = urllib.request.Request(url, headers=CABECALHOS)
    with urllib.request.urlopen(requisicao, timeout=180) as resposta:
        conteudo = resposta.read()
    try:
        zf = zipfile.ZipFile(io.BytesIO(conteudo))
    except zipfile.BadZipFile as erro:
        raise ValueError(
            f"{arquivo.destino}: o Drive não devolveu um ZIP (limite de acesso?). "
            "Baixe manualmente pela página da PRF (ver docstring deste módulo)."
        ) from erro
    csvs = [n for n in zf.namelist() if n.lower().endswith(".csv")]
    if len(csvs) != 1:
        raise ValueError(f"{arquivo.destino}: esperava 1 CSV no ZIP, achei {csvs}")
    caminho = destino / arquivo.destino
    caminho.write_bytes(zf.read(csvs[0]))
    logger.info("  salvo: %s (%.1f MB)", caminho, caminho.stat().st_size / 1e6)
    return caminho


def contar_linhas(caminho: Path) -> int:
    """Conta as linhas de dados de um CSV (sem o cabeçalho).

    Usa o módulo ``csv`` (e não contagem de quebras de linha) para que campos
    entre aspas com quebra de linha não sejam contados em duplicidade.

    Args:
        caminho: CSV a conferir.

    Returns:
        Número de linhas de dados.
    """
    with caminho.open(encoding="latin1", newline="") as f:
        return sum(1 for _ in csv.reader(f, delimiter=";")) - 1


def garantir_base(destino: Path = PASTA_RAW, forcar: bool = False) -> list[Path]:
    """Garante que os quatro arquivos brutos existam (baixa o que faltar).

    Args:
        destino: Pasta ``data/raw``.
        forcar: Se ``True``, baixa de novo mesmo que os arquivos já existam.

    Returns:
        Caminhos dos quatro CSVs.
    """
    caminhos = []
    for arquivo in ARQUIVOS:
        caminho = destino / arquivo.destino
        if forcar or not caminho.exists():
            baixar_arquivo(arquivo, destino)
        n = contar_linhas(caminho)
        if n == arquivo.linhas:
            logger.info("%s: %s linhas de dados (OK)", arquivo.destino, f"{n:,}".replace(",", "."))
        else:
            # Aviso (e não erro): a PRF pode ter revisado o arquivo desde o nosso download.
            logger.warning(
                "%s: %d linhas, esperado %d. A PRF revisou o arquivo? Os números da "
                "documentação podem mudar.",
                arquivo.destino, n, arquivo.linhas,
            )  # fmt: skip
        caminhos.append(caminho)
    return caminhos


def main() -> None:
    """Ponto de entrada de linha de comando do download."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
    parser = argparse.ArgumentParser(description="Baixa os acidentes da PRF (2024 e 2025).")
    parser.add_argument("--forcar", action="store_true", help="baixa mesmo que já exista")
    garantir_base(forcar=parser.parse_args().forcar)


if __name__ == "__main__":
    main()
