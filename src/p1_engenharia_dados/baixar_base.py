"""Download da base *Diabetes 130-US Hospitals for Years 1999-2008* (UCI, id 296).

Uso (a partir da raiz do repositório)::

    python -m src.p1_engenharia_dados.baixar_base            # baixa se faltar
    python -m src.p1_engenharia_dados.baixar_base --forcar   # baixa de novo

O script baixa o ZIP oficial do UCI Machine Learning Repository, extrai
``diabetic_data.csv`` e ``IDS_mapping.csv`` em ``data/raw/`` e confere o número
de linhas esperado (101.766). Dados brutos não são versionados no Git.

Página do dataset: https://archive.ics.uci.edu/dataset/296/diabetes+130-us+hospitals+for+years+1999-2008

Licença da base: CC BY 4.0. Citação: Strack et al. (2014), BioMed Research
International, DOI 10.1155/2014/781670. Dataset DOI: 10.24432/C5230J.
"""

from __future__ import annotations

import argparse
import csv
import io
import logging
import urllib.request
import zipfile
from pathlib import Path

logger = logging.getLogger(__name__)

# URL estática do UCI para o dataset 296 (testada: responde 200 com o ZIP).
# Se algum dia mudar, abra a página do dataset (link no topo) e copie o novo link.
URL_ZIP = (
    "https://archive.ics.uci.edu/static/public/296/"
    "diabetes+130-us+hospitals+for+years+1999-2008.zip"
)

# Pasta de destino, relativa à raiz do repositório.
RAIZ = Path(__file__).resolve().parents[2]
PASTA_RAW = RAIZ / "data" / "raw"

# Arquivos que precisamos do ZIP e a contagem de linhas de dados esperada.
ARQUIVOS = ("diabetic_data.csv", "IDS_mapping.csv")
LINHAS_ESPERADAS = 101_766

# Alguns servidores recusam o User-Agent padrão do urllib; usamos um nome genérico.
CABECALHOS = {"User-Agent": "projeto-final-cd/1.0 (trabalho academico)"}


def baixar_e_extrair(url: str = URL_ZIP, destino: Path = PASTA_RAW) -> list[Path]:
    """Baixa o ZIP do UCI e extrai os arquivos necessários.

    Args:
        url: Endereço do ZIP oficial no UCI.
        destino: Pasta onde os arquivos serão salvos (criada se não existir).

    Returns:
        Lista com os caminhos dos arquivos extraídos.

    Raises:
        FileNotFoundError: Se algum arquivo esperado não estiver no ZIP.
        urllib.error.URLError: Se o download falhar (sem internet, URL mudou...).
    """
    destino.mkdir(parents=True, exist_ok=True)
    logger.info("Baixando %s ...", url)
    requisicao = urllib.request.Request(url, headers=CABECALHOS)
    with urllib.request.urlopen(requisicao, timeout=120) as resposta:
        conteudo = resposta.read()
    logger.info("Download concluído: %.1f MB", len(conteudo) / 1e6)

    extraidos: list[Path] = []
    with zipfile.ZipFile(io.BytesIO(conteudo)) as zf:
        # O ZIP pode guardar os arquivos em subpastas; comparamos só o nome.
        nomes = {Path(n).name: n for n in zf.namelist()}
        for arquivo in ARQUIVOS:
            if arquivo not in nomes:
                raise FileNotFoundError(f"{arquivo} não encontrado no ZIP: {sorted(nomes)}")
            caminho = destino / arquivo
            caminho.write_bytes(zf.read(nomes[arquivo]))
            extraidos.append(caminho)
            logger.info("  salvo: %s", caminho)
    return extraidos


def conferir_linhas(caminho: Path, esperado: int = LINHAS_ESPERADAS) -> int:
    """Conta as linhas de dados de um CSV e avisa se diferir do esperado.

    Usa o módulo ``csv`` (e não contagem de quebras de linha) para que campos
    entre aspas com quebra de linha não sejam contados em duplicidade.

    Args:
        caminho: CSV a conferir (cabeçalho não é contado).
        esperado: Número de linhas de dados esperado.

    Returns:
        Número de linhas de dados encontrado.
    """
    with caminho.open(encoding="utf-8", newline="") as f:
        n = sum(1 for _ in csv.reader(f)) - 1  # -1 = cabeçalho
    if n == esperado:
        logger.info("%s: %s linhas de dados (OK)", caminho.name, f"{n:,}".replace(",", "."))
    else:
        logger.warning("%s: %d linhas, esperado %d (ATENÇÃO)", caminho.name, n, esperado)
    return n


def garantir_base(destino: Path = PASTA_RAW, forcar: bool = False) -> Path:
    """Garante que os arquivos brutos existam em ``destino`` (baixa se faltar).

    Args:
        destino: Pasta ``data/raw``.
        forcar: Se ``True``, baixa de novo mesmo que os arquivos já existam.

    Returns:
        Caminho do ``diabetic_data.csv``.

    Raises:
        ValueError: Se o número de linhas baixado não for o esperado.
    """
    faltando = [a for a in ARQUIVOS if not (destino / a).exists()]
    if forcar or faltando:
        baixar_e_extrair(destino=destino)
    caminho = destino / "diabetic_data.csv"
    if conferir_linhas(caminho) != LINHAS_ESPERADAS:
        raise ValueError(f"{caminho} não tem {LINHAS_ESPERADAS} linhas de dados.")
    return caminho


def main() -> None:
    """Ponto de entrada de linha de comando do download."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
    parser = argparse.ArgumentParser(description="Baixa a base Diabetes 130-US Hospitals.")
    parser.add_argument("--forcar", action="store_true", help="baixa mesmo que já exista")
    garantir_base(forcar=parser.parse_args().forcar)


if __name__ == "__main__":
    main()
