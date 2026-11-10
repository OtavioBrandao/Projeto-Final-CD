"""Download da base *Diabetes 130-US Hospitals for Years 1999-2008* (UCI, id 296).

Uso (a partir da raiz do repositório)::

    python -m src.p1_engenharia_dados.baixar_base

O script baixa o ZIP oficial do UCI Machine Learning Repository, extrai
``diabetic_data.csv`` e ``IDS_mapping.csv`` em ``data/raw/`` e confere o número
de linhas esperado (101.766). Dados brutos não são versionados no Git.

Licença da base: CC BY 4.0. Citação: Strack et al. (2014), BioMed Research
International, DOI 10.1155/2014/781670. Dataset DOI: 10.24432/C5230J.
"""

from __future__ import annotations

import io
import urllib.request
import zipfile
from pathlib import Path

# URL estática do UCI para o dataset 296 (se mudar, veja a página do dataset).
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


def baixar_e_extrair(url: str = URL_ZIP, destino: Path = PASTA_RAW) -> list[Path]:
    """Baixa o ZIP do UCI e extrai os arquivos necessários.

    Args:
        url: Endereço do ZIP oficial no UCI.
        destino: Pasta onde os arquivos serão salvos (criada se não existir).

    Returns:
        Lista com os caminhos dos arquivos extraídos.

    Raises:
        FileNotFoundError: Se algum arquivo esperado não estiver no ZIP.
    """
    destino.mkdir(parents=True, exist_ok=True)
    print(f"Baixando {url} ...")
    with urllib.request.urlopen(url, timeout=120) as resposta:
        conteudo = resposta.read()

    extraidos: list[Path] = []
    with zipfile.ZipFile(io.BytesIO(conteudo)) as zf:
        nomes = {Path(n).name: n for n in zf.namelist()}
        for arquivo in ARQUIVOS:
            if arquivo not in nomes:
                raise FileNotFoundError(f"{arquivo} não encontrado no ZIP: {sorted(nomes)}")
            caminho = destino / arquivo
            caminho.write_bytes(zf.read(nomes[arquivo]))
            extraidos.append(caminho)
            print(f"  salvo: {caminho}")
    return extraidos


def conferir_linhas(caminho: Path, esperado: int = LINHAS_ESPERADAS) -> int:
    """Conta as linhas de dados de um CSV e avisa se diferir do esperado.

    Args:
        caminho: CSV a conferir (cabeçalho não é contado).
        esperado: Número de linhas de dados esperado.

    Returns:
        Número de linhas de dados encontrado.
    """
    with caminho.open(encoding="utf-8") as f:
        n = sum(1 for _ in f) - 1
    status = "OK" if n == esperado else f"ATENÇÃO: esperado {esperado}"
    print(f"{caminho.name}: {n} linhas ({status})")
    return n


if __name__ == "__main__":
    baixados = baixar_e_extrair()
    conferir_linhas(PASTA_RAW / "diabetic_data.csv")
