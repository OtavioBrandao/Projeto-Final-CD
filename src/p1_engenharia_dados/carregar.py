"""Leitura dos arquivos brutos e integração com o ``IDS_mapping.csv``.

Responsabilidades deste módulo:

* ler ``diabetic_data.csv`` com a regra correta de valores ausentes;
* ler uma cópia "texto puro" do mesmo arquivo, só para provar na auditoria que
  os ``?`` existiam (plano, seção 4.1);
* ler o ``IDS_mapping.csv``, que na verdade contém **três tabelas** separadas por
  linhas em branco (plano, seção 4.2);
* juntar (*join*) os nomes legíveis aos códigos numéricos.
"""

from __future__ import annotations

import csv
import logging
import re
from pathlib import Path

import pandas as pd

from src.p1_engenharia_dados import configuracao as cfg

logger = logging.getLogger(__name__)

#: Nome da coluna de código de cada uma das três tabelas do IDS_mapping e a
#: coluna legível que criamos a partir dela (plano, seção 4.2).
MAPEAMENTOS: dict[str, str] = {
    "admission_type_id": "tipo_admissao",
    "discharge_disposition_id": "destino_alta",
    "admission_source_id": "origem_admissao",
}


def carregar_dados_brutos(caminho: Path) -> pd.DataFrame:
    """Lê ``diabetic_data.csv`` transformando **apenas** ``?`` em nulo.

    Por que ``keep_default_na=False``: o pandas considera nulos textos como
    ``"None"``, ``"NA"`` e ``"NULL"``. Em ``max_glu_serum`` e ``A1Cresult`` o
    texto ``"None"`` quer dizer "exame não feito" (D09), que é informação e não
    ausência. Se deixássemos o padrão, 94,7% de ``max_glu_serum`` viraria nulo
    por engano.

    Args:
        caminho: Caminho do ``diabetic_data.csv``.

    Returns:
        DataFrame com os dados brutos, ``?`` convertido em ``NaN``.

    Raises:
        FileNotFoundError: Se o arquivo não existir (rode ``baixar_base`` antes).
    """
    if not caminho.exists():
        raise FileNotFoundError(
            f"{caminho} não existe. Rode: python -m src.p1_engenharia_dados.baixar_base"
        )
    # low_memory=False: lê o arquivo inteiro de uma vez e evita o aviso de tipos
    # mistos (payer_code tem códigos e '?' na mesma coluna).
    df = pd.read_csv(
        caminho,
        na_values=[cfg.VALOR_NULO_BRUTO],
        keep_default_na=False,
        low_memory=False,
    )
    logger.info("Carga: %d linhas x %d colunas", *df.shape)
    return df


def carregar_dados_texto(caminho: Path) -> pd.DataFrame:
    """Lê o mesmo CSV sem nenhuma interpretação (tudo texto, nada vira nulo).

    Serve para a auditoria contar quantos ``?`` e quantos ``None`` existiam de
    fato no arquivo original.

    Args:
        caminho: Caminho do ``diabetic_data.csv``.

    Returns:
        DataFrame em que todas as colunas são texto exatamente como no arquivo.
    """
    return pd.read_csv(caminho, dtype=str, keep_default_na=False, na_filter=False)


def limpar_descricao(texto: str) -> str:
    """Padroniza uma descrição do ``IDS_mapping`` (decisão D04).

    Regras: colapsa espaços repetidos, tira espaços e pontos das pontas,
    corrige erros de digitação conhecidos e unifica ``NULL`` / ``Not Available``
    / ``Not Mapped`` / ``Unknown/Invalid`` em ``Nao_informado``.

    Args:
        texto: Descrição como aparece no arquivo (pode ter espaço à esquerda).

    Returns:
        Descrição limpa.
    """
    limpo = re.sub(r"\s+", " ", texto).strip(" .")
    for errado, certo in cfg.CORRECOES_TEXTO.items():
        limpo = limpo.replace(errado, certo)
    if limpo.lower() in cfg.DESCRICOES_NAO_INFORMADO:
        return cfg.ROTULO_NAO_INFORMADO
    return limpo


def carregar_ids_mapping(caminho: Path) -> dict[str, pd.DataFrame]:
    """Lê as três tabelas do ``IDS_mapping.csv``.

    O arquivo tem o formato::

        admission_type_id,description
        1,Emergency
        ...
        ,                          <- linha em branco separa as tabelas
        discharge_disposition_id,description
        ...

    Args:
        caminho: Caminho do ``IDS_mapping.csv``.

    Returns:
        Dicionário ``{nome_da_coluna_de_codigo: DataFrame}``. Cada DataFrame tem
        ``codigo`` (int), ``descricao_original`` (como no arquivo) e ``descricao``
        (já padronizada por :func:`limpar_descricao`).

    Raises:
        ValueError: Se alguma das três tabelas esperadas não for encontrada.
    """
    tabelas: dict[str, list[tuple[int, str]]] = {}
    atual: str | None = None
    # utf-8-sig: aceita arquivo com ou sem BOM.
    with caminho.open(encoding="utf-8-sig", newline="") as f:
        for linha in csv.reader(f):
            if not any(celula.strip() for celula in linha):
                # Linha em branco (",") = fim da tabela atual.
                atual = None
                continue
            if linha[0].strip().endswith("_id") and linha[1].strip() == "description":
                # Cabeçalho de uma nova tabela.
                atual = linha[0].strip()
                tabelas[atual] = []
                continue
            if atual is None:
                raise ValueError(f"Linha fora de tabela no mapeamento: {linha}")
            tabelas[atual].append((int(linha[0]), linha[1]))

    faltando = set(MAPEAMENTOS) - set(tabelas)
    if faltando:
        raise ValueError(f"Tabelas ausentes no IDS_mapping: {sorted(faltando)}")

    resultado: dict[str, pd.DataFrame] = {}
    for nome, linhas in tabelas.items():
        tabela = pd.DataFrame(linhas, columns=["codigo", "descricao_original"])
        tabela["descricao"] = tabela["descricao_original"].map(limpar_descricao)
        resultado[nome] = tabela
        logger.info("IDS_mapping: %s com %d códigos", nome, len(tabela))
    return resultado


def integrar_mapeamentos(df: pd.DataFrame, mapas: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Faz o *join* dos códigos numéricos com seus nomes legíveis (seção 4.2).

    Cria ``tipo_admissao``, ``destino_alta`` e ``origem_admissao``. As colunas de
    código originais continuam no DataFrame (são úteis para os filtros D02 e
    para o agrupamento F09); a seleção de atributos as remove da base final.

    Args:
        df: Base com ``admission_type_id``, ``discharge_disposition_id`` e
            ``admission_source_id``.
        mapas: Saída de :func:`carregar_ids_mapping`.

    Returns:
        Novo DataFrame (mesmas linhas, mesma ordem) com as três colunas novas.
        Códigos que não existirem no mapeamento viram ``Nao_informado``.
    """
    saida = df.copy()
    n_antes = len(saida)
    for coluna_codigo, coluna_nome in MAPEAMENTOS.items():
        tabela = mapas[coluna_codigo][["codigo", "descricao"]].rename(
            columns={"codigo": coluna_codigo, "descricao": coluna_nome}
        )
        # validate="many_to_one": cada código aparece uma única vez no mapeamento,
        # então o join não pode duplicar linhas.
        saida = saida.merge(tabela, on=coluna_codigo, how="left", validate="many_to_one")
        sem_nome = int(saida[coluna_nome].isna().sum())
        if sem_nome:
            logger.warning("%s: %d linhas com código fora do mapeamento", coluna_codigo, sem_nome)
            saida[coluna_nome] = saida[coluna_nome].fillna(cfg.ROTULO_NAO_INFORMADO)
    if len(saida) != n_antes:  # pragma: no cover - proteção contra join mal feito
        raise RuntimeError("O join com IDS_mapping alterou o número de linhas.")
    return saida
