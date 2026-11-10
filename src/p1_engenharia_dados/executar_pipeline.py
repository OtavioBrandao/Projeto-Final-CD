"""Orquestrador do tratamento de dados: um comando refaz tudo.

Uso (a partir da raiz do repositório)::

    python -m src.p1_engenharia_dados.executar_pipeline

Fluxo (cada etapa registra linhas/colunas para o diagrama da documentação)::

    carga -> auditoria -> pessoas (D04-D06) -> resumo por acidente -> integração (I01)
          -> padronização (D04) -> filtros (D01, D02) -> inconsistências (D03, D07)
          -> atributos (F01-F11) -> outliers -> seleção -> base final + dicionário

Saídas:

* ``data/interim/acidentes_prf_limpo.csv``: base legível, com ID e consequências;
* ``data/processed/acidentes_prf_modelagem.csv`` (+ ``.parquet``): base para modelagem;
* ``data/processed/dicionario_de_dados.csv``;
* ``docs/tabelas/*.csv`` e ``docs/figuras/*.png``: evidências e gráficos;
* ``docs/README.md`` e ``docs/secao_base_de_dados.md``: textos gerados com os
  números das tabelas.

O pipeline é determinístico: rodar duas vezes gera arquivos idênticos.
"""

from __future__ import annotations

import argparse
import logging
import random

import numpy as np
import pandas as pd

from src.p1_engenharia_dados import auditoria, carregar, engenharia_atributos, limpeza
from src.p1_engenharia_dados import configuracao as cfg
from src.p1_engenharia_dados import outliers, relatorios, selecao
from src.p1_engenharia_dados.configuracao import Caminhos

logger = logging.getLogger(__name__)

Metricas = dict[str, object]


def _registrar_etapa(etapas: list[dict[str, object]], nome: str, df: pd.DataFrame) -> None:
    """Guarda o tamanho da base após uma etapa (alimenta o diagrama Mermaid).

    Args:
        etapas: Lista acumuladora.
        nome: Nome curto da etapa.
        df: Base ao final da etapa.
    """
    etapas.append({"etapa": nome, "linhas": len(df), "colunas": df.shape[1]})
    logger.info("Etapa %-32s -> %7d linhas x %2d colunas", nome, len(df), df.shape[1])


def _aplicar_filtro(
    funil: list[dict[str, object]], id_: str, descricao: str, df: pd.DataFrame, depois: pd.DataFrame
) -> pd.DataFrame:
    """Registra uma linha do funil e devolve a base filtrada.

    Args:
        funil: Lista acumuladora do funil de linhas.
        id_: ID da decisão (D01, D02).
        descricao: Texto curto do filtro.
        df: Base antes do filtro.
        depois: Base depois do filtro.

    Returns:
        ``depois`` (para encadear chamadas).
    """
    removidas = len(df) - len(depois)
    funil.append(
        {
            "etapa": id_,
            "descricao": descricao,
            "linhas_antes": len(df),
            "removidas": removidas,
            "pct_do_bruto_removido": round(100 * removidas / int(funil[0]["linhas_antes"]), 3),  # type: ignore[call-overload]
            "linhas_depois": len(depois),
        }
    )
    return depois


def _taxa_fatal(base: pd.DataFrame, mascara: pd.Series) -> float:
    """Percentual de acidentes fatais dentro de um subconjunto.

    Args:
        base: Base com o alvo.
        mascara: Filtro booleano do subconjunto.

    Returns:
        Taxa em % (0 a 100), arredondada a 2 casas.
    """
    return round(100 * float(base.loc[mascara, cfg.COLUNA_ALVO].mean()), 2)


def _metricas_brutas(
    oc: pd.DataFrame, pes: pd.DataFrame, resumo: dict[str, int], perfil_oc: pd.DataFrame, perfil_pes: pd.DataFrame
) -> Metricas:
    """Reúne as evidências medidas nas bases brutas (antes de qualquer tratamento).

    Args:
        oc: Ocorrências brutas (``NA`` como nulo).
        pes: Pessoas brutas.
        resumo: Saída de ``auditoria.resumo_geral``.
        perfil_oc: Auditoria das ocorrências.
        perfil_pes: Auditoria das pessoas.

    Returns:
        Dicionário de métricas que alimenta tabelas e documentação.
    """
    por_ano = oc["ano_arquivo"].value_counts()
    sexo_sem_info = pes["sexo"].isna() | pes["sexo"].isin(cfg.ROTULOS_SEM_INFORMACAO)
    m: Metricas = {
        "linhas_ocorrencias_bruto": resumo["linhas_ocorrencias"],
        "colunas_ocorrencias_bruto": resumo["colunas_ocorrencias"],
        "linhas_pessoas_bruto": resumo["linhas_pessoas"],
        "colunas_pessoas_bruto": resumo["colunas_pessoas"],
        "ocorrencias_sem_pessoas": resumo["ocorrencias_sem_pessoas"],
        "pessoas_sem_ocorrencia": resumo["pessoas_sem_ocorrencia"],
        "ocorrencias_duplicadas_sem_id": resumo["ocorrencias_duplicadas_sem_id"],
        "id_unico": resumo["id_unico"],
        "n_ocorrencias_2024": int(por_ano.get(2024, 0)),
        "n_ocorrencias_2025": int(por_ano.get(2025, 0)),
        "n_na_classificacao": int(oc["classificacao_acidente"].isna().sum()),
        "n_clima_ignorado": int((oc["condicao_metereologica"] == "Ignorado").sum()),
        "n_sentido_nao_informado": int((oc["sentido_via"] == "Não Informado").sum()),
        "n_pessoas_sexo_sem_info": int(sexo_sem_info.sum()),
        "pct_pessoas_sexo_sem_info": round(100 * float(sexo_sem_info.mean()), 2),
        "n_pessoas_tipo_veiculo_nulo": int(pes["tipo_veiculo"].isna().sum()),
        "n_causas_distintas": int(oc["causa_acidente"].nunique()),
        "n_tipos_acidente": int(oc["tipo_acidente"].nunique()),
        "n_municipios": int(oc["municipio"].nunique()),
        "n_combinacoes_tracado": int(oc["tracado_via"].nunique()),
        "n_ufs": int(oc["uf"].nunique()),
        "n_rodovias": int(oc["br"].nunique()),
        "n_latitude_positiva": int((oc["latitude"] > 0).sum()),
        "n_latitude_positiva_valida": limpeza.contar_coordenadas_positivas_validas(oc),
        "idade_maxima_bruta": int(pes["idade"].max()),
        "veiculos_maximo_bruto": int(oc["veiculos"].max()),
        "pessoas_maximo_bruto": int(oc["pessoas"].max()),
        "km_maximo_bruto": float(oc["km"].max()),
        "pct_fatal_bruto": round(100 * float((oc["mortos"] > 0).mean()), 2),
        "n_fatais_bruto": int((oc["mortos"] > 0).sum()),
        "n_colunas_com_nulo_ocorrencias": int((perfil_oc["n_nulos"] > 0).sum()),
        "n_rotulos_equivalentes_caixa_acento": auditoria.contar_rotulos_equivalentes(
            oc, cfg.COLUNAS_CATEGORICAS_PADRONIZAR
        ),
    }
    return m


def _verificar_final(df: pd.DataFrame, modelagem: pd.DataFrame, funil: list[dict[str, object]]) -> None:
    """Confere invariantes do resultado; interrompe o pipeline se algo estiver errado.

    Args:
        df: Base completa (interim).
        modelagem: Base final.
        funil: Linhas do funil.

    Raises:
        RuntimeError: Se alguma verificação falhar.
    """
    problemas: list[str] = []
    if not df[cfg.COLUNA_ID].is_unique:
        problemas.append("id repetido")
    if set(modelagem[cfg.COLUNA_ALVO].unique()) - {0, 1}:
        problemas.append("alvo com valores fora de 0/1")
    vazamento = [c for c in cfg.COLUNAS_RESULTADO if c in modelagem.columns]
    if vazamento:
        problemas.append(f"colunas de consequencia na modelagem: {vazamento}")
    entrada, saida = int(funil[0]["linhas_antes"]), int(funil[-1]["linhas_depois"])  # type: ignore[call-overload]
    removidas = sum(int(f["removidas"]) for f in funil)  # type: ignore[call-overload]
    if entrada - removidas != saida or saida != len(df):
        problemas.append("funil de linhas não fecha")
    if problemas:
        raise RuntimeError("Falha nas verificações finais: " + "; ".join(problemas))


def executar(caminhos: Caminhos | None = None, gerar_documentacao: bool = True) -> Metricas:
    """Executa o pipeline completo e grava todas as saídas.

    Args:
        caminhos: Caminhos de entrada/saída. Se ``None``, usa os do repositório.
        gerar_documentacao: Se ``True``, também gera tabelas, figuras e os
            textos de ``docs/``. Os arquivos de dados são sempre gerados.

    Returns:
        Dicionário com as métricas medidas (as mesmas gravadas em
        ``docs/tabelas/metricas.csv``).

    Raises:
        FileNotFoundError: Se os dados brutos ainda não foram baixados.
        RuntimeError: Se alguma verificação final falhar.
    """
    caminhos = caminhos or Caminhos()
    # Fixa as sementes por segurança. O pipeline não sorteia nada.
    random.seed(cfg.SEMENTE_ALEATORIA)
    np.random.seed(cfg.SEMENTE_ALEATORIA)

    etapas: list[dict[str, object]] = []
    funil: list[dict[str, object]] = []

    # ---- 1) Carga e auditoria inicial (nada foi alterado ainda) ----------------
    oc_bruto = carregar.carregar_ocorrencias(caminhos)
    pes_bruto = carregar.carregar_pessoas(caminhos)
    perfil_oc = auditoria.perfil_qualidade(oc_bruto, carregar.carregar_ocorrencias(caminhos, como_texto=True), "ocorrencias")
    perfil_pes = auditoria.perfil_qualidade(pes_bruto, carregar.carregar_pessoas(caminhos, como_texto=True), "pessoas")
    resumo = auditoria.resumo_geral(oc_bruto, pes_bruto)
    m = _metricas_brutas(oc_bruto, pes_bruto, resumo, perfil_oc, perfil_pes)
    _registrar_etapa(etapas, "Bruto (ocorrências)", oc_bruto.drop(columns="ano_arquivo"))

    # ---- 2) Pessoas: padronização (D04), idade (D05), ano de fabricação (D06) --
    pes = limpeza.padronizar_rotulos(pes_bruto, ["tipo_veiculo", "tipo_envolvido", "estado_fisico", "sexo"])
    m["n_pessoas_idade_invalida"] = int(
        (~pes["idade"].between(cfg.IDADE_MINIMA, cfg.IDADE_MAXIMA) & pes["idade"].notna()).sum()
    )
    condutores = pes["tipo_envolvido"] == "Condutor"
    m["n_condutores_total"] = int(condutores.sum())
    m["n_condutores_idade_menor_10"] = int((condutores & (pes["idade"] < cfg.IDADE_MINIMA_CONDUTOR)).sum())
    m["pct_condutores_idade_menor_10"] = round(100 * m["n_condutores_idade_menor_10"] / m["n_condutores_total"], 2)  # type: ignore[operator]
    pes = limpeza.invalidar_idade(pes)
    ano_fab = pes["ano_fabricacao_veiculo"]
    m["n_pessoas_ano_fab_zero"] = int((ano_fab == 0).sum())
    n_antes_fab = int(ano_fab.isna().sum())
    pes = limpeza.invalidar_ano_fabricacao(pes)
    m["n_pessoas_ano_fab_invalido"] = int(pes["ano_fabricacao_veiculo"].isna().sum()) - n_antes_fab
    m["pct_pessoas_ano_fab_invalido"] = round(100 * m["n_pessoas_ano_fab_invalido"] / len(pes), 2)  # type: ignore[operator]

    # ---- 3) Resumo por acidente e integração (I01) ------------------------------
    resumo_pessoas = engenharia_atributos.resumir_pessoas(pes)
    m["n_colunas_resumo_pessoas"] = resumo_pessoas.shape[1] - 1  # sem a coluna id
    integrado = carregar.integrar_pessoas(oc_bruto, resumo_pessoas)
    _registrar_etapa(etapas, "Integração com pessoas (I01)", integrado.drop(columns="ano_arquivo"))

    # ---- 4) D04: padronização dos rótulos nas ocorrências ------------------------
    padronizado = limpeza.padronizar_rotulos(integrado, cfg.COLUNAS_CATEGORICAS_PADRONIZAR)
    m["celulas_alteradas_d04"] = limpeza.contar_celulas_alteradas(integrado, padronizado)
    _registrar_etapa(etapas, "Padronização de rótulos (D04)", padronizado.drop(columns="ano_arquivo"))

    # ---- 5) Filtros de linhas (funil) -------------------------------------------
    funil.append(
        {
            "etapa": "Entrada",
            "descricao": "Base bruta",
            "linhas_antes": len(padronizado),
            "removidas": 0,
            "pct_do_bruto_removido": 0.0,
            "linhas_depois": len(padronizado),
        }
    )
    d01 = _aplicar_filtro(funil, "D01", "Acidentes duplicados", padronizado, limpeza.remover_duplicados(padronizado))
    m["n_causa_intencional"] = int((d01["causa_acidente"] == cfg.CAUSA_INTENCIONAL).sum())
    m["n_tipo_pessoal"] = int((d01["tipo_acidente"] == cfg.TIPO_PESSOAL).sum())
    d02 = _aplicar_filtro(
        funil, "D02", "Suicídio presumido e sinistro pessoal", d01, limpeza.remover_eventos_intencionais(d01)
    )
    for chave, id_ in (("d01", "D01"), ("d02", "D02")):
        m[f"{chave}_removidas"] = next(int(f["removidas"]) for f in funil if f["etapa"] == id_)  # type: ignore[call-overload]
    m["n_fatais_removidos_d02"] = int((d01["mortos"] > 0).sum() - (d02["mortos"] > 0).sum())
    _registrar_etapa(etapas, "Filtros D01 e D02", d02.drop(columns="ano_arquivo"))

    # ---- 6) Inconsistências que só sinalizamos ou anulamos (D03, D07) -----------
    com_flag = limpeza.sinalizar_contagens_inconsistentes(d02)
    m["n_contagens_inconsistentes"] = int(com_flag["contagens_inconsistentes"].sum())
    m["pct_contagens_inconsistentes"] = round(100 * float(com_flag["contagens_inconsistentes"].mean()), 2)
    coords = limpeza.invalidar_coordenadas(com_flag)
    m["n_coordenadas_invalidas"] = int((coords["latitude"].isna() & com_flag["latitude"].notna()).sum())
    _registrar_etapa(etapas, "Inconsistências (D03, D07)", coords.drop(columns="ano_arquivo"))

    # ---- 7) Engenharia de atributos (F01-F11) -----------------------------------
    com_atributos = engenharia_atributos.criar_atributos(coords)
    m["n_grupos_causa"] = int(com_atributos["grupo_causa"].nunique())
    m["n_feriados_nos_dados"] = int(com_atributos["eh_feriado"].sum())
    m["pct_feriados_nos_dados"] = round(100 * float(com_atributos["eh_feriado"].mean()), 2)
    _registrar_etapa(etapas, "Atributos (F01-F11)", com_atributos.drop(columns="ano_arquivo"))

    # ---- 8) Outliers: só sinalizar ---------------------------------------------
    limites = outliers.calcular_limites_iqr(com_atributos)
    completo = outliers.sinalizar_outliers(com_atributos, limites)
    m["n_variaveis_outlier"] = len(limites)
    m["n_variaveis_com_outlier"] = int((limites["n_outliers"] > 0).sum())
    m["pct_outlier_min"] = float(limites["pct_outliers"].min())
    m["pct_outlier_max"] = float(limites["pct_outliers"].max())
    _registrar_etapa(etapas, "Outliers (sinalização)", completo.drop(columns="ano_arquivo"))

    # ---- 9) Seleção de atributos + dicionário -----------------------------------
    modelagem, removidas, dominancia, correlacoes = selecao.selecionar_para_modelagem(completo)
    dicionario = selecao.montar_dicionario(modelagem)
    _registrar_etapa(etapas, "Seleção (base final)", modelagem)

    m["colunas_interim"] = completo.shape[1]
    m["colunas_modelagem"] = modelagem.shape[1]
    m["n_colunas_removidas"] = len(removidas)
    m["n_colunas_resultado_removidas"] = int((removidas["motivo"] == "consequencia").sum())
    m["n_quase_constantes"] = int(dominancia["quase_constante"].sum())
    m["n_correlacoes_altas"] = int(correlacoes["acima_do_limiar"].sum())
    maior = correlacoes.iloc[0]
    m["correlacao_maxima"] = float(maior["r_absoluto"])
    m["par_correlacao_maxima"] = f"{maior['coluna_a']} x {maior['coluna_b']}"
    m["linhas_final"] = len(modelagem)
    m["pct_linhas_mantidas"] = round(100 * len(modelagem) / int(m["linhas_ocorrencias_bruto"]), 2)  # type: ignore[call-overload]
    m["n_fatais_final"] = int(modelagem[cfg.COLUNA_ALVO].sum())
    m["pct_fatal_final"] = round(100 * float(modelagem[cfg.COLUNA_ALVO].mean()), 2)
    m["variaveis_iqr_zero"] = ", ".join(limites.loc[limites["iqr"] == 0, "variavel"])
    m["n_variaveis_iqr_zero"] = int((limites["iqr"] == 0).sum())
    for coluna in ("idade_media_condutores", "pct_condutores_homens", "idade_media_veiculos", "latitude", "longitude"):
        m[f"n_nulo_final_{coluna}"] = int(modelagem[coluna].isna().sum())
        m[f"pct_nulo_final_{coluna}"] = round(100 * float(modelagem[coluna].isna().mean()), 2)
    nulos = limpeza.nulos_restantes(modelagem)
    m["n_colunas_com_nulo_final"] = len(nulos)
    m["colunas_com_nulo_final"] = ", ".join(nulos.index)
    contagem_tipos = dicionario["tipo"].value_counts()
    for tipo in ("numerica", "categorica", "binaria", "alvo"):
        m[f"n_colunas_{tipo}"] = int(contagem_tipos.get(tipo, 0))
    m["n_colunas_sensiveis"] = int((dicionario["sensivel"] == "sim").sum())
    m["n_colunas_derivadas_modelagem"] = int((dicionario["origem"] == "derivada").sum())
    m["n_acidentes_alagoas"] = int((modelagem["uf"] == "AL").sum())

    _verificar_final(completo, modelagem, funil)

    # ---- 10) Gravação dos dados -------------------------------------------------
    _gravar_dados(caminhos, completo, modelagem, dicionario)

    # ---- 11) Tabelas, figuras e textos de docs/ ---------------------------------
    if gerar_documentacao:
        _gerar_relatorios(
            caminhos, m, oc_bruto, completo, modelagem, perfil_oc, perfil_pes, funil, etapas,
            limites, removidas, dominancia, correlacoes,
        )  # fmt: skip
        from src.p1_engenharia_dados import documentacao  # import tardio: evita ciclo

        documentacao.gerar_documentacao(caminhos)
    logger.info("Pipeline concluído: %d linhas x %d colunas na base de modelagem.", *modelagem.shape)
    return m


def _gravar_dados(
    caminhos: Caminhos, completo: pd.DataFrame, modelagem: pd.DataFrame, dicionario: pd.DataFrame
) -> None:
    """Grava as bases e o dicionário (CSV com ``\\n`` e Parquet).

    Args:
        caminhos: Caminhos de saída.
        completo: Base legível (interim), com ID e consequências.
        modelagem: Base final.
        dicionario: Dicionário de dados.
    """
    for pasta in (caminhos.interim, caminhos.processed):
        pasta.mkdir(parents=True, exist_ok=True)
    opcoes = {"index": False, "encoding": "utf-8", "lineterminator": "\n"}
    completo.drop(columns="ano_arquivo").to_csv(caminhos.csv_limpo, **opcoes)
    modelagem.to_csv(caminhos.csv_modelagem, **opcoes)
    modelagem.to_parquet(caminhos.parquet_modelagem, index=False, engine="pyarrow")
    dicionario.to_csv(caminhos.csv_dicionario, **opcoes)
    logger.info("Dados gravados em %s e %s", caminhos.interim, caminhos.processed)


def _gerar_relatorios(
    caminhos: Caminhos,
    m: Metricas,
    bruto: pd.DataFrame,
    completo: pd.DataFrame,
    modelagem: pd.DataFrame,
    perfil_oc: pd.DataFrame,
    perfil_pes: pd.DataFrame,
    funil: list[dict[str, object]],
    etapas: list[dict[str, object]],
    limites: pd.DataFrame,
    removidas: pd.DataFrame,
    dominancia: pd.DataFrame,
    correlacoes: pd.DataFrame,
) -> None:
    """Grava as tabelas de auditoria (CSV) e as figuras (PNG) em ``docs/``.

    Args:
        caminhos: Caminhos de saída.
        m: Métricas medidas (é completada com números das tabelas).
        bruto: Ocorrências brutas.
        completo: Base final com todas as colunas (interim).
        modelagem: Base de modelagem.
        perfil_oc: Auditoria das ocorrências.
        perfil_pes: Auditoria das pessoas.
        funil: Linhas do funil.
        etapas: Tamanho da base após cada etapa.
        limites: Limites IQR por variável.
        removidas: Colunas removidas e motivo.
        dominancia: Dominância por coluna.
        correlacoes: Pares de correlação.
    """
    r = relatorios
    t = caminhos.tabelas
    funil_df = pd.DataFrame(funil)
    perfil = pd.concat([perfil_oc, perfil_pes], ignore_index=True)
    cardinalidade = r.tabela_cardinalidade(bruto, completo)
    balanceamento = r.tabela_balanceamento(bruto["classificacao_acidente"], completo[cfg.COLUNA_ALVO])
    taxas = r.tabela_taxa_fatal(completo)

    # Números extraídos das tabelas para o texto.
    base = taxas.set_index(["dimensao", "grupo"])["taxa_fatal_pct"]
    for chave, dimensao, grupo in (
        ("taxa_fatal_madrugada", "faixa_horaria", "Madrugada"),
        ("taxa_fatal_manha", "faixa_horaria", "Manha"),
        ("taxa_fatal_tarde", "faixa_horaria", "Tarde"),
        ("taxa_fatal_noite", "faixa_horaria", "Noite"),
        ("taxa_fatal_pedestre", "envolve_pedestre", "1"),
        ("taxa_fatal_moto", "envolve_moto", "1"),
        ("taxa_fatal_sem_moto", "envolve_moto", "0"),
        ("taxa_fatal_alcool", "grupo_causa", "Alcool_e_substancias"),
        ("taxa_fatal_velocidade", "grupo_causa", "Velocidade_e_distancia"),
        ("taxa_fatal_pista_simples", "tipo_pista", "Simples"),
        ("taxa_fatal_pista_dupla", "tipo_pista", "Dupla"),
    ):
        m[chave] = float(base.loc[(dimensao, grupo)])
    m["taxa_fatal_geral"] = float(base.loc[("geral", "todos")])
    card = cardinalidade.set_index("coluna_antes")
    m["n_causas_antes"] = int(card.loc["causa_acidente", "n_distintos_antes"])
    m["n_grupos_causa_depois"] = int(card.loc["causa_acidente", "n_distintos_depois"])

    r.salvar_tabela(perfil, "auditoria_inicial", t)
    r.salvar_tabela(funil_df, "funil_linhas", t)
    r.salvar_tabela(pd.DataFrame(etapas), "etapas_pipeline", t)
    ausentes = r.tabela_ausentes(perfil, completo, m)
    r.salvar_tabela(ausentes, "ausentes_antes_depois", t)
    r.salvar_tabela(r.montar_tabela_decisoes(m), "decisoes", t)
    r.salvar_tabela(r.tabela_atributos_criados(completo), "atributos_criados", t)
    r.salvar_tabela(limites, "outliers", t)
    r.salvar_tabela(removidas, "colunas_removidas", t)
    r.salvar_tabela(dominancia, "dominancia_colunas", t)
    r.salvar_tabela(correlacoes, "correlacoes", t)
    r.salvar_tabela(cardinalidade, "cardinalidade_antes_depois", t)
    r.salvar_tabela(balanceamento, "balanceamento_alvo", t)
    r.salvar_tabela(taxas, "taxa_fatal_por_grupo", t)
    metricas_df = pd.DataFrame({"chave": list(m), "valor": [str(v) for v in m.values()]})
    r.salvar_tabela(metricas_df, "metricas", t)

    f = caminhos.figuras
    r.figura_funil(funil_df, f / "funil_linhas.png")
    r.figura_ausentes(ausentes, f / "mapa_ausentes.png")
    r.figura_outliers(completo, limites, f / "outliers_boxplots.png")
    r.figura_alvo(balanceamento, f / "antes_depois_alvo.png")
    r.figura_outlier_antes_depois(bruto, completo, limites, f / "antes_depois_outliers.png")
    r.figura_causas(bruto, completo, f / "antes_depois_causas.png")
    r.figura_taxa_fatal(taxas, f / "taxa_fatal_por_grupo.png")
    logger.info("Tabelas em %s e figuras em %s", t, f)


def main() -> None:
    """Ponto de entrada de linha de comando."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
    logging.getLogger("matplotlib").setLevel(logging.WARNING)  # silencia avisos de categorias
    parser = argparse.ArgumentParser(description="Executa o tratamento de dados (PRF).")
    parser.add_argument(
        "--sem-documentacao", action="store_true", help="gera só os dados, sem docs/"
    )
    executar(gerar_documentacao=not parser.parse_args().sem_documentacao)


if __name__ == "__main__":
    main()
