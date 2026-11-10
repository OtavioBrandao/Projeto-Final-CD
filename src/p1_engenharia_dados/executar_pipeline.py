"""Orquestrador do pipeline da Pessoa 1: um comando refaz tudo.

Uso (a partir da raiz do repositório)::

    python -m src.p1_engenharia_dados.executar_pipeline

Fluxo (cada etapa registra linhas/colunas para o diagrama da documentação)::

    carga -> auditoria -> integração (IDS_mapping) -> padronização (D04)
          -> filtros (D01, D03, D02) -> ausentes (D05-D09) -> atributos (F01-F11)
          -> outliers (sinalização) -> seleção -> base final + dicionário

Saídas:

* ``data/interim/diabetes_limpo.csv``: base legível, com IDs;
* ``data/processed/diabetes_modelagem.csv`` (+ ``.parquet``): base para modelagem;
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
    logger.info("Etapa %-28s -> %7d linhas x %2d colunas", nome, len(df), df.shape[1])


def _aplicar_filtro(
    funil: list[dict[str, object]], id_: str, descricao: str, df: pd.DataFrame, depois: pd.DataFrame
) -> pd.DataFrame:
    """Registra uma linha do funil e devolve a base filtrada.

    Args:
        funil: Lista acumuladora do funil de linhas.
        id_: ID da decisão (D01, D02, D03).
        descricao: Texto curto do filtro.
        df: Base antes do filtro.
        depois: Base depois do filtro.

    Returns:
        ``depois`` (para encadear chamadas).
    """
    funil.append(
        {
            "etapa": id_,
            "descricao": descricao,
            "linhas_antes": len(df),
            "removidas": len(df) - len(depois),
            "pct_do_bruto_removido": round(
                100 * (len(df) - len(depois)) / int(funil[0]["linhas_antes"]), 3  # type: ignore[call-overload]
            ),
            "linhas_depois": len(depois),
        }
    )
    return depois


def _metricas_brutas(
    bruto: pd.DataFrame, perfil: pd.DataFrame, resumo: dict[str, int]
) -> Metricas:
    """Reúne as evidências medidas na base bruta (antes de qualquer tratamento).

    Args:
        bruto: Base bruta (``?`` como nulo).
        perfil: Saída de ``auditoria.perfil_qualidade``.
        resumo: Saída de ``auditoria.resumo_geral``.

    Returns:
        Dicionário de métricas que alimenta tabelas e documentação.
    """
    m: Metricas = {
        "linhas_bruto": resumo["linhas"],
        "colunas_bruto": resumo["colunas"],
        "pacientes_bruto": resumo["pacientes_unicos"],
        "encontros_repetidos_bruto": resumo["encontros_repetidos"],
        "pacientes_varios_encontros_bruto": resumo["pacientes_com_varios_encontros"],
        "linhas_duplicadas_inteiras_bruto": resumo["linhas_duplicadas_inteiras"],
        "n_interrogacoes_total": int(perfil["n_interrogacoes_no_arquivo"].sum()),
        "n_colunas_com_nulo": int((perfil["n_nulos"] > 0).sum()),
        "n_readmitted_no_maiusculo": int((bruto["readmitted"] == "NO").sum()),
        "pct_positivo_bruto": round(
            100 * float((bruto["readmitted"] == cfg.CLASSE_POSITIVA_READMITIDO).mean()), 2
        ),
    }
    for r in perfil.itertuples():
        if r.n_nulos > 0:
            m[f"n_nulo_{r.coluna}"] = r.n_nulos
            m[f"pct_nulo_{r.coluna}"] = r.pct_nulos
        if r.coluna in ("max_glu_serum", "A1Cresult"):
            m[f"n_none_{r.coluna.lower()}"] = r.n_textos_none
            m[f"pct_none_{r.coluna.lower()}"] = round(100 * r.n_textos_none / resumo["linhas"], 2)
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
    if not df["patient_nbr"].is_unique:
        problemas.append("patient_nbr repetido (D03 falhou)")
    if set(modelagem[cfg.COLUNA_ALVO].unique()) - {0, 1}:
        problemas.append("alvo com valores fora de 0/1")
    if modelagem.isna().any().any():
        problemas.append("base de modelagem ainda tem nulos")
    entrada, saida = funil[0]["linhas_antes"], funil[-1]["linhas_depois"]
    removidas = sum(int(f["removidas"]) for f in funil)  # type: ignore[call-overload]
    if entrada - removidas != saida or saida != len(df):  # type: ignore[operator]
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
    bruto = carregar.carregar_dados_brutos(caminhos.csv_bruto)
    texto_puro = carregar.carregar_dados_texto(caminhos.csv_bruto)
    mapas = carregar.carregar_ids_mapping(caminhos.csv_mapeamento)
    perfil = auditoria.perfil_qualidade(bruto, texto_puro)
    resumo = auditoria.resumo_geral(bruto)
    m = _metricas_brutas(bruto, perfil, resumo)
    _registrar_etapa(etapas, "Bruto", bruto)

    # ---- 2) Integração com o IDS_mapping ---------------------------------------
    integrado = carregar.integrar_mapeamentos(bruto, mapas)
    tab_integracao = auditoria.tabela_integracao(mapas, bruto)
    m["n_codigos_nao_informado"] = int(tab_integracao["unificado_em_nao_informado"].sum())
    m["n_descricoes_com_espaco_ou_ponto"] = int(tab_integracao["tinha_espaco_ou_ponto_extra"].sum())
    nomes_legiveis = list(carregar.MAPEAMENTOS.values())
    m["n_linhas_codigo_nao_informado"] = int(
        (integrado[nomes_legiveis] == cfg.ROTULO_NAO_INFORMADO).any(axis=1).sum()
    )
    _registrar_etapa(etapas, "Integração IDS_mapping", integrado)

    # ---- 3) D04: padronização de texto -----------------------------------------
    padronizado = limpeza.padronizar_texto(integrado)
    m["celulas_alteradas_d04"] = limpeza.contar_celulas_alteradas(integrado, padronizado)
    _registrar_etapa(etapas, "Padronização (D04)", padronizado)

    # ---- 4) Filtros de linhas (funil) ------------------------------------------
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
    codigos = limpeza.identificar_codigos_obito_hospice(mapas["discharge_disposition_id"])
    m["codigos_obito_hospice"] = ", ".join(str(c) for c in sorted(codigos))
    m["codigos_iguais_ao_plano"] = int(codigos == cfg.CODIGOS_ALTA_OBITO_HOSPICE_PLANO)
    if not m["codigos_iguais_ao_plano"]:
        logger.warning("Códigos de óbito/hospice no IDS_mapping diferem do plano: %s", sorted(codigos))
    m["obito_hospice_no_bruto"] = int(padronizado["discharge_disposition_id"].isin(codigos).sum())

    d01 = _aplicar_filtro(
        funil, "D01", "Sexo Unknown/Invalid", padronizado, limpeza.remover_genero_invalido(padronizado)
    )
    # DIVERGÊNCIA DO PLANO: D03 (1º encontro) antes de D02 (óbito/hospice).
    # Assim o paciente cujo PRIMEIRO encontro terminou em óbito/hospice sai
    # inteiro; na ordem do plano, um encontro posterior desse paciente (que
    # aparece em alguns registros) viraria o "primeiro" por engano.
    d03 = _aplicar_filtro(
        funil, "D03", "Só o 1º encontro por paciente", d01, limpeza.manter_primeiro_encontro(d01)
    )
    d02 = _aplicar_filtro(
        funil, "D02", "Óbito/hospice na alta", d03, limpeza.remover_obito_hospice(d03, codigos)
    )
    for chave, id_ in (("d01", "D01"), ("d02", "D02"), ("d03", "D03")):
        m[f"{chave}_removidas"] = next(int(f["removidas"]) for f in funil if f["etapa"] == id_)  # type: ignore[call-overload]
    # Prova numérica de que a ordem importa: refaz na ordem do plano (D02 -> D03).
    ordem_plano = limpeza.manter_primeiro_encontro(limpeza.remover_obito_hospice(d01, codigos))
    m["linhas_final_na_ordem_do_plano"] = len(ordem_plano)
    m["diferenca_ordem_plano"] = len(ordem_plano) - len(d02)
    _registrar_etapa(etapas, "Filtros D01, D03, D02", d02)

    # ---- 5) Ausentes (D05-D09) -------------------------------------------------
    tratado = limpeza.tratar_ausentes(d02)
    restantes = limpeza.nulos_restantes(tratado)
    if len(restantes):
        logger.warning("Nulos restantes (ficam para a Pessoa 3): %s", restantes.to_dict())
    m["nulos_restantes_total"] = int(restantes.sum())
    _registrar_etapa(etapas, "Ausentes (D05-D09)", tratado)

    # ---- 6) Engenharia de atributos (F01-F11) ----------------------------------
    top = engenharia_atributos.calcular_top_especialidades(
        tratado["medical_specialty"], cfg.TOP_N_ESPECIALIDADES
    )
    m["top_especialidades"] = "; ".join(top)
    com_atributos = engenharia_atributos.criar_atributos(tratado, top)
    _registrar_etapa(etapas, "Atributos (F01-F11)", com_atributos)

    # ---- 7) Outliers: só sinalizar ---------------------------------------------
    limites = outliers.calcular_limites_iqr(com_atributos)
    completo = outliers.sinalizar_outliers(com_atributos, limites)
    m["n_variaveis_outlier"] = len(limites)
    m["n_variaveis_com_outlier"] = int((limites["n_outliers"] > 0).sum())
    m["pct_outlier_min"] = float(limites["pct_outliers"].min())
    m["pct_outlier_max"] = float(limites["pct_outliers"].max())
    _registrar_etapa(etapas, "Outliers (sinalização)", completo)

    # ---- 8) Seleção de atributos + dicionário ----------------------------------
    modelagem, removidas, dominancia, correlacoes = selecao.selecionar_para_modelagem(completo)
    peso = pd.DataFrame(
        [
            {
                "coluna": "weight",
                "motivo": "muito_ausente",
                "detalhe": f"{relatorios.fmt_pct(float(m['pct_nulo_weight']))} de nulos (D05)",
            }
        ]
    )
    removidas = pd.concat([peso, removidas], ignore_index=True)
    dicionario = selecao.montar_dicionario(modelagem)
    _registrar_etapa(etapas, "Seleção (base final)", modelagem)

    m["colunas_interim"] = completo.shape[1]
    m["colunas_modelagem"] = modelagem.shape[1]
    m["n_colunas_removidas"] = len(removidas)
    m["n_quase_constantes"] = int(dominancia["quase_constante"].sum())
    m["n_correlacoes_altas"] = int(correlacoes["acima_do_limiar"].sum())
    maior = correlacoes.iloc[0]
    m["correlacao_maxima"] = float(maior["r_absoluto"])
    m["par_correlacao_maxima"] = f"{maior['coluna_a']} x {maior['coluna_b']}"
    m["linhas_final"] = len(modelagem)
    m["pct_linhas_mantidas"] = round(100 * len(modelagem) / int(m["linhas_bruto"]), 2)  # type: ignore[call-overload]
    sem_linhas = sorted(c for c in codigos if int((padronizado["discharge_disposition_id"] == c).sum()) == 0)
    m["codigos_obito_hospice_sem_linhas"] = ", ".join(str(c) for c in sem_linhas)
    m["variaveis_iqr_zero"] = ", ".join(limites.loc[limites["iqr"] == 0, "variavel"])
    m["n_variaveis_iqr_zero"] = int((limites["iqr"] == 0).sum())
    meds_fora = set(removidas.loc[removidas["motivo"] == "quase_constante", "coluna"]) & set(
        cfg.COLUNAS_MEDICAMENTOS
    )
    m["n_meds_total"] = len(cfg.COLUNAS_MEDICAMENTOS)
    m["n_meds_quase_constantes"] = len(meds_fora)
    m["n_meds_mantidos"] = len(cfg.COLUNAS_MEDICAMENTOS) - len(meds_fora)
    contagem_tipos = dicionario["tipo"].value_counts()
    for tipo in ("numerica", "categorica", "binaria", "alvo"):
        m[f"n_colunas_{tipo}"] = int(contagem_tipos.get(tipo, 0))
    m["n_colunas_sensiveis"] = int((dicionario["sensivel"] == "sim").sum())
    m["n_colunas_derivadas_modelagem"] = int((dicionario["origem"] == "derivada").sum())
    m["n_positivo_final"] = int(modelagem[cfg.COLUNA_ALVO].sum())
    m["pct_positivo_final"] = round(100 * float(modelagem[cfg.COLUNA_ALVO].mean()), 2)

    _verificar_final(completo, modelagem, funil)

    # ---- 9) Gravação dos dados -------------------------------------------------
    _gravar_dados(caminhos, completo, modelagem, dicionario)

    # ---- 10) Tabelas, figuras e textos de docs/ --------------------------------
    if gerar_documentacao:
        _gerar_relatorios(
            caminhos, m, bruto, padronizado, completo, modelagem, perfil, tab_integracao, funil,
            etapas, limites, removidas, dominancia, correlacoes, dicionario,
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
        completo: Base legível (interim), com IDs.
        modelagem: Base final.
        dicionario: Dicionário de dados.
    """
    for pasta in (caminhos.interim, caminhos.processed):
        pasta.mkdir(parents=True, exist_ok=True)
    opcoes = {"index": False, "encoding": "utf-8", "lineterminator": "\n"}
    completo.to_csv(caminhos.csv_limpo, **opcoes)
    modelagem.to_csv(caminhos.csv_modelagem, **opcoes)
    modelagem.to_parquet(caminhos.parquet_modelagem, index=False, engine="pyarrow")
    dicionario.to_csv(caminhos.csv_dicionario, **opcoes)
    logger.info("Dados gravados em %s e %s", caminhos.interim, caminhos.processed)


def _gerar_relatorios(
    caminhos: Caminhos,
    m: Metricas,
    bruto: pd.DataFrame,
    padronizado: pd.DataFrame,
    completo: pd.DataFrame,
    modelagem: pd.DataFrame,
    perfil: pd.DataFrame,
    tab_integracao: pd.DataFrame,
    funil: list[dict[str, object]],
    etapas: list[dict[str, object]],
    limites: pd.DataFrame,
    removidas: pd.DataFrame,
    dominancia: pd.DataFrame,
    correlacoes: pd.DataFrame,
    dicionario: pd.DataFrame,
) -> None:
    """Grava as tabelas de auditoria (CSV) e as figuras (PNG) em ``docs/``.

    Args:
        caminhos: Caminhos de saída.
        m: Métricas medidas (é completada com números das tabelas).
        bruto: Base bruta.
        padronizado: Base após integração e D04 (antes dos filtros).
        completo: Base final com todas as colunas (interim).
        modelagem: Base de modelagem.
        perfil: Auditoria inicial.
        tab_integracao: Tabela código -> nome.
        funil: Linhas do funil.
        etapas: Tamanho da base após cada etapa.
        limites: Limites IQR por variável.
        removidas: Colunas removidas e motivo.
        dominancia: Dominância por coluna.
        correlacoes: Pares de correlação.
        dicionario: Dicionário de dados.
    """
    r = relatorios
    t = caminhos.tabelas
    funil_df = pd.DataFrame(funil)
    cardinalidade = r.tabela_cardinalidade(padronizado, completo)
    balanceamento = r.tabela_balanceamento(padronizado["readmitted"], completo[cfg.COLUNA_ALVO])
    limiar = r.tabela_limiar_usuario(completo)

    # Números extraídos das tabelas para o texto.
    diag1 = cardinalidade.set_index("coluna_antes").loc["diag_1"]
    m["n_codigos_diag1_antes"] = int(diag1["n_distintos_antes"])
    # O grupo "Sem_diagnostico" (D08) não é um grupo clínico: não entra na contagem.
    m["n_grupos_diag1_depois"] = int(
        (completo["grupo_diag_1"].drop_duplicates() != cfg.ROTULO_SEM_DIAGNOSTICO).sum()
    )
    m["pct_usuario_frequente"] = round(100 * float(completo["usuario_frequente"].mean()), 2)
    m["taxa_readmissao_0_internacoes"] = float(limiar.iloc[0]["taxa_readmissao_pct"])
    m["taxa_readmissao_1_internacao"] = float(limiar.iloc[1]["taxa_readmissao_pct"])
    m["taxa_readmissao_usuario_frequente"] = round(
        100 * float(completo.loc[completo["usuario_frequente"] == 1, cfg.COLUNA_ALVO].mean()), 2
    )
    m["taxa_readmissao_nao_frequente"] = round(
        100 * float(completo.loc[completo["usuario_frequente"] == 0, cfg.COLUNA_ALVO].mean()), 2
    )

    r.salvar_tabela(perfil, "auditoria_inicial", t)
    r.salvar_tabela(tab_integracao, "integracao_ids", t)
    r.salvar_tabela(funil_df, "funil_linhas", t)
    r.salvar_tabela(pd.DataFrame(etapas), "etapas_pipeline", t)
    r.salvar_tabela(r.tabela_ausentes(perfil, completo), "ausentes_antes_depois", t)
    r.salvar_tabela(r.montar_tabela_decisoes(m), "decisoes", t)
    r.salvar_tabela(r.tabela_atributos_criados(completo), "atributos_criados", t)
    r.salvar_tabela(limites, "outliers", t)
    r.salvar_tabela(removidas, "colunas_removidas", t)
    r.salvar_tabela(dominancia, "dominancia_colunas", t)
    r.salvar_tabela(correlacoes, "correlacoes", t)
    r.salvar_tabela(cardinalidade, "cardinalidade_antes_depois", t)
    r.salvar_tabela(balanceamento, "balanceamento_alvo", t)
    r.salvar_tabela(limiar, "limiar_usuario_frequente", t)
    metricas_df = pd.DataFrame({"chave": list(m), "valor": [str(v) for v in m.values()]})
    r.salvar_tabela(metricas_df, "metricas", t)

    f = caminhos.figuras
    r.figura_funil(funil_df, f / "funil_linhas.png")
    r.figura_ausentes(perfil, int(m["linhas_bruto"]), f / "mapa_ausentes.png")  # type: ignore[call-overload]
    r.figura_outliers(completo, limites, f / "outliers_boxplots.png")
    r.figura_alvo(balanceamento, f / "antes_depois_alvo.png")
    r.figura_outlier_antes_depois(bruto, completo, limites, f / "antes_depois_outliers.png")
    r.figura_diagnosticos(bruto, completo, f / "antes_depois_diagnosticos.png")
    r.figura_limiar_usuario(limiar, f / "limiar_usuario_frequente.png")
    logger.info("Tabelas em %s e figuras em %s", t, f)


def main() -> None:
    """Ponto de entrada de linha de comando."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
    logging.getLogger("matplotlib").setLevel(logging.WARNING)  # silencia avisos de categorias
    parser = argparse.ArgumentParser(description="Executa o pipeline de tratamento (Pessoa 1).")
    parser.add_argument(
        "--sem-documentacao", action="store_true", help="gera só os dados, sem docs/"
    )
    executar(gerar_documentacao=not parser.parse_args().sem_documentacao)


if __name__ == "__main__":
    main()
