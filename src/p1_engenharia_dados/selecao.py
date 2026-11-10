"""Seleção e redução de atributos + dicionário de dados.

Monta a base de modelagem removendo colunas que não devem (ou não precisam)
chegar ao modelo, sempre registrando **o que saiu e por quê**:

1. identificadores (``encounter_id``, ``patient_nbr``): não são características
   do paciente e deixariam o modelo "decorar" linhas;
2. colunas substituídas por uma versão mais útil (código -> nome, 800 códigos
   CID-9 -> 9 grupos, etc.) e o ``readmitted`` original (contém a resposta);
3. colunas constantes ou quase constantes (um único valor em >= 99,9% das linhas);
4. colunas numéricas redundantes (|correlação de Pearson| > 0,9).

Nenhum dos critérios usa o alvo, portanto não há vazamento de informação.
"""

from __future__ import annotations

import logging

import pandas as pd
from pandas.api.types import is_numeric_dtype

from src.p1_engenharia_dados import configuracao as cfg

logger = logging.getLogger(__name__)

#: Colunas que saem da base de modelagem porque uma versão melhor as substitui.
#: ``{coluna_removida: (substituta, motivo)}``.
COLUNAS_SUBSTITUIDAS: dict[str, tuple[str, str]] = {
    "admission_type_id": ("tipo_admissao", "código numérico; o nome legível já está na base"),
    "admission_source_id": ("origem_admissao", "código numérico; o nome legível já está na base"),
    "discharge_disposition_id": (
        "grupo_destino_alta",
        "código numérico; o grupo legível já está na base",
    ),
    "destino_alta": ("grupo_destino_alta", "dezenas de níveis; o agrupamento mantém o sentido"),
    "age": ("idade_media", "10 faixas viram um número (ponto médio da faixa)"),
    "diag_1": ("grupo_diag_1", "centenas de códigos CID-9; o grupo clínico resume"),
    "diag_2": ("grupo_diag_2", "centenas de códigos CID-9; o grupo clínico resume"),
    "diag_3": ("grupo_diag_3", "centenas de códigos CID-9; o grupo clínico resume"),
    "medical_specialty": ("especialidade_agrupada", "dezenas de níveis raros; mantemos o top-N"),
    "readmitted": (cfg.COLUNA_ALVO, "contém a própria resposta: usar como atributo seria vazamento"),
}

#: Colunas criadas pelo pipeline (as demais são originais da base).
COLUNAS_DERIVADAS: frozenset[str] = frozenset(
    {
        cfg.COLUNA_ALVO, "idade_media", "tipo_admissao", "destino_alta", "origem_admissao",
        "grupo_diag_1", "grupo_diag_2", "grupo_diag_3", "payer_code_ausente",
        "medical_specialty_ausente", "total_visitas_previas", "usuario_frequente",
        "n_meds_alterados", "n_meds_em_uso", "a1c_medido", "glicose_medida",
        "grupo_destino_alta", "exames_por_dia", "procedimentos_por_dia",
        "especialidade_agrupada",
        *(f"{c}_outlier" for c in cfg.COLUNAS_OUTLIER),
    }
)  # fmt: skip

#: Colunas sensíveis (ética e viés): usar para análise de equidade, não para discriminar.
COLUNAS_SENSIVEIS: frozenset[str] = frozenset({"race", "gender", "idade_media", "age"})


# ---------------------------------------------------------------------------
# Detecção
# ---------------------------------------------------------------------------


def calcular_dominancia(df: pd.DataFrame, colunas: list[str], limiar: float) -> pd.DataFrame:
    """Mede, para cada coluna, quanto o valor mais frequente domina.

    Coluna com dominância >= ``limiar`` é "quase constante": quase todas as
    linhas têm o mesmo valor, então ela quase não ajuda a separar pacientes e só
    aumenta a dimensão do problema.

    Args:
        df: Base a inspecionar.
        colunas: Colunas candidatas.
        limiar: Fração a partir da qual a coluna é considerada quase constante.

    Returns:
        Uma linha por coluna, ordenada da maior para a menor dominância, com
        ``quase_constante`` (0/1).
    """
    linhas = []
    for coluna in colunas:
        frequencias = df[coluna].value_counts(normalize=True, dropna=False)
        linhas.append(
            {
                "coluna": coluna,
                "valor_dominante": str(frequencias.index[0]),
                "pct_dominancia": round(100 * float(frequencias.iloc[0]), 3),
                "n_distintos": int(df[coluna].nunique(dropna=False)),
                "quase_constante": int(float(frequencias.iloc[0]) >= limiar),
            }
        )
    tabela = pd.DataFrame(linhas)
    return tabela.sort_values(["pct_dominancia", "coluna"], ascending=[False, True]).reset_index(
        drop=True
    )


def calcular_correlacoes(df: pd.DataFrame, colunas: list[str], limiar: float) -> pd.DataFrame:
    """Lista os pares de colunas numéricas por correlação de Pearson (|r|).

    Só entram colunas numéricas com mais de 2 valores (as sinalizações 0/1 não
    fazem sentido nesse cálculo).

    Args:
        df: Base a inspecionar.
        colunas: Colunas candidatas.
        limiar: |r| acima do qual um par é considerado redundante.

    Returns:
        Todos os pares, ordenados por |r| decrescente, com ``acima_do_limiar`` (0/1).
    """
    numericas = [c for c in colunas if is_numeric_dtype(df[c]) and df[c].nunique() > 2]
    corr = df[numericas].corr(method="pearson")
    pares = []
    for i, a in enumerate(numericas):
        for b in numericas[i + 1 :]:
            r = float(corr.loc[a, b])
            pares.append(
                {
                    "coluna_a": a,
                    "coluna_b": b,
                    "r_pearson": round(r, 4),
                    "r_absoluto": round(abs(r), 4),
                    "acima_do_limiar": int(abs(r) > limiar),
                }
            )
    return (
        pd.DataFrame(pares)
        .sort_values(["r_absoluto", "coluna_a", "coluna_b"], ascending=[False, True, True])
        .reset_index(drop=True)
    )


# ---------------------------------------------------------------------------
# Seleção
# ---------------------------------------------------------------------------


def selecionar_para_modelagem(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Gera a base de modelagem e documenta cada coluna removida.

    Args:
        df: Base completa (todas as colunas originais úteis + derivadas).

    Returns:
        Tupla ``(modelagem, removidas, dominancia, correlacoes)``:

        * ``modelagem``: DataFrame final, com o alvo na última coluna;
        * ``removidas``: tabela ``coluna``, ``motivo``, ``detalhe``;
        * ``dominancia`` e ``correlacoes``: tabelas de evidência.
    """
    removidas: list[dict[str, str]] = []
    restantes = list(df.columns)

    def remover(coluna: str, motivo: str, detalhe: str) -> None:
        """Tira ``coluna`` da lista de restantes e registra o motivo."""
        restantes.remove(coluna)
        removidas.append({"coluna": coluna, "motivo": motivo, "detalhe": detalhe})

    # 1) Identificadores: ficam só no interim, para rastrear linhas.
    for coluna in cfg.COLUNAS_ID:
        remover(coluna, "identificador", "não descreve o paciente; evitaria 'decorar' linhas")

    # 2) Colunas substituídas por versões melhores (e o readmitted original).
    for coluna, (substituta, motivo) in COLUNAS_SUBSTITUIDAS.items():
        if coluna in restantes:
            remover(coluna, "substituida", f"{motivo} -> usar {substituta}")

    # 3) Colunas constantes / quase constantes (o alvo nunca é removido aqui).
    candidatas = [c for c in restantes if c != cfg.COLUNA_ALVO]
    dominancia = calcular_dominancia(df, candidatas, cfg.LIMIAR_DOMINANCIA)
    for linha in dominancia.loc[dominancia["quase_constante"] == 1].itertuples():
        remover(
            linha.coluna,
            "quase_constante",
            f"valor '{linha.valor_dominante}' em {str(linha.pct_dominancia).replace('.', ',')}% das linhas",
        )

    # 4) Redundância entre numéricas: de cada par muito correlacionado, sai a 2ª coluna.
    candidatas = [c for c in restantes if c != cfg.COLUNA_ALVO]
    correlacoes = calcular_correlacoes(df, candidatas, cfg.LIMIAR_CORRELACAO)
    for linha in correlacoes.loc[correlacoes["acima_do_limiar"] == 1].itertuples():
        if linha.coluna_a in restantes and linha.coluna_b in restantes:
            remover(
                linha.coluna_b,
                "redundante",
                f"|r| = {linha.r_absoluto} com {linha.coluna_a} (mantida)",
            )

    # Alvo sempre na última coluna: facilita separar X e y.
    ordem = [c for c in restantes if c != cfg.COLUNA_ALVO] + [cfg.COLUNA_ALVO]
    modelagem = df[ordem].copy()
    logger.info(
        "Seleção: %d colunas -> %d colunas (%d removidas)",
        df.shape[1], modelagem.shape[1], len(removidas),
    )  # fmt: skip
    return modelagem, pd.DataFrame(removidas), dominancia, correlacoes


# ---------------------------------------------------------------------------
# Dicionário de dados
# ---------------------------------------------------------------------------

_DESC_MEDICAMENTO = (
    "Mudança de dose do medicamento '{nome}' na internação: "
    "No (não usado), Steady (dose mantida), Up (aumentada), Down (reduzida)."
)

#: ``{coluna: (descricao_negocio, observacao)}``. Linguagem simples, para a Pessoa 4.
DESCRICOES: dict[str, tuple[str, str]] = {
    "race": ("Raça/cor informada no cadastro do paciente.", "Sensível: usar só para análise de equidade."),
    "gender": ("Sexo do paciente (Female ou Male).", "Sensível: não usar para discriminar."),
    "idade_media": (
        "Idade aproximada do paciente: ponto médio da faixa etária (faixa 60-70 vira 65).",
        "Sensível. Derivada de age (F02).",
    ),
    "time_in_hospital": ("Dias de internação (de 1 a 14).", "Cauda longa; ver sinalização de outliers."),
    "payer_code": (
        "Quem paga o atendimento (convênio/seguro), em código; 'Desconhecido' se não informado.",
        "Pode funcionar como proxy de condição socioeconômica (D06).",
    ),
    "num_lab_procedures": ("Quantidade de exames de laboratório feitos na internação.", ""),
    "num_procedures": ("Quantidade de procedimentos (fora exames de laboratório) na internação.", ""),
    "num_medications": ("Quantidade de medicamentos diferentes administrados na internação.", ""),
    "number_outpatient": ("Consultas ambulatoriais do paciente no ano anterior à internação.", "Cauda longa."),
    "number_emergency": ("Visitas à emergência do paciente no ano anterior à internação.", "Cauda muito longa."),
    "number_inpatient": ("Internações do paciente no ano anterior à internação.", "Forte candidato a preditor."),
    "number_diagnoses": ("Quantidade de diagnósticos registrados na internação.", ""),
    "max_glu_serum": (
        "Resultado do exame de glicose no sangue: Norm, >200, >300 ou Nao_medido.",
        "Nao_medido = exame não feito, não é dado faltante (D09).",
    ),
    "A1Cresult": (
        "Resultado da hemoglobina glicada (HbA1c): Norm, >7, >8 ou Nao_medido.",
        "Nao_medido = exame não feito, não é dado faltante (D09).",
    ),
    "change": ("Houve mudança em algum medicamento de diabetes na internação (Ch = sim, No = não).", ""),
    "diabetesMed": ("Algum medicamento para diabetes foi receitado (Yes ou No).", ""),
    "tipo_admissao": ("Como o paciente entrou (emergência, urgência, eletivo...).", "Nome vindo do IDS_mapping; 'Nao_informado' agrupa códigos nulos."),
    "origem_admissao": ("De onde o paciente veio (encaminhamento, emergência, transferência...).", "Nome vindo do IDS_mapping; 'Nao_informado' agrupa códigos nulos."),
    "grupo_destino_alta": ("Para onde o paciente foi na alta: Casa, Transferencia, Outros ou Nao_informado.", "F09. Óbito/hospice já foram removidos (D02)."),
    "grupo_diag_1": ("Grupo clínico do diagnóstico principal (circulatório, diabetes, respiratório...).", "F03. Agrupamento CID-9 de Strack et al."),
    "grupo_diag_2": ("Grupo clínico do 2º diagnóstico.", "F03. 'Sem_diagnostico' quando não há (D08)."),
    "grupo_diag_3": ("Grupo clínico do 3º diagnóstico.", "F03. 'Sem_diagnostico' quando não há (D08)."),
    "payer_code_ausente": ("1 se o convênio não estava informado na base original.", "D06."),
    "medical_specialty_ausente": ("1 se a especialidade do médico não estava informada na base original.", "D06."),
    "especialidade_agrupada": ("Especialidade do médico: as mais comuns, 'Outras' e 'Desconhecido'.", "F11. Lista do top-N calculada só por frequência (sem usar o alvo)."),
    "total_visitas_previas": ("Soma de consultas ambulatoriais, emergências e internações do ano anterior.", "F04."),
    "usuario_frequente": ("1 se o paciente teve ao menos uma internação no ano anterior.", "F05. Limiar de definição clínica."),
    "n_meds_alterados": ("Quantos medicamentos tiveram a dose aumentada ou reduzida.", "F06."),
    "n_meds_em_uso": ("Quantos medicamentos o paciente estava usando (qualquer valor diferente de No).", "F07."),
    "a1c_medido": ("1 se o exame de HbA1c foi feito na internação.", "F08."),
    "glicose_medida": ("1 se o exame de glicose foi feito na internação.", "F08."),
    "exames_por_dia": ("Exames de laboratório por dia de internação.", "F10."),
    "procedimentos_por_dia": ("Procedimentos por dia de internação.", "F10."),
    cfg.COLUNA_ALVO: ("ALVO: 1 se o paciente voltou ao hospital em menos de 30 dias; 0 caso contrário.", "F01. Não use como atributo."),
}  # fmt: skip


def _descricao_coluna(coluna: str) -> tuple[str, str]:
    """Devolve ``(descricao_negocio, observacao)`` de uma coluna.

    Args:
        coluna: Nome da coluna da base de modelagem.

    Returns:
        Par com a descrição em linguagem simples e uma observação.

    Raises:
        KeyError: Se a coluna não tiver descrição (impede coluna sem documentação).
    """
    if coluna in cfg.COLUNAS_MEDICAMENTOS:
        return _DESC_MEDICAMENTO.format(nome=coluna), "Nome do remédio mantido em inglês, como na base."
    if coluna.endswith("_outlier") and coluna[: -len("_outlier")] in cfg.COLUNAS_OUTLIER:
        base = coluna[: -len("_outlier")]
        return (
            f"1 se '{base}' está fora dos limites do IQR (valor atípico).",
            "Só sinaliza; o valor original NÃO foi alterado. Opcional para modelagem.",
        )
    if coluna in DESCRICOES:
        return DESCRICOES[coluna]
    raise KeyError(f"Coluna sem descrição no dicionário de dados: {coluna}")


def classificar_tipo(serie: pd.Series, coluna: str) -> str:
    """Classifica a coluna em ``alvo``, ``binaria``, ``numerica`` ou ``categorica``.

    Args:
        serie: Valores da coluna.
        coluna: Nome da coluna.

    Returns:
        Uma das quatro classes acima.
    """
    if coluna == cfg.COLUNA_ALVO:
        return "alvo"
    if serie.nunique(dropna=True) <= 2:
        return "binaria"
    return "numerica" if is_numeric_dtype(serie) else "categorica"


def montar_dicionario(modelagem: pd.DataFrame) -> pd.DataFrame:
    """Gera o dicionário de dados da base de modelagem.

    Args:
        modelagem: Base final (saída de :func:`selecionar_para_modelagem`).

    Returns:
        Uma linha por coluna: ``coluna``, ``descricao_negocio``, ``tipo``,
        ``origem`` (original/derivada), ``sensivel`` (sim/nao), ``observacao``.
    """
    linhas = []
    for coluna in modelagem.columns:
        descricao, observacao = _descricao_coluna(coluna)
        linhas.append(
            {
                "coluna": coluna,
                "descricao_negocio": descricao,
                "tipo": classificar_tipo(modelagem[coluna], coluna),
                "origem": "derivada" if coluna in COLUNAS_DERIVADAS else "original",
                "sensivel": "sim" if coluna in COLUNAS_SENSIVEIS else "nao",
                "observacao": observacao,
            }
        )
    return pd.DataFrame(linhas)
