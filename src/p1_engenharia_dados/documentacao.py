"""Gera ``docs/README.md`` e ``docs/secao_base_de_dados.md`` a partir das tabelas.

Regra deste módulo: **nenhum número é digitado à mão**. Todos os valores do
texto são lidos de ``docs/tabelas/*.csv`` (principalmente ``metricas.csv``) e
formatados no padrão brasileiro. Assim, se os dados ou as regras mudarem, basta
rodar o pipeline de novo e a documentação acompanha.

O ``docs/README.md`` é a base da apresentação (cerca de 3 minutos) da Pessoa 1.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Mapping
from pathlib import Path

import pandas as pd

from src.p1_engenharia_dados import configuracao as cfg
from src.p1_engenharia_dados.configuracao import Caminhos
from src.p1_engenharia_dados.relatorios import fmt_int, fmt_num, fmt_pct

logger = logging.getLogger(__name__)

Formatador = Callable[[object], str]


# ---------------------------------------------------------------------------
# Leitura das métricas e utilitários de Markdown
# ---------------------------------------------------------------------------


class Numeros:
    """Acesso às métricas do pipeline já formatadas em português.

    Cada método recebe a **chave** de ``metricas.csv`` e devolve o texto pronto
    para entrar na documentação. Chave inexistente gera ``KeyError`` (melhor
    quebrar do que escrever um número errado).
    """

    def __init__(self, metricas: Mapping[str, str]) -> None:
        """Guarda o dicionário ``chave -> valor (texto)`` de ``metricas.csv``.

        Args:
            metricas: Métricas lidas do CSV.
        """
        self._m = dict(metricas)

    def bruto(self, chave: str) -> str:
        """Valor sem formatação (texto como está no CSV).

        Args:
            chave: Nome da métrica.

        Returns:
            Texto original.
        """
        return self._m[chave]

    def i(self, chave: str) -> str:
        """Inteiro com separador de milhar (``101766`` -> ``101.766``).

        Args:
            chave: Nome da métrica.

        Returns:
            Texto formatado.
        """
        return fmt_int(float(self._m[chave]))

    def p(self, chave: str, casas: int = 2) -> str:
        """Percentual com vírgula (``8.97`` -> ``8,97%``); o valor já está em %.

        Args:
            chave: Nome da métrica.
            casas: Casas decimais.

        Returns:
            Texto formatado.
        """
        return fmt_pct(float(self._m[chave]), casas)

    def f(self, chave: str, casas: int = 2) -> str:
        """Decimal com vírgula (``0.8082`` -> ``0,81``).

        Args:
            chave: Nome da métrica.
            casas: Casas decimais.

        Returns:
            Texto formatado.
        """
        return fmt_num(float(self._m[chave]), casas)

    def __getitem__(self, chave: str) -> str:
        """Atalho para :meth:`bruto`."""
        return self.bruto(chave)


def _ler_metricas(pasta_tabelas: Path) -> Numeros:
    """Lê ``metricas.csv`` e devolve o acessor :class:`Numeros`.

    Args:
        pasta_tabelas: Pasta ``docs/tabelas``.

    Returns:
        Objeto com as métricas.
    """
    tabela = pd.read_csv(pasta_tabelas / "metricas.csv", dtype=str, keep_default_na=False)
    return Numeros(dict(zip(tabela["chave"], tabela["valor"])))


def _tabela_md(
    df: pd.DataFrame,
    colunas: list[str],
    cabecalhos: list[str],
    formatadores: Mapping[str, Formatador] | None = None,
) -> str:
    """Converte um DataFrame em tabela Markdown (GitHub).

    Args:
        df: Dados.
        colunas: Colunas a exibir, na ordem.
        cabecalhos: Títulos das colunas na tabela.
        formatadores: Função de formatação opcional por coluna.

    Returns:
        Texto da tabela.
    """
    formatadores = formatadores or {}
    linhas = ["| " + " | ".join(cabecalhos) + " |", "|" + "|".join("---" for _ in colunas) + "|"]
    for _, linha in df.iterrows():
        celulas = []
        for c in colunas:
            valor = linha[c]
            texto = formatadores[c](valor) if c in formatadores else str(valor)
            celulas.append(texto.replace("|", "\\|").replace("\n", " "))
        linhas.append("| " + " | ".join(celulas) + " |")
    return "\n".join(linhas)


def _fmt_int_ou_texto(valor: object) -> str:
    """Formata inteiro; se o valor for texto (ex.: ``removida``), devolve o texto.

    Args:
        valor: Número ou texto.

    Returns:
        Texto para a célula da tabela.
    """
    try:
        return fmt_int(float(valor))  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return str(valor)


def _fmt_pct_celula(valor: object) -> str:
    """Formata percentual (valor já em %) para célula de tabela.

    Args:
        valor: Número em porcentagem.

    Returns:
        Texto com vírgula e ``%``.
    """
    return fmt_pct(float(valor))  # type: ignore[arg-type]


def _fmt_decimal_celula(valor: object) -> str:
    """Formata decimal com 2 casas e vírgula.

    Args:
        valor: Número.

    Returns:
        Texto formatado.
    """
    return fmt_num(float(valor))  # type: ignore[arg-type]


def _diagrama_mermaid(etapas: pd.DataFrame) -> str:
    """Monta o diagrama Mermaid do pipeline com linhas x colunas de cada etapa.

    Args:
        etapas: Tabela ``etapas_pipeline.csv``.

    Returns:
        Bloco de código Mermaid (renderiza no GitHub).
    """
    nos = []
    for k, r in enumerate(etapas.itertuples()):
        rotulo = f"{r.etapa}<br/>{fmt_int(r.linhas)} linhas × {fmt_int(r.colunas)} colunas"
        nos.append(f'    N{k}["{rotulo}"]')
        if k == 0:
            # A auditoria só mede o estado inicial; não altera nenhum dado.
            nos.append('    AUD["Auditoria inicial<br/>só mede, não altera"]')
    ligacoes = ["    N0 --> AUD --> N1"] + [f"    N{k} --> N{k + 1}" for k in range(1, len(etapas) - 1)]
    return "```mermaid\nflowchart LR\n" + "\n".join(nos) + "\n" + "\n".join(ligacoes) + "\n```"


# ---------------------------------------------------------------------------
# docs/README.md
# ---------------------------------------------------------------------------


def _montar_readme(t: Path, n: Numeros) -> str:
    """Monta o texto completo de ``docs/README.md``.

    Args:
        t: Pasta ``docs/tabelas``.
        n: Métricas formatadas.

    Returns:
        Texto Markdown.
    """
    lerc = lambda nome: pd.read_csv(t / f"{nome}.csv", keep_default_na=False)  # noqa: E731
    funil, etapas, decisoes = lerc("funil_linhas"), lerc("etapas_pipeline"), lerc("decisoes")
    ausentes, atributos, outliers = lerc("ausentes_antes_depois"), lerc("atributos_criados"), lerc("outliers")
    top = n["top_especialidades"].split("; ")
    corte = funil.loc[funil["removidas"].idxmax()]  # etapa que mais removeu linhas

    tab_funil = _tabela_md(
        funil,
        ["etapa", "descricao", "linhas_antes", "removidas", "pct_do_bruto_removido", "linhas_depois"],
        ["ID", "O que fez", "Linhas antes", "Removidas", "% do bruto", "Linhas depois"],
        {
            "linhas_antes": _fmt_int_ou_texto,
            "removidas": _fmt_int_ou_texto,
            "pct_do_bruto_removido": lambda v: fmt_pct(float(v), 3),  # type: ignore[arg-type]
            "linhas_depois": _fmt_int_ou_texto,
        },
    )
    tab_ausentes = _tabela_md(
        ausentes,
        ["coluna", "id_decisao", "n_nulos_antes", "pct_nulos_antes", "n_textos_none_antes", "decisao", "n_nulos_depois"],
        ["Coluna", "ID", "Nulos (`?`) antes", "% antes", "Textos `None`", "Decisão", "Nulos depois"],
        {
            "n_nulos_antes": _fmt_int_ou_texto,
            "pct_nulos_antes": _fmt_pct_celula,
            "n_textos_none_antes": _fmt_int_ou_texto,
            "n_nulos_depois": _fmt_int_ou_texto,
        },
    )
    tab_decisoes = _tabela_md(
        decisoes,
        ["id", "problema", "evidencia", "decisao", "motivo", "impacto"],
        ["ID", "O que estava errado", "Evidência (número)", "O que fizemos", "Por quê", "Impacto"],
    )
    tab_atributos = _tabela_md(
        atributos,
        ["id", "coluna", "como_foi_criado", "intuicao_de_negocio", "resumo"],
        ["ID", "Coluna", "Como foi criada", "Intuição de negócio", "Resumo (calculado)"],
    )
    tab_outliers = _tabela_md(
        outliers,
        ["variavel", "q1", "q3", "limite_superior", "n_outliers", "pct_outliers", "maximo", "assimetria"],
        ["Variável", "Q1", "Q3", "Limite superior", "Outliers", "% outliers", "Máximo", "Assimetria"],
        {
            "q1": _fmt_decimal_celula, "q3": _fmt_decimal_celula,
            "limite_superior": _fmt_decimal_celula, "n_outliers": _fmt_int_ou_texto,
            "pct_outliers": _fmt_pct_celula, "maximo": _fmt_int_ou_texto,
            "assimetria": _fmt_decimal_celula,
        },
    )  # fmt: skip
    divergencias = _tabela_divergencias(n)

    secoes = [
        "# Tratamento de dados — Pessoa 1 (Engenharia de Dados e Pré-processamento)\n\n"
        "> **Este arquivo é gerado por código** (`python -m src.p1_engenharia_dados.executar_pipeline`) "
        "e todos os números vêm de [`tabelas/`](tabelas/). Não edite à mão: rode o pipeline de novo.",
        # 1 ---------------------------------------------------------------
        f"""## 1. Resumo em 30 segundos

- **De onde vem:** base *Diabetes 130-US Hospitals* (UCI, 130 hospitais dos EUA, 1999–2008): **{n.i('linhas_bruto')} internações**, **{n.i('colunas_bruto')} colunas**, **{n.i('pacientes_bruto')} pacientes**. Queremos prever se o paciente volta ao hospital em **menos de 30 dias**.
- **O que estava errado:** `?` no lugar de dado ({n.i('n_interrogacoes_total')} ocorrências em {n.i('n_colunas_com_nulo')} colunas), coluna `weight` com {n.p('pct_nulo_weight')} de ausentes, o mesmo paciente repetido ({n.i('encontros_repetidos_bruto')} encontros a mais), pacientes que morreram ou foram para hospice, códigos numéricos sem nome e a armadilha do `None`.
- **O que fizemos:** juntamos os nomes dos códigos, removemos o que não podia entrar ({n.i('d01_removidas')} + {n.i('d03_removidas')} + {n.i('d02_removidas')} linhas), tratamos os ausentes coluna por coluna, criamos atributos novos, sinalizamos outliers e tiramos colunas sem utilidade. **Sem escalonar, sem one-hot, sem imputar média:** isso é da Pessoa 3, depois do split.
- **O que ficou:** **{n.i('linhas_final')} linhas × {n.i('colunas_modelagem')} colunas** em `data/processed/`, com **{n.p('pct_positivo_final')}** de readmissões em < 30 dias (classe desbalanceada).""",
        # 2 ---------------------------------------------------------------
        "## 2. Diagrama do pipeline\n\n"
        "Cada caixa mostra o tamanho da base **depois** da etapa.\n\n" + _diagrama_mermaid(etapas),
        # 3 ---------------------------------------------------------------
        f"""## 3. Funil de linhas

Saímos de **{n.i('linhas_bruto')}** e ficamos com **{n.i('linhas_final')}** ({n.p('pct_linhas_mantidas')} do total). O maior corte foi **{corte['descricao'].lower()}** ({corte['etapa']}).

![Funil de linhas](figuras/funil_linhas.png)

{tab_funil}

- **D01** — sexo `Unknown/Invalid`: não dá para recuperar o valor correto.
- **D03** — o mesmo paciente aparece várias vezes; o modelo não pode ver o mesmo paciente no treino e no teste.
- **D02** — óbito/hospice na alta (códigos {n['codigos_obito_hospice']}): quem faleceu não pode ser readmitido. No bruto havia {n.i('obito_hospice_no_bruto')} encontros assim; depois de ficar com 1 encontro por paciente, sobraram {n.i('d02_removidas')} para remover.""",
        # 4 ---------------------------------------------------------------
        f"""## 4. Mapa de ausentes

![Mapa de ausentes](figuras/mapa_ausentes.png)

{tab_ausentes}

**Ponto didático (D09):** o `pandas` lê o texto `None` como nulo por padrão. Em `max_glu_serum` ({n.p('pct_none_max_glu_serum')}) e `A1Cresult` ({n.p('pct_none_a1cresult')}) `None` quer dizer **"exame não foi feito"**, que é informação, não ausência. Por isso lemos o arquivo com `keep_default_na=False` e só `?` vira nulo. Depois da limpeza sobraram **{n.i('nulos_restantes_total')}** nulos.""",
        # 5 ---------------------------------------------------------------
        "## 5. O que estava errado → o que fizemos → por quê\n\n"
        "Esta é a tabela mais importante. Cada linha é uma decisão registrada no código com o mesmo ID.\n\n" + tab_decisoes,
        # 6 ---------------------------------------------------------------
        f"""## 6. Antes × depois

### 6.1 O alvo (balanceamento de classes)

![Alvo antes e depois](figuras/antes_depois_alvo.png)

O alvo original `readmitted` tinha 3 classes (`No`, `>30`, `<30`) e {n.p('pct_positivo_bruto')} de `<30` no bruto. Viramos `readmit_30d` (1 = `<30`). Depois dos filtros, a classe positiva tem **{n.i('n_positivo_final')}** linhas (**{n.p('pct_positivo_final')}**). É minoria: na modelagem, prefira F1, AUC e *class weights* à acurácia.

### 6.2 Outliers (só sinalizados)

![Boxplots dos outliers](figuras/outliers_boxplots.png)

![Antes e depois de um outlier](figuras/antes_depois_outliers.png)

{tab_outliers}

{n.i('n_variaveis_com_outlier')} de {n.i('n_variaveis_outlier')} variáveis têm outliers pela regra do IQR (de {n.p('pct_outlier_min')} a {n.p('pct_outlier_max')} das linhas). São pacientes graves plausíveis, então **não removemos nem limitamos**: criamos colunas `*_outlier` (0/1) e deixamos os valores intactos. A cauda é longa (assimetria alta): `log1p` é uma opção para modelos lineares.

> **Cuidado ao interpretar:** em {n.i('n_variaveis_iqr_zero')} variáveis ({n['variaveis_iqr_zero']}) o IQR é zero (a maioria dos pacientes tem 0). Nesses casos, qualquer valor acima de zero já aparece como "outlier" pela regra. Por isso a taxa de outliers dessas colunas é alta e a sinalização é só descritiva.

### 6.3 Diagnósticos (alta cardinalidade)

![Diagnósticos antes e depois](figuras/antes_depois_diagnosticos.png)

`diag_1` tinha **{n.i('n_codigos_diag1_antes')}** códigos CID-9 distintos; viraram **{n.i('n_grupos_diag1_depois')}** grupos clínicos (mais `Sem_diagnostico`).

### 6.4 Por que `usuario_frequente = 1` a partir de {cfg.LIMIAR_USUARIO_FREQUENTE} internação prévia

![Limiar de usuário frequente](figuras/limiar_usuario_frequente.png)

Quem tem ao menos 1 internação prévia ({n.p('pct_usuario_frequente')} das linhas) volta em < 30 dias em {n.p('taxa_readmissao_usuario_frequente')} dos casos, contra {n.p('taxa_readmissao_nao_frequente')} dos demais. O limiar vem da definição clínica ("já internou antes"), **não** de uma busca no alvo; a figura só confirma que faz sentido.""",
        # 7 ---------------------------------------------------------------
        f"""## 7. Atributos criados (F01–F11)

Todos usam só informação disponível até a alta e **nunca** o alvo. Lista das {cfg.TOP_N_ESPECIALIDADES} especialidades mantidas em F11: {', '.join(top)}.

{tab_atributos}

Também criamos `payer_code_ausente` e `medical_specialty_ausente` (D06) e as colunas `*_outlier` (subseção "Outliers", na seção 6).""",
        # 8 ---------------------------------------------------------------
        f"""## 8. Cuidados

### Vazamento de dados

- **Só o 1º encontro por paciente:** evita o mesmo paciente no treino e no teste.
- **`readmitted` e os IDs ficam fora da base de modelagem:** o primeiro contém a resposta; os outros deixariam o modelo "decorar" linhas.
- **Nada de escalonar, codificar ou imputar aqui.** Essas contas aprendem com os dados e precisam ser feitas só no treino, dentro de um `Pipeline` do sklearn (Pessoa 3).
- Duas escolhas usam **frequência** na base inteira, mas nunca o alvo: a lista das especialidades mais comuns (F11) e a detecção de colunas quase constantes. Os limites do IQR também. Nenhuma dessas usa a resposta, então não há vazamento do alvo; a Pessoa 3 pode recalcular no treino se quiser rigor máximo.

### Atributos sensíveis

{n.i('n_colunas_sensiveis')} colunas da base de modelagem são sensíveis (`race`, `gender`, `idade_media`) e estão marcadas no dicionário. Use-as para **analisar equidade** entre grupos, não para discriminar. `payer_code` pode funcionar como proxy de condição socioeconômica.

### Limitações dos dados

- Internações de **1999 a 2008**, em hospitais dos EUA: as práticas clínicas mudaram. Recomendamos o **método**, não os números absolutos.
- Readmissão só é observada quando o paciente volta a um hospital da própria base.
- Classe positiva rara ({n.p('pct_positivo_final')}).

### Onde seguimos os dados e não o plano

{divergencias}""",
        # 9 ---------------------------------------------------------------
        f"""## 9. O que entregamos para o resto do grupo

| Arquivo | Para quê |
|---|---|
| `data/processed/diabetes_modelagem.csv` e `.parquet` | Base final: **{n.i('linhas_final')} linhas × {n.i('colunas_modelagem')} colunas**, alvo `readmit_30d` na última coluna |
| `data/processed/dicionario_de_dados.csv` | O que cada coluna significa, tipo ({n.i('n_colunas_numerica')} numéricas, {n.i('n_colunas_categorica')} categóricas, {n.i('n_colunas_binaria')} binárias, {n.i('n_colunas_alvo')} alvo), origem e se é sensível |
| `data/interim/diabetes_limpo.csv` | Versão legível, com IDs e as colunas originais úteis ({n.i('colunas_interim')} colunas) |
| `docs/tabelas/*.csv` | Evidências de cada decisão (auditoria antes/depois) |
| `docs/figuras/*.png` | Gráficos para o artigo e os slides |

Carregar em 2 linhas:

```python
import pandas as pd
df = pd.read_csv("data/processed/diabetes_modelagem.csv")  # ou pd.read_parquet(...)
X, y = df.drop(columns="readmit_30d"), df["readmit_30d"]
```

Para a Pessoa 3: as colunas de texto precisam de *one-hot*; `idade_media` e as contagens são numéricas; as colunas `*_outlier` são opcionais; o *split* treino/teste é de vocês (não geramos).""",
        # 10 --------------------------------------------------------------
        _roteiro_e_perguntas(n),
        # 11 --------------------------------------------------------------
        """## 11. Como reproduzir

```bash
python -m venv .venv
.venv\\Scripts\\activate            # Linux/Mac: source .venv/bin/activate
pip install -r requirements.txt

python -m src.p1_engenharia_dados.baixar_base          # baixa a base para data/raw/
python -m src.p1_engenharia_dados.executar_pipeline    # refaz TUDO (dados, tabelas, figuras, textos)
pytest tests/test_pipeline_p1.py                       # confere as regras
python -m pydoc src.p1_engenharia_dados.limpeza        # documentação das funções
```

O pipeline é determinístico: rodar duas vezes gera arquivos idênticos.""",
    ]
    return "\n\n".join(secoes) + "\n"


def _tabela_divergencias(n: Numeros) -> str:
    """Tabela das decisões em que os dados divergiram do plano.

    Args:
        n: Métricas formatadas.

    Returns:
        Tabela Markdown.
    """
    iguais = n["codigos_iguais_ao_plano"] == "1"
    sem_linhas = n["codigos_obito_hospice_sem_linhas"]
    linhas = [
        ("Ordem dos filtros", "D01 → D02 → D03",
         "D01 → **D03 → D02**",
         f"Se o 1º encontro do paciente terminou em óbito/hospice, ele deve sair inteiro. Na ordem do plano, um encontro posterior (que existe em alguns registros) viraria o 'primeiro'. Diferença: {n.i('diferenca_ordem_plano')} pacientes ({n.i('linhas_final_na_ordem_do_plano')} na ordem do plano contra {n.i('linhas_final')})."),
        ("Leitura de nulos", "`na_values=['?']`", "`na_values=['?']` **e** `keep_default_na=False`",
         f"Sem isso o pandas trataria `None` como nulo ({n.p('pct_none_max_glu_serum')} de `max_glu_serum`), apagando a informação 'exame não feito'."),
        ("Códigos de óbito/hospice", "11, 13, 14, 19, 20, 21",
         ("Conferem com o IDS_mapping" if iguais else "Seguimos o IDS_mapping (difere do plano)")
         + f": {n['codigos_obito_hospice']}",
         f"Lemos a descrição ('Expired' / 'Hospice') no mapeamento. Códigos sem nenhuma linha no bruto: {sem_linhas or 'nenhum'}."),
        ("Destino de alta (F09)", "Casa / Transferência / Outros", "Casa / Transferencia / Outros / **Nao_informado**",
         "Não misturamos 'não sabemos o destino' com destinos conhecidos."),
        ("Grupos de diagnóstico (F03)", "9 grupos", "9 grupos + **Sem_diagnostico**",
         "Ausência de diagnóstico secundário é real (D08) e não pertence a nenhum grupo clínico."),
        ("Colunas quase constantes", "'verificar examide e citoglipton'",
         f"{n.i('n_quase_constantes')} colunas removidas (>= {fmt_pct(100 * cfg.LIMIAR_DOMINANCIA, 1)} num único valor)",
         f"Inclui {n.i('n_meds_quase_constantes')} de {n.i('n_meds_total')} medicamentos; ficaram {n.i('n_meds_mantidos')}. Os contadores F06/F07 foram calculados **antes** da remoção."),
        ("Correlação entre numéricas", "listar pares com |r| > 0,9",
         f"{n.i('n_correlacoes_altas')} pares acima do limiar",
         f"A maior correlação é {n.f('correlacao_maxima')} ({n['par_correlacao_maxima']}); nada foi removido por redundância numérica."),
    ]  # fmt: skip
    cabecalho = "| Tema | Plano | O que fizemos | Motivo |\n|---|---|---|---|\n"
    return cabecalho + "\n".join("| " + " | ".join(c.replace("|", "\\|") for c in l) + " |" for l in linhas)


def _roteiro_e_perguntas(n: Numeros) -> str:
    """Roteiro de fala de 3 minutos e 10 perguntas prováveis da banca.

    Args:
        n: Métricas formatadas.

    Returns:
        Texto Markdown da seção 10.
    """
    return f"""## 10. Roteiro de fala (3 minutos) e perguntas da banca

### Roteiro

| Tempo | O que dizer |
|---|---|
| 0:00–0:20 | Base do UCI: {n.i('linhas_bruto')} internações de diabéticos em 130 hospitais. Objetivo: prever readmissão em < 30 dias. |
| 0:20–0:50 | Problemas encontrados: `?` espalhado, `weight` quase vazio ({n.p('pct_nulo_weight', 1)}), pacientes repetidos, óbitos e códigos sem nome. |
| 0:50–1:30 | **Funil:** {n.i('linhas_bruto')} → {n.i('linhas_final')}. Cortamos {n.i('d03_removidas')} encontros repetidos (1 por paciente), {n.i('d02_removidas')} óbitos/hospice e {n.i('d01_removidas')} sexos inválidos. |
| 1:30–2:00 | **Ausentes:** coluna por coluna, sem imputar média. O `None` dos exames não é nulo: é "exame não feito". |
| 2:00–2:30 | **Atributos novos:** grupos de diagnóstico, idade média, `usuario_frequente` ({n.p('taxa_readmissao_usuario_frequente', 1)} de readmissão contra {n.p('taxa_readmissao_nao_frequente', 1)}), exames medidos. |
| 2:30–3:00 | **Cuidados:** nada de vazamento, atributos sensíveis marcados, classe positiva só {n.p('pct_positivo_final', 1)}. Entrega: base de {n.i('linhas_final')} × {n.i('colunas_modelagem')} + dicionário. |

### 10 perguntas prováveis (resposta curta)

1. **Por que removeu pacientes que morreram ou foram para hospice?** Eles não podem ser readmitidos. Deixá-los faria o alvo ser 0 por um motivo que não tem a ver com o atendimento e subestimaria a readmissão. Foram {n.i('d02_removidas')} linhas.
2. **Por que não imputou a média?** Imputar com estatística aprende com os dados; se feito antes do split, vaza informação do teste. Isso fica no `Pipeline` do sklearn (Pessoa 3). Aqui só aplicamos regras fixas do domínio.
3. **Por que só o primeiro encontro?** Encontros do mesmo paciente não são independentes e o modelo "decoraria" o paciente. Perdemos {n.i('d03_removidas')} linhas, mas ganhamos métricas honestas.
4. **Os outliers não atrapalham?** São pacientes graves reais, e removê-los apagaria justamente os de maior risco. Sinalizamos com `*_outlier` e a Pessoa 3 decide, estimando limites só no treino.
5. **Por que removeu `weight`?** Estava {n.p('pct_nulo_weight', 1)} ausente. Preencher seria inventar o peso de quase todo mundo.
6. **O que é o `None` nos exames de glicose e HbA1c?** Significa que o exame não foi feito. Não é dado faltante: virou a categoria `Nao_medido` e as flags `a1c_medido` e `glicose_medida`.
7. **Por que não escalonou nem fez one-hot?** São transformações que aprendem dos dados (média, categorias) e dependem do split. Ficam no `Pipeline` da Pessoa 3.
8. **Como garantiu que não há vazamento?** Alvo `readmitted` e IDs fora da base de modelagem, um encontro por paciente, atributos só com informação até a alta e nenhuma estatística do teste.
9. **Usar raça e sexo é ético?** Estão marcados como sensíveis ({n.i('n_colunas_sensiveis')} colunas). Servem para auditar equidade entre grupos, não para discriminar; `Desconhecido` em `race` existe para não perder linhas.
10. **Dados de 1999–2008 ainda valem?** O método vale, os números absolutos não: as práticas clínicas mudaram. Dizemos isso como limitação do trabalho."""


# ---------------------------------------------------------------------------
# docs/secao_base_de_dados.md
# ---------------------------------------------------------------------------


def _montar_secao_artigo(n: Numeros) -> str:
    """Monta o texto da seção "Base de Dados" do artigo SBC (~1 página).

    Args:
        n: Métricas formatadas.

    Returns:
        Texto Markdown corrido.
    """
    return f"""# Base de Dados

A base utilizada é a *Diabetes 130-US Hospitals for Years 1999-2008*, disponibilizada pelo UCI Machine Learning Repository sob licença CC BY 4.0 [Clore et al. 2014, DOI 10.24432/C5230J]. Ela reúne {n.i('linhas_bruto')} internações de pacientes diabéticos ocorridas entre 1999 e 2008 em 130 hospitais dos Estados Unidos, descritas por {n.i('colunas_bruto')} atributos que incluem dados demográficos, características da internação, resultados de exames, medicamentos prescritos e diagnósticos em CID-9. Os dados foram originalmente analisados por Strack et al. (2014), que investigaram a relação entre a medição da hemoglobina glicada (HbA1c) e a readmissão hospitalar [DOI 10.1155/2014/781670].

O problema de negócio é identificar os pacientes com maior risco de readmissão em até 30 dias após a alta, evento que penaliza financeiramente os hospitais. A variável-alvo original (`readmitted`) possui três categorias (`<30`, `>30` e `No`); ela foi binarizada em `readmit_30d`, igual a 1 quando a readmissão ocorre em menos de 30 dias. Os atributos são numéricos (por exemplo, tempo de internação e número de medicamentos) e categóricos (faixa etária, tipo de admissão, destino de alta, diagnósticos e {n.i('n_meds_total')} medicamentos para diabetes).

O pré-processamento foi implementado em Python de forma determinística e reprodutível. Na **auditoria inicial** identificamos {n.i('n_interrogacoes_total')} ocorrências do marcador `?` em {n.i('n_colunas_com_nulo')} colunas, com destaque para `weight` ({n.p('pct_nulo_weight', 1)} ausente), `medical_specialty` ({n.p('pct_nulo_medical_specialty', 1)}) e `payer_code` ({n.p('pct_nulo_payer_code', 1)}). Na **integração de dados**, as três tabelas de códigos do arquivo `IDS_mapping.csv` foram combinadas à base para substituir identificadores numéricos por nomes legíveis (tipo de admissão, origem da admissão e destino de alta), unificando as descrições equivalentes (`NULL`, `Not Available`, `Not Mapped`) em uma única categoria.

Na **remoção de inconsistências**, foram excluídos {n.i('d01_removidas')} registros com sexo inválido, {n.i('d02_removidas')} encontros em que o paciente faleceu ou foi encaminhado a hospice (que não podem resultar em readmissão) e {n.i('d03_removidas')} encontros repetidos, mantendo-se apenas o primeiro de cada paciente para garantir a independência das observações e evitar vazamento entre treino e teste. Com isso, a base passou de {n.i('linhas_bruto')} para {n.i('linhas_final')} registros. No **tratamento de ausentes**, a coluna `weight` foi descartada; ausências em `payer_code`, `medical_specialty` e `race` foram tratadas como categoria própria (com indicadores de ausência para `payer_code` e `medical_specialty`, que preservam a informação); e o valor `None` dos exames de glicose e HbA1c, que indica exame não realizado, foi mantido como categoria `Nao_medido`, em vez de tratado como dado faltante. Nenhuma imputação estatística foi feita nesta etapa; estimativas dependentes dos dados (imputação, escalonamento e codificação) foram deixadas para dentro de um *pipeline* ajustado apenas no conjunto de treino.

Na **transformação de variáveis**, os {n.i('n_codigos_diag1_antes')} códigos CID-9 do diagnóstico principal foram agrupados em {n.i('n_grupos_diag1_depois')} grupos clínicos, a faixa etária foi convertida em idade média, e foram criados atributos como o total de visitas prévias, o indicador de usuário frequente, o número de medicamentos em uso e alterados, os indicadores de HbA1c e glicose medidas e a intensidade de cuidado por dia de internação. A **análise de outliers** pelo critério do IQR apontou valores atípicos em {n.i('n_variaveis_com_outlier')} das {n.i('n_variaveis_outlier')} variáveis de contagem; por serem clinicamente plausíveis, eles foram apenas sinalizados. Na **seleção de atributos**, foram removidos identificadores, colunas substituídas por versões mais informativas e {n.i('n_quase_constantes')} colunas quase constantes, resultando em {n.i('colunas_modelagem')} colunas na base de modelagem. A classe positiva representa {n.p('pct_positivo_final')} das {n.i('linhas_final')} observações, caracterizando um problema de classificação desbalanceado.

**Referências.** Strack, B. et al. (2014). Impact of HbA1c Measurement on Hospital Readmission Rates: Analysis of 70,000 Clinical Database Patient Records. *BioMed Research International*. DOI 10.1155/2014/781670. Clore, J.; Cios, K.; DeShazo, J.; Strack, B. (2014). Diabetes 130-US Hospitals for Years 1999-2008 [Dataset]. UCI Machine Learning Repository. DOI 10.24432/C5230J.
"""


# ---------------------------------------------------------------------------
# Ponto de entrada
# ---------------------------------------------------------------------------


def gerar_documentacao(caminhos: Caminhos) -> tuple[Path, Path]:
    """Gera ``docs/README.md`` e ``docs/secao_base_de_dados.md``.

    Pré-requisito: as tabelas de ``docs/tabelas`` já foram gravadas pelo pipeline.

    Args:
        caminhos: Caminhos do projeto.

    Returns:
        Tupla com os caminhos do README e da seção do artigo.

    Raises:
        FileNotFoundError: Se ``docs/tabelas/metricas.csv`` não existir.
    """
    numeros = _ler_metricas(caminhos.tabelas)
    readme = caminhos.docs / "README.md"
    secao = caminhos.docs / "secao_base_de_dados.md"
    readme.write_text(_montar_readme(caminhos.tabelas, numeros), encoding="utf-8", newline="\n")
    secao.write_text(_montar_secao_artigo(numeros), encoding="utf-8", newline="\n")
    logger.info("Documentação gerada: %s e %s", readme, secao)
    return readme, secao
