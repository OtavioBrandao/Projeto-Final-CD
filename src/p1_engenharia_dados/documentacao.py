"""Gera ``docs/README.md`` e ``docs/secao_base_de_dados.md`` a partir das tabelas.

Regra deste módulo: **nenhum número é digitado à mão**. Todos os valores do
texto são lidos de ``docs/tabelas/*.csv`` (principalmente ``metricas.csv``) e
formatados no padrão brasileiro. Assim, se os dados ou as regras mudarem, basta
rodar o pipeline de novo e a documentação acompanha.

O ``docs/README.md`` é o README da parte de Engenharia de Dados e
Pré-processamento (não é o README do repositório) e a base da apresentação.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Mapping
from pathlib import Path

import pandas as pd

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
        """Inteiro com separador de milhar (``145731`` -> ``145.731``).

        Args:
            chave: Nome da métrica.

        Returns:
            Texto formatado.
        """
        return fmt_int(float(self._m[chave]))

    def p(self, chave: str, casas: int = 2) -> str:
        """Percentual com vírgula (``7.05`` -> ``7,05%``); o valor já está em %.

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
    """Formata inteiro; se o valor for texto, devolve o texto.

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
        ["coluna", "origem", "id_decisao", "n_antes", "pct_antes", "decisao", "situacao_final"],
        ["Coluna", "Arquivo", "ID", "Linhas afetadas", "% antes", "O que fizemos", "Situação final"],
        {"n_antes": _fmt_int_ou_texto, "pct_antes": _fmt_pct_celula},
    )
    tab_decisoes = _tabela_md(
        decisoes,
        ["id", "problema", "evidencia", "decisao", "motivo", "impacto"],
        ["ID", "O que estava errado", "Evidência (número)", "O que fizemos", "Por quê", "Impacto"],
    )
    # Uma linha por ID (várias colunas juntas); o resumo numérico de cada coluna fica em
    # docs/tabelas/atributos_criados.csv.
    por_id = (
        atributos.groupby("id", sort=True)
        .agg(colunas=("coluna", lambda s: ", ".join(f"`{c}`" for c in s)), como=("como_foi_criado", "first"), intuicao=("intuicao_de_negocio", "first"))
        .reset_index()
    )
    tab_atributos = _tabela_md(
        por_id,
        ["id", "colunas", "como", "intuicao"],
        ["ID", "Colunas criadas", "Como foi criada", "Intuição de negócio"],
    )
    tab_outliers = _tabela_md(
        outliers,
        ["variavel", "q1", "q3", "limite_superior", "n_outliers", "pct_outliers", "maximo", "assimetria"],
        ["Variável", "Q1", "Q3", "Limite superior", "Outliers", "% outliers", "Máximo", "Assimetria"],
        {
            "q1": _fmt_decimal_celula, "q3": _fmt_decimal_celula,
            "limite_superior": _fmt_decimal_celula, "n_outliers": _fmt_int_ou_texto,
            "pct_outliers": _fmt_pct_celula, "maximo": _fmt_decimal_celula,
            "assimetria": _fmt_decimal_celula,
        },
    )  # fmt: skip

    secoes = [
        "# README — Engenharia de Dados e Pré-processamento\n\n"
        "> Este é o README **da parte de Engenharia de Dados e Pré-processamento** (limpeza, integração e "
        "atributos), não o README do repositório. Ele é a base da apresentação desta etapa. "
        "**É gerado por código** (`python -m src.p1_engenharia_dados.executar_pipeline`) e todos os "
        "números vêm de [`tabelas/`](tabelas/). Não edite à mão: rode o pipeline de novo.",
        # 1 ---------------------------------------------------------------
        f"""## 1. Resumo em 30 segundos

- **De onde vem:** Dados Abertos da **PRF** (Polícia Rodoviária Federal), acidentes em rodovias federais de **2024 e 2025**: **{n.i('linhas_ocorrencias_bruto')} acidentes** em {n.i('colunas_ocorrencias_bruto')} colunas, mais **{n.i('linhas_pessoas_bruto')} linhas de pessoas/veículos** em outro arquivo. Queremos entender e prever quais acidentes terminam em **morte**.
- **O que estava errado:** dados de pessoas em arquivo separado, idades impossíveis (máximo de {n['idade_maxima_bruta']} anos), ano de fabricação 0, "não sei" escrito de várias formas, contagens de vítimas que não fecham ({n.i('n_contagens_inconsistentes')} acidentes), traçado da via com {n.i('n_combinacoes_tracado')} combinações e {n.i('n_causas_distintas')} causas.
- **O que fizemos:** juntamos pessoas aos acidentes, anulamos valores impossíveis, unificamos os rótulos, removemos {n.i('d02_removidas')} linhas (suicídios presumidos e sinistros pessoais; nenhum duplicado), criamos atributos de hora, calendário, causa, via e envolvidos, sinalizamos outliers e tiramos as colunas que **vazam o alvo**. **Sem escalonar, sem one-hot, sem imputar média:** isso é da Pessoa 3, depois do split.
- **O que ficou:** **{n.i('linhas_final')} acidentes × {n.i('colunas_modelagem')} colunas** em `data/processed/`, com **{n.p('pct_fatal_final')}** de acidentes fatais (classe desbalanceada).""",
        # 2 ---------------------------------------------------------------
        "## 2. Diagrama do pipeline\n\n"
        "Cada caixa mostra o tamanho da base **depois** da etapa.\n\n" + _diagrama_mermaid(etapas),
        # 3 ---------------------------------------------------------------
        f"""## 3. Funil de linhas

Saímos de **{n.i('linhas_ocorrencias_bruto')}** e ficamos com **{n.i('linhas_final')}** ({n.p('pct_linhas_mantidas')} do total). Esta base perde pouca coisa: ela já vem organizada pela PRF. O maior corte foi **{corte['descricao'].lower()}** ({corte['etapa']}).

![Funil de linhas](figuras/funil_linhas.png)

{tab_funil}

- **D01**: duplicidade checada **depois** de juntar as pessoas. {n.i('ocorrencias_duplicadas_sem_id')} registro(s) igual(is) no arquivo de ocorrências; após juntar as pessoas, {n.i('d01_removidas')} continuam idênticos (os demais são acidentes distintos).
- **D02**: suicídio presumido e sinistro pessoal não são acidentes de trânsito. Nenhuma ação de segurança viária os evitaria e eles inflariam a taxa de fatais. Foram {n.i('d02_removidas')} acidentes, {n.i('n_fatais_removidos_d02')} deles fatais.
- A integração com o arquivo de pessoas **não perde acidentes**: {n.i('ocorrencias_sem_pessoas')} acidentes ficaram sem pessoas.""",
        # 4 ---------------------------------------------------------------
        f"""## 4. Mapa de ausentes e valores impossíveis

![Mapa de ausentes](figuras/mapa_ausentes.png)

{tab_ausentes}

**Ponto didático:** aqui quase não há nulo de verdade (só {n.i('n_na_classificacao')} textos `NA`). O problema é diferente: **rótulos "não sei"** (`Ignorado`, `Não Informado`, `NA`) e **valores impossíveis** (idade {n['idade_maxima_bruta']}, ano de fabricação 0). Nulo declarado é fácil de ver; valor impossível passa despercebido e estraga a média. Por isso a auditoria olha os valores, não só os nulos.""",
        # 5 ---------------------------------------------------------------
        "## 5. O que estava errado → o que fizemos → por quê\n\n"
        "Esta é a tabela mais importante. Cada linha é uma decisão registrada no código com o mesmo ID.\n\n" + tab_decisoes,
        # 6 ---------------------------------------------------------------
        f"""## 6. Antes × depois

### 6.1 O alvo (balanceamento de classes)

![Alvo antes e depois](figuras/antes_depois_alvo.png)

A PRF classifica o acidente em 3 classes (`Sem Vítimas`, `Com Vítimas Feridas`, `Com Vítimas Fatais`). Criamos `acidente_fatal` (1 = ao menos uma morte). No bruto, {n.p('pct_fatal_bruto')} dos acidentes tinham morte; depois dos filtros, **{n.i('n_fatais_final')}** acidentes (**{n.p('pct_fatal_final')}**). É minoria: na modelagem, prefira F1, AUC e *class weights* à acurácia.

### 6.2 Outliers (só sinalizados)

![Boxplots dos outliers](figuras/outliers_boxplots.png)

![Antes e depois de um outlier](figuras/antes_depois_outliers.png)

{tab_outliers}

{n.i('n_variaveis_com_outlier')} de {n.i('n_variaveis_outlier')} variáveis têm outliers pela regra do IQR (de {n.p('pct_outlier_min')} a {n.p('pct_outlier_max')} das linhas). São acidentes e veículos reais (um acidente chega a {n.i('veiculos_maximo_bruto')} veículos), então **não removemos nem limitamos**: criamos colunas `*_outlier` (0/1) e deixamos os valores intactos.

### 6.3 Causas (alta cardinalidade)

![Causas antes e depois](figuras/antes_depois_causas.png)

`causa_acidente` tinha **{n.i('n_causas_antes')}** causas distintas; viraram **{n.i('n_grupos_causa_depois')}** grupos (atenção do condutor, velocidade, álcool, manobra, via...). O traçado da via tinha **{n.i('n_combinacoes_tracado')}** combinações (ex.: `Reta;Declive`) e virou **8** colunas 0/1.

### 6.4 Padrões que já aparecem (descritivo; a Pessoa 2 testa)

![Taxa de acidentes fatais por grupo](figuras/taxa_fatal_por_grupo.png)

Só uma prévia, **sem teste estatístico**: a taxa geral de acidentes fatais é {n.p('taxa_fatal_geral')}. Na madrugada é {n.p('taxa_fatal_madrugada')}, contra {n.p('taxa_fatal_manha')} de manhã. Quando há pedestre envolvido, é {n.p('taxa_fatal_pedestre')}; com motocicleta, {n.p('taxa_fatal_moto')} (sem moto, {n.p('taxa_fatal_sem_moto')}). Nas causas de álcool, {n.p('taxa_fatal_alcool')}. Estes números justificam criar os atributos de faixa horária, causa e envolvidos.""",
        # 7 ---------------------------------------------------------------
        f"""## 7. Atributos criados (F01–F11)

Todos usam só informação sobre **quem e como** (hora, local, via, envolvidos) e **nunca** a consequência do acidente. O resumo numérico de cada coluna está em `tabelas/atributos_criados.csv`.

{tab_atributos}

Também criamos as colunas `*_outlier` (subseção "Outliers", na seção 6) e `contagens_inconsistentes` (só no interim).""",
        # 8 ---------------------------------------------------------------
        f"""## 8. Cuidados

### Vazamento de dados

- **{n.i('n_colunas_resultado_removidas')} colunas de consequência saem da modelagem** (`mortos`, `feridos_leves`, `feridos_graves`, `ilesos`, `ignorados`, `feridos`, `pessoas` e `classificacao_acidente`). Elas definem o alvo ou só existem depois do acidente. Ficam no interim para rastreio.
- **Nada de escalonar, codificar ou imputar aqui.** Essas contas aprendem com os dados e precisam ser feitas só no treino, dentro de um `Pipeline` do sklearn (Pessoa 3).
- Os limites do IQR e a detecção de colunas quase constantes usam frequência na base inteira, mas nunca o alvo; a Pessoa 3 pode recalcular no treino se quiser rigor máximo.
- **Causa e tipo do acidente só são definidos pela PRF depois do atendimento.** Servem para *explicar* o risco e orientar decisões (onde reforçar o quê), mas um modelo que os use não funcionaria como alerta antes do acidente.

### Nulos que sobram (de propósito)

O plano da equipe fala em base "sem nulos". Na modelagem sobram nulos em **{n.i('n_colunas_com_nulo_final')} colunas** ({n['colunas_com_nulo_final']}): `idade_media_condutores` em {n.p('pct_nulo_final_idade_media_condutores')}, `pct_condutores_homens` em {n.p('pct_nulo_final_pct_condutores_homens')} e `idade_media_veiculos` em {n.p('pct_nulo_final_idade_media_veiculos')} dos acidentes (sem condutor de idade válida, sexo ou ano de fabricação válido). Preenchê-los com média ou mediana aprende com os dados e, antes do split, vaza informação do teste. **A Pessoa 3 imputa dentro do `Pipeline`** (por exemplo, `SimpleImputer(add_indicator=True)`).

### Atributos sensíveis

{n.i('n_colunas_sensiveis')} colunas da base de modelagem vêm de idade e sexo dos condutores (`idade_media_condutores`, `pct_condutores_homens`) e estão marcadas no dicionário. Use-as para **analisar equidade**, não para discriminar.

### Limitações dos dados

- Só entram acidentes **registrados pela PRF em rodovias federais**; ruas urbanas e rodovias estaduais ficam de fora (viés de registro).
- Anos de 2024 e 2025. O ano de 2026 existe, mas é parcial e a PRF atualiza o arquivo todo mês; ficou de fora para a análise ser reprodutível.
- Classe positiva rara ({n.p('pct_fatal_final')}). Os resultados mostram **associação**, não causa.

### O que verificamos e **não** era problema

| Suspeita | O que encontramos |
|---|---|
| Latitude positiva é erro | Das {n.i('n_latitude_positiva')} latitudes positivas, {n.i('n_latitude_positiva_valida')} estão em UFs ao norte do Equador (AP e RR): **são válidas**. Coordenadas anuladas: {n.i('n_coordenadas_invalidas')} |
| Acidentes sem pessoas ou pessoas sem acidente | {n.i('ocorrencias_sem_pessoas')} e {n.i('pessoas_sem_ocorrencia')}: o join casa 100% |
| `id` repetido | {'Não há: cada `id` aparece uma vez' if n['id_unico'] == '1' else 'Há ids repetidos'} |
| Rótulos iguais com acento ou caixa diferente (`Céu Claro` x `CEU CLARO`) | {n.i('n_rotulos_equivalentes_caixa_acento')} casos |""",
        # 9 ---------------------------------------------------------------
        f"""## 9. O que entregamos para o resto do grupo

| Arquivo | Para quê |
|---|---|
| `data/processed/acidentes_prf_modelagem.csv` e `.parquet` | Base final: **{n.i('linhas_final')} linhas × {n.i('colunas_modelagem')} colunas**, alvo `acidente_fatal` na última coluna |
| `data/processed/dicionario_de_dados.csv` | O que cada coluna significa, tipo ({n.i('n_colunas_numerica')} numéricas, {n.i('n_colunas_categorica')} categóricas, {n.i('n_colunas_binaria')} binárias, {n.i('n_colunas_alvo')} alvo), origem e se é sensível |
| `data/interim/acidentes_prf_limpo.csv` | Versão legível, com `id`, consequências e `trecho` ({n.i('colunas_interim')} colunas). Gerada pelo pipeline (não versionada: é grande) |
| `docs/tabelas/*.csv` | Evidências de cada decisão (auditoria antes/depois) |
| `docs/figuras/*.png` | Gráficos para o artigo e os slides |

Carregar em 2 linhas:

```python
import pandas as pd
df = pd.read_csv("data/processed/acidentes_prf_modelagem.csv")  # ou pd.read_parquet(...)
X, y = df.drop(columns="acidente_fatal"), df["acidente_fatal"]
```

Para a Pessoa 3: as colunas de texto precisam de *one-hot*; `km`, `veiculos`, `hora` e as idades são numéricas; as colunas `*_outlier` são opcionais; `ano` serve para **validação temporal** (treinar em 2024, testar em 2025) e não como atributo; o *split* treino/teste é de vocês (não geramos).""",
        # 10 --------------------------------------------------------------
        _roteiro_e_perguntas(n),
        # 11 --------------------------------------------------------------
        """## 11. Como reproduzir

```bash
python -m venv .venv
.venv\\Scripts\\activate            # Linux/Mac: source .venv/bin/activate
pip install -r requirements.txt

python -m src.p1_engenharia_dados.baixar_base          # baixa os 4 arquivos da PRF para data/raw/
python -m src.p1_engenharia_dados.executar_pipeline    # refaz TUDO (dados, tabelas, figuras, textos)
pytest                                                 # confere as regras
python -m pydoc src.p1_engenharia_dados.limpeza        # documentação das funções
```

O pipeline é determinístico: rodar duas vezes gera arquivos idênticos.""",
    ]
    return "\n\n".join(secoes) + "\n"


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
| 0:00–0:20 | **Origem e justificativa da base:** Dados Abertos da PRF, {n.i('linhas_ocorrencias_bruto')} acidentes em rodovias federais (2024–2025), mais um arquivo com {n.i('linhas_pessoas_bruto')} pessoas/veículos. Escolhida por ser recente, brasileira, ter dois arquivos para integrar e um problema com decisão clara: o que leva a um acidente **fatal**. |
| 0:20–0:50 | Problemas: pessoas em outro arquivo, idade impossível (até {n['idade_maxima_bruta']} anos), ano de fabricação 0, "não sei" escrito de várias formas e {n.i('n_contagens_inconsistentes')} acidentes com contagem de vítimas que não fecha. |
| 0:50–1:30 | **Funil:** {n.i('linhas_ocorrencias_bruto')} → {n.i('linhas_final')}. A base perde pouco: {n.i('d01_removidas')} duplicados e {n.i('d02_removidas')} suicídios/sinistros pessoais. Mostrar o que **não** era problema (latitude positiva em AP/RR é válida). |
| 1:30–2:00 | **Integração e inconsistências:** idade, sexo e veículo vêm do arquivo de pessoas; valores impossíveis viram nulo, sem inventar valor. |
| 2:00–2:30 | **Atributos novos:** hora e faixa do dia, feriado, região, grupo de causa, traçado da via e envolvidos (moto, caminhão, pedestre). Prévia: {n.p('taxa_fatal_madrugada', 1)} de fatais na madrugada contra {n.p('taxa_fatal_manha', 1)} de manhã. |
| 2:30–3:00 | **Cuidados:** {n.i('n_colunas_resultado_removidas')} colunas de consequência fora da modelagem (vazamento), classe positiva só {n.p('pct_fatal_final', 1)}. Entrega: base de {n.i('linhas_final')} × {n.i('colunas_modelagem')} + dicionário. |

### 10 perguntas prováveis (resposta curta)

1. **Por que juntou o arquivo de pessoas?** Idade do condutor, tipo de veículo e pedestre só existem lá, e ajudam a explicar o risco. Resumimos por acidente e juntamos pelo `id`; nenhum dos {n.i('linhas_ocorrencias_bruto')} acidentes foi perdido.
2. **Por que removeu suicídios e sinistros pessoais?** Não são acidentes de trânsito: nenhuma fiscalização, sinalização ou obra os evitaria, e eles inflariam a taxa de fatais. Foram {n.i('d02_removidas')} acidentes.
3. **Por que não imputou a idade que faltava?** Não dá para saber a idade real, e imputar média aprende com os dados (vaza informação do teste se feito antes do split). Deixamos o nulo e a Pessoa 3 imputa dentro do `Pipeline`.
4. **Por que `mortos` virou alvo e as outras colunas de vítimas saíram?** `mortos` define o alvo; feridos, ilesos e a classificação são consequência do mesmo evento. Usá-los como atributo seria vazamento: o modelo "colaria" a resposta.
5. **Por que escolheram essa base e só 2024 e 2025?** Ela atende às regras do projeto (mais de cinco mil linhas, variáveis numéricas e categóricas, alvo claro, exige preparação, não é base de lista nem *Hotel Booking*) e é recente, como o planejamento pedia. 2026 é um ano parcial e a PRF atualiza o arquivo todo mês; ficar com anos completos torna a análise reprodutível.
6. **Os outliers não atrapalham?** São acidentes reais (um chega a {n.i('veiculos_maximo_bruto')} veículos) e estão entre os mais graves. Sinalizamos com `*_outlier` e a Pessoa 3 decide, estimando limites só no treino.
7. **Por que não escalonou nem fez one-hot?** São transformações que aprendem dos dados (média, categorias) e dependem do split. Ficam no `Pipeline` da Pessoa 3.
8. **Como garantiu que não há vazamento?** Consequências fora da base de modelagem, atributos só de hora, local, via e envolvidos, e nenhuma estatística do teste. Reconhecemos que causa e tipo só são conhecidos após o atendimento: o modelo explica risco, não funciona como alerta prévio.
9. **Usar idade e sexo é ético?** Estão marcados como sensíveis ({n.i('n_colunas_sensiveis')} colunas, resumidas por acidente). Servem para auditar equidade, não para discriminar.
10. **Os dados são representativos?** Só acidentes registrados pela PRF em rodovias federais. Rodovias estaduais e vias urbanas ficam de fora, então os resultados valem para esse recorte."""


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

A base utilizada reúne os acidentes de trânsito registrados pela Polícia Rodoviária Federal (PRF) em rodovias federais brasileiras nos anos de 2024 e 2025, disponibilizados como dados abertos pelo órgão (Boletim de Acidente de Trânsito, atualização mensal) [PRF 2026]. O conjunto é composto por dois arquivos por ano: o de **ocorrências**, com {n.i('linhas_ocorrencias_bruto')} acidentes descritos por {n.i('colunas_ocorrencias_bruto')} atributos (data, horário, rodovia, quilômetro, município, causa, tipo, condições do tempo, traçado e tipo da pista, e a contagem de vítimas), e o de **pessoas**, com {n.i('linhas_pessoas_bruto')} registros de pessoas e veículos envolvidos (idade, sexo, tipo de envolvimento, tipo e ano de fabricação do veículo).

O problema de negócio é identificar quais condições elevam o risco de um acidente ser **fatal** e, com isso, apoiar a decisão de onde e quando concentrar fiscalização, sinalização e campanhas. A variável-alvo (`acidente_fatal`) foi construída a partir do número de mortos: vale 1 quando há ao menos uma morte, o que ocorre em {n.p('pct_fatal_final')} dos acidentes, caracterizando um problema de classificação desbalanceado. Os atributos são numéricos (quilômetro, número de veículos, hora, idade dos condutores e dos veículos) e categóricos (rodovia, estado, tipo de acidente, fase do dia, condição do tempo, tipo de pista e grupo de causa).

O pré-processamento foi implementado em Python de forma determinística e reprodutível. Na **auditoria inicial**, observou-se que a base é bem preenchida, mas contém rótulos de "não informado" escritos de várias formas (`Ignorado`, `Não Informado`, `NA`) e valores impossíveis, como idades de até {n['idade_maxima_bruta']} anos e ano de fabricação de veículo igual a zero. Na **integração de dados**, o arquivo de pessoas foi resumido por acidente (presença de motocicleta, caminhão, ônibus, bicicleta e pedestre, idade média dos condutores, proporção de condutores homens e idade média dos veículos) e associado ao de ocorrências pelo identificador, sem perda de registros; foram também integrados um calendário de feriados nacionais e a região do país de cada estado.

Na **remoção de inconsistências**, foram excluídos {n.i('d01_removidas')} acidente duplicado e {n.i('d02_removidas')} registros de suicídio presumido ou sinistro pessoal, que não configuram acidentes de trânsito, resultando em {n.i('linhas_final')} registros. Idades e anos de fabricação impossíveis foram convertidos em valores ausentes, sem substituição por estimativas, e {n.i('n_contagens_inconsistentes')} acidentes com contagens de vítimas incoerentes foram sinalizados. Nenhuma imputação estatística foi feita nesta etapa; imputação, escalonamento e codificação foram reservados a um *pipeline* ajustado apenas no conjunto de treino, evitando vazamento de informação.

Na **transformação de variáveis**, as {n.i('n_causas_antes')} causas foram agrupadas em {n.i('n_grupos_causa_depois')} grupos, as {n.i('n_combinacoes_tracado')} combinações de traçado da via foram separadas em indicadores, e foram derivados a hora, a faixa do dia, o mês, o fim de semana e o feriado. A **análise de outliers** pelo critério do IQR apontou valores atípicos em {n.i('n_variaveis_com_outlier')} das {n.i('n_variaveis_outlier')} variáveis numéricas analisadas; por representarem acidentes reais, foram apenas sinalizados. Na **seleção de atributos**, foram removidos o identificador, as colunas administrativas, {n.i('n_colunas_resultado_removidas')} colunas que descrevem a consequência do acidente (para evitar vazamento do alvo) e {n.i('n_quase_constantes')} colunas quase constantes, resultando em {n.i('colunas_modelagem')} colunas na base de modelagem.

**Referências.** Polícia Rodoviária Federal (PRF). *Dados Abertos da PRF: Acidentes* (BAT). Disponível em: https://www.gov.br/prf/pt-br/acesso-a-informacao/dados-abertos/dados-abertos-da-prf. Acesso em: 10 nov. 2026.
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

