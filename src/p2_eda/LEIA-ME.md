# Pessoa 2 — EDA e Estatística

**Entrada:** `data/processed/acidentes_prf_modelagem.csv` (Pessoa 1). A EDA só lê; nunca altera a base.
**Saída:** `reports/figuras/p2_*.png`, `reports/tabelas/p2_*.csv`, `reports/README_eda.md` e a seção "Estatística Descritiva e Inferência" (`reports/secao_estatistica_descritiva_inferencia.md` e `.tex`).

```bash
python -m src.p2_eda.executar_eda    # gera tudo (~10 s)
pytest tests/test_eda_p2.py          # 20 testes (não precisam dos dados brutos)
```

| Arquivo | O que faz |
|---|---|
| `configuracao.py` | Todos os parâmetros (α, IC, hipóteses H1–H9, fatores, cores) com o porquê |
| `estatistica.py` | Funções puras: IC de Wilson, risco relativo, qui-quadrado + V de Cramér, Mann-Whitney, Cochran-Mantel-Haenszel, Holm |
| `analises.py` | Monta as tabelas (descritiva, fatores, testes, estratificação, veredito, ranking de atributos) |
| `graficos.py` | 8 figuras (4 principais + 4 de apoio); o título de cada uma é a conclusão |
| `documentacao.py` | Gera o README da EDA e a seção do artigo (Markdown + LaTeX) a partir das tabelas |
| `executar_eda.py` | Orquestra tudo |

Navegação interativa: `notebooks/p2_eda.ipynb`. Explicação completa dos resultados: `reports/README_eda.md`.
