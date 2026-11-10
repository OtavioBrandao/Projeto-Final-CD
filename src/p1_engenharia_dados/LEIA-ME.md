# Engenharia de Dados e Pré-processamento
Entrega: base limpa para modelagem (data/processed) + documentação em docs/ + seção 'Base de Dados' do artigo.
Plano detalhado: docs/PLANO_IMPLEMENTACAO_ENGENHARIA_DE_DADOS.md · README desta parte (estudo e apresentação): docs/README_pre_processamento.md

Um comando refaz tudo (dados, tabelas, figuras e textos):

    python -m src.p1_engenharia_dados.executar_pipeline

| Módulo | Papel |
|---|---|
| baixar_base.py | baixa os 4 arquivos da PRF (ocorrências e pessoas, 2024 e 2025) |
| configuracao.py | caminhos, constantes, agrupamentos e limiares (um lugar só) |
| carregar.py | leitura dos CSVs (latin1, ';', vírgula decimal) e join ocorrências x pessoas |
| auditoria.py | perfil de qualidade antes de mexer nos dados |
| limpeza.py | D01-D07: duplicados, eventos fora do escopo, rótulos, valores impossíveis |
| engenharia_atributos.py | F01-F11: alvo, tempo, feriado, região, causa, via, resumo de pessoas |
| outliers.py | detecção por IQR e sinalização (sem alterar valores) |
| selecao.py | seleção de colunas (sem vazamento) e dicionário de dados |
| relatorios.py / documentacao.py | tabelas, figuras e textos de docs/ |
| executar_pipeline.py | orquestra tudo |

Documentação das funções: `python -m pydoc src.p1_engenharia_dados.limpeza`
