# Pessoa 1 — Engenharia de Dados e Pré-processamento
Responsável: Pessoa 1. Entrega: dataset limpo (data/processed) + documentação em docs/ + seção 'Base de Dados' do artigo.
Plano detalhado: docs/PLANO_IMPLEMENTACAO_PESSOA1.md · Explicação para estudo/apresentação: docs/README.md

Um comando refaz tudo (dados, tabelas, figuras e textos):

    python -m src.p1_engenharia_dados.executar_pipeline

| Módulo | Papel |
|---|---|
| baixar_base.py | baixa diabetic_data.csv e IDS_mapping.csv do UCI |
| configuracao.py | caminhos, constantes e limiares (um lugar só) |
| carregar.py | leitura dos CSVs e join com IDS_mapping |
| auditoria.py | perfil de qualidade antes de mexer nos dados |
| limpeza.py | D01-D09: inconsistências e valores ausentes |
| engenharia_atributos.py | F01-F11: novos atributos |
| outliers.py | detecção por IQR e sinalização (sem alterar valores) |
| selecao.py | seleção de colunas e dicionário de dados |
| relatorios.py / documentacao.py | tabelas, figuras e textos de docs/ |
| executar_pipeline.py | orquestra tudo |

Documentação das funções: `python -m pydoc src.p1_engenharia_dados.limpeza`
