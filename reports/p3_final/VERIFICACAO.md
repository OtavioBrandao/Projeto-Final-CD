# Verificação da entrega — Etapa 04

- Branch: `p3/modelagem`; HEAD: `80a677ce1abddd3e03e30208b2f7c1e22d62e2d6`.
- Estado inicial limpo. Alterações finais locais, sem commit, push, PR, merge ou rebase.
- Suíte completa antes da avaliação: **167 aprovados em 148,28 s**.
- Cinco testes novos repetidos após os ajustes: **5 aprovados em 25,20 s**.
- Reprodução de 2024 com `--somente-relatorios`: concluída, sem repetir os 42 ajustes.
- Os 32 arquivos de dados/P1/P2 e as 22 tabelas de 2024 têm hashes idênticos aos iniciais.
- Dois XGBoost e um Dummy treinados uma única vez em 73.006 registros de 2024;
  72.336 registros de 2025 avaliados uma vez por modelo, sendo 5.104 positivos.
- Fábrica existente: 100 árvores, profundidade 4, learning_rate 0,1, hist, CPU,
  n_jobs=2, random_state=42, objective=binary:logistic, eval_metric=logloss,
  sem pesos adicionais. Parâmetros completos e pré-processamento no manifesto.
- Limiares A=0.2066718339920044 e B=0.101788230240345, preservados.
- Roundtrip real verificado em amostra de **treino**, com rtol=1e-7/atol=1e-8.
  Os três artefatos reais também foram carregados em processo Python separado nesta auditoria,
  sem nova inferência de 2025; classes, features, nomes transformados e hashes conferidos.
- `--somente-relatorios` final executado sem carregar modelos, fit ou predict.
  Hashes das métricas e previsões finais permaneceram idênticos.
- Alinhamento dos índices originais, distribuição de classes, limiares e classes
  salvas verificados diretamente nas previsões existentes; nenhuma nova avaliação.
- Hashes das fontes usados no treino e dos artefatos conferidos. Relatórios públicos
  e metadados inspecionados sem caminhos locais absolutos. Cinco gráficos inspecionados.

## Arquivos criados e modificados

Modificados: [executor de validação](../../src/p3_modelagem/executar_validacao.py),
[manifesto de 2024](../p3_execucao_2024.json) e [relatório de 2024](../README_modelagem.md).
A correção adiciona referência relativa do CSV e aceita manifestos antigos sem a chave.
Nenhuma métrica ou previsão de 2024 foi alterada.

Criados: [executor final](../../src/p3_modelagem/executar_avaliacao_final.py),
[testes](../../tests/test_avaliacao_final_p3.py),
[verificação prévia](../p3_preparacao_etapa04.json),
[pipelines finais](../../models/), [relatório final](README.md),
[manifesto](manifesto.json), [métricas](metricas_2025.csv),
[previsões](previsoes_2025.csv.gz), [comparação temporal](comparacao_2024_2025.csv),
[calibração](calibracao_2025.csv), [figuras](figuras/) e esta verificação.

## Avisos e falhas técnicas

Nenhuma falha de treinamento, persistência, inferência ou relatório.
O Git emitiu aviso de permissão ao consultar seu arquivo de ignore global no sandbox.
O PowerShell propagou esse stderr como NativeCommandError/código 1, apesar da execução
Python concluída e do manifesto `concluido`. A conclusão foi verificada por hashes,
carregamento dos artefatos e reprodução dos relatórios. Não houve repetição do teste.

Na auditoria externa inicial, o carregamento encontrou `ModuleNotFoundError: src`:
o diretório do repositório precisava estar no sys.path do script externo. A auditoria
foi corrigida adicionando a raiz ao caminho de importação; nenhum modelo foi alterado,
retreinado ou submetido a nova inferência. Consumidores devem executar na raiz usando
`python -m ...` ou garantir que o pacote src esteja disponível no caminho de importação.

## Integração e limites

Pessoa 4: usar o pipeline confiável, seu pré-processamento ajustado e os nomes
transformados do manifesto; manter correspondência com as features de SHAP.
Pessoa 5: consumir as tabelas prontas, preservando modelo + linha_original como chave;
utilizar as classes do limiar congelado, não pipeline.predict implicitamente.
Resultados completos, comparações, calibração e limitações metodológicas estão no
[relatório final](README.md). Não foram implementados SHAP ou dashboard.
