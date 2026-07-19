# Registro de Revisoes do TCC

Arquivo principal revisado: `docs/tcc_abnt_completo.md`  
Versao de leitura/formatacao: `docs/tcc_abnt_completo.html`  
Data: 2026-07-19

## Revisao 1 - Estrutura e completude

Verificado:

- Elementos pre-textuais: capa, folha de rosto, resumo, abstract, lista de tabelas, lista de abreviaturas e siglas, sumario.
- Elementos textuais: introducao, fundamentacao teorica, metodologia, resultados, discussao e conclusao.
- Elementos pos-textuais: referencias.
- Presenca da narrativa central: hipotese parcialmente confirmada e refinada.

Ajustes aplicados:

- Criada versao completa do TCC do inicio ao fim.
- Inseridas lista de tabelas e lista de abreviaturas e siglas.
- Alinhado o sumario aos titulos reais das secoes.

Pendencia proposital:

- Dados pessoais e institucionais permanecem como placeholders, pois nao foram informados pelo autor e nao devem ser inventados.

## Revisao 2 - Coerencia metodologica e resultados

Verificado contra arquivos locais:

- `docs/metodologia_v2_tracker.md`
- `results/metodologia_v2_achados.md`
- `results/comparativo_binary_ablation.md`
- `results/statistical_tests.md`
- `results/metricas_gerativas.md`

Pontos confirmados:

- O protocolo inicial de oito subtipos aparece como diagnostico negativo.
- A virada para classificacao binaria esta explicitada e justificada.
- C100, C25 e C50_full aparecem como ablacao de proporcao sintetica.
- C50_full calibrado esta identificado como principal resultado downstream completo.
- C25 permanece identificado como melhor AUC completa.
- McNemar por imagem aparece com p = 0,017169.
- Bootstrap por paciente aparece com efeitos positivos e intervalos conservadores.
- FID_global = 219,668 aparece como limitacao gerativa.
- As limitacoes obrigatorias foram incluidas.

Ajustes aplicados:

- Reforcado que o McNemar e analise exploratoria pos-selecao.
- Inserida nota sobre pequenas diferencas centesimais de AUC em relatorios auxiliares.
- Trocados termos fortes por formulacoes mais defensaveis.

## Revisao 3 - Citacoes, referencias e linguagem

Verificado:

- Todas as citacoes autor-data no corpo possuem entrada correspondente em referencias.
- Referencia de Osorio et al. foi citada no corpo apos revisao independente.
- Links de referencias foram limpos para remover parametros de rastreamento.
- Nao foram adicionadas citacoes sem fonte consultada.
- Nao foram adicionadas afirmacoes clinicas ou resultados nao observados.

Ajustes aplicados:

- Inserida citacao de Osorio et al. (2023) na fundamentacao sobre LDM em histopatologia.
- Inseridas fontes abaixo das tabelas.
- Especificado que a avaliacao gerativa foi amostrada e que as contagens por subtipo nao equivalem necessariamente ao numero integral usado em cada metrica.
- Substituidas formulacoes coloquiais identificadas pelos revisores por linguagem academica mais sobria.

## Revisoes independentes por subagentes

Foram usados tres subagentes sem memoria compartilhada (`fork_context=false`):

- Revisor ABNT/editorial: apontou placeholders, sumario, fontes de tabelas, referencia nao citada e termos fortes.
- Revisor metodologico-estatistico: confirmou a consistencia geral e apontou cautela sobre AUC e McNemar pos-selecao.
- Revisor de citacoes: confirmou que nao havia citacao sem referencia e apontou apenas Osorio et al. como referencia inicialmente nao citada.

Correcoes derivadas dessas revisoes foram incorporadas em `docs/tcc_abnt_completo.md` e refletidas em `docs/tcc_abnt_completo.html`.