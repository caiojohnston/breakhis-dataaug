# Investigação: instabilidade no AUC binário e padronização do método

Data: 2026-08-14. Origem: sessão de revisão de escrita do TCC encontrou que
`eval_binary_thresholds.py` usa `roc_auc_score(labels, malignant_probs)` e
perguntou se isso também tinha o "bug do AUC" já registrado (uso de
`auc_roc_per_subtype['malignant']` em vez de `auc_macro` em
`results/statistical_tests.json`). Investigação mostrou que a pergunta partia
de premissa errada, e o problema real é mais profundo.

## `eval_binary_thresholds.py` está correto

`roc_auc_score(labels, malignant_probs)` é o cálculo padrão e correto de AUC
para classificação binária. Não existe "macro" a aplicar aqui — macro-AUC só
é um conceito relevante pra multi-classe (>2 classes). A pergunta original
comparou coisas de naturezas diferentes.

## O problema real: instabilidade numérica por saturação do modelo

`train_classifier.py` calcula AUC via one-vs-rest por classe e depois tira a
média (`auc_macro = mean([auc_benign, auc_malignant])`). Para um classificador
binário com softmax de 2 saídas, `auc_benign` e `auc_malignant` deveriam ser
**matematicamente idênticos** (identidade de AUC binário: ranquear por P(benigno)
decrescente é o exato inverso de ranquear por P(maligno) decrescente, logo os
dois AUCs one-vs-rest coincidem sempre — verificado com dado sintético).

Testado ao vivo no checkpoint real do Cenário A (`checkpoints/cenario_A_binary/best.pt`):

```
auc_benign (OvR)            = 0.9012
auc_malignant (OvR)         = 0.8946
auc_malignant via (1-p_benigno) = 0.8966
```

Três valores diferentes pro mesmo modelo, mesma amostra — deveriam ser
idênticos. `p_benigno + p_maligno = 1` confirmado até 1e-7 (softmax genuíno,
sem bug de normalização). Causa raiz: **o modelo está extremamente saturado**.
No conjunto de teste (1503 imagens), **880 (58,5%) têm probabilidade
essencialmente exata em 0 ou 1** (147 com p<1e-6, 733 com p>1-1e-6; só 720
valores únicos de probabilidade em 1503 amostras). Com tanto empate técnico
nas pontas, uma diferença de ponto flutuante de ~1e-7 entre `p_maligno` e
`1-p_benigno` já basta pra desempatar milhares de pares em ordem diferente e
mudar o AUC na 3ª-4ª casa decimal — os AUCs reportados com 4 casas decimais
tinham precisão falsa.

## Evidência de que isso já causava inconsistência interna na Tabela 5 do TCC

Como AUC não depende do threshold de decisão (é métrica de ranking pura), o
AUC de argmax e de calibrado do **mesmo checkpoint** deveria ser idêntico.
No TCC antes desta correção:

| Variante | AUC argmax (Tabela 5 antiga) | AUC calibrado (Tabela 5 antiga) |
|---|---:|---:|
| C25 | 94,05% | 93,76% |
| C50_full | 92,63% | 92,22% |

Dois valores diferentes pro mesmo modelo — logicamente impossível, e não é
coincidência: as linhas "argmax" vinham de `cenario_*_binary*.json`
(`train_classifier.py`, método OvR-macro instável) e as linhas "calibrado"
já vinham de `eval_binary_thresholds.py` (método único, estável). A
inconsistência era sintoma direto do mesmo problema.

## Correção aplicada

Reavaliados A, B, C100, C25 e C50_full com `eval_binary_thresholds.py`
(método único, padrão, estável) usando os checkpoints já treinados — **sem
retreinar nada**. Accuracy, balanced accuracy, F1 macro e matrizes de
confusão não mudaram (esses cálculos via sklearn não sofrem do mesmo
problema). Só o AUC mudou, e só de forma relevante em três variantes:

| Variante | AUC antigo | AUC corrigido | Δ |
|---|---:|---:|---:|
| A | 89,79% | **89,46%** | -0,33pp |
| B | 90,36% | 90,35% | -0,01pp (ruído, não relevante) |
| C100 | 91,14% | 91,13% | -0,01pp (ruído, não relevante) |
| C25 (argmax) | 94,05% | **93,76%** | -0,29pp |
| C50_full (argmax) | 92,63% | **92,22%** | -0,41pp |
| C25 calibrado | 93,76% | 93,76% | sem mudança (já estava certo) |
| C50_full calibrado | 92,22% | 92,22% | sem mudança (já estava certo) |

Agora argmax e calibrado do mesmo modelo batem exatamente (C25: 93,76% nos
dois; C50_full: 92,22% nos dois) — a inconsistência interna desapareceu.

**`results/comparativo_binary_ablation.json` já foi atualizado** com os
valores corrigidos (é a "fonte principal" declarada no método do TCC, seção
3.6). `results/statistical_tests.json` **não precisou de correção** — já
usava o método correto pros dois modelos comparados (A=0,8946,
C50_full_calibrado=0,9222), by design.

## Nenhuma conclusão do TCC muda de direção

- C25 continua sendo o melhor AUC completo (93,76% ainda é claramente o
  maior entre todas as variantes, só o valor exato mudou de 94,05%).
- C50_full calibrado continua sendo o melhor em acurácia/balanced
  accuracy/F1 macro (esses números não mudaram).
- McNemar e bootstrap (`results/statistical_tests.json`) não precisam ser
  re-executados — já usavam o AUC correto.

## O que precisa ser atualizado no texto do TCC (`docs/tcc_abnt_completo.md`/`.html`)

1. Tabela 4 (Resultados binários iniciais): AUC de A muda de 89,79% pra 89,46%.
2. Tabela 5 (Comparativo com ablação): AUC de A (89,79→89,46), C25 argmax
   (94,05→93,76), C50_full argmax (92,63→92,22).
3. Qualquer citação em prosa desses números — checar Resumo, Abstract,
   seção 4.2, 4.3, Discussão.
4. **Sugestão de frase nova nas limitações**, sobre a saturação do modelo:
   ~58% das predições no conjunto de teste do Cenário A ficam com
   confiança extrema (>99,9999% pra uma das classes), indicando um modelo
   pouco calibrado — métricas de ranking (AUC) ficam sensíveis a
   instabilidade numérica nesse regime, o que motivou a padronização do
   método de cálculo nesta revisão.
5. Tabela 7 (testes estatísticos) **não muda** — já usava os números certos.

Checkpoints e dados de reavaliação ficam em `results/v3_auc_fix/` (não
versionado — resultados intermediários grandes, mas os valores finais já
foram propagados pro `comparativo_binary_ablation.json`, que é rastreado).

## Nota sobre um desvio no meio do caminho

Antes desta investigação mais funda, uma orientação minha anterior (passada
pra sessão paralela de revisão de texto) descrevia o problema de forma
incompleta — dizia que `results/statistical_tests.json` estava errado e que
o valor "certo" era o `auc_macro` (0,8979/0,9263). Essa orientação foi
seguida à risca: `src/evaluation/eval_statistical_tests.py` chegou a ser
alterado pra usar OvR-macro, e `results/statistical_tests.json` foi
regravado com os valores antigos (0,8979/0,9263) e o bootstrap recalculado
(p mudou de 0,318072 pra 0,282731). Essa mudança foi descartada
(`git restore`) depois desta investigação mostrar que `statistical_tests.json`
já estava certo desde o início — o script nunca teve o "bug" original
descrito. Registro aqui pra não repetir o mesmo desvio de novo.
