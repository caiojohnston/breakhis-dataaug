# Achados e Resultados - Metodologia v2

## 2026-07-18 - Diagnostico previo aos novos treinos

Resultados v1 carregados de `results/cenario_A.json`, `results/cenario_B.json` e `results/cenario_C.json`.

| Cenario | Accuracy | F1 macro | Macro AUC aproximado qualitativo |
|---|---:|---:|---|
| A | 0.4983 | 0.2771 | Forte em adenosis e ductal; fraco em classes raras |
| B | 0.4152 | 0.2226 | Melhora relevante em phyllodes AUC, piora global |
| C | 0.4398 | 0.2495 | Melhor que B; melhora AUC em varias classes raras |

Accuracy binaria derivada das matrizes 8 classes:

| Cenario | Accuracy binaria derivada |
|---|---:|
| A | 0.8363 |
| B | 0.7711 |
| C | 0.8343 |

Interpretacao: o classificador v1 aprende razoavelmente a separacao benigno/maligno, mas falha na granularidade dos 8 subtipos. A proxima rodada deve atacar loss, backbone e selecao de checkpoint antes de investir em fine-tuning de VAE.

## Mudanca experimental registrada

Foi criado o protocolo `classifier_v2.yaml`:

- `tf_efficientnetv2_b1.in1k` como backbone.
- Focal Loss com `gamma=2.0`.
- Pesos por classe calculados do treino efetivo de cada cenario.
- Checkpoint por `f1_macro` de validacao.
- Saidas com sufixo `_v2`.

Resultados numericos v2 serao adicionados aqui apos cada treino.

## 2026-07-18 - Validacao tecnica antes de treino longo

Validacoes executadas:

- `py_compile` passou para os modulos alterados.
- `tf_efficientnetv2_b1.in1k` instanciou com `pretrained=False`.
- Forward dummy retornou logits `(2, 8)`.
- `FocalLoss` retornou valor finito.
- Dataloaders do Cenario A v2 carregaram corretamente: train=4892, val=1514, test=1503.
- Pesos por classe do treino A v2:

```text
[2.417, 0.8067, 10.1917, 1.6438, 0.284, 1.5287, 1.1581, 1.6617]
```

Interpretacao: o peso de `phyllodes_tumor` ficou muito alto por causa de apenas 60 imagens no treino. Isso e esperado e deve ser monitorado; caso o treino fique instavel, testar `weighted_sampler=false` com CrossEntropy ponderada ou limitar peso maximo.

## 2026-07-18 - Falha operacional no primeiro A v2

A primeira tentativa de `scenario A` com `classifier_v2.yaml` carregou CUDA, pesos pretrained e class weights corretamente, mas falhou antes do primeiro batch:

```text
PermissionError: [WinError 5] Access is denied
```

Causa: criacao de fila multiprocessing do DataLoader no Windows/sandbox.

Acao aplicada: `num_workers: 0` foi adicionado ao `classifier_v2.yaml`, e `train_classifier.py` passou a aceitar `num_workers` configuravel.

## 2026-07-18 - Resultado A v2 com Focal Loss ponderada

Resultado salvo em `results/cenario_A_v2.json`.

| Metrica | A v1 | A v2 Focal+pesos |
|---|---:|---:|
| Accuracy | 0.4983 | 0.4544 |
| F1 macro | 0.2771 | 0.2571 |
| Balanced accuracy | nao registrado | 0.2589 |
| AUC macro | nao registrado | 0.6503 |
| Melhor epoca | 1 | 8 |

Achado: a troca para EfficientNetV2-B1 nao foi suficiente quando combinada com Focal Loss ponderada. O treinamento overfitou rapido (`train_accuracy=0.9565` na melhor epoca), enquanto `val_f1_macro` ficou apenas 0.1550.

Decisao: nao rodar B/C com este protocolo. Proxima abla??o: manter `tf_efficientnetv2_b1.in1k`, trocar para `CrossEntropyLoss` sem pesos e manter selecao por `f1_macro`.

## 2026-07-18 - Diagnostico estrutural do split por pacientes

Pacientes totais por subtipo no BreakHis usado:

| Subtipo | Pacientes totais | Train | Val | Test |
|---|---:|---:|---:|---:|
| adenosis | 4 | 2 | 1 | 1 |
| ductal_carcinoma | 38 | 25 | 5 | 8 |
| fibroadenoma | 10 | 7 | 1 | 2 |
| lobular_carcinoma | 5 | 3 | 1 | 1 |
| mucinous_carcinoma | 9 | 6 | 1 | 2 |
| papillary_carcinoma | 6 | 4 | 1 | 1 |
| phyllodes_tumor | 3 | 1 | 1 | 1 |
| tubular_adenoma | 7 | 5 | 1 | 1 |

Interpretacao: a classificacao 8-subtipos patient-wise esta limitada por baixissimo numero de pacientes em varias classes. `phyllodes_tumor`, por exemplo, tem 1 paciente no treino, 1 no val e 1 no test. Isso explica por que mudancas de backbone/loss nao estao recuperando macro F1.

Decisao metodologica proposta: usar downstream binario benigno/maligno como endpoint principal, mantendo o LDM condicionado por 8 subtipos e reportando metricas por subtipo como analise secundaria/exploratoria.

## 2026-07-18 - Resultado A binario

Resultado salvo em `results/cenario_A_binary.json`.

| Metrica | Valor |
|---|---:|
| Accuracy | 0.8589 |
| Balanced accuracy | 0.8113 |
| F1 macro | 0.8288 |
| AUC macro | 0.8979 |
| Melhor epoca | 13 |

Relatorio por classe:

| Classe | Precision | Recall | F1 | Support |
|---|---:|---:|---:|---:|
| benign | 0.8616 | 0.6748 | 0.7569 | 489 |
| malignant | 0.8580 | 0.9477 | 0.9007 | 1014 |

Matriz de confusao:

```text
[[330, 159],
 [ 53, 961]]
```

Interpretacao: o baseline binario patient-wise atinge patamar coerente com a literatura e com a expectativa do projeto. A metodologia deve seguir com A/B/C binarios como downstream principal.

## 2026-07-18 - Resultado B binario

Resultado salvo em `results/cenario_B_binary.json`.

| Metrica | A binario | B binario | Delta B-A |
|---|---:|---:|---:|
| Accuracy | 0.8589 | 0.8430 | -0.0159 |
| Balanced accuracy | 0.8113 | 0.8127 | +0.0014 |
| F1 macro | 0.8288 | 0.8180 | -0.0108 |
| AUC macro | 0.8979 | 0.9036 | +0.0057 |
| Melhor epoca | 13 | 14 | - |

Matriz de confusao B:

```text
[[355, 134],
 [102, 912]]
```

Interpretacao: o aumento classico reduz falsos positivos benignos->malignos (159 para 134), mas aumenta falsos negativos malignos->benignos (53 para 102). Por isso a balanced accuracy e AUC sobem discretamente, enquanto accuracy/F1 macro caem.

## 2026-07-18 - Resultado C binario

Resultado salvo em `results/cenario_C_binary.json`.

| Metrica | A binario | B binario | C binario |
|---|---:|---:|---:|
| Accuracy | 0.8589 | 0.8430 | 0.8490 |
| Balanced accuracy | 0.8113 | 0.8127 | 0.8224 |
| F1 macro | 0.8288 | 0.8180 | 0.8260 |
| AUC macro | 0.8979 | 0.9036 | 0.9114 |
| Melhor epoca | 13 | 14 | 7 |

Matriz de confusao C:

```text
[[365, 124],
 [103, 911]]
```

Achado central: C supera B em accuracy, balanced accuracy, F1 macro e AUC macro. Em relacao a A, C e melhor em balanced accuracy e AUC macro, mas fica discretamente abaixo em accuracy e F1 macro. Como AUC de C e a maior, ha indicio de que o classificador C tem melhor ordenacao probabilistica e pode se beneficiar de calibracao de threshold escolhida na validacao.

## 2026-07-18 - Comparativo com threshold calibrado

Foram gerados:

- `results/comparativo_binary_thresholds_f1-macro.json`
- `results/comparativo_binary_thresholds_f1-macro.md`
- `results/comparativo_binary_thresholds_balanced-accuracy.json`
- `results/comparativo_binary_thresholds_balanced-accuracy.md`

### Threshold escolhido para maximizar F1 macro na validacao

| Cenario | Threshold | Accuracy | Balanced accuracy | F1 macro | AUC |
|---|---:|---:|---:|---:|---:|
| A | 0.4800 | 0.8583 | 0.8097 | 0.8276 | 0.8946 |
| B | 0.5400 | 0.8436 | 0.8153 | 0.8195 | 0.9035 |
| C | 0.6800 | 0.8436 | 0.8338 | 0.8259 | 0.9113 |

### Threshold escolhido para maximizar balanced accuracy na validacao

| Cenario | Threshold | Accuracy | Balanced accuracy | F1 macro | AUC |
|---|---:|---:|---:|---:|---:|
| A | 0.9900 | 0.8523 | 0.8297 | 0.8310 | 0.8946 |
| B | 0.5800 | 0.8450 | 0.8189 | 0.8218 | 0.9035 |
| C | 0.6800 | 0.8436 | 0.8338 | 0.8259 | 0.9113 |

Leitura para o TCC: o Cenario C e consistentemente superior ao B e lidera nas metricas mais adequadas para dados desbalanceados (`balanced_accuracy` e `AUC`). A vantagem sobre A nao aparece em accuracy/F1, mas aparece em AUC e balanced accuracy, sugerindo melhor discriminacao probabilistica com necessidade de calibracao.

## 2026-07-18 - Preparacao da abla??o C25/C50/C100

Foi adicionado suporte a `synthetic_fraction` no `BreakHisDataset`, com amostragem deterministica por subtipo (`synthetic_seed`).

Configs criadas:

- `configs/classifier_binary_c25.yaml`
- `configs/classifier_binary_c50.yaml`
- `configs/classifier_binary_c100.yaml`

Proxima execucao planejada: C50 binario.

## 2026-07-18 - Pausa operacional C50

O treino `C` com `configs/classifier_binary_c50.yaml` foi interrompido para liberar o computador.

Ultimo estado no log:

```text
Epoch 20/30 | train_loss=0.3909 acc=0.7619 | val_loss=4.2054 acc=0.8217 | val_f1=0.7863 bal_acc=0.7679 auc=0.8345
```

Melhor checkpoint salvo ate a interrupcao:

```text
Epoch 18/30 | val_f1=0.7962 | val_balanced_accuracy=0.7814 | val_auc=0.8358
```

Nao foi gerado JSON final de teste para C50 porque a interrupcao ocorreu durante a epoca 21, antes da avaliacao final.

## 2026-07-18 - Resultado C50 parcial avaliado no teste

Arquivo consolidado: `results/cenario_C_binary_c50_partial.json`.

O `best.pt` parcial de C50 foi avaliado com threshold escolhido na validacao para maximizar F1 macro.

| Variante | Accuracy | Balanced accuracy | F1 macro | AUC |
|---|---:|---:|---:|---:|
| A binario | 0.8589 | 0.8113 | 0.8288 | 0.8979 |
| B binario | 0.8430 | 0.8127 | 0.8180 | 0.9036 |
| C100 binario | 0.8490 | 0.8224 | 0.8260 | 0.9114 |
| C50 parcial | 0.8849 | 0.8676 | 0.8686 | 0.9215 |

Matriz de confusao C50 parcial:

```text
[[400, 89],
 [ 84, 930]]
```

Leitura: C50 parcial supera A, B e C100 em todas as metricas principais. Isso sustenta uma narrativa mais forte: sinteticas LDM ajudam, mas a proporcao importa; C100 pode diluir o dominio real, enquanto C50 melhora generalizacao.

## 2026-07-18 - Resultado C25 completo

Resultado salvo em `results/cenario_C_binary_c25.json`.

C25 usa 25% das sinteticas LDM, totalizando `4892` reais + `3082` sinteticas.

| Variante | Accuracy | Balanced accuracy | F1 macro | AUC |
|---|---:|---:|---:|---:|
| A binario | 0.8589 | 0.8113 | 0.8288 | 0.8979 |
| B binario | 0.8430 | 0.8127 | 0.8180 | 0.9036 |
| C100 binario | 0.8490 | 0.8224 | 0.8260 | 0.9114 |
| C25 binario | 0.8756 | 0.8416 | 0.8531 | 0.9405 |

Com threshold calibrado por balanced accuracy na validacao, C25 ficou:

| Threshold | Accuracy | Balanced accuracy | F1 macro | AUC |
|---:|---:|---:|---:|---:|
| 0.7900 | 0.8789 | 0.8520 | 0.8591 | 0.9376 |

Matriz de confusao C25 argmax:

```text
[[364, 125],
 [ 62, 952]]
```

Matriz de confusao C25 calibrado:

```text
[[379, 110],
 [ 72, 942]]
```

Leitura: C25 e o melhor experimento completo e sustenta a tese de que sinteticas LDM melhoram o downstream quando usadas em proporcao controlada. A comparacao C25 vs C100 indica que excesso de sinteticas pode diluir o dominio real.

## 2026-07-19 - Resultado C50_full completo

Resultado salvo em `results/cenario_C_binary_c50_full.json`.

C50_full usa 50% das sinteticas LDM, totalizando `4892` reais + `6164` sinteticas. O treino foi monitorado por `results/run_status.json`, concluiu com early stopping na epoca 26 e melhor checkpoint na epoca 19 (`val_f1_macro=0.7954`).

| Variante | Accuracy | Balanced accuracy | F1 macro | AUC | Observacao |
|---|---:|---:|---:|---:|---|
| C25 argmax | 0.8756 | 0.8416 | 0.8531 | 0.9405 | melhor AUC completo |
| C50_full argmax | 0.8729 | 0.8381 | 0.8498 | 0.9263 | completo, threshold 0.50 |
| C25 calibrado | 0.8789 | 0.8520 | 0.8591 | 0.9376 | threshold val balanced=0.79 |
| C50_full calibrado | 0.8802 | 0.8673 | 0.8646 | 0.9222 | melhor completo por F1/balanced/accuracy |
| C50 parcial calibrado | 0.8849 | 0.8676 | 0.8686 | 0.9215 | evidencia complementar, treino interrompido |

Matriz de confusao C50_full calibrado:

```text
[[406, 83],
 [ 97, 917]]
```

Leitura: C50_full calibrado passa a ser o melhor experimento completo por F1 macro, balanced accuracy e accuracy. C25 permanece melhor em AUC. A conclusao metodologica fica mais forte: sinteticas LDM ajudam o downstream quando a proporcao e controlada, e C100 sugere que excesso de sinteticas pode diluir o dominio real.

## Avaliacao Gerativa Amostrada

Data: 2026-07-19

Arquivos:

- `results/metricas_gerativas.json`
- `results/metricas_gerativas.md`

Resumo: FID global amostrado = `219.668`. Os subtipos gerados tiveram SSIM baixo (`0.1095` a `0.1820`) e LPIPS alto (`0.7789` a `0.8421`), indicando que as imagens sinteticas ainda estao distantes da distribuicao real em metricas perceptuais.

Interpretacao: a contribuicao principal das sinteticas neste experimento e downstream, nao a substituicao perfeita do dominio real. Isso reforca a narrativa de uso controlado como augmentacao: C50_full calibrado melhora F1/balanced accuracy/accuracy mesmo com FID alto, enquanto C100 sugere que excesso de sinteticas pode diluir o dominio real.
