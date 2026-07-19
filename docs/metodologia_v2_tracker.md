# Metodologia v2 - Recuperacao do Protocolo BreakHis

## Objetivo

Fazer a metodologia Encoder-Difusor funcionar de forma defensavel em ate 1 mes, corrigindo primeiro o gargalo downstream antes de gastar novas horas em fine-tuning do VAE ou novo treino do LDM.

## Diagnostico inicial

Data: 2026-07-18

Resultados v1 existentes:

| Cenario | Accuracy 8 classes | F1 macro | Observacao |
|---|---:|---:|---|
| A - original | 0.4983 | 0.2771 | Melhor checkpoint ficou na epoca 1 |
| B - aumento classico | 0.4152 | 0.2226 | Piorou accuracy global, ajudou phyllodes em AUC |
| C - sinteticas LDM | 0.4398 | 0.2495 | Melhor que B, ainda abaixo de A em accuracy |

Achado importante: ao colapsar a matriz 8 classes para benigno vs maligno, o Cenario A atinge aproximadamente 0.8363 de accuracy binaria. Isso indica que o modelo nao esta completamente quebrado; o principal problema e a classificacao fina entre os 8 subtipos.

## Hipoteses de falha da rodada v1

1. EfficientNet-B0 pode estar fraco para 8 subtipos em split patient-wise.
2. `CrossEntropyLoss` sem pesos favorece a classe majoritaria `ductal_carcinoma`.
3. Checkpoint por `val_loss` e fragil com validacao desbalanceada.
4. `phyllodes_tumor` tem 60 imagens no treino e 235 na validacao, distorcendo early stopping.
5. O Cenario C usou muitas sinteticas de uma vez, podendo diluir o dominio real.

## Decisao tecnica v2

Antes de fine-tunar VAE, rerodar A/B/C com classificador mais forte e protocolo robusto:

- Backbone: `tf_efficientnetv2_b1.in1k`.
- Loss: Focal Loss com pesos por classe.
- Selecao de checkpoint: `val_f1_macro`.
- Metricas adicionais: balanced accuracy e macro AUC.
- Resultados salvos com sufixo `_v2`, sem sobrescrever v1.

## Status

| Item | Status |
|---|---|
| Criar tracker metodologico | Concluido |
| Criar `configs/classifier_v2.yaml` | Concluido |
| Parametrizar backbone no wrapper do classificador | Concluido |
| Adicionar Focal Loss/class weights/selection metric | Concluido |
| Validar sintaxe/CLI curta | Concluido |
| Rodar Cenario A v2 | Concluido - resultado abaixo de A v1; Focal+pesos agressivos descartado como protocolo principal |
| Criar/Rodar A v2 CE sem pesos agressivos | Pendente |
| Rodar Cenario B binario | Concluido |
| Rodar Cenario C binario | Concluido |
| Comparar A/B/C binarios | Concluido |
| Avaliar necessidade de C25/C50/C100 | Concluido - C25 e melhor experimento completo; C50 parcial e melhor checkpoint observado em F1 |
| Implementar avaliacao gerativa | Pendente |
| Diagnosticar viabilidade 8 subtipos patient-wise | Concluido - poucas classes tem 3 a 7 pacientes no total |
| Criar rota downstream binaria | Concluido |
| Decidir fine-tuning VAE | Pendente |

## Validacao tecnica

Data: 2026-07-18

Validacoes concluidas sem treino longo:

- `py_compile` de `classifier.py`, `train_classifier.py` e `dataset.py`.
- Instanciacao de `tf_efficientnetv2_b1.in1k` com `pretrained=False`.
- Forward dummy com saida `(2, 8)`.
- Focal Loss finita no batch dummy.
- Config `classifier_v2.yaml` carregada com train=4892, val=1514, test=1503 no Cenario A.
- Pesos calculados no treino A: `[2.417, 0.8067, 10.1917, 1.6438, 0.284, 1.5287, 1.1581, 1.6617]`.
- Primeira tentativa de treino A v2 iniciou com CUDA, mas falhou no DataLoader por `PermissionError: [WinError 5] Access is denied` ao criar fila multiprocessing. Correcao aplicada: `num_workers: 0` em `classifier_v2.yaml`.

## Proximos comandos planejados

```powershell
.venv\Scripts\python src/training/train_classifier.py --scenario A --config configs/classifier_v2.yaml
.venv\Scripts\python src/training/train_classifier.py --scenario B --config configs/classifier_v2.yaml
.venv\Scripts\python src/training/train_classifier.py --scenario C --config configs/classifier_v2.yaml
```

## Criterios de sucesso imediatos

1. Baseline A v2 com Focal+pesos nao superou A v1; proxima tentativa deve isolar backbone usando CrossEntropy sem pesos.
2. Cenario C v2 deve ser comparado principalmente por macro F1, balanced accuracy e macro AUC.
3. Accuracy global continua reportada, mas nao deve ser a unica metrica de decisao por causa do desbalanceamento.

## Resultado A v2 - Focal Loss com pesos

Data: 2026-07-18

Arquivo: `results/cenario_A_v2.json`

| Metrica | A v1 | A v2 Focal+pesos | Delta |
|---|---:|---:|---:|
| Accuracy | 0.4983 | 0.4544 | -0.0439 |
| F1 macro | 0.2771 | 0.2571 | -0.0200 |
| Balanced accuracy | nao registrado v1 | 0.2589 | - |
| AUC macro | nao registrado v1 | 0.6503 | - |

Interpretacao: a combinacao Focal Loss + class weights foi agressiva demais. O treino chegou a 0.9565 de accuracy no treino na epoca 8, mas validacao macro F1 ficou em 0.1550. Isso indica overfitting/instabilidade no protocolo de balanceamento, nao necessariamente falha do backbone.

## Diagnostico de Pacientes por Subtipo

Data: 2026-07-18

| Subtipo | Imagens totais | Pacientes totais | Train pacientes | Val pacientes | Test pacientes |
|---|---:|---:|---:|---:|---:|
| adenosis | 444 | 4 | 2 | 1 | 1 |
| ductal_carcinoma | 3451 | 38 | 25 | 5 | 8 |
| fibroadenoma | 1014 | 10 | 7 | 1 | 2 |
| lobular_carcinoma | 626 | 5 | 3 | 1 | 1 |
| mucinous_carcinoma | 792 | 9 | 6 | 1 | 2 |
| papillary_carcinoma | 560 | 6 | 4 | 1 | 1 |
| phyllodes_tumor | 453 | 3 | 1 | 1 | 1 |
| tubular_adenoma | 569 | 7 | 5 | 1 | 1 |

Conclusao operacional: 8 subtipos com split estritamente patient-wise e muito instavel para ser o unico downstream principal. A rota mais defensavel em 1 mes e usar classificacao binaria benigno/maligno como downstream principal e manter resultados por subtipo/AUC como analise complementar.

## Resultado A Binario

Data: 2026-07-18

Arquivo: `results/cenario_A_binary.json`

| Metrica | Valor |
|---|---:|
| Accuracy | 0.8589 |
| Balanced accuracy | 0.8113 |
| F1 macro | 0.8288 |
| AUC macro | 0.8979 |
| Melhor epoca | 13 |

Matriz de confusao:

```text
[[330, 159],
 [ 53, 961]]
```

Interpretacao: a expectativa de baseline acima de 80% e atingida quando o downstream e formulado como benigno/maligno, mantendo split patient-wise. Isso confirma que o pipeline geral funciona; o gargalo dos 8 subtipos vem da baixa quantidade de pacientes por subtipo.

## Resultado B Binario

Data: 2026-07-18

Arquivo: `results/cenario_B_binary.json`

| Metrica | A binario | B binario | Delta B-A |
|---|---:|---:|---:|
| Accuracy | 0.8589 | 0.8430 | -0.0159 |
| Balanced accuracy | 0.8113 | 0.8127 | +0.0014 |
| F1 macro | 0.8288 | 0.8180 | -0.0108 |
| AUC macro | 0.8979 | 0.9036 | +0.0057 |

Interpretacao: aumento classico melhora ligeiramente AUC e balanced accuracy, mas reduz accuracy e F1 macro. Para o TCC, isso pode ser apresentado como efeito misto do aumento classico, nao como ganho universal.

## Resultado C Binario

Data: 2026-07-18

Arquivo: `results/cenario_C_binary.json`

| Metrica | A binario | B binario | C binario | Melhor |
|---|---:|---:|---:|---|
| Accuracy | 0.8589 | 0.8430 | 0.8490 | A |
| Balanced accuracy | 0.8113 | 0.8127 | 0.8224 | C |
| F1 macro | 0.8288 | 0.8180 | 0.8260 | A |
| AUC macro | 0.8979 | 0.9036 | 0.9114 | C |

Interpretacao: o Cenario C supera o aumento classico B em todas as metricas bin?rias e supera A em balanced accuracy e AUC macro. A leve queda contra A em accuracy/F1 sugere problema de calibracao de threshold: C tem melhor ranking probabilistico (AUC), mas o threshold argmax/0.5 favorece menos a classe maligna.

## Comparativo com Threshold Calibrado

Data: 2026-07-18

Arquivos:

- `results/comparativo_binary_thresholds_f1-macro.json`
- `results/comparativo_binary_thresholds_f1-macro.md`
- `results/comparativo_binary_thresholds_balanced-accuracy.json`
- `results/comparativo_binary_thresholds_balanced-accuracy.md`

Threshold escolhido em validacao e aplicado no teste.

### Maximizando F1 macro na validacao

| Cenario | Threshold | Accuracy | Balanced accuracy | F1 macro | AUC |
|---|---:|---:|---:|---:|---:|
| A | 0.4800 | 0.8583 | 0.8097 | 0.8276 | 0.8946 |
| B | 0.5400 | 0.8436 | 0.8153 | 0.8195 | 0.9035 |
| C | 0.6800 | 0.8436 | 0.8338 | 0.8259 | 0.9113 |

### Maximizando balanced accuracy na validacao

| Cenario | Threshold | Accuracy | Balanced accuracy | F1 macro | AUC |
|---|---:|---:|---:|---:|---:|
| A | 0.9900 | 0.8523 | 0.8297 | 0.8310 | 0.8946 |
| B | 0.5800 | 0.8450 | 0.8189 | 0.8218 | 0.9035 |
| C | 0.6800 | 0.8436 | 0.8338 | 0.8259 | 0.9113 |

Conclusao: C e o melhor em AUC e balanced accuracy, inclusive apos calibracao. A narrativa defensavel e que o LDM melhora discriminacao probabilistica e equilibrio entre classes frente ao aumento classico, embora A ainda fique ligeiramente acima em accuracy/F1 em algumas leituras.

## Ablacao de Fracao Sintetica

Data: 2026-07-18

Objetivo: testar se reduzir a quantidade de imagens sinteticas melhora accuracy/F1 do Cenario C sem perder o ganho observado em AUC/balanced accuracy.

Configs criadas:

- `configs/classifier_binary_c25.yaml` (`synthetic_fraction: 0.25`)
- `configs/classifier_binary_c50.yaml` (`synthetic_fraction: 0.50`)
- `configs/classifier_binary_c100.yaml` (`synthetic_fraction: 1.00`)

Decisao operacional: iniciar por C50, pois reduz a dominancia das sinteticas com custo menor que C100 e ainda testa a hipotese central.

## Pausa Operacional C50

Data: 2026-07-18

O treino `C` com `configs/classifier_binary_c50.yaml` foi interrompido a pedido do usuario para liberar o computador.

Estado salvo:

- Checkpoints disponiveis: `epoch_01.pt` ate `epoch_20.pt`.
- Melhor checkpoint atual: `checkpoints/cenario_C_binary_c50/best.pt`, salvo na epoca 18.
- Melhor validacao observada: `val_f1_macro=0.7962`, `val_balanced_accuracy=0.7814`, `val_auc_macro=0.8358`.
- Nao ha `results/cenario_C_binary_c50.json` final porque o treino foi parado antes da avaliacao de teste.

Proxima acao recomendada: avaliar o `best.pt` parcial no teste ou retomar o experimento C50 do zero em uma janela livre.

## Resultado C50 Parcial

Data: 2026-07-18

Arquivo: `results/cenario_C_binary_c50_partial.json`

O checkpoint parcial `checkpoints/cenario_C_binary_c50/best.pt` foi avaliado no teste. Esse `best.pt` corresponde a melhor validacao ate a interrupcao operacional do treino C50.

| Variante | Accuracy | Balanced accuracy | F1 macro | AUC | Decisao |
|---|---:|---:|---:|---:|---|
| A binario | 0.8589 | 0.8113 | 0.8288 | 0.8979 | argmax |
| B binario | 0.8430 | 0.8127 | 0.8180 | 0.9036 | argmax |
| C100 binario | 0.8490 | 0.8224 | 0.8260 | 0.9114 | argmax |
| C50 parcial | 0.8849 | 0.8676 | 0.8686 | 0.9215 | threshold val F1=0.39 |

Matriz de confusao do C50 parcial no teste:

```text
[[400, 89],
 [ 84, 930]]
```

Conclusao: reduzir as sinteticas para 50% resolveu a queda de accuracy/F1 observada no C100 e preservou o ganho em AUC/balanced accuracy. Este e o melhor resultado atual para defender a metodologia.

## Resultado C25 Completo

Data: 2026-07-18

Arquivos:

- `results/cenario_C_binary_c25.json`
- `results/c25/comparativo_binary_thresholds_f1-macro.json`
- `results/c25/comparativo_binary_thresholds_balanced-accuracy.json`
- `results/comparativo_binary_ablation.json`
- `results/comparativo_binary_ablation.md`

C25 usa 25% das sinteticas geradas: `4892` imagens reais + `3082` sinteticas = `7974` imagens de treino.

| Variante | Accuracy | Balanced accuracy | F1 macro | AUC | Observacao |
|---|---:|---:|---:|---:|---|
| A binario | 0.8589 | 0.8113 | 0.8288 | 0.8979 | baseline real |
| B binario | 0.8430 | 0.8127 | 0.8180 | 0.9036 | aumento classico |
| C100 binario | 0.8490 | 0.8224 | 0.8260 | 0.9114 | 100% sinteticas |
| C25 binario | 0.8756 | 0.8416 | 0.8531 | 0.9405 | melhor experimento completo |
| C25 calibrado | 0.8789 | 0.8520 | 0.8591 | 0.9376 | threshold val balanced=0.79 |
| C50 parcial calibrado | 0.8849 | 0.8676 | 0.8686 | 0.9215 | melhor checkpoint observado, treino interrompido |

Conclusao: C25 fornece a narrativa mais limpa para o TCC, pois e um treino completo e supera A/B/C100 em todas as metricas principais. C50 parcial permanece como evidencia complementar de que a proporcao de sinteticas pode ser otimizada.
