# Comparativo Binario - Ablacao de Sinteticas

C50_full calibrado e o melhor experimento completo por F1 macro, balanced accuracy e accuracy. C25 permanece como melhor AUC. C50_partial continua como evidencia complementar porque veio de checkpoint interrompido.

| Variante | Tipo | Threshold | Accuracy | Balanced accuracy | F1 macro | AUC | Melhor epoca |
|---|---|---:|---:|---:|---:|---:|---:|
| A | completed_argmax | - | 0.8589 | 0.8113 | 0.8288 | 0.8979 | 13 |
| B | completed_argmax | - | 0.8430 | 0.8127 | 0.8180 | 0.9036 | 14 |
| C100 | completed_argmax | - | 0.8490 | 0.8224 | 0.8260 | 0.9114 | 7 |
| C25 | completed_argmax | - | 0.8756 | 0.8416 | 0.8531 | 0.9405 | 24 |
| C50_full | completed_argmax | - | 0.8729 | 0.8381 | 0.8498 | 0.9263 | 19 |
| C25 calibrated | threshold_balanced_accuracy | 0.79 | 0.8789 | 0.8520 | 0.8591 | 0.9376 | - |
| C50_full calibrated | threshold_balanced_accuracy | 0.97 | 0.8802 | 0.8673 | 0.8646 | 0.9222 | - |
| C50_partial calibrated | threshold_f1_macro_partial | 0.39 | 0.8849 | 0.8676 | 0.8686 | 0.9215 | - |

## Leitura principal

- Melhor experimento completo por F1 macro: `C50_full calibrated` (0.8646).
- Melhor experimento completo por AUC: `C25` (0.9405).
- Melhor resultado observado por F1 macro: `C50_partial calibrated` (0.8686).
- A narrativa mais defensavel: sinteticas LDM ajudam, mas a proporcao precisa ser controlada; C25 e C50_full superam A/B/C100 em metricas-chave.

## Matrizes de Confusao

### A

```text
[[330, 159], [53, 961]]
```

### B

```text
[[355, 134], [102, 912]]
```

### C100

```text
[[365, 124], [103, 911]]
```

### C25

```text
[[364, 125], [62, 952]]
```

### C50_full

```text
[[361, 128], [63, 951]]
```

### C25 calibrated

```text
[[379, 110], [72, 942]]
```

### C50_full calibrated

```text
[[406, 83], [97, 917]]
```

### C50_partial calibrated

```text
[[400, 89], [84, 930]]
```
