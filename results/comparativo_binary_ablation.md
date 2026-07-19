# Comparativo Binario - Ablacao de Sinteticas

C25 e o melhor experimento completo. C50 parcial e uma avaliacao de checkpoint interrompido, mantida como evidencia complementar.

| Variante | Tipo | Threshold | Accuracy | Balanced accuracy | F1 macro | AUC | Melhor epoca |
|---|---|---:|---:|---:|---:|---:|---:|
| A | completed_argmax | - | 0.8589 | 0.8113 | 0.8288 | 0.8979 | 13 |
| B | completed_argmax | - | 0.8430 | 0.8127 | 0.8180 | 0.9036 | 14 |
| C100 | completed_argmax | - | 0.8490 | 0.8224 | 0.8260 | 0.9114 | 7 |
| C25 | completed_argmax | - | 0.8756 | 0.8416 | 0.8531 | 0.9405 | 24 |
| C25 calibrated | threshold_balanced_accuracy | 0.79 | 0.8789 | 0.8520 | 0.8591 | 0.9376 | - |
| C50 partial calibrated | threshold_f1_macro_partial | 0.39 | 0.8849 | 0.8676 | 0.8686 | 0.9215 | - |

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

### C25 calibrated

```text
[[379, 110], [72, 942]]
```

### C50 partial calibrated

```text
[[400, 89], [84, 930]]
```
