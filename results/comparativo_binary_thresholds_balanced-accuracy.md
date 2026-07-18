# Comparativo Binario com Threshold Calibrado

Threshold escolhido em validacao para maximizar `balanced_accuracy`; metricas reportadas no teste.

| Cenario | Threshold | Accuracy | Balanced accuracy | F1 macro | AUC |
|---|---:|---:|---:|---:|---:|
| A | 0.9900 | 0.8523 | 0.8297 | 0.8310 | 0.8946 |
| B | 0.5800 | 0.8450 | 0.8189 | 0.8218 | 0.9035 |
| C | 0.6800 | 0.8436 | 0.8338 | 0.8259 | 0.9113 |

## Matrizes de Confusao

### Cenario A

```text
[[374, 115], [107, 907]]
```

### Cenario B

```text
[[364, 125], [108, 906]]
```

### Cenario C

```text
[[394, 95], [140, 874]]
```

