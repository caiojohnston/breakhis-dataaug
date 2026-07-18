# Comparativo Binario com Threshold Calibrado

Threshold escolhido em validacao para maximizar F1 macro; metricas reportadas no teste.

| Cenario | Threshold | Accuracy | Balanced accuracy | F1 macro | AUC |
|---|---:|---:|---:|---:|---:|
| A | 0.4800 | 0.8583 | 0.8097 | 0.8276 | 0.8946 |
| B | 0.5400 | 0.8436 | 0.8153 | 0.8195 | 0.9035 |
| C | 0.6800 | 0.8436 | 0.8338 | 0.8259 | 0.9113 |

## Matrizes de Confusao

### Cenario A

```text
[[328, 161], [52, 962]]
```

### Cenario B

```text
[[359, 130], [105, 909]]
```

### Cenario C

```text
[[394, 95], [140, 874]]
```

