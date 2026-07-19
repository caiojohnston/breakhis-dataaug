# Testes Estatisticos - Downstream Binario

Comparacao pareada no conjunto de teste entre baseline binario A e C50_full calibrado.
O bootstrap foi feito por paciente para respeitar a estrutura patient-wise do BreakHis.

## Modelos Comparados

| Modelo | Threshold | Accuracy | Balanced accuracy | F1 macro | AUC |
|---|---:|---:|---:|---:|---:|
| A_binary_argmax | 0.5000 | 0.8589 | 0.8113 | 0.8288 | 0.8946 |
| C50_full_calibrated | 0.9700 | 0.8802 | 0.8673 | 0.8646 | 0.9222 |

## McNemar Exato

- A correto / C50_full errado: `69`
- A errado / C50_full correto: `101`
- Pares discordantes: `170`
- p-valor exato bicaudal: `0.017169`
- Significativo a 5%: `True`

## Bootstrap por Paciente - Diferenca C50_full menos A

| Metrica | Media diff | IC 95% baixo | IC 95% alto | p bootstrap |
|---|---:|---:|---:|---:|
| accuracy | 0.0219 | -0.0357 | 0.0812 | 0.4602 |
| balanced_accuracy | 0.0618 | -0.0005 | 0.1230 | 0.0526 |
| f1_macro | 0.0411 | -0.0295 | 0.1105 | 0.2550 |
| auc | 0.0325 | -0.0125 | 0.1211 | 0.3181 |

## Leitura

O ganho de C50_full calibrado sobre A e positivo nas metricas principais. A significancia deve ser interpretada com cautela porque ha apenas 17 pacientes no teste; por isso o bootstrap por paciente e mais conservador do que uma analise por imagem.
