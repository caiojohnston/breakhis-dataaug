# Metricas Gerativas - Avaliacao Amostrada

Avaliacao executada em 2026-07-19 com amostragem deterministicamente controlada (`seed=42`).

- FID global: `219.668`
- Escopo global: `generated_subtypes_only_sampled`
- Maximo por subtipo para FID: `300`
- Maximo de pares por subtipo para SSIM/LPIPS/PSNR: `100`

| Subtipo | FID | SSIM | LPIPS | PSNR | Real | Sinteticas | Status |
|---|---:|---:|---:|---:|---:|---:|---|
| adenosis | 284.7316 | 0.1820 | 0.8421 | 8.5844 | 253 | 1900 | ok |
| fibroadenoma | 235.7016 | 0.1341 | 0.8399 | 8.2475 | 758 | 1395 | ok |
| phyllodes_tumor | 308.5863 | 0.1251 | 0.8216 | 7.7415 | 60 | 2093 | ok |
| tubular_adenoma | 257.8646 | 0.1095 | 0.8070 | 8.2711 | 372 | 1781 | ok |
| ductal_carcinoma | - | - | - | - | 2153 | 0 | skipped_no_synthetic_images |
| lobular_carcinoma | 230.1126 | 0.1557 | 0.7789 | 9.0130 | 400 | 1753 | ok |
| mucinous_carcinoma | 259.4122 | 0.1179 | 0.7957 | 7.6713 | 528 | 1625 | ok |
| papillary_carcinoma | 247.6999 | 0.1439 | 0.7801 | 8.4586 | 368 | 1785 | ok |

## Leitura

- O FID global alto indica distancia distribucional relevante entre imagens reais e sinteticas.
- Mesmo assim, os experimentos downstream C25/C50_full melhoraram F1/accuracy, sugerindo que as sinteticas funcionam melhor como regularizacao/augmentacao controlada do que como substituto perfeito do dominio real.
- `ductal_carcinoma` nao possui sinteticas porque ja era a classe majoritaria usada como alvo de equalizacao; por isso foi marcado como `skipped_no_synthetic_images`.
