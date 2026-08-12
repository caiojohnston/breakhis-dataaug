# Guia — retreino do LDM na GPU emprestada (16GB VRAM / 32GB RAM)

Pra quem vai rodar isso na máquina emprestada. Contexto completo em
`docs/investigacao_ldm_v2.md` se quiser entender o porquê; este guia é só o
passo a passo prático. Segue as instruções na ordem — cada passo diz o que
commitar antes de ir pro próximo, pra nunca perder trabalho.

## 0. Pré-requisitos

- Git instalado.
- Python 3.11 (o projeto foi feito com 3.11.8; outra versão 3.10+ provavelmente
  funciona mas não foi testada).
- Driver NVIDIA atualizado. Confirma com:
  ```powershell
  nvidia-smi
  ```
  Anota a versão de CUDA que aparece no canto (ex. "CUDA Version: 12.4") — vai
  precisar pra instalar o PyTorch certo no passo 3.

## 1. Pegar o código

```powershell
git clone https://github.com/caiojohnston/breakhis-dataaug.git
cd breakhis-dataaug
```

Se já tiver uma cópia (ex. passada por pen drive), só confirma que está na
branch `main` e atualizada:
```powershell
git status
git pull
```

## 2. Pegar o dataset (NÃO vem pelo git — é ~4GB, fica de fora do repositório)

O Caio vai te mandar um **link de download** (nuvem — Drive, WeTransfer, etc.)
com a pasta `data/raw/BreaKHis_v1/` já organizada (a extração original do
BreakHis, ~4GB, ~7.900 imagens PNG). Não é o dataset oficial cru — é a versão
já processada que o pipeline deste projeto espera, então baixa exatamente o
que ele mandar, não procura o dataset original por conta própria.

Depois de baixar e extrair, a pasta tem que ficar **exatamente** neste
caminho, relativo à raiz do projeto (repare no nome duplicado
`BreaKHis_v1/BreaKHis_v1/` — é assim mesmo, não é engano):

```
data/raw/BreaKHis_v1/BreaKHis_v1/histology_slides/breast/...
```

Os arquivos `data/splits/train.csv`, `val.csv`, `test.csv` (esses sim vêm no
git) apontam pros caminhos dentro dessa estrutura — se a pasta não bater
exatamente com esse caminho, o treino não vai achar as imagens.

**Confirma antes de seguir pro passo 3:**
```powershell
Get-ChildItem -Recurse -File data\raw\BreaKHis_v1 | Measure-Object | Select-Object -ExpandProperty Count
```
Tem que dar **7909**. Se der 0 ou bem menos, a pasta não está no lugar certo
ou o download não veio completo — para aqui e resolve antes de continuar.

## 3. Ambiente Python

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Depois, instala o PyTorch com suporte a CUDA (a versão do `requirements.txt`
pode vir sem CUDA). Usa o link certo pra tua versão de CUDA em
https://pytorch.org/get-started/locally/ — geralmente algo como:

```powershell
pip install torch --index-url https://download.pytorch.org/whl/cu124
```

(troca `cu124` pela versão de CUDA que apareceu no `nvidia-smi`.)

Confirma que a GPU aparece pro PyTorch:
```powershell
$env:PYTHONUTF8 = "1"
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```
Precisa imprimir `True` e o nome da tua placa. Se der `False`, para aqui e
resolve isso antes de continuar (driver ou instalação do torch errada).

## 4. Achar o batch máximo seguro (ANTES de treinar de verdade)

```powershell
$env:PYTHONUTF8 = "1"
python scripts_diag/ldm_vram_probe.py --channels 256,512,512,512 --find-max-batch
```

Isso testa a arquitetura nova (UNet com o dobro da largura da versão original)
e acha o maior batch que cabe em 85% da tua VRAM, sem chutar número. Demora
uns 1-2 minutos. No final imprime algo como:

```
Maior batch seguro (85% da VRAM) pra channels=(256, 512, 512, 512): 24
```

Abre `configs/ldm_v2_16gb.yaml` e substitui a linha:
```yaml
ldm_batch_size: 8          # PLACEHOLDER — ...
```
pelo número que apareceu (ex. `ldm_batch_size: 24`).

### → Commit já (não espera terminar o treino todo)

```powershell
git add configs/ldm_v2_16gb.yaml
git commit -m "chore: define batch size do LDM v2 achado na GPU emprestada"
git push
```

Isso garante que, mesmo se o treino demorado falhar por qualquer motivo depois,
o resultado desse teste rápido (que já é útil por si só) não se perde.

## 5. Rodar o treino de verdade (demorado — roda em background, monitorado)

```powershell
$env:PYTHONUTF8 = "1"
.venv\Scripts\python src\utils\start_monitored.py `
  --name ldm_v2_16gb `
  --log logs/ldm_v2_16gb.log `
  --status results/run_status_ldm_v2.json `
  --checkpoint-dir checkpoints/ldm_v2_16gb/ldm `
  --results-path results/ldm_v2_16gb_loss_history.json `
  -- .venv\Scripts\python -u src/training/train_ldm.py --stage ldm --config configs/ldm_v2_16gb.yaml --checkpoints-dir checkpoints/ldm_v2_16gb
```

Repara no `--checkpoints-dir checkpoints/ldm_v2_16gb` — isso é de propósito,
pra não sobrescrever os checkpoints do treino original que já geraram os
resultados do TCC (`checkpoints/ldm/`). Não muda esse caminho.

O comando volta rápido, o treino continua rodando por trás. Pra acompanhar:
```powershell
.venv\Scripts\python src\utils\show_run_status.py
```

Pode fechar o terminal e voltar depois — o treino continua rodando (é
processo em background). Só não desliga o PC.

**Se precisar parar antes de terminar:** o checkpoint é salvo ao final de cada
época, então é seguro parar. `.venv\Scripts\python src\utils\stop_monitored.py`
(ou `--force` se não responder).

## 6. Quando terminar

Vai ter uma mensagem de "LDM training concluído" no log, e o arquivo
`checkpoints/ldm_v2_16gb/ldm/training_complete.txt` vai existir.

### Gerar sintéticas com o checkpoint novo:

```powershell
$env:PYTHONUTF8 = "1"
python src/generation/generate.py --config configs/ldm_v2_16gb.yaml `
  --checkpoints-dir checkpoints/ldm_v2_16gb `
  --synthetic-dir data/synthetic_v2
```

(`--synthetic-dir data/synthetic_v2` também é de propósito — não sobrescreve
`data/synthetic/`, que é o resultado usado no TCC atual.)

### Avaliar qualidade gerativa:

```powershell
python src/evaluation/eval_generative.py `
  --config configs/ldm_v2_16gb.yaml `
  --synthetic-dir data/synthetic_v2 `
  --output results/metricas_gerativas_v2.json
```

### → Commit final

```powershell
git add results/ldm_v2_16gb_loss_history.json results/metricas_gerativas_v2.json
git commit -m "feat: retreino LDM v2 (UNet maior, GPU 16GB) — resultado da avaliacao generativa"
git push
```

**Não** dá `git add` em `checkpoints/` nem `data/synthetic_v2/` — são
propositalmente ignorados pelo `.gitignore` (checkpoint é gigante, sintética é
gerada, dá pra recriar). O que importa versionar é o config final usado, o
histórico de loss e os números da avaliação — isso sim é permanente.

## Se algo der errado

- **`CUDA out of memory` mesmo depois do find-max-batch:** algo mais estava
  usando a GPU (outro programa, jogo, etc.) na hora do teste. Fecha tudo e
  roda o find-max-batch de novo.
- **Terminal fecha sozinho / perde conexão:** o processo continua rodando
  (é background de verdade, não depende do terminal ficar aberto). Só reabre
  outro PowerShell na mesma pasta e roda `show_run_status.py` de novo.
- **Loss vira `nan`:** já aconteceu antes (ver `docs/investigacao_ldm_v2.md` e
  `CLAUDE.md`) — foi bug de FP16/AMP. Esse config usa FP32 (mesma solução de
  antes), não deveria acontecer. Se acontecer mesmo assim, para o treino e
  chama o Caio antes de tentar de novo.
