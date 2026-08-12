# Investigação LDM v2 — diagnóstico de gargalo e preparação pra retreino em GPU maior

Data: 2026-08-11. Motivação: qualidade gerativa do TCC ficou baixa (FID global 219,668,
SSIM ~0,10-0,18, LPIPS ~0,78-0,84 — ver `results/metricas_gerativas.json` e
`docs/tcc_abnt_completo.md` seção 4.6). Inspeção visual confirmou: sintéticas têm cor
de coloração H&E plausível mas nenhuma estrutura celular reconhecível (sem núcleo,
sem glândula). Um amigo se ofereceu para emprestar PC com GPU de 16GB VRAM / 32GB RAM.
Antes de usar esse recurso, foi feito diagnóstico local pra saber ONDE está o gargalo
real, evitando gastar a máquina emprestada em tentativa às cegas.

## Fase 0 — VAE congelado é o gargalo? (resposta: não)

Script: `scripts_diag/vae_reconstruction_check.py`. Método: encode→decode de imagens
reais do BreakHis (8 imagens, 4 subtipos) usando o `sd-vae-ft-mse` **congelado**
(o mesmo usado no pipeline, sem nenhum treino), medindo SSIM/PSNR da reconstrução
contra o original.

**Resultado:**

| Subtipo | SSIM médio | PSNR médio |
|---|---:|---:|
| adenosis | 0,8412 | 32,03 |
| ductal_carcinoma | 0,5797 | 22,91 |
| fibroadenoma | 0,7210 | 27,38 |
| mucinous_carcinoma | 0,7649 | 28,77 |
| **Média geral** | **0,7267** | **27,77** |

Inspeção visual (`scripts_diag/vae_recon_out/*.png`) confirma: núcleo, glândula e
estroma preservados mesmo no pior caso (ductal_carcinoma, SSIM 0,54). Compare com o
SSIM das sintéticas do LDM contra real, que fica em 0,10-0,18 — a reconstrução do VAE
sozinho é 4-7x melhor que a saída final do pipeline completo.

**Conclusão:** o VAE congelado não é o fator limitante da qualidade gerativa. A
decisão metodológica de não fazer fine-tuning do VAE (registrada no TCC como
limitação por restrição de hardware) provavelmente não seria a alavanca certa mesmo
com mais GPU disponível — pelo menos não a prioritária.

## Causa raiz identificada: UNet do LDM subdimensionado

Em `configs/ldm.yaml` / `src/models/ldm.py` (antes desta investigação), a arquitetura
tinha comentário explícito: `ldm_batch_size: 8  # conservador para GTX 1660 SUPER 6GB`
e `block_out_channels=(128, 256, 256, 256)` — cerca de 40% da largura do UNet padrão
do Stable Diffusion original (320/640/1280/1280), redimensionado pra caber em 6GB de
VRAM. Isso é consistente com o padrão visual observado: cor certa (o VAE resolve
isso), estrutura errada (o difusor não tem capacidade suficiente pra aprender a
distribuição latente completa de tecido histológico).

## Mudanças de código feitas hoje

- `src/models/ldm.py`: `ConditionedLDM` deixou de ter arquitetura hardcoded.
  Agora aceita `block_out_channels`, `layers_per_block`, `attention_head_dim` e
  `gradient_checkpointing` como parâmetros, com os valores antigos como default
  (backward-compatible — testado: config antigo `ldm.yaml` continua gerando
  exatamente 39,70M parâmetros, igual antes).
- `src/training/train_ldm.py`: lê esses parâmetros do YAML (`unet_block_out_channels`
  etc.) em vez de usar os valores fixos da classe. Também loga a contagem de
  parâmetros no início do treino.
- `src/training/train_ldm.py`: o VAE congelado ficava carregado na GPU durante todo
  o loop de treino do LDM mesmo sem ser usado após pré-codificar os latentes
  (só é necessário uma vez, no início). Corrigido: `del vae` + `torch.cuda.empty_cache()`
  logo após `precompute_latents`, liberando VRAM pro UNet/batch maior.
- `src/generation/generate.py`: tinha o mesmo problema — construía o `ConditionedLDM`
  com arquitetura hardcoded (128,256,256,256) antes de carregar o checkpoint.
  Carregar um checkpoint da arquitetura nova (256,512,512,512) ali ia quebrar por
  incompatibilidade de shape no `load_state_dict`. Corrigido pra ler a arquitetura
  do mesmo config usado no treino, igual `train_ldm.py`.
- Novo script `scripts_diag/ldm_vram_probe.py`: mede VRAM de pico e tempo/passo de
  uma arquitetura+batch usando latentes aleatórios (sem precisar do dataset real).
  Tem modo `--find-max-batch` que faz busca binária pelo maior batch seguro
  (85% da VRAM total) numa GPU qualquer — pra usar direto na máquina emprestada em
  vez de chutar número.
- `src/training/train_ldm.py`: agora salva histórico de loss por época em
  `results/{nome_do_config}_loss_history.json` (GPU usada, arquitetura, batch,
  loss e tempo por época). Tampa uma lacuna que a revisão do TCC apontou —
  o texto atual não tem nenhuma curva de treino. Esse JSON é leve, versionável,
  e vira figura de curva de aprendizado depois. Testado isoladamente (dataset
  falso, 3 épocas) — funciona e não mexe nos checkpoints reais.

## Calibração de VRAM feita na GTX 1660 SUPER (6GB, ~4,7GB livres no momento do teste)

| Config | Params | Batch | VRAM pico | Obs |
|---|---:|---:|---:|---|
| Atual (128,256,256,256) | 39,7M | 8 | 1.075 MB | baseline, bem abaixo do limite — havia folga não usada |
| Atual (128,256,256,256) | 39,7M | 64 | 4.088 MB | escala sub-linear, cabe folgado em 6GB |
| Full SD (320,640,1280,1280) | 700,2M | 8 | ~13.421 MB (estourou os 6GB físicos, foi pra memória compartilhada — por isso 18,7s/passo, número de tempo não é confiável, mas o de VRAM alocada é) | grande demais pra essa GPU, mas dá noção de escala pra 16GB |

**Config nova proposta** (`configs/ldm_v2_16gb.yaml`): `block_out_channels=(256,512,512,512)`
— largura dobrada em relação ao original, ~158M parâmetros (medido, ver smoke test
abaixo). Meio-termo entre o atual (subdimensionado, confirmado) e o full-SD
(provavelmente pesado demais pro tempo disponível na máquina emprestada).
`ldm_batch_size` no arquivo está como placeholder — **rodar
`python scripts_diag/ldm_vram_probe.py --channels 256,512,512,512 --find-max-batch`
direto na máquina de 16GB antes do treino real**, não assumir número.

## Smoke test (2026-08-11, CPU + GPU local)

- Construção do modelo via YAML → `ConditionedLDM` com os novos parâmetros: ok.
- Forward + backward com batch=2 em CPU: ok, sem erro de shape.
- Params contados: 158,11M pra `configs/ldm_v2_16gb.yaml` (256,512,512,512).
- Regressão: `configs/ldm.yaml` original continua gerando 39,70M parâmetros
  (idêntico a antes da refatoração) — mudança não quebrou o pipeline existente.

## Fase 0.5 — teste real de fine-tuning do VAE (2026-08-12, enquanto LDM v2 treina na máquina emprestada)

Enquanto o retreino do LDM roda na GPU do amigo, sobrou tempo (~12h) pra testar
uma coisa que a Fase 0 não respondeu sozinha: será que fine-tunar o VAE, mesmo
não sendo o gargalo principal, ainda ajudaria um pouco? A estimativa antiga de
"54h pra fine-tuning do VAE" (usada pra justificar pular essa etapa no TCC)
nunca foi medida de verdade — só um chute conservador.

**Calibração real** (`scripts_diag/vae_finetune_probe.py`, medido na GTX 1660
SUPER local):

| Batch | s/passo | VRAM | min/época |
|---|---:|---:|---:|
| 2 | 1,55s | 3,1GB | **63min** |
| 4 | 4,21s | 5,2GB | 86min (pior por imagem, apesar de menos passos) |
| 8 | 39s+ (travando) | ~5,8GB | descartado — perto do limite físico da placa |

Batch=2 é o melhor ponto (mais rápido por imagem, não só o que cabe). Com
`vae_epochs: 10` já configurado, 10 épocas ≈ **10,5h**, não 54h. A estimativa
antiga documentada em `CLAUDE.md` estava ~5x pessimista.

**Baseline em split de validação** (evita circularidade — o fine-tuning otimiza
no train; testar reconstrução no val mede generalização de verdade):
SSIM médio = **0,6481** (mais baixo que os 0,7267 medidos em train na Fase 0
original — esperado, val é mais difícil por definição).

**Treino real iniciado** (`checkpoints/vae/`, monitorado via
`start_monitored.py`, log em `logs/vae_finetune.log`, config
`configs/ldm.yaml` sem alteração — já tinha `vae_epochs: 10, vae_batch_size: 2`
prontos). A cada época, `scripts_diag/vae_reconstruction_check.py --vae-ckpt
checkpoints/vae/epoch_XX.pt --tag epoch_XX --split val` roda pra comparar
contra o baseline 0,6481 — decide se vale continuar ou se estabilizou/piorou.

**Bug de infraestrutura #1 encontrado ao lançar o monitor:** `start_monitored.py`
falhava com `WinError 2` (`FileNotFoundError` no `subprocess.Popen`) quando
chamado via Git Bash (Bash tool), mesmo com caminho absoluto e `.exe`
explícito. Resolvido rodando o mesmo comando via PowerShell — a ferramenta
`run_monitored.py`/`start_monitored.py` parece depender de resolução de path
que só funciona de forma confiável a partir do PowerShell (coerente com
`docs/operacao_runs_monitoradas.md`, cujos exemplos já eram só em PowerShell).
Registro pra não perder tempo de novo: **sempre lançar `start_monitored.py`
via PowerShell, nunca via Git Bash.**

**Bug real #2 — o treino do VAE morria silenciosamente (sem traceback) logo
nos primeiros passos.** Primeira hipótese (errada, registrada aqui pra
transparência): suspeitei do `num_workers=4` do DataLoader brigando com o
`DETACHED_PROCESS` do `start_monitored.py` no Windows. Mudei pra
`num_workers=0` em `train_vae()` e `precompute_latents()` (esse segundo
também usado pelo LDM — mudança mantida por ser mais segura de qualquer
forma) e o crash **continuou** — provando que não era essa a causa.

Causa real: `train_vae()` usava `torch.amp.autocast` + `GradScaler` (FP16),
e o estágio do VAE nunca tinha recebido o mesmo tratamento que o LDM/UNet
(que já tinha AMP desativado desde 2026-05-14 por instabilidade em FP16 —
ver seção de bugs no `CLAUDE.md`). Removido `autocast`/`scaler` do
`train_vae()`, treino em FP32 puro. Confirmado estável: 83+ passos sem
queda, GPU 94% de uso, ~1,26s/passo (mais rápido até que a calibração
inicial de 1,55s/passo) — ~51min/época real, 10 épocas ≈ **8,6h**, ainda
melhor que a estimativa de 10,5h.

## Próximos passos (nesta ordem, ver também `CLAUDE.md` seção "A Fazer")

1. Na máquina de 16GB: rodar `find-max-batch` pra `configs/ldm_v2_16gb.yaml`,
   preencher `ldm_batch_size` de verdade no config.
2. Retreinar o LDM do zero com essa config (não dá pra continuar do checkpoint
   antigo — arquitetura mudou). Manter FP32 na primeira tentativa (testar BF16
   depois, com cautela, só se sobrar tempo — GPU de 16GB provavelmente tem tensor
   core de verdade, ao contrário da 1660 SUPER, mas não assumir sem testar).
3. Regenerar sintéticas com o checkpoint novo, rodar `eval_generative.py` de novo,
   comparar contra a baseline atual (FID global 219,668) antes de tocar em
   qualquer coisa do downstream.
4. Só se a Fase 3 mostrar ganho real (FID menor, SSIM maior, núcleo/glândula
   visíveis nas amostras): refazer ablação C25/C50/C100 e calibração de threshold
   com as sintéticas novas, atualizar `docs/tcc_abnt_completo.md`.

## Para a escrita do TCC (nota pra depois)

Esse achado é material bom pro TCC independente do resultado final do retreino:
- Se o retreino melhorar a qualidade gerativa: vira uma seção nova de "iteração
  metodológica" — mostra que o diagnóstico (VAE ok, UNet subdimensionado) estava
  certo, reforça o valor do split gerativo vs. downstream que o TCC já defende.
- Se não melhorar o suficiente: ainda assim é limitação bem diagnosticada (não só
  "faltou hardware", mas "identificamos exatamente qual componente e por quê"),
  mais forte que o texto atual pra seção de limitações/trabalhos futuros.
- De qualquer forma, documentar: reconstrução do VAE congelado (SSIM 0,73) como
  evidência de que a arquitetura de encoder está adequada, isolando a discussão de
  limitação pro componente de difusão especificamente — mais preciso cientificamente
  do que a frase genérica atual ("VAE não fine-tunado pode ter limitado a qualidade").
