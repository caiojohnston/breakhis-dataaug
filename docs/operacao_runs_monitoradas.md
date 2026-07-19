# Operacao de Runs Monitoradas

Este fluxo evita treinos longos "as cegas". Toda run longa deve ser iniciada
com `src/utils/start_monitored.py`, que cria um processo em background,
mantem um log vivo e atualiza `results/run_status.json`.

## Iniciar uma run longa

Exemplo para o C50 binario completo, sem sobrescrever o C50 parcial:

```powershell
.venv\Scripts\python src/utils/start_monitored.py `
  --name cenario_C_binary_c50_full `
  --log logs/cenario_C_binary_c50_full.log `
  --checkpoint-dir checkpoints/cenario_C_binary_c50_full `
  --results-path results/cenario_C_binary_c50_full.json `
  -- .venv\Scripts\python -u src/training/train_classifier.py --scenario C --config configs/classifier_binary_c50_full.yaml
```

O comando retorna rapido. O treino continua em background.

## Acompanhar progresso

```powershell
.venv\Scripts\python src/utils/show_run_status.py
```

## Painel visivel em uma janela separada

Para deixar um painel atualizando em outro terminal:

```powershell
.\watch_status.ps1
```

Para abrir uma nova janela do PowerShell com o painel:

```powershell
.\open_status_window.ps1
```

O painel mostra barra de progresso por epoca, metricas atuais, checkpoint mais recente,
ultimas linhas do log e dica de pausa segura.

Para ver mais linhas finais do log:

```powershell
.venv\Scripts\python src/utils/show_run_status.py --tail 15
```

O arquivo bruto tambem fica disponivel em:

```text
results/run_status.json
```

## Pausar ou interromper

O classificador salva checkpoint ao fim de cada epoca. A janela mais segura de
pausa e logo depois de aparecer uma linha `Epoch XX/YY` no status/log.

```powershell
.venv\Scripts\python src/utils/stop_monitored.py
```

Se o processo nao responder:

```powershell
.venv\Scripts\python src/utils/stop_monitored.py --force
```

## Politica operacional

- Antes de qualquer run longa: commitar resultados/documentos pendentes.
- Durante a run: consultar `show_run_status.py` em vez de esperar o processo acabar.
- Se o usuario precisar do PC: parar depois da ultima epoca registrada.
- Depois da run: salvar resultados em `results/`, atualizar os MDs e commitar.
