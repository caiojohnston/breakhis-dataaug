# Resume pipeline: pula LDM (ja treinado), roda generate.py + Cenario C
# Uso: .\resume_pipeline.ps1
# Prereq: checkpoints\ldm\best.pt deve existir

$env:PYTHONUTF8 = "1"
$logFile = "resume_run_$(Get-Date -Format 'yyyyMMdd_HHmmss').log"

"=== INICIO: $(Get-Date) ===" | Out-File $logFile -Encoding utf8

# Verifica best.pt
if (-not (Test-Path "checkpoints\ldm\best.pt")) {
    "ERRO: checkpoints\ldm\best.pt nao encontrado. Rode train_ldm.py primeiro." | Out-File $logFile -Append -Encoding utf8
    Write-Host "ERRO: best.pt nao encontrado."
    exit 1
}

Write-Host "best.pt encontrado. Iniciando geracao de sinteticas..."
"=== Fase 4.3 - Geracao de sinteticas: $(Get-Date) ===" | Out-File $logFile -Append -Encoding utf8

$genOut = & .venv\Scripts\python src/generation/generate.py --config configs/ldm.yaml 2>&1
$gen_exit = $LASTEXITCODE
$genOut | Out-File $logFile -Append -Encoding utf8

if ($gen_exit -eq 0) {
    $gen_ok = $true
    Write-Host "Geracao concluida."
} else {
    $gen_ok = $false
    "ERRO: generate.py saiu com codigo $gen_exit. Abortando." | Out-File $logFile -Append -Encoding utf8
    Write-Host "ERRO: generate.py falhou (exit=$gen_exit)."
}

if ($gen_ok) {
    Write-Host "Iniciando Cenario C..."
    "=== Fase 6 - Cenario C: $(Get-Date) ===" | Out-File $logFile -Append -Encoding utf8

    $clsOut = & .venv\Scripts\python src/training/train_classifier.py --scenario C --config configs/classifier.yaml 2>&1
    $cls_exit = $LASTEXITCODE
    $clsOut | Out-File $logFile -Append -Encoding utf8

    if ($cls_exit -eq 0) {
        $cls_ok = $true
        Write-Host "Cenario C concluido."
    } else {
        $cls_ok = $false
        "ERRO: train_classifier.py saiu com codigo $cls_exit." | Out-File $logFile -Append -Encoding utf8
        Write-Host "ERRO: Cenario C falhou (exit=$cls_exit)."
    }
} else {
    $cls_ok = $false
}

"=== FIM: $(Get-Date) | GEN=$gen_ok CLS=$cls_ok ===" | Out-File $logFile -Append -Encoding utf8
Write-Host "Pipeline finalizado. Log: $logFile"
