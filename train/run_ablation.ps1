# Скрипт последовательного запуска экспериментов A/B/C/D
$env:WANDB_MODE="offline"
$GlobalTimer = [System.Diagnostics.Stopwatch]::StartNew()

function Run-Experiment {
    param (
        [string]$Name,
        [string]$Command
    )
    $startTime = Get-Date
    Write-Host "`n" + ("=" * 60) -ForegroundColor Cyan
    Write-Host "ЗАПУСК ЭКСПЕРИМЕНТА: [$Name]" -ForegroundColor Cyan
    Write-Host "Время начала: $($startTime.ToString('HH:mm:ss'))" -ForegroundColor Gray
    Write-Host ("-" * 60) -ForegroundColor DarkGray
    
    # Запуск самой команды
    Invoke-Expression $Command
    
    $endTime = Get-Date
    $duration = $endTime - $startTime
    Write-Host ("-" * 60) -ForegroundColor DarkGray
    Write-Host "ЗАВЕРШЕНО: [$Name]" -ForegroundColor Green
    Write-Host "Время окончания: $($endTime.ToString('HH:mm:ss'))" -ForegroundColor Gray
    Write-Host "Длительность этапа: $($duration.ToString('hh\:mm\:ss'))" -ForegroundColor Yellow
}

# --- СПИСОК ЭКСПЕРИМЕНТОВ ---

$CommonArgs = "--use-transformer --use-attention --amp --detailed-eval --epochs 50 --batch-size 1 --scale 0.1 --classes 9 --num-workers 8 --prefetch-factor 4 --persistent-workers"

# A) Baseline
Run-Experiment -Name "A_Baseline" `
    -Command "python train.py $CommonArgs --no-class-weights --no-rare-oversampling --results-dir results_ablation_A_baseline"

# B) Только class weights
Run-Experiment -Name "B_Weights" `
    -Command "python train.py $CommonArgs --use-class-weights --no-rare-oversampling --results-dir results_ablation_B_weights"

# C) Только oversampling
Run-Experiment -Name "C_Sampler" `
    -Command "python train.py $CommonArgs --no-class-weights --use-rare-oversampling --results-dir results_ablation_C_sampler"

# D) Weights + oversampling
Run-Experiment -Name "D_Both" `
    -Command "python train.py $CommonArgs --use-class-weights --use-rare-oversampling --results-dir results_ablation_D_both"

# --- ФИНАЛ ---
$GlobalTimer.Stop()
$totalTime = $GlobalTimer.Elapsed
Write-Host "`n" + ("=" * 60) -ForegroundColor Magenta
Write-Host "ВСЕ ЭКСПЕРИМЕНТЫ ВЫПОЛНЕНЫ!" -ForegroundColor Magenta
Write-Host "Общее затраченное время: $($totalTime.ToString('hh\:mm\:ss'))" -ForegroundColor Magenta
Write-Host ("=" * 60) -ForegroundColor Magenta
