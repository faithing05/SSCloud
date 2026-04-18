param(
    [string]$ResultsDir = "results_quality_baseline",
    [string]$InputDir = "data/imgs",
    [string]$OutputDir = "data/test/output_local",
    [double]$Scale = 0.1,
    [int]$Classes = 9,
    [double]$MaskThreshold = 0.5,
    [int]$ChunkSize = 20,
    [Nullable[int]]$Epoch = $null,
    [switch]$NoAttentionMaps,
    [switch]$NoTransformer,
    [switch]$NoAttention,
    [switch]$NoColorized,
    [string]$ColorizedDir = "",
    [string]$ClassNamesPath = ""
)

$ErrorActionPreference = "Stop"

function Resolve-FullPath {
    param([string]$PathValue)
    if ([System.IO.Path]::IsPathRooted($PathValue)) {
        return $PathValue
    }
    return (Join-Path (Get-Location) $PathValue)
}

$scriptRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$resultsDirFull = Resolve-FullPath $ResultsDir
$inputDirFull = Resolve-FullPath $InputDir
$outputDirFull = Resolve-FullPath $OutputDir
$checkpointsDir = Join-Path $resultsDirFull "checkpoints"

if (-not (Test-Path -Path $checkpointsDir -PathType Container)) {
    throw "Checkpoints directory not found: $checkpointsDir"
}

if (-not (Test-Path -Path $inputDirFull -PathType Container)) {
    throw "Input directory not found: $inputDirFull"
}

if ($ChunkSize -lt 1) {
    throw "ChunkSize must be >= 1"
}

if (-not $Epoch.HasValue) {
    $latestCheckpoint = Get-ChildItem -Path $checkpointsDir -Filter "checkpoint_epoch*.pth" -File |
        Where-Object { $_.BaseName -match '^checkpoint_epoch(\d+)$' } |
        Sort-Object { [int]([regex]::Match($_.BaseName, '^checkpoint_epoch(\d+)$')).Groups[1].Value } |
        Select-Object -Last 1

    if (-not $latestCheckpoint) {
        throw "No checkpoint files found in: $checkpointsDir"
    }

    $Epoch = [int]([regex]::Match($latestCheckpoint.BaseName, '^checkpoint_epoch(\d+)$')).Groups[1].Value
    Write-Host "Auto-selected latest epoch checkpoint: $Epoch"
}

$checkpointPath = Join-Path $checkpointsDir ("checkpoint_epoch{0}.pth" -f $Epoch)
if (-not (Test-Path -Path $checkpointPath -PathType Leaf)) {
    throw "Checkpoint file not found: $checkpointPath"
}

$classNamesFull = $null
if (-not [string]::IsNullOrWhiteSpace($ClassNamesPath)) {
    $classNamesFull = Resolve-FullPath $ClassNamesPath
    if (-not (Test-Path -Path $classNamesFull -PathType Leaf)) {
        throw "Class names file not found: $classNamesFull"
    }
} else {
    $defaultClassNames = Join-Path $scriptRoot "class_names.json"
    if (Test-Path -Path $defaultClassNames -PathType Leaf) {
        $classNamesFull = $defaultClassNames
    }
}

New-Item -ItemType Directory -Force -Path $outputDirFull | Out-Null

$patterns = @("*.png", "*.jpg", "*.jpeg", "*.bmp", "*.tif", "*.tiff")
$images = foreach ($pattern in $patterns) {
    Get-ChildItem -Path $inputDirFull -Filter $pattern -File
}
$images = $images | Sort-Object Name

if (-not $images -or $images.Count -eq 0) {
    throw "No input images found in: $inputDirFull"
}

Write-Host "Found $($images.Count) images. Starting batch inference (chunk size: $ChunkSize)..."

Push-Location $scriptRoot
try {
    $totalBatches = [math]::Ceiling($images.Count / [double]$ChunkSize)
    for ($start = 0; $start -lt $images.Count; $start += $ChunkSize) {
        $end = [Math]::Min($start + $ChunkSize - 1, $images.Count - 1)
        $currentBatch = @($images[$start..$end])
        $batchIndex = [int]($start / $ChunkSize) + 1

        $inputFiles = @()
        $outputFiles = @()
        foreach ($image in $currentBatch) {
            $inputFiles += $image.FullName
            $outputFiles += (Join-Path $outputDirFull ("{0}_mask.png" -f $image.BaseName))
        }

        $predictArgs = @(
            "predict.py",
            "--results-dir", $resultsDirFull,
            "--epoch", $Epoch,
            "--scale", $Scale,
            "--classes", $Classes,
            "--mask-threshold", $MaskThreshold,
            "-i"
        )
        $predictArgs += $inputFiles
        $predictArgs += "-o"
        $predictArgs += $outputFiles

        if ($NoTransformer) {
            $predictArgs += "--no-transformer"
        } else {
            $predictArgs += "--use-transformer"
        }

        if ($NoAttention) {
            $predictArgs += "--no-attention"
        } else {
            $predictArgs += "--use-attention"
        }

        if (-not $NoAttentionMaps) {
            $predictArgs += "--save-attention"
        }

        Write-Host "Processing batch $batchIndex/$totalBatches ($($currentBatch.Count) images)..."
        & python @predictArgs
        if ($LASTEXITCODE -ne 0) {
            throw "predict.py failed for batch $batchIndex/$totalBatches (exit code: $LASTEXITCODE)"
        }
    }

    if (-not $NoColorized) {
        $colorizedDirPath = $ColorizedDir
        if ([string]::IsNullOrWhiteSpace($colorizedDirPath)) {
            $colorizedDirPath = Join-Path $outputDirFull "colorized"
        }
        $colorizedDirFull = Resolve-FullPath $colorizedDirPath

        $colorizeArgs = @(
            "colorize_masks_with_legend.py",
            "--mask-dir", $outputDirFull,
            "--output-dir", $colorizedDirFull,
            "--classes", $Classes,
            "--checkpoint", $checkpointPath
        )
        if ($classNamesFull) {
            $colorizeArgs += "--class-names"
            $colorizeArgs += $classNamesFull
        }
        & python @colorizeArgs
        if ($LASTEXITCODE -ne 0) {
            throw "colorize_masks_with_legend.py failed (exit code: $LASTEXITCODE)"
        }

        Write-Host "Colorized masks and legend saved to: $colorizedDirFull"
    }
}
finally {
    Pop-Location
}

Write-Host "Batch inference finished. Outputs saved to: $outputDirFull"
