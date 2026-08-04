# Corre Demucs (modelo htdemucs, 4 stems: drums/bass/other/vocals) sobre todos los
# audios de una carpeta y deja el resultado en output/<nombre_track>/
#
# Uso:
#   .\run_demucs.ps1 -InputDir .\input_instrumental
#   .\run_demucs.ps1 -InputDir .\input_coral
#
# Nota: htdemucs separa en 4 stems genéricos (drums/bass/other/vocals), no en las
# categorías finales del proyecto (cuerda/viento-madera/viento-metal/percusion, o SATB).
# Para Fase 0 el objetivo es evaluar CALIDAD de separación cruda, no el mapeo final
# de categorías (eso viene en fases posteriores con fine-tuning).

param(
    [Parameter(Mandatory = $true)]
    [string]$InputDir,

    [string]$OutDir = "$PSScriptRoot\output",

    [string]$Model = "htdemucs"
)

$py = "$PSScriptRoot\.venv\Scripts\python.exe"

if (-not (Test-Path $py)) {
    Write-Error "No se encuentra el entorno virtual en $py. Corré primero la instalacion."
    exit 1
}

if (-not (Test-Path $InputDir)) {
    Write-Error "No existe la carpeta de entrada: $InputDir"
    exit 1
}

$tracks = Get-ChildItem -Path $InputDir -File | Where-Object {
    $_.Extension -in ".wav", ".mp3", ".flac", ".m4a", ".aiff"
}

if ($tracks.Count -eq 0) {
    Write-Warning "No se encontraron archivos de audio en $InputDir (.wav/.mp3/.flac/.m4a/.aiff)"
    exit 0
}

Write-Host "Separando $($tracks.Count) archivo(s) con modelo '$Model' -> $OutDir"

& $py -m demucs -n $Model -o $OutDir --mp3 $tracks.FullName
