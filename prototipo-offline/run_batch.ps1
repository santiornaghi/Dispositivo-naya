# Corre procesar.py sobre todos los tracks ya separados en Fase 0, produciendo
# una mezcla final (separacion + limitador) por track para escuchar de punta a punta.

$py = "$PSScriptRoot\..\demucs-fase0\.venv\Scripts\python.exe"
$stemsBase = "$PSScriptRoot\..\demucs-fase0\output\htdemucs"
$salidas = "$PSScriptRoot\salidas"

New-Item -ItemType Directory -Force -Path $salidas | Out-Null

$instrumental = @(
    "1721_Bach_Brandenburg_Concerto_No3",
    "1725_Vivaldi_Four_Seasons_Spring",
    "1928_Ravel_Bolero",
    "1867_Strauss_Blue_Danube_Waltz"
)
$coral = @(
    "Rowan_University_Chamber_Choir_Song_of_Democracy",
    "Stamford_Glory_to_God_a_cappella"
)

foreach ($track in $instrumental) {
    Write-Host "=== $track (config instrumental) ==="
    & $py procesar.py --input "$stemsBase\$track" --config config_ejemplo.json --out "$salidas\$track.wav"
}

foreach ($track in $coral) {
    Write-Host "=== $track (config coral) ==="
    & $py procesar.py --input "$stemsBase\$track" --config config_coral.json --out "$salidas\$track.wav"
}

Write-Host "Listo. Salidas en $salidas"
