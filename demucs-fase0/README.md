# Fase 0 — Validación de separación con Demucs

Primer paso técnico del roadmap (sin gastar en hardware): correr Demucs sobre grabaciones
reales de ensayo/orquesta y de coro, y evaluar la calidad de separación por categoría.

## Setup (ya hecho)

- Python 3.11 instalado.
- Entorno virtual en `.venv/` con `demucs`, `torch` (CPU, no hay GPU NVIDIA en esta
  máquina — corre más lento que con CUDA pero funciona para evaluar calidad offline).

## Estructura

```
demucs-fase0/
  input_instrumental/   <- poné acá 3-5 grabaciones de ensamble instrumental
  input_coral/           <- poné acá 3-5 grabaciones de coro
  output/                 <- ahí caen los stems separados (se genera solo)
  run_demucs.ps1          <- script para correr la separación
  evaluacion.csv           <- tabla para completar a mano tras escuchar los resultados
```

## Cómo correr

Desde PowerShell, parado en esta carpeta:

```powershell
.\run_demucs.ps1 -InputDir .\input_instrumental
.\run_demucs.ps1 -InputDir .\input_coral
```

Esto separa cada track en 4 stems genéricos del modelo `htdemucs`: `drums`, `bass`,
`other`, `vocals`. Quedan en `output\htdemucs\<nombre_del_track>\`.

**Importante:** estos 4 stems NO son las categorías finales del proyecto (cuerda /
viento-madera / viento-metal / percusión, o SATB). Para Fase 0 el objetivo es evaluar
qué tan limpia es la separación *en general* (fugas, artefactos) como punto de partida
para el fine-tuning posterior — no juzgar el mapeo de categorías todavía.

Para grabaciones corales, probablemente lo más útil sea mirar el stem `vocals` (separa
voz del resto) — separar dentro de las voces por cuerda (SATB) va a requerir el
fine-tuning mencionado en el roadmap, htdemucs de fábrica no distingue eso.

## Datasets de referencia (si hace falta más material de prueba)

- Instrumental: MedleyDB, Slakh2100.
- Coral / SATB: Choral Singing Dataset, Bach Chorals, Barbershop Quartet dataset.

## Evaluación

Después de escuchar cada resultado, completar una fila en `evaluacion.csv`:

| columna | qué poner |
|---|---|
| audio | nombre del archivo original |
| tipo_de_ensamble | instrumental / coral, y detalle (ej. "orquesta cuerdas", "coro SATB") |
| calidad_1_5 | 1 (mala) a 5 (excelente) separación percibida |
| fugas_entre_categorias | qué se filtra de una categoría a otra (ej. "viento-metal se cuela en other") |
| artefactos | ruidos/distorsión introducidos por la separación |
| comentario | libre |

## Estado (actualizado)

Ya hay 6 grabaciones de prueba separadas en `output/htdemucs/`:

- Instrumental (4): Bach Brandenburg No.3 (cuerdas), Vivaldi Four Seasons - Spring
  (cuerdas + solista), Ravel Bolero (orquesta completa, única con percusión real),
  J. Strauss II Blue Danube (orquesta completa).
- Coral (2): Rowan University Chamber Choir (coro + orquesta, no a cappella), Stamford
  High School A Cappella Choir - Glory to God (SATB a cappella, grabación histórica
  ~1920s de 78rpm, tiene ruido de superficie original de la fuente).

`evaluacion.csv` ya tiene las 6 filas con mediciones objetivas (nivel RMS por stem,
clipping detectado) precargadas — falta la columna `calidad_1_5`, que requiere
escuchar cada resultado.

## Próximo paso

1. Escuchar los 6 resultados y completar `calidad_1_5` en `evaluacion.csv`.
2. Con la tabla completa, decidir si htdemucs de fábrica alcanza como base o si hace
   falta arrancar el fine-tuning (Demucs / Conv-TasNet) mencionado en el roadmap — ver
   `fine-tuning-brief.md` para el plan técnico de esa siguiente fase (datasets,
   cómputo, próximos pasos concretos).
