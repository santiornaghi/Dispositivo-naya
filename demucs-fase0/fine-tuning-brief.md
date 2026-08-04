# Brief técnico: fine-tuning para separación por categoría real

Confirmado en Fase 0: el modelo pretrained `htdemucs` separa en 4 stems fijos
(`drums`/`bass`/`other`/`vocals`, entrenados sobre música popular), no en las categorías
del proyecto (cuerda / viento-madera / viento-metal / percusión / instrumento propio,
o SATB coral). Todo el contenido orquestal cae en `other`; todo el contenido coral cae
en `vocals`. Sirve como confirmación de calidad base del motor, pero no como producto.
Hace falta fine-tuning para llegar a las categorías reales.

## 1. Qué hay que reentrenar

Demucs (arquitectura `htdemucs`, tipo U-Net + transformer en el dominio tiempo-frecuencia)
soporta entrenar con un número arbitrario de stems/fuentes, no está atado a 4. La
alternativa más liviana en latencia es Conv-TasNet (dominio tiempo puro, sin STFT/iSTFT,
mejor punto de partida para <20-30ms si se llega a correr en el DSP en vivo más adelante).

Dos líneas de fine-tuning separadas (categorías distintas, no se pueden mezclar en un
mismo modelo sin curar las etiquetas):

- **Instrumental**: cuerda / viento-madera / viento-metal / percusión / "resto".
- **Coral**: soprano / alto / tenor / bajo (SATB).

## 2. Datasets — qué falta resolver antes de arrancar

| Dataset | Contenido | Acceso | Nota |
|---|---|---|---|
| MedleyDB | Multipistas etiquetadas por instrumento | Requiere solicitud de acceso a los autores (formulario), no es descarga directa | Buena cobertura de vientos y percusión orquestal |
| Slakh2100 | Mezclas sintéticas (renderizadas de MIDI con samples reales), separadas por instrumento | Descarga pública (~run con licencia CC BY 4.0) | Sintético → puede no generalizar bien a grabación real de sala/ensayo; sirve para pre-entrenar antes de afinar con datos reales |
| Choral Singing Dataset (Universitat Pompeu Fabra) | Grabaciones multipista de coro real, por cuerda SATB con mics cercanos | Descarga pública vía Zenodo | El más directamente aplicable al caso SATB |
| Bach Chorales (varios corpus, ej. JSB Chorales) | La mayoría son datasets simbólicos (MIDI/partitura), no audio multipista | Verificar caso por caso | Puede no servir para audio real sin generar el render |
| Barbershop Quartet dataset | Mencionado en el roadmap, no confirmado acceso | Pendiente de localizar fuente concreta | Queda como pendiente de investigación |

**Pendiente de decisión del founder**: solicitar acceso a MedleyDB (tiene proceso de
aprobación, no es inmediato) y confirmar la fuente exacta del "Barbershop Quartet
dataset" referenciado — no se encontró un repositorio público inequívoco en esta pasada.

## 3. Etiquetado / preparación de datos

- Slakh2100 y MedleyDB ya vienen con stems por instrumento — hay que mapear sus
  etiquetas de instrumento individual a las 5 categorías del proyecto (ej. "violin",
  "viola", "cello" → `cuerda`).
- Choral Singing Dataset ya viene separado por cuerda SATB — mapeo directo, sin trabajo
  de agrupación.
- Ningún dataset trae grabaciones tomadas con micrófono ambiental único simulando
  ensayo real (todos son estudio/multipista) — conviene simular la mezcla mono/estéreo
  a partir de los stems como paso de preprocesamiento, para que el modelo aprenda a
  partir de una señal parecida a lo que va a recibir en producción (un solo micrófono).

## 4. Cómputo

- Demucs recomienda GPU para entrenar (no solo para inferencia). Esta máquina tiene GPU
  AMD integrada sin soporte CUDA — no sirve para entrenar en tiempos razonables.
- Opciones: alquilar GPU en la nube (ej. instancia con NVIDIA T4/A10, por horas) o usar
  Google Colab/similar para una primera prueba de fine-tuning a escala reducida.
- No se estimó costo/tiempo todavía — depende del tamaño final del dataset curado.

## 5. Próximo paso concreto

1. Founder: gestionar acceso a MedleyDB y confirmar dataset de coro/barbershop.
2. Bajar y explorar Choral Singing Dataset (acceso público, sin trámite) como primer
   caso de prueba de fine-tuning SATB — es el más simple de arrancar ya.
3. Definir métrica de evaluación objetiva (ej. SDR/SI-SDR por categoría) para no
   depender solo de evaluación auditiva subjetiva en las siguientes iteraciones.
