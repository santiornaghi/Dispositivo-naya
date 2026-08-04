"""
Prototipo offline del motor de regulación (Fase 2 del roadmap).

Toma stems ya separados (por ahora, los 4 stems genéricos de Demucs; el día que
exista el modelo fine-tuneado, esto se aplica igual sobre las categorías reales
cuerda/viento-madera/viento-metal/percusión o SATB — el limitador no sabe ni le
importa qué separó cada stem).

Dos modos por categoría, según el proyecto:
- Modo Volumen: ganancia relativa fija (mezcla para ensayo).
- Modo dB: techo absoluto — limitador con attack rápido / release lento, nunca deja
  pasar más nivel que el umbral, sin importar cuánto suba la fuente real.

IMPORTANTE: el umbral en Modo dB acá está en dBFS (relativo a la escala digital),
NO en dB SPL real. La conversión dBFS -> SPL real depende de la curva de calibración
por modelo de auricular (Fase 1 del roadmap, pendiente de medición física con
oído artificial). Hasta tener esa curva, este prototipo sirve para validar la
LÓGICA del limitador (attack/release, no bombeo, no overshoot), no niveles de
protección auditiva reales.
"""
from __future__ import annotations

import numpy as np


def db_to_lin(db: float) -> float:
    return 10 ** (db / 20)


def aplicar_ganancia(audio: np.ndarray, ganancia_db: float) -> np.ndarray:
    """Modo Volumen: ganancia relativa fija, sin limitar nada."""
    return audio * db_to_lin(ganancia_db)


def _minimo_hacia_adelante(valores: np.ndarray, ventana: int) -> np.ndarray:
    """Para cada i, el mínimo de valores[i : i+ventana] (mira hacia adelante)."""
    if ventana <= 1:
        return valores
    resultado = valores.copy()
    for desplazo in range(1, ventana):
        corrido = np.empty_like(valores)
        corrido[:-desplazo] = valores[desplazo:]
        corrido[-desplazo:] = valores[-1]
        np.minimum(resultado, corrido, out=resultado)
    return resultado


def aplicar_techo_db(
    audio: np.ndarray,
    sr: int,
    techo_db: float,
    attack_ms: float = 3.0,
    release_ms: float = 100.0,
) -> np.ndarray:
    """
    Modo dB: limitador feedforward con lookahead + attack/release asimétrico.

    - attack_ms (1-5ms según el proyecto): ventana de lookahead (cuánto "mira
      hacia adelante" antes de que llegue el pico). Como ya sabemos con
      anticipación cuánto hay que bajar, la BAJADA de ganancia es inmediata
      (no exponencial) en cuanto el lookahead detecta un pico próximo — un
      suavizado exponencial puro en la bajada asienta ~5x más lento que su
      "constante de tiempo" nominal, y dejaría pasar transitorios breves
      (platillos/metales) casi sin atenuar. La SUBIDA de vuelta a ganancia
      unidad sí es exponencial (ver release_ms), porque ahí sí conviene lenta.
    - release_ms (50-200ms según el proyecto): qué tan rápido vuelve a ganancia
      unidad después de que el nivel baja (lento, para evitar sonido "bombeado").

    En un sistema en vivo (streaming) el lookahead exige guardar un buffer de
    `attack_ms` de audio futuro antes de poder emitir cada muestra, lo que
    suma esa misma latencia al presupuesto de <20-30ms de la Fase 3. Acá,
    como procesamos el archivo completo de una, no hace falta modelar ese
    delay explícitamente — el efecto de "mirar hacia adelante" ya está
    resuelto en `ganancia_objetivo` al tomar el mínimo sobre la ventana
    futura, y se aplica sobre la muestra original en el mismo índice.

    Trabaja en mono por canal; si el audio es estéreo, aplica el mismo cálculo
    de ganancia a ambos canales usando el máximo entre canales (evita que el
    limitador se desincronice entre L/R y mueva la imagen estéreo).
    """
    mono_ref = np.max(np.abs(audio), axis=-1) if audio.ndim > 1 else np.abs(audio)
    techo_lin = db_to_lin(techo_db)

    ventana_lookahead = max(1, round(sr * attack_ms / 1000.0))
    release_coef = np.exp(-1.0 / (sr * release_ms / 1000.0))

    ganancia_objetivo = np.minimum(1.0, techo_lin / np.maximum(mono_ref, 1e-9))
    ganancia_objetivo = _minimo_hacia_adelante(ganancia_objetivo, ventana_lookahead)

    ganancia_suavizada = np.empty_like(ganancia_objetivo)
    actual = 1.0
    for i in range(len(ganancia_objetivo)):
        objetivo = ganancia_objetivo[i]
        if objetivo < actual:
            actual = objetivo  # bajada inmediata: el lookahead ya avisó
        else:
            actual = release_coef * actual + (1 - release_coef) * objetivo
        ganancia_suavizada[i] = actual

    if audio.ndim > 1:
        return audio * ganancia_suavizada[:, None]
    return audio * ganancia_suavizada


def procesar_categoria(audio: np.ndarray, sr: int, config: dict) -> np.ndarray:
    modo = config["modo"]
    if modo == "volumen":
        return aplicar_ganancia(audio, config["valor_db"])
    if modo == "db":
        return aplicar_techo_db(
            audio,
            sr,
            techo_db=config["techo_db"],
            attack_ms=config.get("attack_ms", 3.0),
            release_ms=config.get("release_ms", 100.0),
        )
    raise ValueError(f"Modo desconocido: {modo!r} (esperado 'volumen' o 'db')")


def mezclar(stems_procesados: dict[str, np.ndarray]) -> np.ndarray:
    """
    Suma todos los stems ya procesados. Avisa si el resultado clippea.

    El techo de Modo dB aplicado en `procesar_categoria` es por categoría, no
    del mix total: si varias categorías están cerca de su techo al mismo
    tiempo, la suma puede superar cualquiera de esos techos individuales
    (es el comportamiento esperado — cada categoría se protege por separado).
    Si además hace falta un límite sobre el mix ya sumado, aplicar
    `aplicar_techo_db` de nuevo sobre el resultado de esta función (ver la
    clave opcional "master" en procesar.py).
    """
    largo = max(a.shape[0] for a in stems_procesados.values())
    mezcla = None
    for audio in stems_procesados.values():
        if audio.shape[0] < largo:
            pad = largo - audio.shape[0]
            audio = np.pad(audio, [(0, pad)] + [(0, 0)] * (audio.ndim - 1))
        mezcla = audio if mezcla is None else mezcla + audio

    pico = np.max(np.abs(mezcla))
    if pico > 1.0:
        print(f"[aviso] la mezcla final clippea: pico={pico:.3f} (>1.0). "
              f"Bajá la ganancia de alguna categoría o agregá un limitador maestro.")
    return mezcla
