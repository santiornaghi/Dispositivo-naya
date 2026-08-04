"""
CLI del prototipo offline: separación (ya hecha por Demucs en Fase 0) + limitador.

Uso:
    python procesar.py --input <carpeta con bass.mp3/drums.mp3/other.mp3/vocals.mp3> \
                        --config config_ejemplo.json \
                        --out salida.wav
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import soundfile as sf

from limiter import procesar_categoria, mezclar, aplicar_techo_db
from perfiles import PerfilManager


def cargar_stems(carpeta: Path, categorias: list[str]) -> dict[str, tuple[np.ndarray, int]]:
    stems = {}
    for cat in categorias:
        ruta = carpeta / f"{cat}.mp3"
        if not ruta.exists():
            ruta = carpeta / f"{cat}.wav"
        if not ruta.exists():
            raise FileNotFoundError(f"No encontré el stem '{cat}' en {carpeta}")
        audio, sr = sf.read(ruta)
        stems[cat] = (audio, sr)
    return stems


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input", required=True, type=Path,
                     help="Carpeta con los stems separados (salida de Demucs)")
    grupo_config = ap.add_mutually_exclusive_group(required=True)
    grupo_config.add_argument("--config", type=Path,
                               help="JSON con la config de ganancia/techo por categoría")
    grupo_config.add_argument("--perfil", type=str,
                               help="Nombre del perfil de sesión (Modo Volumen). "
                                    "Se crea si no existe, se auto-guarda con cada uso.")
    ap.add_argument("--perfiles-dir", type=Path, default=Path("perfiles_data"),
                     help="Carpeta donde viven los perfiles (default: ./perfiles_data)")
    ap.add_argument("--out", required=True, type=Path, help="Archivo .wav de salida")
    args = ap.parse_args()

    if args.perfil:
        stems_disponibles = sorted(
            p.stem for p in args.input.glob("*.mp3")
        ) or sorted(p.stem for p in args.input.glob("*.wav"))

        manager = PerfilManager(args.perfiles_dir)
        perfil, mensaje = manager.cargar(args.perfil)
        print(mensaje)

        for categoria in stems_disponibles:
            if manager.registrar_categoria_nueva(perfil, categoria):
                print(f"Categoría nueva '{categoria}' detectada — entra en 0dB (neutro) "
                      f"hasta que la ajustes.")

        config = perfil.a_config_limiter()
        config_master = None
    else:
        config = json.loads(args.config.read_text(encoding="utf-8"))
        config_master = config.pop("master", None)

    stems_crudos = cargar_stems(args.input, list(config.keys()))

    sr_ref = next(iter(stems_crudos.values()))[1]
    for cat, (_, sr) in stems_crudos.items():
        if sr != sr_ref:
            raise ValueError(f"Sample rate inconsistente en '{cat}': {sr} != {sr_ref}")

    stems_procesados = {}
    for cat, (audio, sr) in stems_crudos.items():
        cfg = config[cat]
        print(f"Procesando '{cat}': modo={cfg['modo']} "
              f"{cfg.get('valor_db', cfg.get('techo_db'))}dB")
        stems_procesados[cat] = procesar_categoria(audio, sr, cfg)

    mezcla = mezclar(stems_procesados)

    if config_master is not None:
        print(f"Aplicando límite maestro sobre el mix: techo={config_master['techo_db']}dB")
        mezcla = aplicar_techo_db(
            mezcla,
            sr_ref,
            techo_db=config_master["techo_db"],
            attack_ms=config_master.get("attack_ms", 3.0),
            release_ms=config_master.get("release_ms", 100.0),
        )
        pico_final = np.max(np.abs(mezcla))
        print(f"Pico final del mix tras límite maestro: {pico_final:.4f} "
              f"({20*np.log10(pico_final):+.2f} dBFS)")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    sf.write(args.out, mezcla, sr_ref)
    print(f"Listo: {args.out}")


if __name__ == "__main__":
    main()
