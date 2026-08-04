"""
Aplica un checkpoint de SeparadorSATB a un audio nuevo y escribe 4 wav
(soprano/alto/tenor/bajo.wav) — el equivalente, para coro, a lo que hace
`demucs` con los 4 stems genéricos.

Uso:
    python infer.py --checkpoint modelo_entrenado/satb.pt --audio entrada.wav --out salida_dir
"""
from __future__ import annotations

import argparse
from pathlib import Path

import soundfile as sf
import torch

from model import SeparadorSATB, VOCES, espectrograma, audio_desde_espectrograma, SR


def separar(checkpoint: Path, audio_path: Path, out_dir: Path):
    audio, sr = sf.read(audio_path, dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if sr != SR:
        raise ValueError(f"este modelo espera {SR}Hz, el archivo tiene {sr}Hz "
                          f"(resamplear antes de separar)")

    modelo = SeparadorSATB(canales_base=16)
    modelo.load_state_dict(torch.load(checkpoint, map_location="cpu"))
    modelo.eval()

    audio_t = torch.from_numpy(audio).unsqueeze(0)  # (1, muestras)
    with torch.no_grad():
        spec = espectrograma(audio_t)
        mag_log = torch.log1p(spec.abs()).unsqueeze(1)
        mascaras = modelo(mag_log)  # (1, 4, freq, tiempo)

        h = min(mascaras.shape[-2], spec.shape[-2])
        w = min(mascaras.shape[-1], spec.shape[-1])
        spec = spec[..., :h, :w]
        mascaras = mascaras[..., :h, :w]

        out_dir.mkdir(parents=True, exist_ok=True)
        for i, voz in enumerate(VOCES):
            spec_voz = spec * mascaras[:, i]
            audio_voz = audio_desde_espectrograma(spec_voz, largo=audio_t.shape[-1])
            ruta = out_dir / f"{voz}.wav"
            sf.write(ruta, audio_voz.squeeze(0).numpy(), SR)
            print(f"  {voz}.wav escrito")

    print(f"Listo: {out_dir}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--audio", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    separar(args.checkpoint, args.audio, args.out)
