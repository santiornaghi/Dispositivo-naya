"""
Fine-tuning del SeparadorSATB sobre el Choral Singing Dataset (Cuesta et al.).

Cada pieza del dataset trae las 16 pistas individuales (4 cantantes por voz:
soprano/alto/tenor/bajo). Para entrenar:
- mezcla = suma de las 16 pistas (simula el micrófono ambiental único de v1)
- objetivo por voz = suma de las 4 pistas de esa voz

Se recorta cada pieza en fragmentos cortos (SEGMENTO_SEG) para tener más
ejemplos de entrenamiento con pocas piezas, y para que quepa en memoria en CPU.
"""
from __future__ import annotations

import argparse
import re
import time
from pathlib import Path

import soundfile as sf
import torch
import torch.nn.functional as F

from model import SeparadorSATB, VOCES, espectrograma, N_FFT, HOP

SR = 44100
SEGMENTO_SEG = 6.0
SEGMENTO_MUESTRAS = int(SR * SEGMENTO_SEG)

# nombres de archivo típicos: algo con S/A/T/B (mayúscula, seguida de un
# número de cantante) en algún punto del nombre, ej "..._S1.wav", "..._A2.wav"
PATRON_VOZ = re.compile(r"(?:^|[_\-])([SATB])\d*(?:[_\-.]|$)", re.IGNORECASE)
MAPA_VOZ = {"S": "soprano", "A": "alto", "T": "tenor", "B": "bajo"}


def detectar_voz(nombre_archivo: str) -> str | None:
    m = PATRON_VOZ.search(nombre_archivo)
    if not m:
        return None
    return MAPA_VOZ[m.group(1).upper()]


def cargar_pieza(carpeta: Path) -> dict[str, torch.Tensor] | None:
    """Devuelve {'soprano': tensor, 'alto': ..., ...} sumando las pistas de cada voz."""
    archivos = list(carpeta.glob("*.wav")) + list(carpeta.glob("*.WAV"))
    por_voz: dict[str, list[torch.Tensor]] = {v: [] for v in VOCES}
    for archivo in archivos:
        voz = detectar_voz(archivo.stem)
        if voz is None:
            continue
        audio, sr = sf.read(archivo, dtype="float32")
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        if sr != SR:
            raise ValueError(f"sample rate inesperado en {archivo}: {sr} (esperaba {SR})")
        por_voz[voz].append(torch.from_numpy(audio))

    if any(len(v) == 0 for v in por_voz.values()):
        faltantes = [v for v, lst in por_voz.items() if not lst]
        print(f"  [aviso] {carpeta.name}: sin pistas detectadas para {faltantes}, salteo esta pieza")
        return None

    largo = min(min(t.shape[0] for t in lst) for lst in por_voz.values())
    sumas = {voz: torch.stack([t[:largo] for t in lst]).sum(dim=0) for voz, lst in por_voz.items()}
    return sumas


def segmentar(sumas_por_voz: dict[str, torch.Tensor]) -> list[dict[str, torch.Tensor]]:
    largo = next(iter(sumas_por_voz.values())).shape[0]
    n_segmentos = largo // SEGMENTO_MUESTRAS
    segmentos = []
    for i in range(n_segmentos):
        ini, fin = i * SEGMENTO_MUESTRAS, (i + 1) * SEGMENTO_MUESTRAS
        segmentos.append({voz: audio[ini:fin] for voz, audio in sumas_por_voz.items()})
    return segmentos


def preparar_dataset(dataset_dir: Path) -> list[dict[str, torch.Tensor]]:
    piezas = [p for p in dataset_dir.iterdir() if p.is_dir()]
    print(f"Piezas encontradas: {[p.name for p in piezas]}")
    segmentos = []
    for pieza in piezas:
        sumas = cargar_pieza(pieza)
        if sumas is None:
            continue
        nuevos = segmentar(sumas)
        print(f"  {pieza.name}: {nuevos} segmentos de {SEGMENTO_SEG}s" if False else
              f"  {pieza.name}: {len(nuevos)} segmentos de {SEGMENTO_SEG}s")
        segmentos.extend(nuevos)
    return segmentos


def perdida(mascaras: torch.Tensor, spec_mezcla: torch.Tensor, objetivos_mag: torch.Tensor) -> torch.Tensor:
    """L1 entre |mezcla|*mascara (estimación) y |objetivo| real, sumado en las 4 voces."""
    mag_mezcla = spec_mezcla.abs().unsqueeze(1)  # (batch, 1, freq, tiempo)
    estimado = mascaras * mag_mezcla  # (batch, 4, freq, tiempo)
    return F.l1_loss(estimado, objetivos_mag)


def entrenar(dataset_dir: Path, epocas: int, lr: float, checkpoint_out: Path, batch_size: int = 4):
    segmentos = preparar_dataset(dataset_dir)
    if not segmentos:
        raise RuntimeError("no se armó ningún segmento de entrenamiento — revisar detección de voces")
    print(f"Total segmentos de entrenamiento: {len(segmentos)}")

    modelo = SeparadorSATB(canales_base=16)
    opt = torch.optim.Adam(modelo.parameters(), lr=lr)

    for epoca in range(1, epocas + 1):
        t0 = time.time()
        perm = torch.randperm(len(segmentos))
        perdida_epoca = 0.0
        n_batches = 0

        for i in range(0, len(perm), batch_size):
            idx = perm[i:i + batch_size]
            lote = [segmentos[j] for j in idx]

            mezcla = torch.stack([sum(seg[v] for v in VOCES) for seg in lote])
            spec_mezcla = espectrograma(mezcla)
            mag_log = torch.log1p(spec_mezcla.abs()).unsqueeze(1)

            objetivos_mag = torch.stack([
                torch.stack([espectrograma(seg[v].unsqueeze(0)).abs().squeeze(0) for v in VOCES])
                for seg in lote
            ])

            mascaras = modelo(mag_log)
            # recortar a tamaño común por si el u-net redondeó distinto
            h = min(mascaras.shape[-2], objetivos_mag.shape[-2], spec_mezcla.shape[-2])
            w = min(mascaras.shape[-1], objetivos_mag.shape[-1], spec_mezcla.shape[-1])

            loss = perdida(
                mascaras[..., :h, :w],
                spec_mezcla[..., :h, :w],
                objetivos_mag[..., :h, :w],
            )

            opt.zero_grad()
            loss.backward()
            opt.step()

            perdida_epoca += loss.item()
            n_batches += 1

        print(f"época {epoca}/{epocas}: loss={perdida_epoca/n_batches:.4f} "
              f"({time.time()-t0:.1f}s, {n_batches} batches)")

    checkpoint_out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(modelo.state_dict(), checkpoint_out)
    print(f"Checkpoint guardado en {checkpoint_out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", type=Path, required=True)
    ap.add_argument("--epocas", type=int, default=5)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--out", type=Path, default=Path("modelo_entrenado/satb.pt"))
    ap.add_argument("--batch-size", type=int, default=4)
    args = ap.parse_args()

    entrenar(args.dataset, args.epocas, args.lr, args.out, args.batch_size)
