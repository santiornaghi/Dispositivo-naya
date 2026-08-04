"""
Modelo liviano de separación SATB por máscara espectral.

No es el HTDemucs completo (ese está pensado para entrenarse en GPU; en esta
máquina no hay CUDA). Es un modelo más chico, en la misma familia de idea que
Conv-TasNet (que ya figura en el roadmap como alternativa liviana a Demucs):
una CNN chica predice, para cada una de las 4 voces (soprano/alto/tenor/bajo),
una máscara sobre el espectrograma de la mezcla — no reconstruye la señal
desde cero, aprende "qué parte del espectro de la mezcla le corresponde a
cada voz".

Entra: espectrograma (magnitud, log) de la mezcla, mono (un solo micrófono,
como especifica el proyecto para v1).
Sale: 4 máscaras en [0,1], una por voz, mismo tamaño que el espectrograma.
"""
from __future__ import annotations

import torch
import torch.nn as nn

VOCES = ["soprano", "alto", "tenor", "bajo"]
N_FFT = 1024
HOP = 256


def espectrograma(audio: torch.Tensor) -> torch.Tensor:
    """audio: (batch, muestras) -> complejo (batch, freq, tiempo)"""
    ventana = torch.hann_window(N_FFT, device=audio.device)
    return torch.stft(audio, n_fft=N_FFT, hop_length=HOP, window=ventana, return_complex=True)


def audio_desde_espectrograma(spec: torch.Tensor, largo: int) -> torch.Tensor:
    ventana = torch.hann_window(N_FFT, device=spec.device)
    return torch.istft(spec, n_fft=N_FFT, hop_length=HOP, window=ventana, length=largo)


class BloqueConv(nn.Module):
    def __init__(self, canales_in: int, canales_out: int):
        super().__init__()
        self.red = nn.Sequential(
            nn.Conv2d(canales_in, canales_out, kernel_size=3, padding=1),
            nn.BatchNorm2d(canales_out),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.red(x)


class SeparadorSATB(nn.Module):
    """
    U-Net chico sobre el espectrograma. Encoder de 3 niveles, decoder
    simétrico con skip connections, cabeza final con 4 máscaras (sigmoid).
    Deliberadamente chico (pocos canales) para que entrenar en CPU sea viable.
    """

    def __init__(self, canales_base: int = 16):
        super().__init__()
        c = canales_base
        self.enc1 = BloqueConv(1, c)
        self.enc2 = BloqueConv(c, c * 2)
        self.enc3 = BloqueConv(c * 2, c * 4)
        self.pool = nn.MaxPool2d(2, ceil_mode=True)

        self.cuello = BloqueConv(c * 4, c * 4)

        self.up3 = nn.Upsample(scale_factor=2, mode="nearest")
        self.dec3 = BloqueConv(c * 4 + c * 4, c * 2)
        self.up2 = nn.Upsample(scale_factor=2, mode="nearest")
        self.dec2 = BloqueConv(c * 2 + c * 2, c)
        self.up1 = nn.Upsample(scale_factor=2, mode="nearest")
        self.dec1 = BloqueConv(c + c, c)

        self.cabeza = nn.Conv2d(c, len(VOCES), kernel_size=1)

    def forward(self, mag_log: torch.Tensor) -> torch.Tensor:
        """mag_log: (batch, 1, freq, tiempo) -> (batch, 4, freq, tiempo) mascaras en [0,1]"""
        e1 = self.enc1(mag_log)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))

        m = self.cuello(self.pool(e3))

        d3 = self._combinar(self.up3(m), e3)
        d3 = self.dec3(d3)
        d2 = self._combinar(self.up2(d3), e2)
        d2 = self.dec2(d2)
        d1 = self._combinar(self.up1(d2), e1)
        d1 = self.dec1(d1)

        return torch.sigmoid(self.cabeza(d1))

    @staticmethod
    def _combinar(arriba: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        # el upsample con tamaños impares puede quedar 1px desalineado, recortamos
        h = min(arriba.shape[-2], skip.shape[-2])
        w = min(arriba.shape[-1], skip.shape[-1])
        return torch.cat([arriba[..., :h, :w], skip[..., :h, :w]], dim=1)
