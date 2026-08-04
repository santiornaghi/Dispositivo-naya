"""
Motor de perfiles de sesión para Modo Volumen.

Un perfil = nombre libre (coro, lugar de concierto, grupo de ensayo) + ganancia
en dB por categoría. Reglas del proyecto:

- Se actualiza automáticamente con cada cambio del usuario — sin botón de
  "guardar" explícito (ver `PerfilManager.actualizar_categoria`).
- Al abrir un perfil ya usado, se cargan y aplican solos los últimos ajustes
  conocidos por categoría (ver `PerfilManager.cargar`).
- Categorías nuevas detectadas a mitad de sesión entran con valor neutro (0dB)
  hasta que el usuario las ajuste (ver `PerfilManager.registrar_categoria_nueva`).
- Red de seguridad: "Restaurar ajuste anterior" disponible tras cambios grandes
  (ver `PerfilManager.restaurar_anterior` y `Perfil.ultimo_cambio_fue_grande`).

Este módulo es puro (sin UI): expone el estado y las reglas de negocio para que
una futura app (o este mismo CLI) decida cómo mostrarlas.
"""
from __future__ import annotations

import json
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

GANANCIA_NEUTRA_DB = 0.0
UMBRAL_CAMBIO_GRANDE_DB = 6.0  # a partir de acá se considera "cambio grande"


@dataclass
class Perfil:
    nombre: str
    ajustes_db: dict[str, float] = field(default_factory=dict)
    creado_ts: float = field(default_factory=time.time)
    actualizado_ts: float = field(default_factory=time.time)
    _ajustes_previos: dict[str, float] | None = field(default=None, repr=False)
    _ultimo_cambio_grande: bool = field(default=False, repr=False)

    def dias_desde_actualizacion(self) -> float:
        return (time.time() - self.actualizado_ts) / 86400

    def hay_deshacer_disponible(self) -> bool:
        return self._ajustes_previos is not None

    def ultimo_cambio_fue_grande(self) -> bool:
        return self._ultimo_cambio_grande

    def a_config_limiter(self) -> dict[str, dict]:
        """Convierte el perfil al formato de config que espera procesar.py."""
        return {
            categoria: {"modo": "volumen", "valor_db": db}
            for categoria, db in self.ajustes_db.items()
        }

    def to_dict(self) -> dict:
        return {
            "nombre": self.nombre,
            "ajustes_db": self.ajustes_db,
            "creado_ts": self.creado_ts,
            "actualizado_ts": self.actualizado_ts,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Perfil":
        return cls(
            nombre=data["nombre"],
            ajustes_db=dict(data["ajustes_db"]),
            creado_ts=data["creado_ts"],
            actualizado_ts=data["actualizado_ts"],
        )


class PerfilManager:
    """
    Persiste perfiles como JSON, uno por archivo, en `base_dir`.

    Pensado para una app real: los métodos públicos (`actualizar_categoria`,
    `registrar_categoria_nueva`, `restaurar_anterior`) SIEMPRE releen el estado
    más reciente de disco antes de aplicar su cambio, bajo un lock. Esto evita
    dos problemas reales que aparecen con varios sliders moviéndose casi al
    mismo tiempo (confirmado con un test de escrituras concurrentes):

    1. Corrupción del archivo: `Path.write_text` no es atómica: si dos
       requests escriben el mismo archivo casi al mismo tiempo, un lector
       puede encontrarse el archivo a medio escribir y romper el JSON. Se
       arregla escribiendo a un archivo temporal y haciendo `replace()`
       (atómico a nivel de sistema de archivos).
    2. Cambios perdidos ("lost update"): si dos categorías se actualizan casi
       al mismo tiempo y cada una parte de una copia del perfil ya vieja en
       memoria, la que guarda último pisa el cambio de la otra. Se arregla
       releyendo el estado actual de disco dentro del lock, no confiando en
       el objeto `Perfil` que pasó el que llama.
    """

    def __init__(self, base_dir: Path | str):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    def _ruta(self, nombre: str) -> Path:
        return self.base_dir / f"{nombre}.json"

    def existe(self, nombre: str) -> bool:
        return self._ruta(nombre).exists()

    def _leer(self, nombre: str) -> Perfil | None:
        ruta = self._ruta(nombre)
        if not ruta.exists():
            return None
        return Perfil.from_dict(json.loads(ruta.read_text(encoding="utf-8")))

    def cargar(self, nombre: str) -> tuple[Perfil, str]:
        """
        Carga (o crea) un perfil. Devuelve (perfil, mensaje_ux) — el mensaje es
        el aviso que hay que mostrarle al usuario ("se crea por primera vez" o
        "cargando ajustes de hace X días"), según la regla de UX del proyecto.
        """
        with self._lock:
            perfil = self._leer(nombre)
            if perfil is None:
                perfil = Perfil(nombre=nombre)
                self._guardar(perfil)
                mensaje = (
                    f"Perfil '{nombre}' creado. Se va a actualizar solo con cada "
                    f"ajuste que hagas durante la sesión."
                )
                return perfil, mensaje

        dias = perfil.dias_desde_actualizacion()
        if dias < 1:
            hace = "hoy"
        elif dias < 2:
            hace = "hace 1 día"
        else:
            hace = f"hace {int(dias)} días"
        mensaje = f"Cargando ajustes de tu último ensayo en '{nombre}', {hace}."
        return perfil, mensaje

    def _ajustes_frescos(self, perfil: Perfil) -> dict[str, float]:
        """
        Última versión de `ajustes_db` conocida (de disco si existe, si no la
        que ya tenía `perfil` en memoria). Traer esto de disco es lo que evita
        que dos categorías actualizadas casi al mismo tiempo se pisen entre sí.
        OJO: no toca `_ajustes_previos` — ese campo es a propósito transitorio
        (no se persiste), porque el "deshacer" es del cambio que está haciendo
        ESTA sesión en memoria, no algo que tenga sentido reconstruir de disco.
        """
        disco = self._leer(perfil.nombre)
        return dict(disco.ajustes_db) if disco is not None else dict(perfil.ajustes_db)

    def actualizar_categoria(self, perfil: Perfil, categoria: str, valor_db: float) -> None:
        """Aplica un cambio de ganancia y lo persiste inmediatamente (auto-save)."""
        with self._lock:
            ajustes = self._ajustes_frescos(perfil)
            valor_anterior = ajustes.get(categoria, GANANCIA_NEUTRA_DB)

            perfil._ajustes_previos = dict(ajustes)
            perfil._ultimo_cambio_grande = (
                abs(valor_db - valor_anterior) >= UMBRAL_CAMBIO_GRANDE_DB
            )
            ajustes[categoria] = valor_db
            perfil.ajustes_db = ajustes
            perfil.actualizado_ts = time.time()
            self._guardar(perfil)

    def registrar_categoria_nueva(self, perfil: Perfil, categoria: str) -> bool:
        """
        Si `categoria` no existe todavía en el perfil, la agrega en 0dB
        (neutro) sin pisar las categorías ya ajustadas. Devuelve True si la
        agregó, False si ya existía.
        """
        with self._lock:
            ajustes = self._ajustes_frescos(perfil)
            perfil.ajustes_db = ajustes
            if categoria in ajustes:
                return False
            ajustes[categoria] = GANANCIA_NEUTRA_DB
            perfil.actualizado_ts = time.time()
            self._guardar(perfil)
            return True

    def restaurar_anterior(self, perfil: Perfil) -> bool:
        """Deshace el último cambio hecho por ESTA sesión. Devuelve False si no había nada que deshacer."""
        with self._lock:
            if perfil._ajustes_previos is None:
                return False
            perfil.ajustes_db = perfil._ajustes_previos
            perfil._ajustes_previos = None
            perfil._ultimo_cambio_grande = False
            perfil.actualizado_ts = time.time()
            self._guardar(perfil)
            return True

    def _guardar(self, perfil: Perfil) -> None:
        """Escritura atómica: a un .tmp primero, después replace() (no deja archivos a medio escribir)."""
        destino = self._ruta(perfil.nombre)
        tmp = destino.with_suffix(".json.tmp")
        tmp.write_text(
            json.dumps(perfil.to_dict(), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        tmp.replace(destino)
