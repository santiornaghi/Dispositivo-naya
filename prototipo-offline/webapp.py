"""
App web local del prototipo: elegís un track ya separado, cargás/creás un
perfil de sesión, y movés sliders de Modo Volumen escuchando el resultado en
vivo (la mezcla en tiempo real la hace el navegador con Web Audio API — el
servidor solo sirve los stems y persiste el perfil, no reprocesa audio en
cada movimiento de slider).

Uso:
    python webapp.py
    -> abrir http://127.0.0.1:5000

Requiere que ya exista output de Demucs en STEMS_DIR (ver demucs-fase0), y/o
separaciones SATB reales en SATB_DIR (ver fine-tuning-satb/infer.py).

Si un track tiene separación SATB real (soprano/alto/tenor/bajo), la app usa
esa por sobre los 4 stems genéricos de Demucs (bass/drums/other/vocals) —
son las categorías de verdad del proyecto para coro, no un placeholder.
"""
from __future__ import annotations

import subprocess
import threading
import time
from pathlib import Path

from flask import Flask, jsonify, request, send_from_directory

from perfiles import PerfilManager

STEMS_DIR = Path(__file__).parent.parent / "demucs-fase0" / "output" / "htdemucs"
FINE_TUNING_DIR = Path(__file__).parent.parent / "fine-tuning-satb"
SATB_DIR = FINE_TUNING_DIR / "separaciones"
GRABACIONES_DIR = Path(__file__).parent / "grabaciones"
PERFILES_DIR = Path(__file__).parent / "perfiles_data"
VENV_PYTHON = Path(__file__).parent.parent / "demucs-fase0" / ".venv" / "Scripts" / "python.exe"
CHECKPOINT_SATB = FINE_TUNING_DIR / "modelo_entrenado" / "satb_v2.pt"

app = Flask(__name__, static_folder="static", static_url_path="")
manager = PerfilManager(PERFILES_DIR)

_estado_grabaciones: dict[str, str] = {}  # id -> "procesando" | "listo" | "error"
_estado_lock = threading.Lock()


def _resolver_track(track: str) -> tuple[Path, str] | tuple[None, None]:
    """Devuelve (carpeta, extension) priorizando SATB real sobre stems genéricos."""
    carpeta_satb = SATB_DIR / track
    if carpeta_satb.exists():
        return carpeta_satb, "wav"
    carpeta_generica = STEMS_DIR / track
    if carpeta_generica.exists():
        return carpeta_generica, "mp3"
    return None, None


def _categorias_de(track: str) -> list[str]:
    carpeta, ext = _resolver_track(track)
    if carpeta is None:
        return []
    return sorted(p.stem for p in carpeta.glob(f"*.{ext}"))


@app.get("/")
def index():
    return send_from_directory(app.static_folder, "index.html")


@app.get("/api/tracks")
def listar_tracks():
    tracks = set()
    if SATB_DIR.exists():
        tracks.update(p.name for p in SATB_DIR.iterdir() if p.is_dir())
    if STEMS_DIR.exists():
        tracks.update(p.name for p in STEMS_DIR.iterdir() if p.is_dir())
    return jsonify(sorted(tracks))


@app.get("/api/tracks/<track>/categorias")
def categorias_del_track(track: str):
    carpeta, _ = _resolver_track(track)
    if carpeta is None:
        return jsonify({"error": f"no existe el track '{track}'"}), 404
    return jsonify(_categorias_de(track))


@app.get("/api/audio/<track>/<categoria>")
def audio_stem(track: str, categoria: str):
    carpeta, ext = _resolver_track(track)
    if carpeta is None or not (carpeta / f"{categoria}.{ext}").exists():
        return jsonify({"error": "stem no encontrado"}), 404
    return send_from_directory(carpeta, f"{categoria}.{ext}")


@app.get("/api/perfil/<nombre>")
def cargar_perfil(nombre: str):
    track = request.args.get("track")
    if not track:
        return jsonify({"error": "falta ?track="}), 400

    perfil, mensaje = manager.cargar(nombre)

    categorias_nuevas = []
    for categoria in _categorias_de(track):
        if manager.registrar_categoria_nueva(perfil, categoria):
            categorias_nuevas.append(categoria)

    return jsonify({
        "nombre": perfil.nombre,
        "ajustes_db": perfil.ajustes_db,
        "mensaje": mensaje,
        "categorias_nuevas": categorias_nuevas,
        "hay_deshacer": perfil.hay_deshacer_disponible(),
    })


@app.post("/api/perfil/<nombre>/ajuste")
def actualizar_ajuste(nombre: str):
    body = request.get_json(force=True)
    categoria = body["categoria"]
    valor_db = float(body["valor_db"])

    perfil, _ = manager.cargar(nombre)
    manager.actualizar_categoria(perfil, categoria, valor_db)

    return jsonify({
        "ajustes_db": perfil.ajustes_db,
        "cambio_grande": perfil.ultimo_cambio_fue_grande(),
        "hay_deshacer": perfil.hay_deshacer_disponible(),
    })


@app.post("/api/perfil/<nombre>/restaurar")
def restaurar(nombre: str):
    perfil, _ = manager.cargar(nombre)
    ok = manager.restaurar_anterior(perfil)
    return jsonify({
        "restaurado": ok,
        "ajustes_db": perfil.ajustes_db,
        "hay_deshacer": perfil.hay_deshacer_disponible(),
    })


def _separar_en_background(id_grabacion: str, ruta_wav: Path) -> None:
    resultado = subprocess.run(
        [str(VENV_PYTHON), "infer.py",
         "--checkpoint", str(CHECKPOINT_SATB),
         "--audio", str(ruta_wav),
         "--out", str(SATB_DIR / id_grabacion)],
        cwd=str(FINE_TUNING_DIR),
        capture_output=True, text=True,
    )
    with _estado_lock:
        if resultado.returncode == 0:
            _estado_grabaciones[id_grabacion] = "listo"
        else:
            _estado_grabaciones[id_grabacion] = "error"
            print(f"[error separando {id_grabacion}]\n{resultado.stderr}")


@app.post("/api/grabaciones")
def subir_grabacion():
    archivo = request.files.get("audio")
    if archivo is None:
        return jsonify({"error": "falta el archivo 'audio'"}), 400

    GRABACIONES_DIR.mkdir(parents=True, exist_ok=True)
    id_grabacion = f"grabacion_{int(time.time())}"
    ruta_wav = GRABACIONES_DIR / f"{id_grabacion}.wav"
    archivo.save(ruta_wav)

    with _estado_lock:
        _estado_grabaciones[id_grabacion] = "procesando"

    hilo = threading.Thread(target=_separar_en_background, args=(id_grabacion, ruta_wav), daemon=True)
    hilo.start()

    return jsonify({"id": id_grabacion, "estado": "procesando"})


@app.get("/api/grabaciones/<id_grabacion>/estado")
def estado_grabacion(id_grabacion: str):
    with _estado_lock:
        estado = _estado_grabaciones.get(id_grabacion, "desconocido")
    return jsonify({"id": id_grabacion, "estado": estado})


if __name__ == "__main__":
    # host=0.0.0.0 + ssl para que sea alcanzable (y el mic del celular funcione,
    # getUserMedia exige contexto seguro) desde otro dispositivo en la misma red.
    # debug=False a propósito: el debugger interactivo de Flask permite ejecutar
    # código arbitrario si alguien más en la red lo alcanza — no vale la pena
    # exponerlo aunque sea una red doméstica.
    app.run(host="0.0.0.0", port=5000, debug=False, threaded=True, ssl_context="adhoc")
