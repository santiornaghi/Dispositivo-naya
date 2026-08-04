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

from pathlib import Path

from flask import Flask, jsonify, request, send_from_directory

from perfiles import PerfilManager

STEMS_DIR = Path(__file__).parent.parent / "demucs-fase0" / "output" / "htdemucs"
SATB_DIR = Path(__file__).parent.parent / "fine-tuning-satb" / "separaciones"
PERFILES_DIR = Path(__file__).parent / "perfiles_data"

app = Flask(__name__, static_folder="static", static_url_path="")
manager = PerfilManager(PERFILES_DIR)


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


if __name__ == "__main__":
    app.run(debug=True, port=5000)
