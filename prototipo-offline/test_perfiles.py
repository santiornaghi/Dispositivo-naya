import threading
import time

import pytest

from perfiles import PerfilManager, GANANCIA_NEUTRA_DB, UMBRAL_CAMBIO_GRANDE_DB


@pytest.fixture
def manager(tmp_path):
    return PerfilManager(tmp_path)


def test_perfil_nuevo_arranca_vacio_y_avisa_primera_vez(manager):
    perfil, mensaje = manager.cargar("Coro San Martín")
    assert perfil.ajustes_db == {}
    assert "creado" in mensaje.lower()


def test_perfil_existente_avisa_hace_cuantos_dias(manager):
    perfil, _ = manager.cargar("Ensayo Jueves")
    manager.actualizar_categoria(perfil, "cuerda", -3.0)

    # simulamos que pasaron 3 días desde el último ajuste
    perfil.actualizado_ts = time.time() - 3 * 86400
    manager._guardar(perfil)

    _, mensaje = manager.cargar("Ensayo Jueves")
    assert "hace 3 días" in mensaje
    assert "Ensayo Jueves" in mensaje


def test_actualizar_categoria_persiste_sin_boton_guardar(manager):
    perfil, _ = manager.cargar("Orquesta Municipal")
    manager.actualizar_categoria(perfil, "viento-metal", -8.0)

    # "reabrimos" el perfil como si fuera otra sesión
    recargado, _ = manager.cargar("Orquesta Municipal")
    assert recargado.ajustes_db["viento-metal"] == -8.0


def test_categoria_nueva_a_mitad_de_sesion_entra_neutra_sin_pisar_otras(manager):
    perfil, _ = manager.cargar("Coro SATB")
    manager.actualizar_categoria(perfil, "soprano", -5.0)

    agregada = manager.registrar_categoria_nueva(perfil, "solista")
    assert agregada is True
    assert perfil.ajustes_db["solista"] == GANANCIA_NEUTRA_DB
    assert perfil.ajustes_db["soprano"] == -5.0  # no se pisó

    # si ya existe, no la vuelve a tocar
    manager.actualizar_categoria(perfil, "solista", -2.0)
    agregada_de_nuevo = manager.registrar_categoria_nueva(perfil, "solista")
    assert agregada_de_nuevo is False
    assert perfil.ajustes_db["solista"] == -2.0  # se mantiene el ajuste del usuario


def test_restaurar_anterior_deshace_el_ultimo_cambio(manager):
    perfil, _ = manager.cargar("Ensayo Sabado")
    manager.actualizar_categoria(perfil, "percusion", -4.0)
    manager.actualizar_categoria(perfil, "percusion", -20.0)

    assert perfil.hay_deshacer_disponible()
    restaurado = manager.restaurar_anterior(perfil)
    assert restaurado is True
    assert perfil.ajustes_db["percusion"] == -4.0
    assert perfil.hay_deshacer_disponible() is False  # un solo nivel de undo


def test_restaurar_anterior_sin_cambios_previos_no_hace_nada(manager):
    perfil, _ = manager.cargar("Perfil Vacio")
    restaurado = manager.restaurar_anterior(perfil)
    assert restaurado is False


def test_flag_de_cambio_grande_segun_umbral(manager):
    perfil, _ = manager.cargar("Test Umbral")

    manager.actualizar_categoria(perfil, "cuerda", -2.0)  # cambio chico (0 -> -2)
    assert perfil.ultimo_cambio_fue_grande() is False

    manager.actualizar_categoria(perfil, "cuerda", -2.0 - UMBRAL_CAMBIO_GRANDE_DB)
    assert perfil.ultimo_cambio_fue_grande() is True


def test_actualizaciones_concurrentes_no_corrompen_ni_pierden_cambios(manager):
    """
    Regresión: en la app real, mover 4 sliders casi al mismo tiempo (cada uno
    dispara su propio POST/request) rompía el archivo del perfil
    (`json.JSONDecodeError` por escritura no atómica) y podía perder el
    cambio de alguna categoría (cada request partía de una copia vieja en
    memoria). Simulamos eso: 8 threads, cada uno hace su propio `cargar()`
    (como haría cada request HTTP) y después actualiza UNA categoría.
    """
    categorias = [f"cat{i}" for i in range(8)]
    errores = []

    def trabajo(categoria, valor):
        try:
            perfil, _ = manager.cargar("Concurrencia")
            manager.actualizar_categoria(perfil, categoria, valor)
        except Exception as e:  # noqa: BLE001
            errores.append(e)

    hilos = [
        threading.Thread(target=trabajo, args=(cat, -float(i)))
        for i, cat in enumerate(categorias)
    ]
    for h in hilos:
        h.start()
    for h in hilos:
        h.join()

    assert not errores, f"fallaron {len(errores)} actualizaciones: {errores}"

    final, _ = manager.cargar("Concurrencia")
    assert set(final.ajustes_db.keys()) == set(categorias)
    for i, cat in enumerate(categorias):
        assert final.ajustes_db[cat] == -float(i)


def test_a_config_limiter_da_formato_compatible_con_procesar_categoria(manager):
    perfil, _ = manager.cargar("Compat")
    manager.actualizar_categoria(perfil, "vocals", -6.0)

    config = perfil.a_config_limiter()
    assert config == {"vocals": {"modo": "volumen", "valor_db": -6.0}}

    from limiter import procesar_categoria
    import numpy as np

    audio = np.array([1.0, -1.0, 0.5])
    salida = procesar_categoria(audio, 44100, config["vocals"])
    assert np.allclose(salida, audio * (10 ** (-6 / 20)))
