import numpy as np
import pytest

from limiter import aplicar_ganancia, aplicar_techo_db, mezclar, db_to_lin

SR = 44100


def test_aplicar_ganancia_es_lineal_y_exacta():
    audio = np.array([0.5, -0.5, 0.25, -0.25])
    salida = aplicar_ganancia(audio, -6)
    assert np.allclose(salida, audio * db_to_lin(-6))


def test_techo_db_salto_sostenido_converge_exacto_al_techo():
    techo_lin = db_to_lin(-12)
    sig = np.zeros(SR * 2)
    sig[SR // 4:] = 1.0
    out = aplicar_techo_db(sig, SR, techo_db=-12, attack_ms=3, release_ms=100)
    assert out.max() == pytest.approx(techo_lin, rel=1e-3)


def test_techo_db_transitorio_breve_no_hace_overshoot():
    """
    Regresión: la primera versión (sin lookahead) dejaba pasar hasta ~8dB por
    encima del techo en transitorios breves porque el suavizado exponencial
    puro tarda ~5x su 'attack_ms' nominal en asentarse. Con lookahead, hasta
    un pico de una sola muestra debe quedar clampeado exacto.
    """
    techo_lin = db_to_lin(-12)
    sig = np.zeros(SR)
    inicio = SR // 4
    sig[inicio] = 1.0  # pico de una sola muestra, el peor caso posible
    out = aplicar_techo_db(sig, SR, techo_db=-12, attack_ms=3, release_ms=100)
    assert out.max() == pytest.approx(techo_lin, rel=1e-3)
    assert out.max() <= techo_lin * 1.01  # nunca por encima del techo (margen de redondeo)


def test_techo_db_nunca_amplifica():
    """Si la señal ya está por debajo del techo, no la toca."""
    techo_lin = db_to_lin(-6)
    sig = np.random.default_rng(0).uniform(-0.01, 0.01, SR)
    out = aplicar_techo_db(sig, SR, techo_db=-6, attack_ms=3, release_ms=100)
    assert np.allclose(out, sig)


def test_mezclar_suma_stems_de_igual_largo():
    a = np.array([0.1, 0.2, 0.3])
    b = np.array([0.05, -0.05, 0.0])
    resultado = mezclar({"a": a, "b": b})
    assert np.allclose(resultado, a + b)


def test_mezclar_rellena_con_ceros_si_los_largos_difieren():
    a = np.array([0.1, 0.2, 0.3, 0.4])
    b = np.array([0.05, -0.05])
    resultado = mezclar({"a": a, "b": b})
    assert len(resultado) == 4
    assert resultado[-1] == pytest.approx(0.4)  # b ya no aporta, se rellenó con 0


def test_mezclar_avisa_si_clippea(capsys):
    a = np.array([0.9, 0.9])
    b = np.array([0.9, 0.9])
    mezclar({"a": a, "b": b})
    salida = capsys.readouterr().out
    assert "aviso" in salida.lower()
    assert "clippea" in salida.lower()
