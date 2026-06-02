"""Tests de fumée : la librairie s'importe et expose ses constantes."""

import compariawatch as cw


def test_version() -> None:
    assert cw.__version__ == "0.1.0"


def test_random_state() -> None:
    assert cw.RANDOM_STATE == 42
