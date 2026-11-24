"""The symbolic and numerical checks of this folder, at small sizes."""
import pytest

pytest.importorskip("sympy")
pytest.importorskip("scipy")

import bounds_sympy
import equivariance_check
import invariance_check


@pytest.mark.parametrize("part", ["gate", "squash", "switch", "taper"])
def test_bounds_are_proved_and_match_the_code(part):
    getattr(bounds_sympy, part)()


@pytest.mark.parametrize("dmin", ["1.8", "1.0"], ids=["open", "core_switch_active"])
def test_symmetries_and_finite_differences(dmin):
    assert invariance_check.main(["--atoms", "16", "--dmin", dmin]) == 0


@pytest.mark.parametrize("args", [[], ["--config", "XS"]], ids=["small", "XS"])
def test_internal_features_are_equivariant(args):
    assert equivariance_check.main(args) == 0
