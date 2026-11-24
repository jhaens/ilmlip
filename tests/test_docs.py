"""docs/reference.md names every command-line option and every load_calculator keyword."""
import argparse
import inspect
import re
import runpy
from pathlib import Path

import pytest

from ilmlip import data, evaluate, finetune, pretrained, probe, replay, train
from ilmlip.calculator import load_calculator
from ilmlip.jax_md import JaxCalculator
from ilmlip.verlet import VerletCalculator

DOC = Path(__file__).resolve().parents[1] / "docs" / "reference.md"
pytestmark = pytest.mark.skipif(not DOC.exists(), reason="no docs/reference.md")


class Parser(Exception):
    pass


def parser_of(call, monkeypatch):
    """The ArgumentParser that call() builds, taken at its parse_args."""
    def grab(self, *args, **kwargs):
        raise Parser(self)
    monkeypatch.setattr(argparse.ArgumentParser, "parse_args", grab)
    with pytest.raises(Parser) as got:
        call()
    return got.value.args[0]


ENTRY = {
    "train": lambda: train.parse_args([]),
    "finetune": lambda: finetune.parser().parse_args([]),
    "prepare-replay": lambda: replay.prepare_cli([]),
    "data": lambda: data.main([]),
    "evaluate": lambda: evaluate.main([]),
    "pretrained": lambda: pretrained.main(),
    "probe": lambda: probe.main([]),
    "lammps": lambda: runpy.run_module("ilmlip.lammps", run_name="__main__"),
}

HEADING = {"prepare-replay": "finetune prepare-replay"}


def section(heading):
    """The text under a heading of reference.md, up to the next heading of the same or a higher level."""
    level = heading.split()[0]
    lines = DOC.read_text().splitlines()
    start = lines.index(heading) + 1
    end = next((k for k in range(start, len(lines))
                if lines[k].startswith("#") and len(lines[k].split()[0]) <= len(level)), len(lines))
    return "\n".join(lines[start:end])


@pytest.mark.parametrize("command", ENTRY)
def test_every_option_is_documented(command, monkeypatch):
    text = section(f"### {HEADING.get(command, command)}")
    p = parser_of(ENTRY[command], monkeypatch)
    shown = [a for a in p._actions if a.help is not argparse.SUPPRESS]
    names = [s for a in shown for s in a.option_strings if s.startswith("--") and s != "--help"]
    names += [a.dest for a in shown if not a.option_strings]
    missing = [n for n in names if not re.search(f"`{re.escape(n)}[` ]", text)]
    assert not missing, f"{command}: not in {DOC.name}: {missing}"


def test_every_calculator_keyword_is_documented():
    text = DOC.read_text()
    own = lambda f, skip: [k for k in inspect.signature(f).parameters if k not in skip]
    names = (own(load_calculator, {"options"}) + own(JaxCalculator.__init__, {"self", "model", "kw"})
             + own(VerletCalculator.__init__, {"self", "species", "rcut", "dtype", "kw"}))
    missing = sorted({n for n in names if f"`{n}`" not in text})
    assert not missing, f"not in {DOC.name}: {missing}"
