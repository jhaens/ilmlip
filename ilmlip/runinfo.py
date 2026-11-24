"""The lines a command-line tool writes first: command, host, versions."""
import platform
import shlex
import subprocess
import sys
from pathlib import Path


def git_hash():
    """Short hash if the package folder's parent is a git repository, else None."""
    root = Path(__file__).resolve().parent.parent
    try:
        run = lambda *a: subprocess.run(["git", "-C", str(root), *a], capture_output=True, text=True,
                                        timeout=5, check=True).stdout.strip()
        return run("rev-parse", "--short", "HEAD") if Path(run("rev-parse", "--show-toplevel")).resolve() == root else None
    except (OSError, subprocess.SubprocessError):
        return None


_called = None


def called(module, argv):
    """Record main(argv) of module: an in-process call (argv given) is named instead of sys.argv."""
    global _called
    _called = None if argv is None else [sys.executable, "-m", module, *map(str, argv)]


def header():
    """The calling command (quoted so that it can be pasted), host, ilmlip and torch versions."""
    import torch
    from . import __version__
    argv = _called or getattr(sys, "orig_argv", None) or [sys.executable, *sys.argv]
    git = git_hash()
    return [f"command    {shlex.join(argv)}",
            f"host       {platform.node()}; python {platform.python_version()}, torch {torch.__version__}, "
            f"ilmlip {__version__}" + (f" ({git})" if git else "")]


def show(log=print):
    for line in header():
        log(line)
