"""Run an unmodified upstream run.py on CPU.

Optional third-party packages that are missing locally are replaced by inert
stubs. Every attribute access or call on a stub after import time is counted
and written to <stub_report>, so a run that silently depended on a stub can be
rejected afterwards.

usage: python cpu_launch.py <repo_dir> <work_dir> -- <run.py args...>
The stub report is written to <work_dir>/stub_report.json; relative paths used
by run.py resolve against <work_dir>, so the upstream checkout stays untouched.
"""
import importlib.abc
import importlib.machinery
import importlib.util
import json
import os
import sys
import types

STUB_ROOTS = {
    "cupy", "cuml", "sktime", "torch_geometric", "torch_scatter", "torch_sparse",
    "torch_cluster", "torch_spline_conv", "tsdistances", "robustica", "pyrpca",
    "ot", "torch_dct", "reformer_pytorch", "thop", "torchinfo", "torchmetrics",
    "sklearnex", "patoolib", "pywt", "einops", "setproctitle", "ptflops",
}
USAGE = {"import_phase": True, "created": [], "runtime_use": {}}


class _Stub:
    def __init__(self, name):
        object.__setattr__(self, "_name", name)

    def _hit(self, what):
        if not USAGE["import_phase"]:
            key = f"{self._name}{what}"
            USAGE["runtime_use"][key] = USAGE["runtime_use"].get(key, 0) + 1

    def __getattr__(self, item):
        if item.startswith("__") and item.endswith("__"):
            raise AttributeError(item)
        self._hit("." + item)
        return _Stub(f"{self._name}.{item}")

    def __call__(self, *a, **k):
        self._hit("()")
        return _Stub(f"{self._name}()")

    def __mro_entries__(self, bases):
        return (object,)


class _StubModule(types.ModuleType):
    def __getattr__(self, item):
        if item.startswith("__") and item.endswith("__"):
            raise AttributeError(item)
        return _Stub(f"{self.__name__}.{item}")


class _StubLoader(importlib.abc.Loader):
    def create_module(self, spec):
        m = _StubModule(spec.name)
        m.__path__ = []
        return m

    def exec_module(self, module):
        USAGE["created"].append(module.__name__)


class _StubFinder(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path, target=None):
        root = fullname.split(".")[0]
        if root not in STUB_ROOTS:
            return None
        if fullname == root and importlib.machinery.PathFinder.find_spec(root) is not None:
            return None
        if fullname != root and root in sys.modules and not isinstance(sys.modules[root], _StubModule):
            return None
        return importlib.util.spec_from_loader(fullname, _StubLoader(), is_package=True)


def main():
    repo, work = os.path.abspath(sys.argv[1]), os.path.abspath(sys.argv[2])
    assert sys.argv[3] == "--"
    os.makedirs(work, exist_ok=True)
    report = os.path.join(work, "stub_report.json")
    os.environ.setdefault("MPLBACKEND", "Agg")
    sys.meta_path.insert(0, _StubFinder())
    os.chdir(work)
    sys.path.insert(0, repo)
    sys.argv = ["run.py"] + sys.argv[4:]

    import numpy as np
    for alias, value in (("Inf", np.inf), ("NaN", np.nan)):
        if not hasattr(np, alias):
            setattr(np, alias, value)

    import torch
    orig_load = torch.load

    def load(*a, **k):
        # checkpoints are written by the same run; torch>=2.6 defaults to weights_only=True
        k.setdefault("weights_only", False)
        return orig_load(*a, **k)

    torch.load = load

    import argparse
    orig_parse = argparse.ArgumentParser.parse_args

    def parse_args(self, *a, **k):
        USAGE["import_phase"] = False
        return orig_parse(self, *a, **k)

    argparse.ArgumentParser.parse_args = parse_args
    if os.environ.get("Q1_SIGMA_MODE"):
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        import fixed_sigma_patch
        fixed_sigma_patch.install()

    code = 0
    try:
        run_py = os.path.join(repo, "run.py")
        src = open(run_py, encoding="utf-8").read()
        g = {"__name__": "__main__", "__file__": run_py}
        exec(compile(src, run_py, "exec"), g)
    except SystemExit as e:
        code = e.code or 0
    finally:
        with open(report, "w", encoding="utf-8") as f:
            json.dump(USAGE, f, indent=1)
    sys.exit(code)


if __name__ == "__main__":
    main()
