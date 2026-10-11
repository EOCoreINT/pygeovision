"""OptunaHPO must fail loudly instead of returning an empty result or inventing a constant."""
import sys
import types

import pytest

from pygeovision.training.hpo import OptunaHPO


def test_missing_optuna_is_an_error(monkeypatch):
    monkeypatch.setitem(sys.modules, "optuna", None)          # makes `import optuna` raise ImportError
    with pytest.raises(ImportError, match="pip install optuna"):
        OptunaHPO(n_trials=1).run(lambda p: 0.0, {"lr": ("float", 0.0, 1.0)})


def test_unknown_search_space_kind_is_an_error(monkeypatch):
    class Trial:
        def suggest_float(self, *a, **k):
            return 0.5

    class Study:
        trials: list = []

        def optimize(self, fn, **kw):
            fn(Trial())

    fake = types.SimpleNamespace(
        logging=types.SimpleNamespace(set_verbosity=lambda *_: None, WARNING=0),
        pruners=types.SimpleNamespace(MedianPruner=lambda **k: None, HyperbandPruner=lambda: None,
                                      NopPruner=lambda: None),
        create_study=lambda **k: Study(),
    )
    monkeypatch.setitem(sys.modules, "optuna", fake)
    with pytest.raises(ValueError, match="unknown kind"):
        OptunaHPO(n_trials=1).run(lambda p: 0.0, {"lr": ("float", 0.0, 1.0), "x": ("weird", 1)})
