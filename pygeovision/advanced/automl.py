"""
AutoML for geospatial models (D6) — automated architecture search and HPO.
"""
from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any

logger = logging.getLogger(__name__)


class GeoAutoML:
    """Hyperparameter search for any training function (D6), with Optuna (default) or Ray Tune.

    You supply a function that trains with a given configuration and returns one number; this class
    proposes configurations, keeps the best and tells you what mattered. It does not know anything
    about models: what is searched (learning rate, backbone name, loss name, ...) is whatever your
    function does with the configuration.

    Example::

        automl = GeoAutoML(metric="val_iou", direction="maximize", n_trials=50)
        result = automl.search(
            train_fn=my_train_function,           # def my_train_function(config: dict) -> float
            search_space={
                "lr": ("float", 1e-5, 1e-2, "log"),
                "backbone": ("categorical", ["resnet50", "resnet101", "efficientnet_b4"]),
                "epochs": ("int", 5, 30),
                "loss": ("categorical", ["combo", "tversky", "focal"]),
            },
        )
        result["best_params"], result["best_value"]
    """

    SEARCH_BACKENDS = ["optuna", "ray_tune"]
    PRUNERS = ("median", "hyperband", "none")
    SAMPLERS = ("tpe", "cmaes", "random")

    def __init__(
        self,
        metric: str = "val_iou",
        direction: str = "maximize",
        n_trials: int = 50,
        timeout_s: int | None = None,
        backend: str = "optuna",
        n_jobs: int = 1,
        storage: str | None = None,
        study_name: str | None = None,
        seed: int | None = 42,
        show_progress_bar: bool = True,
    ) -> None:
        if direction not in ("maximize", "minimize"):
            raise ValueError(f"direction must be 'maximize' or 'minimize'; got {direction!r}")
        if backend not in self.SEARCH_BACKENDS:
            raise ValueError(f"backend must be one of {self.SEARCH_BACKENDS}; got {backend!r}")
        self.metric = metric
        self.direction = direction
        self.n_trials = n_trials
        self.timeout_s = timeout_s
        self.backend = backend
        self.n_jobs = n_jobs
        self.storage = storage
        self.study_name = study_name or "pgv_automl"
        self.seed = seed
        self.show_progress_bar = show_progress_bar
        self._study = None
        self.errors: list[str] = []

    def search(
        self,
        train_fn: Callable[..., float],
        search_space: dict[str, Any],
        pruner: str | None = "median",
        sampler: str | None = "tpe",
    ) -> dict[str, Any]:
        """Run the search.

        Args:
            train_fn: ``train_fn(config) -> float``, the metric to optimise (higher is better for
                ``maximize``). If it takes a second positional argument it also receives the Optuna
                ``trial`` so it can report intermediate values and be pruned::

                    def train_fn(config, trial):
                        for epoch in range(config["epochs"]):
                            ...
                            trial.report(val_iou, epoch)
                            if trial.should_prune():
                                raise optuna.TrialPruned()
                        return val_iou

                Without that, pruners never see anything and no trial is pruned. An exception, ``None``
                or NaN from ``train_fn`` marks the trial **failed** (it can never become the best).
            search_space: ``{name: spec}`` with ``spec`` one of ``("float", low, high)``,
                ``("float", low, high, "log")``, ``("int", low, high)``, ``("categorical", [choices])``
                or ``("constant", value)``. Any other kind raises ``ValueError``.
            pruner: ``"median"`` (default), ``"hyperband"`` or ``None``/``"none"`` for no pruning.
            sampler: ``"tpe"`` (default), ``"cmaes"`` (numeric parameters only) or ``"random"``.

        Returns:
            ``{"success": True, "best_params", "best_value", "n_trials_completed", "n_trials_pruned",
            "n_trials_failed", "metric", "direction", "study_name", "errors"}``, or
            ``{"success": False, "error": ...}`` when Optuna is missing or no trial completed.
        """
        self._validate_space(search_space)
        if self.backend == "optuna":
            return self._search_optuna(train_fn, search_space, pruner, sampler)
        return self._search_ray(train_fn, search_space)

    @staticmethod
    def _validate_space(search_space: dict[str, Any]) -> None:
        if not search_space:
            raise ValueError("search_space is empty")
        for name, spec in search_space.items():
            kinds = {"float": 3, "int": 3, "categorical": 2, "constant": 2}
            if not isinstance(spec, (tuple, list)) or not spec or spec[0] not in kinds:
                raise ValueError(f"search_space[{name!r}] = {spec!r}: expected a tuple starting with one of "
                                 f"{list(kinds)}")
            if len(spec) < kinds[spec[0]]:
                raise ValueError(f"search_space[{name!r}] = {spec!r}: too few elements for {spec[0]!r}")
            if spec[0] in ("float", "int") and not spec[1] < spec[2]:
                raise ValueError(f"search_space[{name!r}]: low ({spec[1]}) must be below high ({spec[2]})")
            if spec[0] == "categorical" and not len(spec[1]):
                raise ValueError(f"search_space[{name!r}]: no choices")

    def _search_optuna(self, train_fn, search_space, pruner, sampler) -> dict[str, Any]:
        try:
            import optuna
        except ImportError:
            return {"success": False, "error": "pip install optuna"}

        pruner_key = "none" if pruner is None else pruner
        sampler_key = "tpe" if sampler is None else sampler
        if pruner_key not in self.PRUNERS:
            raise ValueError(f"pruner must be one of {list(self.PRUNERS)} or None; got {pruner!r}")
        if sampler_key not in self.SAMPLERS:
            raise ValueError(f"sampler must be one of {list(self.SAMPLERS)}; got {sampler!r}")
        pruner_obj = {
            "median":    lambda: optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=5),
            "hyperband": lambda: optuna.pruners.HyperbandPruner(),
            "none":      lambda: optuna.pruners.NopPruner(),
        }[pruner_key]()
        sampler_obj = {
            "tpe":    lambda: optuna.samplers.TPESampler(seed=self.seed),
            "cmaes":  lambda: optuna.samplers.CmaEsSampler(seed=self.seed),
            "random": lambda: optuna.samplers.RandomSampler(seed=self.seed),
        }[sampler_key]()

        import inspect
        try:
            positional = [p for p in inspect.signature(train_fn).parameters.values()
                          if p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD)]
        except (TypeError, ValueError):
            positional = []
        wants_trial = len(positional) >= 2

        self.errors = []

        def _objective(trial: optuna.Trial) -> float:
            config = {}
            for name, spec in search_space.items():
                kind = spec[0]
                if kind == "float":
                    config[name] = trial.suggest_float(name, spec[1], spec[2],
                                                       log=len(spec) > 3 and spec[3] == "log")
                elif kind == "int":
                    config[name] = trial.suggest_int(name, spec[1], spec[2])
                elif kind == "categorical":
                    config[name] = trial.suggest_categorical(name, list(spec[1]))
                else:
                    config[name] = spec[1]
            try:
                value = train_fn(config, trial) if wants_trial else train_fn(config)
            except optuna.exceptions.TrialPruned:
                raise
            except Exception as exc:
                if len(self.errors) < 5:
                    self.errors.append(f"{type(exc).__name__}: {exc}")
                logger.warning("Trial %d failed: %s: %s", trial.number, type(exc).__name__, exc)
                raise
            if value is None or value != value:
                msg = f"train_fn returned {value!r} for {config}"
                if len(self.errors) < 5:
                    self.errors.append(msg)
                raise ValueError(msg)
            return float(value)

        # Optuna prints a full traceback for every failed trial; we log one line per failure ourselves
        # and keep the reasons in ``errors``.
        previous_verbosity = optuna.logging.get_verbosity()
        optuna.logging.set_verbosity(optuna.logging.ERROR)
        try:
            self._study = optuna.create_study(
                direction=self.direction,
                pruner=pruner_obj,
                sampler=sampler_obj,
                study_name=self.study_name,
                storage=self.storage,
                load_if_exists=True,
            )
            self._study.optimize(
                _objective,
                n_trials=self.n_trials,
                timeout=self.timeout_s,
                n_jobs=self.n_jobs,
                show_progress_bar=self.show_progress_bar,
                catch=(Exception,),
            )
        finally:
            optuna.logging.set_verbosity(previous_verbosity)

        states = optuna.trial.TrialState
        trials = self._study.trials
        n_ok = sum(t.state == states.COMPLETE for t in trials)
        n_pruned = sum(t.state == states.PRUNED for t in trials)
        n_failed = sum(t.state == states.FAIL for t in trials)
        if n_ok == 0:
            return {"success": False,
                    "error": "no trial completed" + (f"; first error: {self.errors[0]}" if self.errors else ""),
                    "n_trials_pruned": n_pruned, "n_trials_failed": n_failed, "errors": list(self.errors)}

        best = self._study.best_trial
        return {
            "success": True,
            "best_params": best.params,
            "best_value": round(best.value, 6),
            "n_trials_completed": n_ok,
            "n_trials_pruned": n_pruned,
            "n_trials_failed": n_failed,
            "metric": self.metric,
            "direction": self.direction,
            "study_name": self.study_name,
            "errors": list(self.errors),
        }

    def _search_ray(self, train_fn, search_space) -> dict[str, Any]:
        try:
            from ray import air, tune
        except ImportError:
            return {"success": False, "error": "pip install ray[tune]"}

        ray_space = {}
        for name, spec in search_space.items():
            kind = spec[0]
            if kind == "float":
                ray_space[name] = tune.loguniform(spec[1], spec[2]) if (len(spec) > 3 and spec[3] == "log") else tune.uniform(spec[1], spec[2])
            elif kind == "int":
                ray_space[name] = tune.randint(spec[1], spec[2] + 1)
            elif kind == "categorical":
                ray_space[name] = tune.choice(spec[1])

        def _ray_objective(config):
            val = train_fn(config)
            tune.report(**{self.metric: val})

        analysis = tune.run(
            _ray_objective,
            config=ray_space,
            num_samples=self.n_trials,
            metric=self.metric,
            mode=self.direction[:3],  # "max" or "min"
        )

        best = analysis.best_config
        best_val = analysis.best_result.get(self.metric)
        return {
            "success": True,
            "best_params": best,
            "best_value": best_val,
            "n_trials_completed": self.n_trials,
        }

    def importance(self) -> dict[str, float]:
        """Hyperparameter importance (shares that sum to 1) from the completed Optuna study.

        Returns ``{}`` before a search has run or when Optuna cannot compute it (it needs at least two
        completed trials and a parameter that varies). With few trials the numbers are noise.
        """
        if self._study is None:
            return {}
        try:
            import optuna
            imp = optuna.importance.get_param_importances(self._study)
            return {k: round(v, 4) for k, v in imp.items()}
        except Exception:
            return {}

    def plot_optimization_history(self, save_path: str | None = None) -> Any:
        """Plot the best-so-far value over trials (needs ``plotly``).

        With ``save_path`` ending in ``.html`` the interactive figure is written there (no extra packages);
        any other extension (``.png``, ``.svg``) needs ``kaleido``. Without ``save_path`` the figure is
        returned for you to show. Raises ``RuntimeError`` before a search and ``ImportError`` without plotly.
        """
        if self._study is None:
            raise RuntimeError("no study yet: call search() first")
        try:
            import optuna.visualization as viz
            fig = viz.plot_optimization_history(self._study)
        except ImportError as exc:
            raise ImportError("plotting needs plotly: pip install plotly") from exc
        if save_path:
            if str(save_path).lower().endswith(".html"):
                fig.write_html(save_path)
            else:
                fig.write_image(save_path)       # raises with a clear message if kaleido is missing
        return fig
