"""AIEngine.train argument routing and the CloudDeployer base class."""
import pytest


def test_engine_train_routes_arguments_to_the_real_geotrainer_signature(monkeypatch):
    """`GeoTrainer(model=, train_dataset=, val_dataset=)` raised TypeError: the constructor takes
    (model, config) and the datasets go to fit()."""
    from pygeovision.ai.engine import AIEngine
    from pygeovision.ai.training import trainer as tr

    seen = {}

    class Spy:
        def __init__(self, model, config=None):
            seen["init"] = (model, config)

        def fit(self, **kw):
            seen["fit"] = kw
            return {"ok": True}

    monkeypatch.setattr(tr, "GeoTrainer", Spy)
    eng = AIEngine(pgv_client=None)
    assert eng.train("M", "TRAIN", "VAL", max_epochs=3, loss_fn="L") == {"ok": True}
    assert seen["init"][0] == "M" and seen["init"][1].max_epochs == 3
    assert seen["fit"] == {"train_dataset": "TRAIN", "val_dataset": "VAL", "loss_fn": "L"}
    with pytest.raises(TypeError, match="bogus"):
        eng.train("M", "T", bogus=1)


def test_cloud_deployer_base_is_abstract_and_all_providers_are_concrete():
    from pygeovision.cloud.deploy import AWSDeployer, AzureDeployer, CloudDeployer, GCPDeployer
    with pytest.raises(TypeError):
        CloudDeployer("aws")
    for cls in (AWSDeployer, AzureDeployer, GCPDeployer):
        assert not getattr(cls, "__abstractmethods__", None), cls
