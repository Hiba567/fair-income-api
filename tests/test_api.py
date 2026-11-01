import pytest
from fastapi.testclient import TestClient

from training import train

VALID = {
    "age": 38, "education_num": 13, "hours_per_week": 40,
    "capital_gain": 0, "capital_loss": 0,
    "workclass": "Private", "marital_status": "Married", "occupation": "Tech",
}


@pytest.fixture(scope="module")
def client(tmp_path_factory, monkeypatch_module):
    model_dir = tmp_path_factory.mktemp("models")
    train.main(synthetic=True, out_dir=model_dir)
    monkeypatch_module.setenv("MODEL_DIR", str(model_dir))

    import importlib

    import app.main as main_module
    importlib.reload(main_module)
    with TestClient(main_module.app) as c:
        yield c


@pytest.fixture(scope="module")
def monkeypatch_module():
    mp = pytest.MonkeyPatch()
    yield mp
    mp.undo()


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200 and r.json()["model_loaded"] is True


def test_predict_returns_valid_probability(client):
    r = client.post("/predict", json=VALID)
    assert r.status_code == 200
    body = r.json()
    assert 0.0 <= body["probability"] <= 1.0
    assert body["label"] in (0, 1)


def test_predict_rejects_invalid_input(client):
    bad = {**VALID, "age": -5}
    assert client.post("/predict", json=bad).status_code == 422


def test_unknown_category_still_predicts(client):
    r = client.post("/predict", json={**VALID, "occupation": "Astronaut"})
    assert r.status_code == 200


def test_fairness_report_has_both_attributes(client):
    report = client.get("/fairness").json()
    assert {"sex", "race"} <= set(report["fairness"])
    assert "equalized_odds_diff" in report["fairness"]["sex"]["gaps"]


def test_metrics_counts_predictions(client):
    before = client.get("/metrics").json()["predictions"]
    client.post("/predict", json=VALID)
    assert client.get("/metrics").json()["predictions"] == before + 1
