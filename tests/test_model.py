import pandas as pd
import mlflow
from pathlib import Path
import pytest

mlflow.set_tracking_uri("sqlite:///mlflow.db")

root_path = Path(__file__).parent.parent
model_name = "delivery_time_pred_model"
stage = "Staging"

TARGET = "time_taken"

# acceptable performance threshold — if MAE exceeds this, something regressed
MAX_ACCEPTABLE_MAE = 5.0
MIN_ACCEPTABLE_R2 = 0.70


@pytest.fixture(scope="module")
def model():
    client = mlflow.MlflowClient()
    latest_version = client.get_latest_versions(name=model_name, stages=[stage])[0].version
    return mlflow.sklearn.load_model(f"models:/{model_name}/{latest_version}")


@pytest.fixture(scope="module")
def test_data():
    test_path = root_path / "data" / "processed" / "test_trans.csv"
    df = pd.read_csv(test_path)
    X_test = df.drop(columns=[TARGET])
    y_test = df[TARGET]
    return X_test, y_test


def test_model_loads_successfully(model):
    """Sanity check: the registered model actually loads without error."""
    assert model is not None


def test_model_produces_predictions(model, test_data):
    """Sanity check: model can predict on real test data and returns the right shape."""
    X_test, y_test = test_data
    predictions = model.predict(X_test)
    assert len(predictions) == len(y_test)


def test_model_mae_within_threshold(model, test_data):
    """The core check: does the model still perform well enough to be trustworthy?"""
    from sklearn.metrics import mean_absolute_error
    X_test, y_test = test_data
    predictions = model.predict(X_test)
    mae = mean_absolute_error(y_test, predictions)
    assert mae < MAX_ACCEPTABLE_MAE, f"Model MAE {mae:.3f} exceeds threshold {MAX_ACCEPTABLE_MAE}"


def test_model_r2_within_threshold(model, test_data):
    """Checks the model explains a reasonable amount of variance."""
    from sklearn.metrics import r2_score
    X_test, y_test = test_data
    predictions = model.predict(X_test)
    r2 = r2_score(y_test, predictions)
    assert r2 > MIN_ACCEPTABLE_R2, f"Model R2 {r2:.3f} below threshold {MIN_ACCEPTABLE_R2}"


def test_predictions_are_positive(model, test_data):
    """Domain sanity check: delivery time can never be negative."""
    X_test, _ = test_data
    predictions = model.predict(X_test)
    assert (predictions > 0).all(), "Model produced negative delivery time predictions!"