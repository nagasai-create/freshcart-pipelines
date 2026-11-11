
from pathlib import Path
import pandas as pd

if 'data_exporter' not in globals():
    from mage_ai.data_preparation.decorators import data_exporter

if 'test' not in globals():
    from mage_ai.data_preparation.decorators import test


@data_exporter
def save_results(model_output, *args, **kwargs):

    predictions = model_output["predictions"]

    output_path = (
        Path.cwd()
        / "iris_project"
        / "iris_predictions.csv"
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)

    predictions.to_csv(output_path, index=False)

    print(f"Predictions saved to: {output_path}")
    print(f"Total predictions: {len(predictions)}")

    return str(output_path)


@test
def test_output(output, *args, **kwargs):
    """
    Check that the predictions CSV was created correctly.
    """

    output_path = Path(output)

    assert output_path.exists(), "CSV file was not created"

    saved_data = pd.read_csv(output_path)

    assert len(saved_data) == 30, "Expected 30 predictions"

    assert "actual" in saved_data.columns
    assert "predicted" in saved_data.columns

