
import pandas as pd

from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

if 'transformer' not in globals():
    from mage_ai.data_preparation.decorators import transformer

if 'test' not in globals():
    from mage_ai.data_preparation.decorators import test


@transformer
def evaluate_model(model_output, *args, **kwargs):
    """
    Evaluate the trained Iris Decision Tree model.
    Generate a confusion matrix and metrics table.
    """

    predictions = model_output["predictions"]

    y_true = predictions["actual"]
    y_pred = predictions["predicted"]

    # Calculate confusion matrix for the three Iris classes.
    labels = [0, 1, 2]

    cm = confusion_matrix(
        y_true,
        y_pred,
        labels=labels
    )

    confusion_df = pd.DataFrame(
        cm,
        index=["Actual 0", "Actual 1", "Actual 2"],
        columns=["Predicted 0", "Predicted 1", "Predicted 2"]
    )

    # Calculate evaluation metrics.
    metrics_df = pd.DataFrame({
        "Metric": [
            "Accuracy",
            "Precision (Macro)",
            "Recall (Macro)",
            "F1-Score (Macro)"
        ],
        "Value": [
            accuracy_score(y_true, y_pred),
            precision_score(y_true, y_pred, average="macro", zero_division=0),
            recall_score(y_true, y_pred, average="macro", zero_division=0),
            f1_score(y_true, y_pred, average="macro", zero_division=0)
        ]
    })

    # Display results.
    print("\nCONFUSION MATRIX")
    print(confusion_df.to_string())

    print("\nMETRICS TABLE")
    print(metrics_df.to_string(index=False))

    return {
        "confusion_matrix": confusion_df,
        "metrics": metrics_df
    }


@test
def test_output(output, *args, **kwargs):
    """
    Validate the evaluation results.
    """

    assert output is not None

    cm = output["confusion_matrix"]
    metrics = output["metrics"]

    assert cm.shape == (3, 3)
    assert cm.values.sum() == 30

    assert len(metrics) == 4
    assert metrics["Value"].between(0, 1).all()
