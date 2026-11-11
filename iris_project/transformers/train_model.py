
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import pandas as pd

if 'transformer' not in globals():
    from mage_ai.data_preparation.decorators import transformer

if 'test' not in globals():
    from mage_ai.data_preparation.decorators import test

@transformer
def train_model(df, *args, **kwargs):
    """
    Train a Decision Tree classifier using the prepared Iris dataset.
    """

    # Read model depth from Mage runtime variables.
    max_depth = int(kwargs.get("max_depth", 3))

    # Separate training and testing rows.
    train_df = df[df["split"] == "train"]
    test_df = df[df["split"] == "test"]

    # Separate input features and target.
    feature_columns = [
        col for col in df.columns
        if col not in ["target", "split"]
    ]

    X_train = train_df[feature_columns]
    y_train = train_df["target"]

    X_test = test_df[feature_columns]
    y_test = test_df["target"]

    # Create and train the Decision Tree model.
    model = DecisionTreeClassifier(
        max_depth=max_depth,
        random_state=42
    )

    model.fit(X_train, y_train)

    # Predict test data.
    y_pred = model.predict(X_test)

    # Calculate accuracy.
    test_accuracy = accuracy_score(y_test, y_pred)
    print(f"Test accuracy: {test_accuracy:.4f}")

    predictions = pd.DataFrame({
        "actual": y_test.to_numpy(),
        "predicted": y_pred
    })

    return {
        "predictions": predictions,
        "test_accuracy": float(test_accuracy),
        "max_depth": max_depth
    }


@test
def test_output(output, *args, **kwargs):
    """
    Check whether the trained model achieves at least 90% accuracy.
    """

    assert output is not None, "Model output is empty"

    assert "test_accuracy" in output, "Accuracy is missing"

    assert output["test_accuracy"] >= 0.90, (
        f"Test accuracy is too low: {output['test_accuracy']:.2f}"
    )

    assert len(output["predictions"]) == 30, (
        "Expected 30 test predictions"
    )
