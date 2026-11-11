
import pandas as pd

from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score

if 'transformer' not in globals():
    from mage_ai.data_preparation.decorators import transformer

if 'test' not in globals():
    from mage_ai.data_preparation.decorators import test


@transformer
def knn_model(df, *args, **kwargs):
    """
    Train a KNN model as a parallel branch and compare
    its accuracy with the Decision Tree classifier.
    """

    max_depth = int(kwargs.get("max_depth", 3))

    # Separate training and testing records.
    train_df = df[df["split"] == "train"]
    test_df = df[df["split"] == "test"]

    feature_columns = [
        col for col in df.columns
        if col not in ["target", "split"]
    ]

    X_train = train_df[feature_columns]
    y_train = train_df["target"]

    X_test = test_df[feature_columns]
    y_test = test_df["target"]

    # Train the KNN model.
    knn = make_pipeline(
        StandardScaler(),
        KNeighborsClassifier(n_neighbors=5)
    )

    knn.fit(X_train, y_train)
    knn_predictions = knn.predict(X_test)

    knn_accuracy = accuracy_score(y_test, knn_predictions)

    # Train a matching Decision Tree for comparison.
    decision_tree = DecisionTreeClassifier(
        max_depth=max_depth,
        random_state=42
    )

    decision_tree.fit(X_train, y_train)
    tree_predictions = decision_tree.predict(X_test)

    tree_accuracy = accuracy_score(y_test, tree_predictions)

    # Display both model accuracies side by side.
    comparison = pd.DataFrame({
        "Model": [
            "Decision Tree",
            "K-Nearest Neighbors"
        ],
        "Accuracy": [
            tree_accuracy,
            knn_accuracy
        ]
    })

    print("\nMODEL ACCURACY COMPARISON")
    print(comparison.to_string(index=False))

    predictions = pd.DataFrame({
        "actual": y_test.to_numpy(),
        "predicted": knn_predictions
    })

    return {
        "predictions": predictions,
        "knn_accuracy": float(knn_accuracy),
        "decision_tree_accuracy": float(tree_accuracy),
        "comparison": comparison
    }


@test
def test_output(output, *args, **kwargs):
    """
    Validate the KNN model output.
    """

    assert output is not None

    predictions = output["predictions"]
    comparison = output["comparison"]

    assert len(predictions) == 30
    assert set(predictions.columns) == {"actual", "predicted"}

    assert 0 <= output["knn_accuracy"] <= 1
    assert 0 <= output["decision_tree_accuracy"] <= 1

    assert len(comparison) == 2
    assert comparison["Accuracy"].between(0, 1).all()
