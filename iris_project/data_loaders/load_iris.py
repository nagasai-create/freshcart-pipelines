
from sklearn.datasets import load_iris as sklearn_load_iris

if 'data_loader' not in globals():
    from mage_ai.data_preparation.decorators import data_loader

if 'test' not in globals():
    from mage_ai.data_preparation.decorators import test


@data_loader
def load_iris(*args, **kwargs):
    """
    Load the Iris dataset from scikit-learn.
    Returns a DataFrame containing 150 rows and 5 columns.
    """
    iris = sklearn_load_iris(as_frame=True)
    return iris.frame


@test
def test_output(output, *args):
    """
    Validate the loaded Iris dataset.
    """
    assert output is not None, "Dataset is empty"
    assert output.shape == (150, 5), "Expected 150 rows and 5 columns"
    assert "target" in output.columns, "Target column is missing"
    assert output["target"].nunique() == 3, "Expected exactly 3 classes"
