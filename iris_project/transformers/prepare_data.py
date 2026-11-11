
# Import Mage decorators
# Fallback decorators allow importing outside Mage.


if 'transformer' not in globals():
    try:
        from mage_ai.data_preparation.decorators import transformer
    except ImportError:
        def transformer(func):
            return func

if 'test' not in globals():
    try:
        from mage_ai.data_preparation.decorators import test
    except ImportError:
        def test(func):
            return func



from sklearn.model_selection import train_test_split


@transformer
def transform(df, *args, **kwargs):

    # Split dataset into training and testing indices
    train_indices, test_indices = train_test_split(
        df.index,
        test_size=0.2,
        random_state=42,
        stratify=df["target"]
    )

    # Copy original dataframe
    output = df.copy()

    # Assign training label
    output["split"] = "train"

    # Assign testing label
    output.loc[test_indices, "split"] = "test"

    return output

@test
def test_output(output, *args):
    """
    Check that the dataset is correctly split into 80% training
    and 20% testing data.
    """
    assert output is not None, "Output is empty"
    assert "split" in output.columns, "Split column is missing"

    train_count = (output["split"] == "train").sum()
    test_count = (output["split"] == "test").sum()

    assert train_count == 120, "Expected 120 training rows"
    assert test_count == 30, "Expected 30 testing rows"
    assert output["split"].notna().all(), "Some rows have no split"
