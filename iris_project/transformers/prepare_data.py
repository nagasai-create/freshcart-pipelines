
# Import Mage decorators
# Fallback decorators allow importing outside Mage.

try:
    from mage_ai.data_preparation.decorators import transformer, test

except ImportError:

    def transformer(func):
        return func

    def test(func):
        return func


from sklearn.model_selection import train_test_split


@transformer
def transform(df, *args, **kwargs):

    # Split dataset into training and testing indices
    train_indices, test_indices = train_test_split(
        df.index,
        test_size=0.3,
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
