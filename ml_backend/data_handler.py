import pandas as pd


def process_file(file):
    if file is None:
        return "No file uploaded"

    try:
        df = pd.read_csv(file)
        X, Y = clean_and_parse(df)

    except Exception as e:
        return f"The file has bad format or is corrupt, error: {e.args}"

    return df, X, Y


def clean_and_parse(df: pd.DataFrame):
    df = df.iloc[:].astype(float)
    if df.isnull().values.any():
        raise ValueError("There are null values.")

    X = df.iloc[:, :-1].values.T
    Y = df.iloc[:, -1].values.reshape(1, -1)

    return X, Y


if __name__ == "__main__":
    print(process_file("sample_data/sample_training_data.csv"))
