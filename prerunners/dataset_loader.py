import pandas as pd

def dataset_loader(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    return df