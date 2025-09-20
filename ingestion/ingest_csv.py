import pandas as pd

def ingest_csv(path: str) -> pd.DataFrame:
    try:
        df = pd.read_csv(path)
        print(f"[INFO] CSV ingested: {path}, Shape={df.shape}")
        return df
    except Exception as e:
        print(f"[ERROR] Failed CSV ingestion: {e}")
        return pd.DataFrame()
