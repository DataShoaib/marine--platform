import pandas as pd
import json

def ingest_json(path: str) -> pd.DataFrame:
    try:
        with open(path) as f:
            data = json.load(f)
        df = pd.DataFrame(data)
        print(f"[INFO] JSON ingested: {path}, Shape={df.shape}")
        return df
    except Exception as e:
        print(f"[ERROR] Failed JSON ingestion: {e}")
        return pd.DataFrame()
