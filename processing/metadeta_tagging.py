import hashlib

def add_metadata(df, source="unknown"):
    df["source"] = source
    df["record_id"] = df.apply(lambda row: hashlib.md5(str(row.values).encode()).hexdigest(), axis=1)
    return df
