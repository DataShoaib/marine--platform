from Bio import SeqIO
import pandas as pd

def ingest_fasta(path: str) -> pd.DataFrame:
    records = []
    for record in SeqIO.parse(path, "fasta"):
        records.append({"id": record.id, "sequence": str(record.seq)})
    df = pd.DataFrame(records)
    print(f"[INFO] FASTA ingested: {len(df)} sequences")
    return df
