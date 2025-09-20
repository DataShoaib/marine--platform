import pandas as pd
from difflib import SequenceMatcher

def match_species(seq_df: pd.DataFrame, reference_db: pd.DataFrame, threshold=0.8):
    matches = []
    for _, query in seq_df.iterrows():
        best_match, best_score = None, 0
        for _, ref in reference_db.iterrows():
            score = SequenceMatcher(None, query["sequence"], ref["sequence"]).ratio()
            if score > best_score:
                best_match, best_score = ref["species"], score
        if best_score >= threshold:
            matches.append({"query_id": query["id"], "species": best_match, "score": best_score})
    return pd.DataFrame(matches)
