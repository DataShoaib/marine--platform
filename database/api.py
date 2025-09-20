from fastapi import FastAPI
import sqlite3
import json

app = FastAPI()

@app.get("/ocean_data/{record_id}")
def get_record(record_id: str):
    conn = sqlite3.connect("marine_data.db")
    cursor = conn.cursor()
    cursor.execute("SELECT data FROM ocean_data WHERE record_id=?", (record_id,))
    row = cursor.fetchone()
    conn.close()
    return {"record_id": record_id, "data": json.loads(row[0]) if row else None}
