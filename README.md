                 ┌────────────────────────┐
                 │  User Opens Platform   │
                 │   (Streamlit App)     │
                 └───────────┬──────────┘
                             │
             ┌───────────────┴─────────────────┐
             │                                 │
 ┌─────────────────────┐             ┌─────────────────────┐
 │ Upload Data & Files │             │ Navigation Sidebar  │
 │ CSV / JSON / FASTA │             │  Taxonomy / eDNA /  │
 │  Otolith Images     │             │ Otolith / Visualization │
 └─────────┬──────────┘             └─────────┬───────────┘
           │                                    │
           ▼                                    ▼
 ┌───────────────────────────┐        ┌────────────────────────────┐
 │ Data Ingestion & Preview  │        │ Select AI Model or Visuals │
 │ - CSV → Pandas DataFrame  │        └───────────┬────────────────┘
 │ - JSON → Readable format  │                    │
 │ - FASTA → Seq DataFrame   │                    │
 │ - Images → PIL / NumPy    │                    │
 └─────────┬─────────────────┘                    │
           │                                      │
           ▼                                      ▼
 ┌───────────────────────────┐       ┌─────────────────────────────┐
 │  Data Processing / ML      │       │ Visualization / Plots        │
 │ - Taxonomy Prediction      │       │ - Ocean Temp, Salinity, etc │
 │   (RandomForest)           │       │ - Trend Charts / Graphs     │
 │ - eDNA Species Matching    │       └─────────────┬─────────────┘
 │   (Sequence Comparison)    │                     │
 │ - Otolith Classification   │                     │
 │   (CNN on uploaded images) │                     │
 └─────────┬─────────────────┘                     │
           │                                       │
           ▼                                       ▼
 ┌───────────────────────────┐        ┌─────────────────────────────┐
 │ Generate Prediction Report │        │ Display Interactive Charts  │
 │ - CSV / TXT download      │        │ Users can view / analyze    │
 │ - Include confidence /    │        │ trends in real time         │
 │   predicted species       │        └─────────────────────────────┘
 └─────────┬─────────────────┘
           │
           ▼
 ┌───────────────────────────┐
 │  Scientist Downloads       │
 │  CSV / TXT Report          │
 │ - Can share / store locally│
 └───────────────────────────┘
