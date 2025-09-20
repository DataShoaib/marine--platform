import streamlit as st
import pandas as pd
from ingestion.ingest_csv import ingest_csv
from ingestion.ingest_fasta import ingest_fasta
from models.taxonomy_classifier import train_taxonomy_model
from models.edna_species_matching import match_species
from models.otolith_cnn import build_otolith_model, predict_otolith
from visualization.ocean_trends import plot_ocean_trend
import io

st.set_page_config(page_title="Marine AI Platform", layout="wide")
st.title("🌊 AI-Driven Unified Marine Data Platform")
st.markdown("### Integrated platform for Oceanography, Taxonomy, eDNA, and Otolith Analysis")

menu = st.sidebar.radio("Navigation", ["Upload Data & Reports", "AI Models", "eDNA", "Otolith", "Visualization"])

# ------------------- Helper: Downloadable CSV Report -------------------
def download_csv_report(df, report_name="report.csv"):
    buffer = io.BytesIO()
    df.to_csv(buffer, index=False)
    buffer.seek(0)
    st.download_button(
        label="📥 Download Report",
        data=buffer,
        file_name=report_name,
        mime="text/csv"
    )

# ------------------- Upload Data -------------------
if menu == "Upload Data & Reports":
    uploaded = st.file_uploader("Upload Marine Dataset (CSV) for Analysis", type=["csv"])
    if uploaded:
        df = ingest_csv(uploaded)
        st.subheader("Preview of Uploaded Data")
        st.dataframe(df.head())
        st.info("✅ File uploaded successfully. You can proceed to run models or generate reports.")

# ------------------- AI Models – Taxonomy -------------------
elif menu == "AI Models":
    st.subheader("Taxonomy Prediction for New Fish Samples")
    uploaded_csv = st.file_uploader("Upload CSV of new fish samples", type=["csv"])
    
    if uploaded_csv:
        new_data = pd.read_csv(uploaded_csv)
        st.dataframe(new_data.head())

        # Train or load model
        df_train = pd.read_csv("sample_data/fish_data.csv")
        model, scaler = train_taxonomy_model(df_train)

        X_new = new_data.drop("species", axis=1, errors='ignore')
        X_scaled = scaler.transform(X_new)
        predictions = model.predict(X_scaled)
        
        # Confidence / probability
        probabilities = model.predict_proba(X_scaled)
        confidence = probabilities.max(axis=1)  # max probability for predicted class

        # Add to report
        new_data["predicted_species"] = predictions
        new_data["confidence"] = confidence.round(3)  # rounded for neat report

        st.success("✅ Prediction complete with confidence!")
        st.dataframe(new_data)

        # Download report
        download_csv_report(new_data, report_name="taxonomy_predictions_report.csv")

# ------------------- eDNA Matching -------------------
elif menu == "eDNA":
    st.subheader("Environmental DNA (eDNA) Species Matching")
    fasta_file = st.file_uploader("Upload eDNA Sequences (FASTA)", type=["fasta"])
    if fasta_file:
        seq_df = ingest_fasta(fasta_file)
        # Reference database
        ref_db = pd.DataFrame({
            "species":["SpeciesA","SpeciesB","SpeciesC","SpeciesD","SpeciesE"],
            "sequence":["ACTGTTAG","ACTGGGAC","ACTGTTAC","ACTGGAAG","ACTGTTAA"]
        })
        matches = match_species(seq_df, ref_db)
        st.dataframe(matches)
        download_csv_report(matches, report_name="edna_species_matching_report.csv")

# ------------------- Otolith CNN -------------------
elif menu == "Otolith":
    st.subheader("Otolith Morphology Classification")
    otolith_file = st.file_uploader("Upload Otolith Image (PNG/JPG)", type=["png","jpg"])
    if otolith_file:
        model = build_otolith_model()
        labels = ["SpeciesA","SpeciesB","SpeciesC","SpeciesD","SpeciesE"]
        pred = predict_otolith(model, otolith_file, labels)
        st.success(f"Predicted Species: {pred}")

        # Create a professional text report
        report_text = f"""
        🐟 Otolith Prediction Report
        ----------------------------
        File: {otolith_file.name}
        Predicted Species: {pred}
        """
        st.download_button(
            label="📥 Download Otolith Report",
            data=report_text,
            file_name="otolith_prediction_report.txt",
            mime="text/plain"
        )

# ------------------- Visualization -------------------
elif menu == "Visualization":
    st.subheader("Oceanographic Data Trends")
    df = pd.read_json("sample_data/ocean_params.json")
    img_path = plot_ocean_trend(df, param="temperature")
    st.image(img_path, caption="Ocean Temperature Trend")
    
    with open(img_path, "rb") as f:
        st.download_button(
            label="📥 Download Plot Image",
            data=f,
            file_name="ocean_temperature_trend.png",
            mime="image/png"
        )
