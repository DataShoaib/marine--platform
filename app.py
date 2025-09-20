import streamlit as st
import pandas as pd
import numpy as np
import io
import pydeck as pdk

from ingestion.ingest_csv import ingest_csv
from ingestion.ingest_fasta import ingest_fasta
from models.taxonomy_classifier import train_taxonomy_model
from models.edna_species_matching import match_species
from models.otolith_cnn import build_otolith_model, predict_otolith
from visualization.ocean_trends import plot_ocean_trend

st.set_page_config(page_title="Marine AI Platform", layout="wide")
st.title("🌊 AI-Driven Unified Marine Data Platform")
st.markdown("### Integrated platform for Oceanography, Taxonomy, eDNA, Otolith, and Fish Abundance Map")

menu = st.sidebar.radio(
    "Navigation", 
    ["Upload Data & Reports", "AI Models", "eDNA", "Otolith", "Visualization", "Fish Abundance Map"]
)

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
        # Convert categorical to numeric if needed
        X_new = pd.get_dummies(X_new)
        X_scaled = scaler.transform(X_new)
        predictions = model.predict(X_scaled)
        probabilities = model.predict_proba(X_scaled)
        confidence = probabilities.max(axis=1)

        new_data["predicted_species"] = predictions
        new_data["confidence"] = confidence.round(3)

        st.success("✅ Prediction complete with confidence!")
        st.dataframe(new_data)
        download_csv_report(new_data, report_name="taxonomy_predictions_report.csv")

# ------------------- eDNA Matching -------------------
elif menu == "eDNA":
    st.subheader("Environmental DNA (eDNA) Species Matching")
    fasta_file = st.file_uploader("Upload eDNA Sequences (FASTA)", type=["fasta"])
    if fasta_file:
        seq_df = ingest_fasta(fasta_file)
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
    st.subheader("Oceanographic Data & Fish Abundance Map")
    
    # Existing ocean trend
    df_ocean = pd.read_json("sample_data/ocean_params.json")
    img_path = plot_ocean_trend(df_ocean, param="temperature")
    st.image(img_path, caption="Ocean Temperature Trend")
    
    with open(img_path, "rb") as f:
        st.download_button(
            label="📥 Download Ocean Plot Image",
            data=f,
            file_name="ocean_temperature_trend.png",
            mime="image/png"
        )
    
    # ------------------- Fish Abundance Map -------------------
    st.markdown("---")
    st.subheader("Fish Abundance Map")
    
    uploaded_map_csv = st.file_uploader("Upload Fish Abundance CSV", type=["csv"])
    if uploaded_map_csv:
        df_fish = pd.read_csv(uploaded_map_csv)
        
        import folium
        from streamlit_folium import st_folium

        # Create folium map centered on average lat/lon
        map_center = [df_fish['latitude'].mean(), df_fish['longitude'].mean()]
        fish_map = folium.Map(location=map_center, zoom_start=6)

        # Add circle markers for each fish data point
        for idx, row in df_fish.iterrows():
            folium.CircleMarker(
                location=[row['latitude'], row['longitude']],
                radius=5 + row['count']/20,  # size proportional to count
                color='red',
                fill=True,
                fill_opacity=0.6,
                popup=f"Species: {row['species']}<br>Count: {row['count']}"
            ).add_to(fish_map)
        
        # Display map in Streamlit
        st_folium(fish_map, width=700, height=500)
        
        # Download map as HTML
        map_html = "fish_abundance_map.html"
        fish_map.save(map_html)
        with open(map_html, "rb") as f:
            st.download_button(
                label="📥 Download Map as HTML",
                data=f,
                file_name="fish_abundance_map.html",
                mime="text/html"
            )


# ------------------- Fish Abundance Map -------------------
elif menu == "Fish Abundance Map":
    st.subheader("Fish Abundance Locations")
    uploaded_csv = st.file_uploader("Upload CSV with 'latitude', 'longitude', 'count'", type=["csv"])
    
    if uploaded_csv:
        map_data = pd.read_csv(uploaded_csv)
        st.dataframe(map_data.head())

        # Use PyDeck for map
        st.pydeck_chart(pdk.Deck(
            map_style='mapbox://styles/mapbox/light-v9',
            initial_view_state=pdk.ViewState(
                latitude=15.0,
                longitude=75.0,
                zoom=5,
                pitch=0
            ),
            layers=[
                pdk.Layer(
                    "ScatterplotLayer",
                    data=map_data,
                    get_position='[longitude, latitude]',
                    get_color='[200, 30, 0, 160]',
                    get_radius='count * 200',  # scale radius by count
                    pickable=True
                )
            ]
        ))
        st.info("📍 Map shows fish abundance with circle size proportional to count.")
