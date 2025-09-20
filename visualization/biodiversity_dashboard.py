import matplotlib.pyplot as plt

def biodiversity_dashboard(df, species_col="species"):
    counts = df[species_col].value_counts()
    plt.figure(figsize=(8,4))
    counts.plot(kind="bar")
    plt.title("Species Distribution")
    plt.xlabel("Species")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig("biodiversity.png")
    return "biodiversity.png"
