import matplotlib.pyplot as plt

def plot_ocean_trend(df, param="temperature"):
    plt.figure(figsize=(8,4))
    plt.plot(df["date"], df[param])
    plt.title(f"{param.capitalize()} Over Time")
    plt.xlabel("Date")
    plt.ylabel(param.capitalize())
    plt.grid()
    plt.tight_layout()
    plt.savefig("trend.png")
    return "trend.png"
