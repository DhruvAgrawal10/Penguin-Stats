import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

def read_csv(file_path):
    return pd.read_csv(file_path)

def normalize_data(df):
    scaler = MinMaxScaler()
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df[numeric_cols] = scaler.fit_transform(df[numeric_cols])
    return df

def calculate_statistics(df):
    stats = {}
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        stats[col] = {
            "mean": df[col].mean(),
            "variance": df[col].var(),
            "std_dev": df[col].std()
        }
    return stats

def plot_scatter(df, col_x, col_y):
    if col_x not in df.columns or col_y not in df.columns:
        raise ValueError("Columns not found in the DataFrame")
    plt.figure(figsize=(8, 6))
    plt.scatter(df[col_x], df[col_y], alpha=0.7, c='blue', edgecolors='k')
    plt.title(f"Scatter Plot: {col_x} vs {col_y}")
    plt.xlabel(col_x)
    plt.ylabel(col_y)
    plt.grid(True)
    plt.show()

def main():
    file_path ="https://raw.githubusercontent.com/mwaskom/seaborn-data/refs/heads/master/penguins.csv"
    col_x = "bill_depth_mm"
    col_y = "body_mass_g"

    df = read_csv(file_path)

    df_normalized = normalize_data(df)
    print("\nNormalized Data:")
    print(df_normalized.head())

    stats = calculate_statistics(df_normalized)
    print("\nBasic Statistics:")
    for col, stat in stats.items():
        print(f"{col}: {stat}")

    plot_scatter(df_normalized, col_x, col_y)

if __name__ == "__main__":
    main()
