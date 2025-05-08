import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt

# === Load and concatenate CSV parts ===
files = ['part1.csv', 'part2.csv', 'part3.csv', 'part4.csv', 'part5.csv']
dfs = [pd.read_csv(files[0])] + [pd.read_csv(f, skiprows=1) for f in files[1:]]
df = pd.concat(dfs, ignore_index=True)

# Drop missing values
df = df.dropna(subset=["NUMBER OF INDIVIDUALS", "ASSOCIATION COUNT"])

# === UI ===
st.title("🧬 GWAS Evolution Predictor")
trait = st.text_input("Enter Trait", value="Diabetes")
ancestry = st.text_input("Enter Ancestry", value="African American")
option = st.radio("What do you want to predict?", ("Associations", "Sample Size"))
input_value = st.number_input("Enter known value:", min_value=1)

# === Filter data ===
filtered = df[
    (df["DISEASE/TRAIT"].str.contains(trait, case=False, na=False)) &
    (df["BROAD ANCESTRAL CATEGORY"].str.contains(ancestry, case=False, na=False))
]

if not filtered.empty:
    X_assoc = filtered[["NUMBER OF INDIVIDUALS"]]
    y_assoc = filtered["ASSOCIATION COUNT"]
    X_sample = filtered[["ASSOCIATION COUNT"]]
    y_sample = filtered["NUMBER OF INDIVIDUALS"]

    model_assoc = RandomForestRegressor(random_state=42)
    model_sample = RandomForestRegressor(random_state=42)

    model_assoc.fit(X_assoc, y_assoc)
    model_sample.fit(X_sample, y_sample)

    if option == "Associations":
        prediction = model_assoc.predict([[input_value]])[0]
        st.success(f"Predicted number of associations: {round(prediction)}")
    else:
        prediction = model_sample.predict([[input_value]])[0]
        st.success(f"Predicted sample size needed: {round(prediction)}")

    # === Plot trends ===
    filtered["DATE"] = pd.to_numeric(filtered["DATE"], errors="coerce")

    if option == "Associations":
        trend = (
            filtered.groupby("DATE")["NUMBER OF INDIVIDUALS"]
            .mean()
            .reset_index()
            .rename(columns={"DATE": "Year", "NUMBER OF INDIVIDUALS": "Avg Sample Size"})
        )
        st.subheader("📈 Sample Size Evolution Over Time")
    else:
        trend = (
            filtered.groupby("DATE")["ASSOCIATION COUNT"]
            .mean()
            .reset_index()
            .rename(columns={"DATE": "Year", "ASSOCIATION COUNT": "Avg Associations"})
        )
        st.subheader("📈 Association Count Evolution Over Time")

    st.line_chart(trend.set_index("Year"))

    # === Download CSV ===
    csv = trend.to_csv(index=False).encode("utf-8")
    filename = "trend_data.csv" if option == "Associations" else "trend_data_association.csv"
    st.download_button("📥 Download Trend Data", csv, filename, "text/csv")

else:
    st.warning("No matching data for this trait and ancestry.")
