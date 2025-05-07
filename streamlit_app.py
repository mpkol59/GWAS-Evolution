import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt


df1 = pd.read_csv("cleaned_no_decimal_part1.csv")
df2 = pd.read_csv("cleaned_no_decimal_part2.csv")
df = pd.concat([df1, df2], ignore_index=True)

# Drop missing
df = df.dropna(subset=["NUMBER OF INDIVIDUALS", "ASSOCIATION COUNT"])

# Sidebar
st.title("🧬 GWAS Evolution Predictor")
trait = st.text_input("Enter Trait", value="Diabetes")
ancestry = st.text_input("Enter Ancestry", value="African American")
known_value = st.number_input("Enter known sample size:", min_value=1)

# Filter data for training
filtered = df[
    (df["DISEASE/TRAIT"].str.contains(trait, case=False, na=False)) &
    (df["BROAD ANCESTRAL CATEGORY"].str.contains(ancestry, case=False, na=False))
]

if not filtered.empty:
    X = filtered[["NUMBER OF INDIVIDUALS"]]
    y = filtered["ASSOCIATION COUNT"]

    model = RandomForestRegressor(random_state=42)
    model.fit(X, y)

    predicted = model.predict([[known_value]])[0]
    st.success(f"Predicted number of associations: {round(predicted)}")

    # Plot trend
    filtered["DATE"] = pd.to_numeric(filtered["DATE"], errors="coerce")
    trend = (
        filtered.groupby("DATE")["NUMBER OF INDIVIDUALS"]
        .mean()
        .reset_index()
        .rename(columns={"DATE": "Year", "NUMBER OF INDIVIDUALS": "Avg Sample Size"})
    )

    st.subheader("📈 Sample Size Evolution Over Time")
    st.line_chart(trend.set_index("Year"))

    csv = trend.to_csv(index=False).encode("utf-8")
    st.download_button("📥 Download Trend Data", csv, "sample_size_trend.csv", "text/csv")
else:
    st.warning("No matching data for this trait and ancestry.")
