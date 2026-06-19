import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt



st.set_page_config(
    page_title="AI-Powered Heart Disease Dashboard",
    page_icon="",
    layout="wide"
)



df = pd.read_csv("heart.csv")



st.title(" AI-Powered Heart Disease Analysis Dashboard")

st.markdown("""
Analyze patient health data, discover hidden patterns,
visualize correlations, and generate AI-powered insights
for heart disease prediction.
""")

# SUMMARY

st.header("📋 Dataset Summary")

c1, c2, c3, c4 = st.columns(4)

c1.metric("Patients", len(df))
c2.metric("Features", len(df.columns))
c3.metric("Heart Disease Cases", int(df["target"].sum()))
c4.metric("No Disease Cases", int(len(df) - df["target"].sum()))



st.sidebar.header("Dashboard Controls")

st.sidebar.write(f"Rows: {df.shape[0]}")
st.sidebar.write(f"Columns: {df.shape[1]}")


# DATA PREVIEW


st.header("📄 Dataset Preview")

rows = st.slider(
    "Select Rows to View",
    min_value=5,
    max_value=50,
    value=10
)

st.dataframe(df.head(rows))



st.header("Dataset Information")

info_df = pd.DataFrame({
    "Column": df.columns,
    "Data Type": df.dtypes.astype(str)
})

st.dataframe(info_df)


# DATASET STATISTICS


st.header("📈 Dataset Statistics")

col1, col2, col3, col4 = st.columns(4)

col1.metric("Rows", df.shape[0])
col2.metric("Columns", df.shape[1])
col3.metric("Missing Values", int(df.isnull().sum().sum()))
col4.metric("Duplicate Rows", int(df.duplicated().sum()))

st.subheader("Statistical Summary")
st.dataframe(df.describe())


# MISSING VALUE HANDLING


st.header("🛠 Missing Value Handling")

missing = df.isnull().sum()

st.dataframe(missing)

if missing.sum() > 0:

    option = st.selectbox(
        "Choose Cleaning Method",
        [
            "Drop Rows",
            "Fill Mean",
            "Fill Median"
        ]
    )

    if st.button("Apply Cleaning"):

        if option == "Drop Rows":
            df = df.dropna()

        elif option == "Fill Mean":

            numeric_cols = df.select_dtypes(
                include=np.number
            ).columns

            for col in numeric_cols:
                df[col] = df[col].fillna(df[col].mean())

        elif option == "Fill Median":

            numeric_cols = df.select_dtypes(
                include=np.number
            ).columns

            for col in numeric_cols:
                df[col] = df[col].fillna(df[col].median())

        st.success("Dataset cleaned successfully!")

else:
    st.success("✅ No missing values found in dataset.")


# TARGET DISTRIBUTION


st.header(" Heart Disease Distribution")

fig, ax = plt.subplots()

sns.countplot(
    x="target",
    data=df,
    ax=ax
)

ax.set_title("Heart Disease Target Distribution")

st.pyplot(fig)


# PIE CHART


st.subheader("Heart Disease Percentage")

fig, ax = plt.subplots()

df["target"].value_counts().plot(
    kind="pie",
    autopct="%1.1f%%",
    ax=ax
)

ax.set_ylabel("")

st.pyplot(fig)


# VISUALIZATIONS


st.header("📊 Interactive Visualizations")

numeric_cols = df.select_dtypes(
    include=np.number
).columns

chart_type = st.selectbox(
    "Select Chart Type",
    [
        "Histogram",
        "Box Plot",
        "Scatter Plot"
    ]
)

if chart_type == "Histogram":

    col = st.selectbox(
        "Choose Column",
        numeric_cols
    )

    fig, ax = plt.subplots()

    sns.histplot(
        df[col],
        kde=True,
        ax=ax
    )

    st.pyplot(fig)

elif chart_type == "Box Plot":

    col = st.selectbox(
        "Choose Column",
        numeric_cols
    )

    fig, ax = plt.subplots()

    sns.boxplot(
        x=df[col],
        ax=ax
    )

    st.pyplot(fig)

elif chart_type == "Scatter Plot":

    x_col = st.selectbox(
        "Select X-axis",
        numeric_cols
    )

    y_col = st.selectbox(
        "Select Y-axis",
        numeric_cols,
        index=1
    )

    fig, ax = plt.subplots()

    sns.scatterplot(
        x=df[x_col],
        y=df[y_col],
        ax=ax
    )

    st.pyplot(fig)


# FEATURE VS TARGET


st.header(" Feature vs Heart Disease")

feature = st.selectbox(
    "Select Feature",
    numeric_cols
)

fig, ax = plt.subplots(figsize=(8, 5))

sns.boxplot(
    x="target",
    y=feature,
    data=df,
    ax=ax
)

st.pyplot(fig)


# CORRELATION MATRIX


st.header(" Correlation Analysis")

corr = df.corr(numeric_only=True)

fig, ax = plt.subplots(
    figsize=(12, 8)
)

sns.heatmap(
    corr,
    annot=True,
    cmap="coolwarm",
    ax=ax
)

st.pyplot(fig)


# TARGET CORRELATION


st.subheader(
    "Features Correlated With Heart Disease"
)

target_corr = corr["target"].sort_values(
    ascending=False
)

st.dataframe(target_corr)





corr_pairs = (
    corr.abs()
    .unstack()
    .sort_values(ascending=False)
)

corr_pairs = corr_pairs[corr_pairs < 1]

strongest = corr_pairs.head(1)

st.success(
    f"Strongest Correlation: "
    f"{strongest.index[0][0]} ↔ "
    f"{strongest.index[0][1]} "
    f"({strongest.values[0]:.2f})"
)


# AI GENERATED INSIGHTS


st.header("🤖 AI-Generated Insights")

avg_age = round(df["age"].mean(), 2)

disease_percent = round(
    (df["target"].sum() / len(df)) * 100,
    2
)

highest_corr = (
    corr["target"]
    .drop("target")
    .abs()
    .idxmax()
)

corr_value = corr["target"][highest_corr]

st.info(
    f"Average patient age is {avg_age} years."
)

st.info(
    f"{disease_percent}% of patients show signs of heart disease."
)

st.success(
    f"'{highest_corr}' is the feature most strongly associated with heart disease "
    f"(correlation = {corr_value:.2f})."
)

st.warning(
    "Chest pain type, maximum heart rate, and ST depression are among the most important indicators."
)



st.header("⬇ Download Dataset")

csv = df.to_csv(
    index=False
).encode("utf-8")

st.download_button(
    label="Download CSV",
    data=csv,
    file_name="cleaned_heart_dataset.csv",
    mime="text/csv"
)
