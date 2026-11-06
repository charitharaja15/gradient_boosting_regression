
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# -----------------------------------
# PAGE CONFIGURATION
# -----------------------------------

st.set_page_config(
    page_title="Customer Lifetime Value Prediction",
    page_icon="📊",
    layout="wide"
)

# -----------------------------------
# CUSTOM CSS
# -----------------------------------

st.markdown("""
<style>
.main {
    background-color: #f5f7fb;
}
.metric-card {
    background-color: #ffffff;
    padding: 20px;
    border-radius: 12px;
    box-shadow: 0px 2px 8px rgba(0,0,0,0.08);
}
h1 {
    color: #17365D;
}
</style>
""", unsafe_allow_html=True)

# -----------------------------------
# HEADER
# -----------------------------------

st.title("📊 Customer Lifetime Value Estimation")

st.subheader("Gradient Boosting Regression-Based Prediction System")

st.write("""
This application predicts Customer Lifetime Value (CLV)
using customer purchase history and machine learning.
It helps businesses understand customer value and
make better marketing decisions.
""")

st.divider()

# -----------------------------------
# SIDEBAR
# -----------------------------------

st.sidebar.title("Project Information")

st.sidebar.info("""
**Project Domain:** Machine Learning & Predictive Analytics

**Algorithm:** Gradient Boosting Regression

**Application:** Customer Lifetime Value Prediction
""")

st.sidebar.write("Upload your customer dataset to begin.")

# -----------------------------------
# DATASET UPLOAD
# -----------------------------------

st.header("1. Upload Customer Dataset")

uploaded_file = st.file_uploader(
    "Upload your training dataset (CSV)",
    type=["csv"]
)

if uploaded_file is not None:

    try:
        df = pd.read_csv(uploaded_file)

        st.success("Dataset uploaded successfully!")

        st.subheader("Dataset Preview")
        st.dataframe(df.head(10), use_container_width=True)

        # Dataset information
        col1, col2, col3 = st.columns(3)

        col1.metric("Total Records", len(df))
        col2.metric("Total Columns", len(df.columns))
        col3.metric("Missing Values", int(df.isnull().sum().sum()))

        st.subheader("Dataset Statistics")
        st.dataframe(df.describe(include="all").T,
                     use_container_width=True)

        # -----------------------------------
        # TARGET COLUMN
        # -----------------------------------

        st.header("2. Select Target Variable")

        target_column = st.selectbox(
            "Select Customer Lifetime Value column",
            options=df.columns
        )

        if st.button("Train Gradient Boosting Model",
                     type="primary"):

            with st.spinner("Training machine learning model..."):

                data = df.copy()

                # Convert target to numeric
                data[target_column] = pd.to_numeric(
                    data[target_column], errors="coerce"
                )

                # Remove rows with missing target
                data = data.dropna(subset=[target_column])

                if len(data) < 10:
                    st.error(
                        "At least 10 valid records are required for training."
                    )
                    st.stop()

                X = data.drop(columns=[target_column])
                y = data[target_column]

                if X.shape[1] == 0:
                    st.error("Please select a valid target column.")
                    st.stop()

                # Convert date and other unsupported values to strings
                for col in X.columns:
                    if (
                        pd.api.types.is_object_dtype(X[col])
                        or pd.api.types.is_datetime64_any_dtype(X[col])
                    ):
                        X[col] = X[col].astype("string")

                # Identify numeric and categorical columns
                numeric_features = X.select_dtypes(
                    include=np.number
                ).columns.tolist()

                categorical_features = [
                    col for col in X.columns
                    if col not in numeric_features
                ]

                # Numeric preprocessing
                numeric_transformer = Pipeline(steps=[
                    ("imputer", SimpleImputer(strategy="median"))
                ])

                # Categorical preprocessing
                categorical_transformer = Pipeline(steps=[
                    ("imputer", SimpleImputer(
                        strategy="constant",
                        fill_value="Unknown"
                    )),
                    ("onehot", OneHotEncoder(
                        handle_unknown="ignore",
                        sparse_output=False
                    ))
                ])

                # Combine preprocessing
                transformers = []

                if numeric_features:
                    transformers.append(
                        ("num", numeric_transformer, numeric_features)
                    )

                if categorical_features:
                    transformers.append(
                        ("cat", categorical_transformer,
                         categorical_features)
                    )

                preprocessor = ColumnTransformer(
                    transformers=transformers,
                    sparse_threshold=0
                )

                # Gradient Boosting model
                model = Pipeline(steps=[
                    ("preprocessor", preprocessor),
                    ("regressor", GradientBoostingRegressor(
                        n_estimators=100,
                        learning_rate=0.05,
                        max_depth=3,
                        random_state=42
                    ))
                ])

                # Split dataset
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y,
                    test_size=0.2,
                    random_state=42
                )

                # Train model
                model.fit(X_train, y_train)

                # Predictions
                y_pred = model.predict(X_test)

                # Evaluation
                mae = mean_absolute_error(y_test, y_pred)
                rmse = np.sqrt(mean_squared_error(y_test, y_pred))
                r2 = r2_score(y_test, y_pred)

                # Save model and training information
                st.session_state["model"] = model
                st.session_state["features"] = X.columns.tolist()
                st.session_state["target"] = target_column
                st.session_state["test_actual"] = y_test.to_numpy()
                st.session_state["test_predicted"] = y_pred

            st.success("Model trained successfully!")

        # -----------------------------------
        # MODEL PERFORMANCE
        # -----------------------------------

        if "model" in st.session_state:

            st.header("3. Model Performance")

            actual = st.session_state["test_actual"]
            predicted = st.session_state["test_predicted"]

            mae = mean_absolute_error(actual, predicted)
            rmse = np.sqrt(mean_squared_error(actual, predicted))
            r2 = r2_score(actual, predicted)

            c1, c2, c3 = st.columns(3)

            c1.metric("MAE", f"{mae:,.2f}")
            c2.metric("RMSE", f"{rmse:,.2f}")
            c3.metric("R² Score", f"{r2:.4f}")

            st.caption(
                "MAE and RMSE measure prediction errors. "
                "R² measures how well the model explains variation "
                "in the target variable."
            )

            # -----------------------------------
            # ACTUAL VS PREDICTED
            # -----------------------------------

            st.header("4. Actual vs Predicted CLV")

            fig, ax = plt.subplots(figsize=(9, 5))

            ax.scatter(
                actual, predicted,
                alpha=0.7
            )

            minimum = min(np.min(actual), np.min(predicted))
            maximum = max(np.max(actual), np.max(predicted))

            ax.plot(
                [minimum, maximum],
                [minimum, maximum],
                linestyle="--"
            )

            ax.set_xlabel("Actual Customer Lifetime Value")
            ax.set_ylabel("Predicted Customer Lifetime Value")
            ax.set_title("Actual vs Predicted CLV")

            st.pyplot(fig)
            plt.close(fig)

            # -----------------------------------
            # PREDICTION RESULTS
            # -----------------------------------

            st.header("5. Customer CLV Prediction")

            st.write(
                "Upload a separate CSV file containing new customers "
                "with the same feature columns used during training."
            )

            new_file = st.file_uploader(
                "Upload new customer data",
                type=["csv"],
                key="new_customers"
            )

            if new_file is not None:

                new_data = pd.read_csv(new_file)

                features = st.session_state["features"]

                missing = [
                    col for col in features
                    if col not in new_data.columns
                ]

                if missing:
                    st.error(
                        "Missing required feature columns: "
                        + ", ".join(missing)
                    )

                else:
                    new_X = new_data[features].copy()

                    for col in new_X.columns:
                        if (
                            pd.api.types.is_object_dtype(new_X[col])
                            or pd.api.types.is_datetime64_any_dtype(new_X[col])
                        ):
                            new_X[col] = new_X[col].astype("string")

                    new_predictions = st.session_state[
                        "model"
                    ].predict(new_X)

                    results = new_data.copy()

                    results["Predicted_CLV"] = new_predictions

                    st.success("Customer CLV predictions generated!")

                    st.dataframe(
                        results,
                        use_container_width=True
                    )

                    csv = results.to_csv(index=False).encode("utf-8")

                    st.download_button(
                        label="Download Prediction Results",
                        data=csv,
                        file_name="customer_clv_predictions.csv",
                        mime="text/csv"
                    )

    except Exception as e:
        st.error(f"An error occurred: {e}")

else:
    st.info("Please upload your customer dataset in CSV format.")

# -----------------------------------
# FOOTER
# -----------------------------------

st.divider()

st.markdown("""
<center>
**Customer Lifetime Value Estimation**

Machine Learning & Predictive Analytics

Developed using Python, Streamlit and Gradient Boosting Regression
</center>
""", unsafe_allow_html=True)

