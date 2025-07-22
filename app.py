import streamlit as st
import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

st.set_page_config(page_title="ILP Consumption Forecast", layout="wide")
st.title("ILP Consumption Weight Forecasting App (XGBoost)")

# Step 0: Upload multiple monthly CSV files
uploaded_files = st.file_uploader("Upload Monthly CSV Files", type="csv", accept_multiple_files=True)

if uploaded_files:
    df_list = []
    for file in uploaded_files:
        df = pd.read_csv(file)

        # Ensure required columns exist
        required_cols = [
            'Posting Date', 'Work center', 'Customer Segmant', 'Grade Series number',
            'Thickness (mm)', 'Output Weight (101)', 'ILP Usage', 'ILP Weight'
        ]
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            st.error(f"Missing columns in file {file.name}: {missing_cols}")
            st.stop()

        # Step 1: Filter only FILP/Fresh_ILP rows and drop NaNs
        df = df[df['ILP Usage'].isin(['FILP', 'Fresh_ILP'])].copy()
        df = df.dropna(subset=required_cols)

        # Step 2: Calculate ILP Tonnage Weight
        df['ILP Tonnage Weight'] = (df['Output Weight (101)'] * 0.035) / (df['Thickness (mm)'] * 7.85)

        # Step 3: Drop original ILP Weight
        df.drop(columns=['ILP Weight'], inplace=True)

        df_list.append(df)

    # Combine all months
    df_year = pd.concat(df_list, ignore_index=True)
    st.success(f"Merged {len(df_year)} rows from {len(uploaded_files)} files.")

    forecast_days = st.number_input("Enter number of days to forecast:", min_value=1, max_value=60, value=17)


    # ================== Forecasting Starts Here ==================

    # Preprocessing
    df = df_year.copy()
    df['Posting Date'] = pd.to_datetime(df['Posting Date'])
    df = df.sort_values('Posting Date') 

    # Encode categorical features
    for col in ['Customer Segmant', 'Grade Series number']:
        df[col] = LabelEncoder().fit_transform(df[col].astype(str))

    # Group per day
    df_daily = df.groupby('Posting Date').agg({
        'ILP Tonnage Weight': 'sum',
        'Thickness (mm)': 'mean',
        'Output Weight (101)': 'sum',
        'Customer Segmant': 'mean',
        'Grade Series number': 'mean',
        'Work center': 'nunique'
    }).reset_index()

    # Create lag features
    for lag in range(1, 15):
        df_daily[f'lag_{lag}'] = df_daily['ILP Tonnage Weight'].shift(lag)

    df_daily.dropna(inplace=True)

    # Train-test split
    X = df_daily.drop(columns=['Posting Date', 'ILP Tonnage Weight'])
    y = df_daily['ILP Tonnage Weight']

    # Train XGBoost
    model = XGBRegressor(n_estimators=1000, learning_rate=0.05, max_depth=4, random_state=42)
    model.fit(X,y)

    # Evaluate
    y = model.predict(X)

    # Forecast 30 days
    future_preds = []
    last_row = df_daily.iloc[-1:].copy()

    for _ in range(forecast_days):
        input_features = last_row.drop(columns=['Posting Date', 'ILP Tonnage Weight']).copy()
        pred = model.predict(input_features)[0]
        future_preds.append(pred)

        # Slide lag window
        for lag in range(14, 1, -1):
            last_row[f'lag_{lag}'] = last_row[f'lag_{lag - 1}']
        last_row['lag_1'] = pred
        last_row['ILP Tonnage Weight'] = pred
        last_row['Posting Date'] += pd.Timedelta(days=1)

    # Show forecast
    future_dates = pd.date_range(start=df_daily['Posting Date'].max() + pd.Timedelta(days=1), periods=forecast_days)
    df_forecast = pd.DataFrame({
        'Posting Date': future_dates,
        'Forecasted ILP Tonnage Weight': future_preds
    })


    st.write(f"📦 **Total Forecasted ILP Tonnage Weight (Next {forecast_days} days)**")
    st.success(f"{df_forecast['Forecasted ILP Tonnage Weight'].sum():.2f} tons")

    # Show forecast table
    st.subheader(f"📅 Next {forecast_days} days Forecast")
    st.dataframe(df_forecast)

    # 📈 Add Chart
    import matplotlib.pyplot as plt

    st.subheader("📈 Forecast Trend")

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(df_forecast['Posting Date'], df_forecast['Forecasted ILP Tonnage Weight'],
            marker='o', linestyle='-', color='blue', label='Forecast')

    ax.set_title(f"Forecasted ILP Tonnage Weight for Next {forecast_days} days", fontsize=16)
    ax.set_xlabel("Date", fontsize=12)
    ax.set_ylabel("ILP Tonnage Weight (tons)", fontsize=12)

    plt.xticks(rotation=45, ha='right')
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.legend(loc='upper right', fontsize=10)
    fig.tight_layout()

    st.pyplot(fig)

    # 📦 Total Forecast
    st.write(f"📦 **Total Forecasted ILP Tonnage Weight (Next {forecast_days} days)**")
    st.success(f"{df_forecast['Forecasted ILP Tonnage Weight'].sum():.2f} tons")
