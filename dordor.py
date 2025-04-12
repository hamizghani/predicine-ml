import pandas as pd
import xgboost as xgb
import math
from sklearn.preprocessing import LabelEncoder
import joblib

users = pd.read_csv("users.csv")
medicine = pd.read_csv("medicine.csv")
medicinedisease = pd.read_csv("medicinedisease.csv")
transaction_history = pd.read_csv("transaction_history.csv")
disease = pd.read_csv("disease.csv")

def generate_features(users, medicine, medicinedisease, disease, transaction_history):
    transaction_history["timestamp"] = pd.to_datetime(transaction_history["timestamp"])
    cpy = transaction_history.copy()
    cpy["date"] = pd.to_datetime(cpy["timestamp"]).dt.date
    cpy.drop(columns = ["timestamp"])

    # Rename kolom 'user_id' agar match dengan 'cpy'
    users_renamed = users.rename(columns={"user_id": "userId"})
    # Merge region ke cpy berdasarkan userId
    cpy = cpy.merge(users_renamed[["userId", "region"]], on="userId", how="left")

    # Hitung AvgPriceByDayGlobal
    avg_price_global = cpy.groupby("medicineId").agg({
        "price": "sum",
        "amount": "sum"
    }).reset_index()
    
    avg_price_global["AvgPriceByDayGlobal"] = avg_price_global["price"] / avg_price_global["amount"]
    avg_price_global = avg_price_global[["medicineId", "AvgPriceByDayGlobal"]]
    
    # Merge ke cpy
    cpy = cpy.merge(avg_price_global, on="medicineId", how="left")

    # Hitung total price dan amount per medicineId & region
    avg_price_region = cpy.groupby(["medicineId", "region"]).agg({
        "price": "sum",
        "amount": "sum"
    }).reset_index()
    
    # Hitung average price per unit per region
    avg_price_region["AvgPriceByDayRegion"] = avg_price_region["price"] / avg_price_region["amount"]
    
    # Ambil kolom yang relevan saja
    avg_price_region = avg_price_region[["medicineId", "region", "AvgPriceByDayRegion"]]
    
    # Merge kembali ke cpy
    cpy = cpy.merge(avg_price_region, on=["medicineId", "region"], how="left")

    # Hitung total price dan amount per medicineId & userId
    avg_price_store = cpy.groupby(["medicineId", "userId"]).agg({
        "price": "sum",
        "amount": "sum"
    }).reset_index()
    
    # Hitung average price per unit per user (store)
    avg_price_store["AvgPriceByDayStore"] = avg_price_store["price"] / avg_price_store["amount"]
    
    # Ambil kolom yang relevan
    avg_price_store = avg_price_store[["medicineId", "userId", "AvgPriceByDayStore"]]
    
    # Merge ke cpy
    cpy = cpy.merge(avg_price_store, on=["medicineId", "userId"], how="left")
    # Langkah 1: Grup berdasarkan medicineId dan date, hitung total amount per hari
    daily_sold = cpy.groupby(["medicineId", "date"]).agg({"amount": "sum"}).reset_index()
    
    # Langkah 2: Untuk setiap medicineId, hitung rata-rata total amount harian
    real_sold_global = daily_sold.groupby("medicineId").agg({"amount": "mean"}).rename(
        columns={"amount": "RealSoldByDayGlobal"}
    ).reset_index()
    
    # Gabungkan hasil ke cpy
    cpy = cpy.merge(real_sold_global, on="medicineId", how="left")

    # Step 1: Hitung jumlah amount per medicine-region per hari
    daily_sold_region = cpy.groupby(["medicineId", "region", "date"]).agg({"amount": "sum"}).reset_index()
    
    # Step 2: Hitung rata-rata harian untuk setiap medicineId-region
    real_sold_region = daily_sold_region.groupby(["medicineId", "region"]).agg({"amount": "mean"}).rename(
        columns={"amount": "RealSoldByDayRegion"}
    ).reset_index()

    # Step 1: Hitung total amount harian per medicine-user
    daily_sold_store = cpy.groupby(["medicineId", "userId", "date"]).agg({"amount": "sum"}).reset_index()
    
    # Step 2: Hitung rata-rata harian
    real_sold_store = daily_sold_store.groupby(["medicineId", "userId"]).agg({"amount": "mean"}).rename(
        columns={"amount": "RealSoldByDayStore"}
    ).reset_index()
    
    # Gabungkan ke cpy
    cpy = cpy.merge(real_sold_store, on=["medicineId", "userId"], how="left")
    cpy = cpy.drop(columns=["timestamp", "price"])
    return cpy

def train_and_save_model(cpy, model_path='xgb_model.json', encoder_path='label_encoders.pkl'):
    target_column = "RealSoldByDayStore"
    feature_columns = [
        "medicineId",
        "userId",
        "price_per_unit",
        "date",
        "region",
        "AvgPriceByDayGlobal",
        "AvgPriceByDayRegion",
        "AvgPriceByDayStore",
        "RealSoldByDayGlobal"
    ]
    
    # Salin dataframe agar aman
    data = cpy[feature_columns + [target_column]].copy()

    # Encode 'region' dan 'date'
    encoders = {}
    for col in ["region", "date"]:
        le = LabelEncoder()
        data[col] = le.fit_transform(data[col])
        encoders[col] = le

    # Split features and target
    X = data[feature_columns]
    y = data[target_column]

    # Inisialisasi dan training model
    model = xgb.XGBRegressor(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=6,
        random_state=42
    )
    model.fit(X, y)

    # Save model
    model.save_model(model_path)

    # Save encoders
    joblib.dump(encoders, encoder_path)

    print(f"Model saved to: {model_path}")
    print(f"Encoders saved to: {encoder_path}")

def predict_from_model(input_data, model_path='xgb_model.json', encoder_path='label_encoders.pkl'):
    # Load model dan encoder yang sudah disimpan
    model = xgb.XGBRegressor()
    model.load_model(model_path)
    
    # Load encoders
    encoders = joblib.load(encoder_path)

    # Salin input_data agar aman (pastikan input_data adalah DataFrame)
    data = input_data.copy()

    data["date"] = pd.to_datetime(data["date"]).dt.date
    # Lakukan encoding pada 'region' dan 'date' dengan encoder yang sudah disimpan
    for col in ["region", "date"]:
        if col in data.columns:
            data[col] = encoders[col].transform(data[col])

    # Tentukan fitur yang digunakan dalam model
    feature_columns = [
        "medicineId",
        "userId",
        "price_per_unit",
        "date",
        "region",
        "AvgPriceByDayGlobal",
        "AvgPriceByDayRegion",
        "AvgPriceByDayStore",
        "RealSoldByDayGlobal"
    ]
    
    # Ekstrak fitur dari input_data
    X_new = data[feature_columns]

    # Prediksi dengan model
    predictions = model.predict(X_new)

    return predictions

def process_data(input_json, final_df):
    # Convert input JSON to DataFrame
    input_data = pd.DataFrame([input_json])

    # Ensure 'date' is in datetime format
    input_data['date'] = pd.to_datetime(input_data['date'])

    final_df['date'] = pd.to_datetime(final_df['date']).dt.date

    # Filter final_df for dates between 2025-04-09 and 2025-04-11
    start_date = pd.to_datetime('2025-04-09').date()
    end_date = pd.to_datetime('2025-04-11').date()

    filtered_df = final_df[(final_df['date'] >= start_date) & (final_df['date'] <= end_date)]

    # Calculate AvgPriceByDayGlobal
    avg_price_global = filtered_df.groupby('medicineId').agg({
        'price_per_unit': 'mean'
    }).rename(columns={'price_per_unit': 'AvgPriceByDayGlobal'}).reset_index()

    # Calculate AvgPriceByDayRegion
    avg_price_region = filtered_df.groupby(['medicineId', 'region']).agg({
        'price_per_unit': 'mean'
    }).rename(columns={'price_per_unit': 'AvgPriceByDayRegion'}).reset_index()

    # Calculate AvgPriceByDayStore
    avg_price_store = filtered_df.groupby(['medicineId', 'userId']).agg({
        'price_per_unit': 'mean'
    }).rename(columns={'price_per_unit': 'AvgPriceByDayStore'}).reset_index()

    # Calculate RealSoldByDayGlobal
    real_sold_global = filtered_df.groupby('medicineId').agg({
        'amount': 'sum'
    }).rename(columns={'amount': 'RealSoldByDayGlobal'}).reset_index()

    # Merge calculated features
    input_data = input_data.merge(avg_price_global, on='medicineId', how='left')
    input_data = input_data.merge(avg_price_region, on=['medicineId', 'region'], how='left')
    input_data = input_data.merge(avg_price_store, on=['medicineId', 'userId'], how='left')
    input_data = input_data.merge(real_sold_global, on='medicineId', how='left')

    # Encode categorical features (region and date)
    input_data['region'] = input_data['region']
    input_data['date'] = input_data['date']

    # The resulting DataFrame is now ready for prediction
    return input_data

def get_predictions(stok, hasil_prediksi):

    # Hitung hari hingga stok habis
    days_until_stockout = stok / hasil_prediksi

    # Tanggal hari ini
    today = datetime.now().date()

    # Tanggal stok habis
    stockout_date = today + timedelta(days=days_until_stockout)

    # Prediksi penjualan 30 hari ke depan
    prediksi_30_hari = 30 * hasil_prediksi

    return {
        "stockout_date": stockout_date.strftime("%Y-%m-%d"),
        "predicted_sales_30_days": round(prediksi_30_hari, 2)
    }

def get_recom(generated_data, top_n=10):
    # Ambil rata-rata global per medicineId
    top_global_avg = (
        generated_data
        .groupby("medicineId")["AvgPriceByDayGlobal"]
        .mean()
        .sort_values(ascending=False)
        .head(top_n)
        .reset_index()
    )

    # Return an array (list) of the top 10 medicineId's
    return top_global_avg["medicineId"].to_numpy()

def calculate_recom(recom_list, stok):
    # List to store stock requirements and replenishment needs
    stock_requirements = []

    # Iterate over each medicine in recom_list with the corresponding current stock from stok
    for medicine_id, current_stock in zip(recom_list, stok):
        # Filter the generated_data for the given medicineId
        medicine_data = generated_data[generated_data["medicineId"] == medicine_id]
        
        # Calculate the average daily sales for the medicine (RealSoldByDayGlobal)
        avg_daily_sales = medicine_data["RealSoldByDayGlobal"].mean()
        
        # Calculate the stock required for the next 30 days (for 1 month)
        stock_for_month = avg_daily_sales * 30
        
        # Calculate the stock needed (ensure that it cannot be negative)
        stock_needed = max(0, stock_for_month - current_stock)
        stock_needed = math.floor(stock_needed)
        
        # Append the result as a tuple (medicineId, stock_for_month, stock_needed)
        stock_requirements.append(stock_needed)
    
    return stock_requirements

## CONTOH FLOW
generated_data = generate_features(users, medicine, medicinedisease, disease, transaction_history)
train_and_save_model(generated_data)

input_json = {
    "userId": 1,
    "medicineId": 3,
    "price_per_unit": 6070,
    "date": "2025-04-11",
    "region": "Jawa"
}

processed_data = process_data(input_json, generated_data)
predictions = predict_from_model(processed_data)
get_predictions(200, predictions[0])
