import pandas as pd
import numpy as np
import yfinance as yf
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from datetime import timedelta

# ----------------------------------
# 1. FETCH LIVE GOLD DATA
# ----------------------------------
# ----------------------------------
# 1. FETCH LIVE DATA & RATES
# ----------------------------------
gold = yf.download("GC=F", period="6mo", interval="1d")
gold.reset_index(inplace=True)

# 2026 ke realistic rates (Adjust according to current market)
USD_TO_INR = 83.5 # Live rate lena better hai
IMPORT_DUTY_GST = 1.18 # 18% approx total tax/duty (India context)
OUNCE_TO_GRAM = 31.1035

# Price for 10 grams in INR with Taxes
gold["Price"] = ((gold["Close"] * USD_TO_INR) / OUNCE_TO_GRAM)  * IMPORT_DUTY_GST
df = gold[["Date", "Price"]].dropna().copy()


# Ensure float
df["Price"] = df["Price"].astype(float)

# ----------------------------------
# 2. FEATURE ENGINEERING
# ----------------------------------
df["lag_1"] = df["Price"].shift(1)
df["lag_2"] = df["Price"].shift(2)
df["diff"] = df["lag_1"] - df["lag_2"]
df_train = df.dropna().copy()

X = df_train[["lag_1", "lag_2", "diff"]].values
y = df_train["Price"].values

# ----------------------------------
# 3. TRAIN MODEL
# ----------------------------------
model = LinearRegression()
model.fit(X, y)

# ----------------------------------
# 4. NEXT DAY PREDICTION
# ----------------------------------
last_date = df["Date"].iloc[-1]
next_date = last_date + timedelta(days=1)

last_price = float(df["Price"].iloc[-1])
prev_price = float(df["Price"].iloc[-2])
input_diff = last_price - prev_price

input_data = np.array([[last_price, prev_price, input_diff]])
prediction = float(model.predict(input_data)[0])

final_prediction = prediction * 1.001

# ----------------------------------
# 5. OUTPUT
# ----------------------------------
print("✅ Gold Model Auto-Updated (LIVE DATA)")
print(f"Last Available Date: {last_date.date()}")
print(f"Gold Rate 10g {last_date.date()}: ₹{round(last_price * 10, 2)}")
print("-" * 35)

pred_date_str = next_date.strftime("%b %d")
print(f"📈 Predicted {pred_date_str} Gold Rate 10g: ₹{round(final_prediction * 10, 2)}")

# ----------------------------------
# 6. GRAPH
# ----------------------------------
plt.figure(figsize=(10, 6))
last_5 = df.tail(5)

plt.plot(last_5["Date"], last_5["Price"],
         marker="o", linewidth=2, label="Past 5 Days")

plt.scatter(next_date, final_prediction,
            color="red", s=120, label="Prediction")

plt.plot(
    [last_5["Date"].iloc[-1], next_date],
    [last_price, final_prediction],
    "r--"
)

plt.title(f"Gold Price Prediction for {pred_date_str}")
plt.xlabel("Date")
plt.ylabel("Price per Gram (INR)")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_percentage_error

# ----------------------------------
# 3. ACCURACY CHECK & TRAINING
# ----------------------------------
# Hum data ko split karenge: 80% training ke liye, 20% testing ke liye
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model Training
model = LinearRegression()
model.fit(X_train, y_train)

# Accuracy Calculation
y_pred = model.predict(X_test)
r2 = r2_score(y_test, y_pred)
mape = mean_absolute_percentage_error(y_test, y_pred)
accuracy_pct = (1 - mape) * 100

print(f"📊 Model Accuracy Score: {r2:.4f}")
print(f"🎯 Prediction Confidence: {accuracy_pct:.2f}%")