import yfinance as yf
import pandas as pd
import numpy as np
from ta.trend import SMAIndicator, MACD
from ta.momentum import RSIIndicator
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn
import os
from datetime import datetime

app = FastAPI(title="Forex Direction Predictor (Educational)")

MODEL_PATH = "forex_model.pkl"
FEATURES_PATH = "forex_features.pkl"
DATA_PATH = "forex_data.pkl"

def fetch_and_prepare(symbol="EURUSD=X", period="3y", interval="1d"):
    """Download data & engineer features"""
    df = yf.download(symbol, period=period, interval=interval, progress=False)
    if df.empty:
        raise ValueError("No data fetched. Check symbol or network.")
    
    df = df.dropna()
    df = df.copy()
    
    # Target: next candle close > current close ? 1 : 0
    df['Returns'] = df['Close'].pct_change()
    df['Target'] = (df['Returns'].shift(-1) > 0).astype(int)
    
    # Technical Indicators
    df['SMA_20'] = SMAIndicator(df['Close'], window=20).sma_indicator()
    df['RSI_14'] = RSIIndicator(df['Close'], window=14).rsi()
    macd = MACD(df['Close'], window_slow=26, window_fast=12, window_sign=9)
    df['MACD'] = macd.macd()
    df['MACD_Signal'] = macd.macd_signal()
    
    df.dropna(inplace=True)
    return df

def train_model():
    """Train & save model"""
    print("Fetching data & engineering features...")
    df = fetch_and_prepare()
    
    features = ['SMA_20', 'RSI_14', 'MACD', 'MACD_Signal', 'Returns']
    X = df[features]
    y = df['Target']
    
    # Time-series split (no shuffle)
    split_idx = int(len(X) * 0.8)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
    
    print("Training Random Forest...")
    model = RandomForestClassifier(n_estimators=150, max_depth=5, random_state=42, class_weight='balanced')
    model.fit(X_train, y_train)
    
    acc = accuracy_score(y_test, model.predict(X_test))
    print(f"📊 Test Accuracy: {acc:.3f} (Note: ~0.50-0.55 is normal in forex)")
    
    joblib.dump(model, MODEL_PATH)
    joblib.dump(features, FEATURES_PATH)
    df.to_pickle(DATA_PATH)
    print("✅ Model & data saved.")
    return acc

def get_latest_prediction():
    """Load model & predict on most recent bar"""
    if not os.path.exists(MODEL_PATH):
        raise HTTPException(500, "Model not found. Run training first.")
        
    model = joblib.load(MODEL_PATH)
    features = joblib.load(FEATURES_PATH)
    df = pd.read_pickle(DATA_PATH)
    
    latest = df.iloc[-1][features].values.reshape(1, -1)
    pred = model.predict(latest)[0]
    proba = model.predict_proba(latest)[0]
    
    return {
        "symbol": "EUR/USD",
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "prediction": "UP" if pred == 1 else "DOWN",
        "confidence": round(float(proba[pred]) * 100, 2),
        "note": "Educational only. Not financial advice. Past performance ≠ future results."
    }

# FastAPI Endpoints
@app.get("/")
def root():
    return {"message": "Forex ML Predictor API. Visit /predict or /train"}

@app.post("/train")
def trigger_train():
    try:
        acc = train_model()
        return {"status": "success", "test_accuracy": acc}
    except Exception as e:
        raise HTTPException(500, str(e))

@app.get("/predict")
def predict():
    try:
        return get_latest_prediction()
    except Exception as e:
        raise HTTPException(500, str(e))

if __name__ == "__main__":
    # Train once, then start API
    if not os.path.exists(MODEL_PATH):
        train_model()
    print("🚀 Starting API on http://localhost:8000")
    uvicorn.run(app, host="0.0.0.0", port=8000)