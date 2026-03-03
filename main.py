from fastapi import FastAPI, HTTPException
import pandas as pd
from datetime import datetime, timedelta
import uvicorn
import threading
import time
import os

app = FastAPI(title="Pharmacy Agentic Backend")

# Helper function to load CSV with standardized columns
def load_medicine_data():
    if not os.path.exists("medicine.csv"):
        raise FileNotFoundError("medicine.csv file nahi mili!")
    df = pd.read_csv("medicine.csv")
    df.columns = df.columns.str.strip().str.lower()
    return df

@app.get("/")
async def root():
    return {"message": "Pharmacy Backend API", "status": "running", "version": "1.0.0"}

# --- 1. STOCK CHECK (Agent uses this to verify inventory) ---
@app.get("/check-stock/{med_name}")
async def check_stock(med_name: str):
    try:
        df = load_medicine_data()
        # Medicine search (case-insensitive)
        med_filter = df['medicine_name'].str.lower() == med_name.lower()

        if not df[med_filter].empty:
            row = df[med_filter].iloc[0]
            # 'stock_level' use kiya hai jo aapke CSV mein hai
            return {
                "medicine_name": row['medicine_name'],
                "stock_level": int(row['stock_level']), #
                "unit": row['unit'],
                "prescription_required": row['prescription_required'],
                "price": float(row['price'])
            }
        raise HTTPException(status_code=404, detail="Medicine not found in inventory")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# --- 2. REFILL ALERTS (Predictive Intelligence Requirement) ---
@app.get("/refill-alerts")
async def get_refill_alerts():
    try:
        if not os.path.exists("orders.csv"):
            return {"alerts": [], "message": "No order history found"}
            
        orders = pd.read_csv("orders.csv")
        medicines = load_medicine_data()
        today = datetime.now()
        alerts = []

        # Merge to get medicine names
        merged = orders.merge(medicines, on='medicine_id', how='left')

        for index, row in merged.iterrows():
            last_date = datetime.strptime(row['order_date'], '%Y-%m-%d')
            # Assuming 10 days dosage cycle
            expiry_date = last_date + timedelta(days=10)

            # Alert if only 2 days left
            if (expiry_date - timedelta(days=2)) <= today <= expiry_date:
                alerts.append({
                    "customer": row['customer_name'],
                    "medicine": row['medicine_name'],
                    "status": "Refill Needed Proactively"
                })
        return {"alerts": alerts}
    except Exception as e:
        return {"error": str(e), "alerts": []}

# --- 3. PLACE ORDER (Real-world Action/Tool Use) ---
@app.post("/place-order/{med_name}")
async def place_order(med_name: str):
    try:
        df = load_medicine_data()
        med_filter = df['medicine_name'].str.lower() == med_name.lower()

        if not df[med_filter].empty:
            current_stock = df.loc[med_filter, 'stock_level'].values[0]

            # STRICT STOCK CHECK: Reject if stock is 0 or less
            if current_stock <= 0:
                raise HTTPException(status_code=400, detail="Medicine is out of stock")

            # 1. Update Inventory Action
            df.loc[med_filter, 'stock_level'] -= 1
            df.to_csv("medicine.csv", index=False)

            # 2. Update Order History (Predictive Intelligence)
            try:
                orders_file = "orders.csv"
                new_order = {
                    "customer_id": "C001",
                    "customer_name": "Pooja", # Mock Name
                    "medicine_id": df.loc[med_filter, 'medicine_id'].values[0],
                    "order_date": datetime.now().strftime('%Y-%m-%d'),
                    "dosage_frequency": "1 per day",
                    "quantity_bought": 1
                }

                if os.path.exists(orders_file):
                    orders_df = pd.read_csv(orders_file)
                    orders_df = pd.concat([orders_df, pd.DataFrame([new_order])], ignore_index=True)
                else:
                    orders_df = pd.DataFrame([new_order])

                orders_df.to_csv(orders_file, index=False)
            except Exception as ex:
                print(f"History update error: {ex}")

            return {"status": "success", "remaining_stock": int(current_stock - 1)}
        return {"status": "error", "message": "Medicine not found"}
    except HTTPException:
        raise
    except Exception as e:
        return {"status": "error", "message": str(e)}

# --- 4. SAVE ORDER (For pending orders from chat) ---
@app.post("/save_order")
async def save_order(order_data: dict):
    try:
        orders_file = "orders.csv"
        new_order = {
            "customer_id": "C001",
            "customer_name": order_data.get("customer_name", "User"),
            "medicine_id": order_data.get("medicine_id", "Unknown"),
            "order_date": datetime.now().strftime('%Y-%m-%d'),
            "dosage_frequency": "As needed",
            "quantity_bought": order_data.get("quantity", 1),
            "medicine_name": order_data.get("medicine_name", "Unknown"),
            "status": "Pending"
        }

        if os.path.exists(orders_file):
            orders_df = pd.read_csv(orders_file)
            orders_df = pd.concat([orders_df, pd.DataFrame([new_order])], ignore_index=True)
        else:
            orders_df = pd.DataFrame([new_order])

        orders_df.to_csv(orders_file, index=False)
        return {"status": "success", "message": "Order saved successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

def start_server():
    uvicorn.run(app, host="127.0.0.1", port=8000)

if __name__ == "__main__":
    server_thread = threading.Thread(target=start_server, daemon=True)
    server_thread.start()
    print("✅ Backend Live at http://127.0.0.1:8000")
    try:
        while True: time.sleep(1)
    except KeyboardInterrupt:
        print("🛑 Stopped")