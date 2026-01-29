# 🏠 California House Price Prediction API

An end-to-end Machine Learning project that predicts house prices using the **California Housing Dataset**.  
The model is trained using **XGBoost Regressor**, served via **FastAPI**, containerized with **Docker**, and ready for deployment.

---

## 🚀 Project Overview

This project demonstrates a complete ML workflow:

- Data loading and preprocessing  
- Model training using XGBoost  
- Model saving & loading  
- REST API using FastAPI  
- Docker containerization  
- Deployment-ready structure  

---

## 🧠 Tech Stack

- Python
- Scikit-learn
- XGBoost
- FastAPI
- Uvicorn
- Docker
- Pandas / NumPy

---

## 📁 Project Structure

california-house-price-ml/
│
├── app/
│ ├── main.py # FastAPI app
│ ├── model.py # Loads trained model
│ ├── train.py # Trains and saves model
│
├── model/
│ └── xgb_model.pkl # Saved model file
│
├── requirements.txt
├── Dockerfile
├── .gitignore
└── README.md


---

## ⚙️ Setup Instructions (Local Run)

### 1️⃣ Clone Repository

```bash
git clone https://github.com/YOUR_USERNAME/house-price-api.git
cd house-price-api
2️⃣ Create Virtual Environment
python -m venv venv
3️⃣ Activate Virtual Environment
Windows

venv\Scripts\activate
Mac/Linux

source venv/bin/activate
4️⃣ Install Dependencies
pip install -r requirements.txt
5️⃣ Train Model
python app/train.py
6️⃣ Run API Server
uvicorn app.main:app --reload
Open in browser:

👉 http://127.0.0.1:8000/docs

🔮 API Endpoint
POST /predict
Predict house price.

Sample Input JSON

{
  "MedInc": 8.3252,
  "HouseAge": 41,
  "AveRooms": 6.984,
  "AveBedrms": 1.023,
  "Population": 322,
  "AveOccup": 2.555,
  "Latitude": 37.88,
  "Longitude": -122.23
}
Response

{
  "predicted_price": 4.12
}
🐳 Run with Docker
Build Image
docker build -t house-price-api .
Run Container
docker run -p 8000:8000 house-price-api
☁️ Deployment
This project can be deployed easily on:

Render

Railway

AWS

Azure

GCP

