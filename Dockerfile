FROM python:3.10

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# 🔥 Train model when container builds
RUN python app/train.py

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]

