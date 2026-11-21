# Контрольная работа №1 MLOps — задание 47

**Датасет:** [Telco Customer Churn (Kaggle)](https://www.kaggle.com/datasets/blastchar/telco-customer-churn): 7043 клиента, 19 признаков, целевая переменная `Churn`.

FastAPI-сервис (`app.py`) с эндпоинтами `GET /health` и `POST /predict`, упакованный в Docker.

```bash
docker build -t telco-churn:1.0 .
docker run -d -p 8000:8000 --name churn-api telco-churn:1.0

curl localhost:8000/health
curl -X POST localhost:8000/predict -H "Content-Type: application/json" -d @sample.json
# [{"churn":true,"churn_probability":0.5244},{"churn":false,"churn_probability":0.0199}]
```

Swagger UI: http://localhost:8000/docs
