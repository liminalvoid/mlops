# Контрольная работа №1 MLOps — задания 46 и 47

**Датасет:** [Telco Customer Churn (Kaggle)](https://www.kaggle.com/datasets/blastchar/telco-customer-churn): 7043 клиента, 19 признаков, целевая переменная `Churn`.

## Задание 46 — обучение модели

```bash
pip install -r requirements.txt
kaggle datasets download -d blastchar/telco-customer-churn -p data --unzip
python train.py
```

Пайплайн `scikit-learn`: `StandardScaler` для числовых признаков и `OneHotEncoder` для категориальных.
Сравниваются логистическая регрессия и градиентный бустинг; лучшая модель выбирается по ROC-AUC
на отложенной выборке (20%, стратификация).

| Модель | Accuracy | F1 (churn) | ROC-AUC |
|---|---|---|---|
| Logistic Regression | 0.8055 | 0.6040 | 0.8421 |
| **Gradient Boosting** (выбрана) | 0.8048 | 0.5877 | **0.8441** |

Артефакты: `model/model.joblib` (весь пайплайн: предобработка и модель), `model/metrics.json`.

## Задание 47 — модель в контейнере

FastAPI-сервис (`app.py`) с эндпоинтами `GET /health` и `POST /predict`, упакованный в Docker.

```bash
docker build -t telco-churn:1.0 .
docker run -d -p 8000:8000 --name churn-api telco-churn:1.0

curl localhost:8000/health
curl -X POST localhost:8000/predict -H "Content-Type: application/json" -d @sample.json
# [{"churn":true,"churn_probability":0.5244},{"churn":false,"churn_probability":0.0199}]
```

Swagger UI: http://localhost:8000/docs
