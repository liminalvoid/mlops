"""Задание 47. REST API для модели оттока клиентов (запускается в Docker-контейнере)."""
import os
from typing import Literal

import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, Field

MODEL_PATH = os.getenv("MODEL_PATH", "model/model.joblib")
model = joblib.load(MODEL_PATH)

app = FastAPI(title="Telco Churn Model", version="1.0.0")

YesNo = Literal["Yes", "No"]


class Customer(BaseModel):
    gender: Literal["Male", "Female"]
    SeniorCitizen: int = Field(ge=0, le=1)
    Partner: YesNo
    Dependents: YesNo
    tenure: int = Field(ge=0)
    PhoneService: YesNo
    MultipleLines: Literal["Yes", "No", "No phone service"]
    InternetService: Literal["DSL", "Fiber optic", "No"]
    OnlineSecurity: Literal["Yes", "No", "No internet service"]
    OnlineBackup: Literal["Yes", "No", "No internet service"]
    DeviceProtection: Literal["Yes", "No", "No internet service"]
    TechSupport: Literal["Yes", "No", "No internet service"]
    StreamingTV: Literal["Yes", "No", "No internet service"]
    StreamingMovies: Literal["Yes", "No", "No internet service"]
    Contract: Literal["Month-to-month", "One year", "Two year"]
    PaperlessBilling: YesNo
    PaymentMethod: Literal[
        "Electronic check", "Mailed check",
        "Bank transfer (automatic)", "Credit card (automatic)",
    ]
    MonthlyCharges: float = Field(ge=0)
    TotalCharges: float = Field(ge=0)


class Prediction(BaseModel):
    churn: bool
    churn_probability: float


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "model_path": MODEL_PATH}


@app.post("/predict", response_model=list[Prediction])
def predict(customers: list[Customer]) -> list[Prediction]:
    X = pd.DataFrame([c.model_dump() for c in customers])
    proba = model.predict_proba(X)[:, 1]
    return [Prediction(churn=bool(p >= 0.5), churn_probability=round(float(p), 4)) for p in proba]
