"""Задание 46. Обучение ML-модели на датасете Kaggle.

Датасет: Telco Customer Churn (https://www.kaggle.com/datasets/blastchar/telco-customer-churn)
Задача: бинарная классификация — уйдёт ли клиент (Churn = Yes/No).

Скачать данные:  kaggle datasets download -d blastchar/telco-customer-churn -p data --unzip
Запуск:          python train.py
Результат:       model/model.joblib + model/metrics.json
"""
import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

DATA_PATH = Path("data/WA_Fn-UseC_-Telco-Customer-Churn.csv")
MODEL_DIR = Path("model")
RANDOM_STATE = 42

NUMERIC = ["tenure", "MonthlyCharges", "TotalCharges", "SeniorCitizen"]
CATEGORICAL = [
    "gender", "Partner", "Dependents", "PhoneService", "MultipleLines",
    "InternetService", "OnlineSecurity", "OnlineBackup", "DeviceProtection",
    "TechSupport", "StreamingTV", "StreamingMovies", "Contract",
    "PaperlessBilling", "PaymentMethod",
]


def load_data(path: Path) -> tuple[pd.DataFrame, pd.Series]:
    df = pd.read_csv(path)
    # TotalCharges содержит пробелы у новых клиентов (tenure = 0) -> NaN -> 0
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce").fillna(0.0)
    y = (df["Churn"] == "Yes").astype(int)
    X = df[NUMERIC + CATEGORICAL]
    return X, y


def build_pipeline(estimator) -> Pipeline:
    preprocess = ColumnTransformer([
        ("num", StandardScaler(), NUMERIC),
        ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL),
    ])
    return Pipeline([("preprocess", preprocess), ("model", estimator)])


def evaluate(pipe: Pipeline, X, y) -> dict:
    proba = pipe.predict_proba(X)[:, 1]
    pred = (proba >= 0.5).astype(int)
    return {
        "accuracy": round(accuracy_score(y, pred), 4),
        "f1": round(f1_score(y, pred), 4),
        "roc_auc": round(roc_auc_score(y, proba), 4),
    }


def main() -> None:
    X, y = load_data(DATA_PATH)
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
    )
    print(f"train: {X_tr.shape}, test: {X_te.shape}, доля оттока: {y.mean():.3f}")

    candidates = {
        "logistic_regression": LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
        "gradient_boosting": GradientBoostingClassifier(
            n_estimators=200, learning_rate=0.05, max_depth=3, random_state=RANDOM_STATE
        ),
    }

    results = {}
    for name, est in candidates.items():
        pipe = build_pipeline(est).fit(X_tr, y_tr)
        results[name] = (pipe, evaluate(pipe, X_te, y_te))
        print(f"{name:20s} {results[name][1]}")

    best_name = max(results, key=lambda n: results[n][1]["roc_auc"])
    best_pipe, best_metrics = results[best_name]
    print(f"\nЛучшая модель: {best_name}")
    print(classification_report(y_te, best_pipe.predict(X_te), target_names=["No churn", "Churn"]))

    MODEL_DIR.mkdir(exist_ok=True)
    joblib.dump(best_pipe, MODEL_DIR / "model.joblib")
    (MODEL_DIR / "metrics.json").write_text(json.dumps({
        "dataset": "kaggle: blastchar/telco-customer-churn",
        "model": best_name,
        "test_metrics": best_metrics,
        "all_models": {n: m for n, (_, m) in results.items()},
        "features": {"numeric": NUMERIC, "categorical": CATEGORICAL},
    }, indent=2, ensure_ascii=False))
    print(f"Модель сохранена в {MODEL_DIR / 'model.joblib'}")


if __name__ == "__main__":
    main()
