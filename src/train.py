import mlflow
import mlflow.sklearn
import pandas as pd
import yaml
import json
import joblib
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from pathlib import Path

# Load biến môi trường từ file .env
env_file = Path(__file__).parent.parent / ".env"
if env_file.exists():
    with open(env_file) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, value = line.split("=", 1)
                os.environ[key] = value

EVAL_THRESHOLD = 0.70


def train(
    params: dict,
    data_path: str = "data/train_phase1.csv",
    eval_path: str = "data/eval.csv",
    use_mlflow: bool = True,
) -> float:
    """
    Huấn luyện mô hình và ghi nhận kết quả vào MLflow.

    Tham số:
        params: dict chứa các siêu tham số cho RandomForestClassifier
        data_path: đường dẫn đến file dữ liệu huấn luyện
        eval_path: đường dẫn đến file dữ liệu đánh giá
        use_mlflow: có sử dụng MLflow tracking không (mặc định True)

    Trả về:
        accuracy (float): độ chính xác trên tập đánh giá
    """

    # TODO 1.5.1: Đọc dữ liệu huấn luyện và đánh giá
    df_train = pd.read_csv(data_path)
    df_eval = pd.read_csv(eval_path)

    # TODO 1.5.2: Tách đặc trưng và nhãn
    X_train = df_train.drop(columns=["target"])
    y_train = df_train["target"]
    X_eval = df_eval.drop(columns=["target"])
    y_eval = df_eval["target"]

    # TODO 1.5.3: Bắt đầu MLflow run (nếu enabled)
    if use_mlflow:
        mlflow.start_run()
    # TODO 1.5.3: Bắt đầu MLflow run (nếu enabled)
    if use_mlflow:
        mlflow.start_run()
        
    # TODO 1.5.4: Ghi nhận các siêu tham số vào MLflow
    if use_mlflow:
        mlflow.log_params(params)

    # TODO 1.5.5: Khởi tạo và huấn luyện mô hình RandomForestClassifier
    model = RandomForestClassifier(**params, random_state=42)
    model.fit(X_train, y_train)

    # TODO 1.5.6: Tính accuracy và f1_score trên tập đánh giá
    preds = model.predict(X_eval)
    acc = accuracy_score(y_eval, preds)
    f1 = f1_score(y_eval, preds, average="weighted")

    # TODO 1.5.7: Ghi nhận các chỉ số vào MLflow
    if use_mlflow:
        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("f1_score", f1)

    # TODO 1.5.8: Log mô hình vào MLflow artifact
    if use_mlflow:
        mlflow.sklearn.log_model(model, "model")

    # TODO 1.5.9: In kết quả ra màn hình
    print(f"Accuracy: {acc:.4f} | F1: {f1:.4f}")

    # TODO 1.5.10: Lưu metrics ra file outputs/metrics.json
    os.makedirs("outputs", exist_ok=True)
    with open("outputs/metrics.json", "w") as f:
        json.dump({"accuracy": acc, "f1_score": f1}, f)

    # TODO 1.5.11: Lưu mô hình ra file models/model.pkl
    os.makedirs("models", exist_ok=True)
    joblib.dump(model, "models/model.pkl")
    
    # Kết thúc MLflow run
    if use_mlflow:
        mlflow.end_run()

    # TODO 1.5.12: Trả về acc
    return acc


if __name__ == "__main__":
    # Đọc siêu tham số từ params.yaml và gọi hàm train()
    with open("params.yaml") as f:
        params = yaml.safe_load(f)
    train(params)
