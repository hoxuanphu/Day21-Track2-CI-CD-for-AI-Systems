import os
import json
import numpy as np
import pandas as pd
from src.train import train


FEATURE_NAMES = [
    "fixed_acidity", "volatile_acidity", "citric_acid", "residual_sugar",
    "chlorides", "free_sulfur_dioxide", "total_sulfur_dioxide", "density",
    "pH", "sulphates", "alcohol", "wine_type",
]


def _make_temp_data(tmp_path):
    """
    Tạo dataset nhỏ với cùng schema Wine Quality để sử dụng trong test.
    """
    rng = np.random.default_rng(0)
    n = 200
    
    # Tạo mảng X với kích thước (n, 12) với giá trị ngẫu nhiên [0, 1)
    X = rng.random((n, len(FEATURE_NAMES)))
    
    # Tạo mảng y với n phần tử, mỗi phần tử là số nguyên ngẫu nhiên trong [0, 3)
    y = rng.integers(0, 3, size=n)
    
    # Tạo DataFrame
    df = pd.DataFrame(X, columns=FEATURE_NAMES)
    df["target"] = y
    
    # Lưu 160 dòng đầu vào train.csv và 40 dòng cuối vào eval.csv
    train_path = tmp_path / "train.csv"
    eval_path = tmp_path / "eval.csv"
    
    df.iloc[:160].to_csv(train_path, index=False)
    df.iloc[160:].to_csv(eval_path, index=False)
    
    return str(train_path), str(eval_path)


def test_train_returns_float(tmp_path):
    """Kiểm tra hàm train() trả về một số thực trong khoảng [0, 1]."""
    train_path, eval_path = _make_temp_data(tmp_path)
    
    # Gọi hàm train với siêu tham số nhỏ và tắt MLflow
    acc = train(
        {"n_estimators": 10, "max_depth": 3, "min_samples_split": 2},
        data_path=train_path,
        eval_path=eval_path,
        use_mlflow=False,
    )
    
    # Kiểm tra kết quả
    assert isinstance(acc, float), f"Expected float, got {type(acc)}"
    assert 0.0 <= acc <= 1.0, f"Accuracy {acc} not in range [0, 1]"


def test_metrics_file_created(tmp_path):
    """Kiểm tra file outputs/metrics.json được tạo sau khi huấn luyện."""
    train_path, eval_path = _make_temp_data(tmp_path)
    
    train(
        {"n_estimators": 10, "max_depth": 3, "min_samples_split": 2},
        data_path=train_path,
        eval_path=eval_path,
        use_mlflow=False,
    )
    
    # Kiểm tra file tồn tại
    metrics_file = "outputs/metrics.json"
    assert os.path.exists(metrics_file), f"File {metrics_file} not found"
    
    # Đọc và kiểm tra nội dung
    with open(metrics_file) as f:
        metrics = json.load(f)
    
    assert "accuracy" in metrics, "Missing 'accuracy' in metrics.json"
    assert "f1_score" in metrics, "Missing 'f1_score' in metrics.json"


def test_model_file_created(tmp_path):
    """Kiểm tra file models/model.pkl được tạo sau khi huấn luyện."""
    train_path, eval_path = _make_temp_data(tmp_path)
    
    train(
        {"n_estimators": 10, "max_depth": 3, "min_samples_split": 2},
        data_path=train_path,
        eval_path=eval_path,
        use_mlflow=False,
    )
    
    # Kiểm tra file tồn tại
    model_file = "models/model.pkl"
    assert os.path.exists(model_file), f"File {model_file} not found"
