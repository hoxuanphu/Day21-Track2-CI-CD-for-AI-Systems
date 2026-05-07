from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import boto3
import joblib
import os

app = FastAPI()

# Đọc tên bucket từ biến môi trường
S3_BUCKET = os.environ.get("S3_BUCKET", "phu-cicd-test")
S3_MODEL_KEY = "models/latest/model.pkl"
MODEL_PATH = os.path.expanduser("~/models/model.pkl")


def download_model():
    """Tải file model.pkl từ S3 về máy khi server khởi động."""
    print(f"Downloading model from s3://{S3_BUCKET}/{S3_MODEL_KEY}...")
    
    # Tạo S3 client
    s3 = boto3.client('s3')
    
    # Tạo thư mục models nếu chưa có
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    
    # Tải file từ S3
    s3.download_file(S3_BUCKET, S3_MODEL_KEY, MODEL_PATH)
    
    print(f"Model downloaded successfully to {MODEL_PATH}")


# Gọi hàm này khi module được import (chạy khi server khởi động)
download_model()
model = joblib.load(MODEL_PATH)


class PredictRequest(BaseModel):
    features: list[float]


@app.get("/health")
def health():
    """Endpoint kiểm tra sức khỏe server."""
    return {"status": "ok"}


@app.post("/predict")
def predict(req: PredictRequest):
    """
    Endpoint suy luận.

    Đầu vào: JSON {"features": [f1, f2, ..., f12]}
    Đầu ra:  JSON {"prediction": <0|1|2>, "label": <"thấp"|"trung_bình"|"cao">}
    """
    # Kiểm tra số lượng features
    if len(req.features) != 12:
        raise HTTPException(
            status_code=400,
            detail="Expected 12 features (wine quality)"
        )
    
    # Dự đoán
    prediction = int(model.predict([req.features])[0])
    
    # Map prediction sang label
    label_map = {
        0: "thấp",
        1: "trung_bình",
        2: "cao"
    }
    
    return {
        "prediction": prediction,
        "label": label_map.get(prediction, "unknown")
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
