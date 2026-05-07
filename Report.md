# Báo Cáo Kết Quả Lab - CI/CD for AI Systems

## 1. Bộ siêu tham số đã chọn (Best Hyperparameters)

Dựa trên các thí nghiệm đã thực hiện với MLflow ở Bước 1, bộ siêu tham số sau đây đã được lựa chọn để triển khai:

*   **n_estimators**: 500
*   **max_depth**: 25
*   **min_samples_split**: 5

**Lý do lựa chọn:** 
Qua quá trình so sánh trên MLflow UI, bộ tham số này mang lại độ chính xác (Accuracy) cao nhất và chỉ số F1-Score cân bằng nhất trên tập dữ liệu kiểm định (eval dataset). Việc tăng `n_estimators` giúp mô hình ổn định hơn, trong khi `max_depth` đủ lớn để bắt được các đặc trưng phức tạp của dữ liệu Wine Quality.

## 2. Khó khăn gặp phải và Cách giải quyết

Trong quá trình thực hiện Bước 2 và Bước 3, tôi đã gặp một số thách thức kỹ thuật và đã giải quyết như sau:

| Khó khăn | Nguyên nhân | Cách giải quyết |
|---|---|---|
| **Lỗi SSH Handshake Failed** | Do chuẩn Ubuntu 24 vô hiệu hóa `ssh-rsa` SHA-1 và có ký tự xuống dòng (`\n`) thừa trong GitHub Secret. | Chuyển sang sử dụng khóa **ED25519** và chuẩn hóa lại giá trị các Secret (`VM_USER`, `VM_HOST`) trên GitHub. |
| **Unit cicd-server not found** | Tên service trong file pipeline (`mlops.yml`) không khớp với tên service thực tế được tạo trên VM. | Tạo file cấu hình `cicd-server.service` đồng bộ trong hệ thống systemd của Ubuntu. |
| **ModuleNotFoundError: fastapi** | Môi trường Python trên VM chưa được cài đặt các thư viện cần thiết và bị chặn bởi PEP 668. | Sử dụng lệnh `pip3 install ... --break-system-packages` để cài đặt trực tiếp các thư viện (FastAPI, Boto3, Scikit-learn) lên VM. |
| **Pipeline không tự động trigger** | Branch mặc định là `master` nhưng file cấu hình để `main`, và filter `paths` không bao gồm file workflow. | Sửa cấu hình branch thành `master` và thực hiện thay đổi nhỏ trong `params.yaml` để kích hoạt trigger hợp lệ. |

## 3. Kết luận
Hệ thống MLOps đã hoạt động ổn định. Mỗi khi có dữ liệu mới được bổ sung, chỉ cần thực hiện `dvc push` và `git push`, mô hình sẽ tự động được huấn luyện lại, kiểm tra chất lượng và triển khai lên Cloud VM mà không cần thao tác thủ công.
