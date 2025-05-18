import os
import cv2
import numpy as np
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
import joblib

# Thiết lập đường dẫn thư mục dữ liệu và nơi lưu mô hình
DATASET_DIR = 'dehazing_data'
MODEL_DIR = 'models'
os.makedirs(MODEL_DIR, exist_ok=True)  # Tạo thư mục models nếu chưa tồn tại

# Đường dẫn đến ảnh mờ và ảnh rõ
hazy_dir = os.path.join(DATASET_DIR, 'hazy_images')
clear_dir = os.path.join(DATASET_DIR, 'clear_images')

# Kiểm tra tồn tại thư mục ảnh
if not os.path.isdir(hazy_dir):
    raise FileNotFoundError(f"Không tìm thấy thư mục: {hazy_dir}")
if not os.path.isdir(clear_dir):
    raise FileNotFoundError(f"Không tìm thấy thư mục: {clear_dir}")

# Hàm trích xuất đặc trưng cơ bản từ ảnh (ảnh xám):
# - Giá trị trung bình độ sáng
# - Độ lệch chuẩn
# - Entropy (mức độ ngẫu nhiên)
def extract_features(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    mean = np.mean(gray)
    std = np.std(gray)
    hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
    hist = hist.flatten() / hist.sum()
    entropy = -np.sum(hist * np.log2(hist + 1e-10))
    return np.array([mean, std, entropy])

# Hàm tìm giá trị gamma tốt nhất bằng phương pháp thử nhiều gamma khác nhau,
# áp dụng correction gamma, rồi tính MSE giữa ảnh mờ sau chỉnh gamma và ảnh rõ
def find_best_gamma(img_hazy, img_clear):
    best_gamma = 1.0
    min_mse = float('inf')
    for gamma in np.arange(0.5, 2.5, 0.05):  # Thử các giá trị gamma từ 0.5 đến 2.45
        invGamma = 1.0 / gamma
        table = np.array([((i / 255.0) ** invGamma) * 255 for i in np.arange(256)]).astype("uint8")
        img_gamma = cv2.LUT(img_hazy, table)  # Chỉnh gamma
        mse = np.mean((img_gamma.astype(np.float32) - img_clear.astype(np.float32)) ** 2)
        if mse < min_mse:
            min_mse = mse
            best_gamma = gamma
    return best_gamma

# Khởi tạo tập dữ liệu đầu vào (X) và nhãn đầu ra (y)
X = []
y = []

# Duyệt từng ảnh mờ trong thư mục
for fname in os.listdir(hazy_dir):
    hazy_path = os.path.join(hazy_dir, fname)
    clear_path = os.path.join(clear_dir, fname)
    
    if not os.path.exists(clear_path):
        continue

    img_hazy = cv2.imread(hazy_path)
    img_clear = cv2.imread(clear_path)

    if img_hazy is None or img_clear is None:
        continue

    # Trích xuất đặc trưng từ ảnh mờ
    features = extract_features(img_hazy)
    # Tìm giá trị gamma tốt nhất khớp với ảnh rõ
    gamma = find_best_gamma(img_hazy, img_clear)

    # Thêm vào tập dữ liệu
    X.append(features)
    y.append(gamma)
    print(f"{fname}: gamma={gamma:.2f}")

# Chuyển sang mảng numpy
X = np.array(X)
y = np.array(y)

# Chuẩn hóa đặc trưng đầu vào
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Khởi tạo và huấn luyện mô hình SVR
svr = SVR(kernel='rbf')
svr.fit(X_scaled, y)

# Lưu mô hình và scaler để dùng dự đoán sau này
joblib.dump(svr, os.path.join(MODEL_DIR, 'gamma_svr.pkl'))
joblib.dump(scaler, os.path.join(MODEL_DIR, 'scaler.pkl'))

print('Đã lưu model vào thư mục models/')
