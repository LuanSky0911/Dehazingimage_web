from flask import Flask, request, jsonify, render_template
import os, cv2, numpy as np, joblib
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler

# Khởi tạo ứng dụng Flask
app = Flask(__name__)
# Cấu hình đường dẫn thư mục
UPLOAD_FOLDER = 'static/uploads/'
PROCESSED_FOLDER = 'static/processed/'
MODEL_FOLDER = 'models/'
# Tạo các thư mục trên nếu chưa tồn tại
for folder in [UPLOAD_FOLDER, PROCESSED_FOLDER, MODEL_FOLDER]:
    os.makedirs(folder, exist_ok=True)

# Lớp xử lý ảnh bằng mô hình SVR
class ImageEnhancer:
    def __init__(self):
        # Đường dẫn file mô hình và scaler
        self.model_path = os.path.join(MODEL_FOLDER, 'gamma_svr.pkl')
        self.scaler_path = os.path.join(MODEL_FOLDER, 'scaler.pkl')
        # Tải mô hình đã huấn luyện nếu tồn tại, nếu không thì tạo mô hình trống
        self.gamma_model = joblib.load(self.model_path) if os.path.exists(self.model_path) else SVR(kernel='rbf')
        self.scaler = joblib.load(self.scaler_path) if os.path.exists(self.scaler_path) else StandardScaler()
    
    # Trích xuất đặc trưng từ ảnh: mean, std, entropy
    def extract_features(self, img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) # Chuyển sang ảnh xám
        mean, std = np.mean(gray), np.std(gray) # Trung bình và độ lệch chuẩn
        hist = cv2.calcHist([gray], [0], None, [256], [0, 256]).flatten()
        hist /= hist.sum() # Chuẩn hóa histogram
        entropy = -np.sum(hist * np.log2(hist + 1e-10)) # Entropy
        return np.array([mean, std, entropy])

# Hàm xử lý chính: dự đoán gamma và áp dụng gamma correction
    def enhance_image(self, img):
        feat = self.extract_features(img) # Trích đặc trưng
        gamma = self.gamma_model.predict(self.scaler.transform(feat.reshape(1, -1)))[0] # Dự đoán gamma
        gamma = np.clip(gamma, 0.5, 2.5)  # Giới hạn giá trị gamma hợp lý
        invGamma = 1.0 / gamma 
        table = np.array([((i / 255.0) ** invGamma) * 255 for i in range(256)]).astype("uint8") 
        return cv2.LUT(img, table)  # Áp dụng gamma correction

# Khởi tạo đối tượng xử lý ảnh
enhancer = ImageEnhancer()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    file = request.files.get('file')
    if not file or file.filename == '':
        return jsonify({'error': 'No file uploaded'}), 400
    path = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(path)
    filesize = os.path.getsize(path) / 1024  # tính KB
    return jsonify({
        'filepath': f'/{path}',
        'filename': file.filename,
        'size': f"{filesize:.1f}KB"
    })

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    filepath = data.get('filepath', '').lstrip('/')
    if not os.path.exists(filepath):
        return jsonify({'error': 'File not found'}), 404
    img = cv2.imread(filepath)
    result = enhancer.enhance_image(img)
    out_name = f"enhanced_{os.path.basename(filepath)}"
    out_path = os.path.join(PROCESSED_FOLDER, out_name)
    cv2.imwrite(out_path, result)
    return jsonify({'processed_filepath': f'/static/processed/{out_name}'})

if __name__ == '__main__':
    app.run(debug=True)
