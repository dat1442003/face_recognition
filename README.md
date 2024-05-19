Các bước chạy chương trình
- Bước 1: Chạy câu lệnh 'pip install -r requirements.txt' để có thể tải các thư viện cần thiết
- Bước 2: Chạy câu lệnh 'python src/align_dataset_mtcnn.py  Dataset/FaceData/raw Dataset/FaceData/processed --image_size 160--margin 32  --random_order --gpu_memory_fraction 0.25'
dùng để cắt ảnh giữ lại khuôn mặt qua data
+ Bước 3:Tạo thư mục Model trong backend và tải thư mục Model tại đây [https://drive.google.com/drive/folders/1rIQJMTd5xaZc-4BL4PTXNW58CY-fCO4O?usp=sharing](https://drive.google.com/drive/folders/1ZCqVwc4kU9xSymva_PgDOOuOuocbv1Q4?usp=sharing)
Sau đó dán vào thư mục Model trong code 
* Bước 4: Chạy câu lệnh 'python src/classifier.py TRAIN Dataset/FaceData/processed Models/20180402-114759.pb Models/facemodel.pkl --batch_size 1000' dùng để train model data
* Bước 5: 'python src/face_rec_cam.py' để test chương trình
  
Chạy giao diện chương trình
* Bước 1: Chạy câu lệnh 'cd font-end'
* Bước 2: Để cài đặt thư viện cần thiết 'npm install'
* Bước 3: Chạy câu lệnh để load sever 'python src/face_rec_flask.py'
* Bước 4: 'npm start'
