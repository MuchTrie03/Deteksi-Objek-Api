import sys
from PyQt5.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QPushButton, QLabel, QFileDialog, QHBoxLayout, QGroupBox, QFrame
)
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import QTimer, Qt
import cv2
from ultralytics import YOLO


class YOLOApp(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Fire Detection")
        self.setGeometry(100, 100, 1200, 700)

        # Inisialisasi model YOLO
        self.model = YOLO('best.pt')

        # Layout utama
        main_layout = QHBoxLayout()

        # Bagian kiri: Kontrol
        control_group = QGroupBox("Kontrol")
        control_layout = QVBoxLayout()

        self.open_file_btn = QPushButton("Pilih Gambar")
        self.open_file_btn.clicked.connect(self.open_file)
        self.open_file_btn.setStyleSheet("padding: 10px;")

        self.open_camera_btn = QPushButton("Nyalakan Kamera")
        self.open_camera_btn.clicked.connect(self.start_camera)
        self.open_camera_btn.setStyleSheet("padding: 10px;")

        self.stop_camera_btn = QPushButton("Matikan Kamera")
        self.stop_camera_btn.clicked.connect(self.stop_camera)
        self.stop_camera_btn.setStyleSheet("padding: 10px;")

        control_layout.addWidget(self.open_file_btn)
        control_layout.addWidget(self.open_camera_btn)
        control_layout.addWidget(self.stop_camera_btn)
        control_layout.addStretch()

        control_group.setLayout(control_layout)

        # Bagian kanan: Tampilan kamera/gambar
        display_group = QGroupBox("Tampilan")
        display_layout = QVBoxLayout()

        self.display_label = QLabel("Tampilan akan muncul di sini.")
        self.display_label.setStyleSheet("background-color: #000; color: #fff; border: 1px solid #ccc;")
        self.display_label.setAlignment(Qt.AlignCenter)

        display_layout.addWidget(self.display_label)
        display_group.setLayout(display_layout)

        # Garis pembatas antara kontrol dan tampilan
        divider = QFrame()
        divider.setFrameShape(QFrame.VLine)
        divider.setFrameShadow(QFrame.Sunken)

        # Tambahkan ke layout utama
        main_layout.addWidget(control_group, 1)
        main_layout.addWidget(divider)
        main_layout.addWidget(display_group, 4)

        self.setLayout(main_layout)

        # Inisialisasi variabel kamera
        self.cap = None
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)

    def open_file(self):
        # Hentikan kamera jika aktif
        self.stop_camera()

        # Pilih file gambar
        file_path, _ = QFileDialog.getOpenFileName(self, "Pilih Gambar", "", "Image Files (*.png *.jpg *.jpeg)")
        if file_path:
            # Baca dan proses gambar menggunakan YOLO
            img = cv2.imread(file_path)
            results = self.model.predict(source=img, imgsz=640, conf=0.6)
            annotated_img = results[0].plot()

            # Konversi gambar ke QPixmap
            height, width, channel = annotated_img.shape
            bytes_per_line = channel * width
            q_img = QImage(annotated_img.data, width, height, bytes_per_line, QImage.Format_RGB888).rgbSwapped()
            pixmap = QPixmap.fromImage(q_img)

            # Tampilkan gambar di label
            self.display_label.setPixmap(pixmap)

    def start_camera(self):
        # Hentikan kamera jika sudah aktif sebelumnya
        self.stop_camera()

        # Nyalakan kamera
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            self.display_label.setText("Kamera tidak terdeteksi.")
            return

        # Bersihkan tampilan sebelumnya
        self.display_label.clear()
        self.timer.start(30)  # Perbarui setiap 30 ms

    def update_frame(self):
        # Ambil frame dari kamera
        ret, frame = self.cap.read()
        if not ret:
            self.display_label.setText("Gagal membaca frame dari kamera.")
            return

        # Proses frame dengan YOLO
        results = self.model.predict(source=frame, imgsz=640, conf=0.6)
        annotated_frame = results[0].plot()

        # Konversi frame ke QPixmap
        height, width, channel = annotated_frame.shape
        bytes_per_line = channel * width
        q_img = QImage(annotated_frame.data, width, height, bytes_per_line, QImage.Format_RGB888).rgbSwapped()
        pixmap = QPixmap.fromImage(q_img)

        # Tampilkan frame di label
        self.display_label.setPixmap(pixmap)

    def stop_camera(self):
        # Matikan kamera dan hentikan timer
        if self.cap and self.cap.isOpened():
            self.cap.release()
        self.timer.stop()
        self.display_label.clear()
        self.display_label.setText("Kamera dimatikan.")

    def closeEvent(self, event):
        # Bersihkan resource saat aplikasi ditutup
        self.stop_camera()
        cv2.destroyAllWindows()
        event.accept()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = YOLOApp()
    window.show()
    sys.exit(app.exec_())
