import sys
import os
import cv2
import numpy as np

from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QLabel, QPushButton, QFileDialog,
    QVBoxLayout, QHBoxLayout, QGridLayout, QGroupBox, QCheckBox,
    QDoubleSpinBox, QSpinBox, QMessageBox, QSlider
)

# ----------------------------
# Slider + Spinbox helpers
# ----------------------------
def make_int_control(label_text, vmin, vmax, vinit, step=1):
    """
    int 파라미터용: (라벨, 슬라이더, 스핀박스) 묶음 생성
    """
    lbl = QLabel(label_text)

    sld = QSlider(Qt.Horizontal)
    sld.setRange(vmin, vmax)
    sld.setSingleStep(step)
    sld.setPageStep(step)
    sld.setValue(vinit)

    spn = QSpinBox()
    spn.setRange(vmin, vmax)
    spn.setSingleStep(step)
    spn.setValue(vinit)

    # 동기화(루프 방지)
    def sld_to_spn(v):
        if spn.value() != v:
            spn.blockSignals(True)
            spn.setValue(v)
            spn.blockSignals(False)

    def spn_to_sld(v):
        if sld.value() != v:
            sld.blockSignals(True)
            sld.setValue(v)
            sld.blockSignals(False)

    sld.valueChanged.connect(sld_to_spn)
    spn.valueChanged.connect(spn_to_sld)

    return lbl, sld, spn


def make_float_control(label_text, vmin, vmax, vinit, step=0.01, decimals=2):
    """
    float 파라미터용: QSlider는 int만 되므로 scale로 변환해서 동기화
    """
    lbl = QLabel(label_text)

    scale = int(round(1.0 / step))
    imin = int(round(vmin * scale))
    imax = int(round(vmax * scale))
    iinit = int(round(vinit * scale))

    sld = QSlider(Qt.Horizontal)
    sld.setRange(imin, imax)
    sld.setSingleStep(1)
    sld.setPageStep(max(1, int(scale * step)))  # 대충 한 step 정도
    sld.setValue(iinit)

    spn = QDoubleSpinBox()
    spn.setRange(vmin, vmax)
    spn.setDecimals(decimals)
    spn.setSingleStep(step)
    spn.setValue(vinit)

    def sld_to_spn(iv):
        fv = iv / scale
        if abs(spn.value() - fv) > 1e-12:
            spn.blockSignals(True)
            spn.setValue(fv)
            spn.blockSignals(False)

    def spn_to_sld(fv):
        iv = int(round(fv * scale))
        if sld.value() != iv:
            sld.blockSignals(True)
            sld.setValue(iv)
            sld.blockSignals(False)

    sld.valueChanged.connect(sld_to_spn)
    spn.valueChanged.connect(spn_to_sld)

    return lbl, sld, spn


# ----------------------------
# Your processing functions
# ----------------------------
def laplacian_pyramid_sharpen(inp_img, w0=1.5, w1=1.25, w2=1.1):
    img = inp_img.astype(np.float32)

    G0 = img
    G1 = cv2.pyrDown(G0)
    G2 = cv2.pyrDown(G1)
    G3 = cv2.pyrDown(G2)

    L0 = G0 - cv2.pyrUp(G1, dstsize=(G0.shape[1], G0.shape[0]))
    L1 = G1 - cv2.pyrUp(G2, dstsize=(G1.shape[1], G1.shape[0]))
    L2 = G2 - cv2.pyrUp(G3, dstsize=(G2.shape[1], G2.shape[0]))
    L3 = G3

    L0 = L0 * w0
    L1 = L1 * w1
    L2 = L2 * w2

    current = L3
    current = cv2.pyrUp(current, dstsize=(L2.shape[1], L2.shape[0]))
    current = current + L2

    current = cv2.pyrUp(current, dstsize=(L1.shape[1], L1.shape[0]))
    current = current + L1

    current = cv2.pyrUp(current, dstsize=(L0.shape[1], L0.shape[0]))
    current = current + L0

    sharp = np.clip(current, 0, 255).astype(np.uint8)
    return sharp


def gaussian_denoise(img, ksize=5, sigma=0, grayscale=False):
    if ksize % 2 == 0:
        ksize += 1
    denoised = cv2.GaussianBlur(img, (ksize, ksize), sigmaX=sigma, sigmaY=sigma)
    return denoised


def apply_clahe(img, clipLimit=2.0, tileGridSize=(8, 8)):
    clahe = cv2.createCLAHE(clipLimit=float(clipLimit), tileGridSize=tileGridSize)
    return clahe.apply(img)


# ----------------------------
# Helpers
# ----------------------------
def cv_bgr_to_qpixmap(bgr: np.ndarray, max_w=640, max_h=360) -> QPixmap:
    if bgr is None:
        return QPixmap()

    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    h, w, ch = rgb.shape
    bytes_per_line = ch * w
    qimg = QImage(rgb.data, w, h, bytes_per_line, QImage.Format_RGB888)
    pix = QPixmap.fromImage(qimg)
    return pix.scaled(max_w, max_h, Qt.KeepAspectRatio, Qt.SmoothTransformation)


def ensure_gray_u8(img_bgr: np.ndarray) -> np.ndarray:
    if img_bgr.ndim == 2:
        gray = img_bgr
    else:
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    if gray.dtype != np.uint8:
        gray = np.clip(gray, 0, 255).astype(np.uint8)
    return gray


def gray_to_bgr(gray: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)


# ----------------------------
# Main GUI
# ----------------------------
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("PyQt Image/Video Processor (Gaussian + CLAHE + Laplacian Sharpen)")
        self.resize(1400, 720)

        # State
        self.input_path = None
        self.mode = None  # "image" or "video"
        self.orig_img = None  # BGR
        self.proc_img = None  # BGR

        self.cap = None
        self.timer = QTimer(self)
        self.timer.timeout.connect(self._video_tick)

        self.video_writer = None
        self.saving_video = False
        self.video_out_path = None
        self.video_fps = None
        self.video_size = None

        # UI
        central = QWidget()
        self.setCentralWidget(central)

        self.lbl_orig = QLabel("Original")
        self.lbl_orig.setAlignment(Qt.AlignCenter)
        self.lbl_orig.setStyleSheet("background:#222; color:#ddd; border:1px solid #444;")
        self.lbl_orig.setMinimumSize(640, 360)

        self.lbl_proc = QLabel("Processed")
        self.lbl_proc.setAlignment(Qt.AlignCenter)
        self.lbl_proc.setStyleSheet("background:#222; color:#ddd; border:1px solid #444;")
        self.lbl_proc.setMinimumSize(640, 360)

        # Buttons
        btn_open = QPushButton("Open Image/Video")
        btn_open.clicked.connect(self.open_file)

        self.btn_play = QPushButton("Play")
        self.btn_play.clicked.connect(self.play_video)
        self.btn_pause = QPushButton("Pause")
        self.btn_pause.clicked.connect(self.pause_video)
        self.btn_play.setEnabled(False)
        self.btn_pause.setEnabled(False)

        self.btn_save_img = QPushButton("Save Processed Image")
        self.btn_save_img.clicked.connect(self.save_image)
        self.btn_save_img.setEnabled(False)

        self.btn_save_vid = QPushButton("Start Video Save")
        self.btn_save_vid.clicked.connect(self.toggle_video_save)
        self.btn_save_vid.setEnabled(False)

        # Controls group
        controls = self._build_controls()

        # Layouts
        top_btns = QHBoxLayout()
        top_btns.addWidget(btn_open)
        top_btns.addWidget(self.btn_play)
        top_btns.addWidget(self.btn_pause)
        top_btns.addStretch(1)
        top_btns.addWidget(self.btn_save_img)
        top_btns.addWidget(self.btn_save_vid)

        views = QHBoxLayout()
        views.addWidget(self.lbl_orig, 1)
        views.addWidget(self.lbl_proc, 1)

        left = QVBoxLayout()
        left.addLayout(top_btns)
        left.addLayout(views)

        root = QHBoxLayout()
        root.addLayout(left, 3)
        root.addWidget(controls, 1)

        central.setLayout(root)

        self._connect_param_signals()

    # ----------------------------
    # UI builders
    # ----------------------------
    def _build_controls(self) -> QWidget:
        box = QGroupBox("Processing Controls")
        layout = QVBoxLayout()

        # Enable stages
        self.chk_denoise = QCheckBox("Enable Gaussian Denoise")
        self.chk_denoise.setChecked(True)

        self.chk_clahe = QCheckBox("Enable CLAHE (on Gray)")
        self.chk_clahe.setChecked(True)

        self.chk_sharpen = QCheckBox("Enable Laplacian Pyramid Sharpen")
        self.chk_sharpen.setChecked(True)

        layout.addWidget(self.chk_denoise)
        layout.addWidget(self.chk_clahe)
        layout.addWidget(self.chk_sharpen)

        # ---------------- Gaussian ----------------
        g_gauss = QGroupBox("Gaussian")
        gl = QGridLayout()

        lbl_ksize, self.sld_ksize, self.sp_ksize = make_int_control("ksize", 1, 99, 5, step=2)
        lbl_sigma, self.sld_sigma, self.sp_sigma = make_float_control("sigma", 0.0, 50.0, 0.0, step=0.25, decimals=2)

        gl.addWidget(lbl_ksize, 0, 0); gl.addWidget(self.sld_ksize, 0, 1); gl.addWidget(self.sp_ksize, 0, 2)
        gl.addWidget(lbl_sigma, 1, 0); gl.addWidget(self.sld_sigma, 1, 1); gl.addWidget(self.sp_sigma, 1, 2)
        g_gauss.setLayout(gl)

        # ---------------- CLAHE ----------------
        g_clahe = QGroupBox("CLAHE")
        cl = QGridLayout()

        lbl_clip, self.sld_clip, self.sp_clip = make_float_control("clipLimit", 0.1, 20.0, 2.0, step=0.1, decimals=2)
        lbl_tx, self.sld_tile_x, self.sp_tile_x = make_int_control("tileGrid X", 2, 64, 8, step=1)
        lbl_ty, self.sld_tile_y, self.sp_tile_y = make_int_control("tileGrid Y", 2, 64, 8, step=1)

        cl.addWidget(lbl_clip, 0, 0); cl.addWidget(self.sld_clip, 0, 1); cl.addWidget(self.sp_clip, 0, 2)
        cl.addWidget(lbl_tx,   1, 0); cl.addWidget(self.sld_tile_x, 1, 1); cl.addWidget(self.sp_tile_x, 1, 2)
        cl.addWidget(lbl_ty,   2, 0); cl.addWidget(self.sld_tile_y, 2, 1); cl.addWidget(self.sp_tile_y, 2, 2)
        g_clahe.setLayout(cl)

        # ---------------- Sharpen ----------------
        g_sharp = QGroupBox("Laplacian Pyramid Sharpen")
        sl = QGridLayout()

        lbl_w0, self.sld_w0, self.sp_w0 = make_float_control("w0 (fine)",   0.0, 5.0, 1.50, step=0.05, decimals=2)
        lbl_w1, self.sld_w1, self.sp_w1 = make_float_control("w1 (mid)",    0.0, 5.0, 1.25, step=0.05, decimals=2)
        lbl_w2, self.sld_w2, self.sp_w2 = make_float_control("w2 (coarse)", 0.0, 5.0, 1.10, step=0.05, decimals=2)

        sl.addWidget(lbl_w0, 0, 0); sl.addWidget(self.sld_w0, 0, 1); sl.addWidget(self.sp_w0, 0, 2)
        sl.addWidget(lbl_w1, 1, 0); sl.addWidget(self.sld_w1, 1, 1); sl.addWidget(self.sp_w1, 1, 2)
        sl.addWidget(lbl_w2, 2, 0); sl.addWidget(self.sld_w2, 2, 1); sl.addWidget(self.sp_w2, 2, 2)
        g_sharp.setLayout(sl)

        layout.addSpacing(8)
        layout.addWidget(g_gauss)
        layout.addWidget(g_clahe)
        layout.addWidget(g_sharp)
        layout.addStretch(1)

        box.setLayout(layout)
        return box

    def _connect_param_signals(self):
        widgets = [
            self.chk_denoise, self.chk_clahe, self.chk_sharpen,

            # sliders + spinboxes
            self.sld_ksize, self.sp_ksize,
            self.sld_sigma, self.sp_sigma,

            self.sld_clip, self.sp_clip,
            self.sld_tile_x, self.sp_tile_x,
            self.sld_tile_y, self.sp_tile_y,

            self.sld_w0, self.sp_w0,
            self.sld_w1, self.sp_w1,
            self.sld_w2, self.sp_w2
        ]
        for w in widgets:
            if isinstance(w, QCheckBox):
                w.stateChanged.connect(self.refresh_once)
            else:
                w.valueChanged.connect(self.refresh_once)

    # ----------------------------
    # File handling
    # ----------------------------
    def open_file(self):
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Open Image or Video",
            "",
            "Media Files (*.png *.jpg *.jpeg *.bmp *.tif *.tiff *.mp4 *.avi *.mov *.mkv);;All Files (*.*)"
        )
        if not path:
            return

        self.stop_video()
        self.input_path = path

        ext = os.path.splitext(path)[1].lower()
        if ext in [".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"]:
            self.mode = "image"
            self.load_image(path)
        else:
            self.mode = "video"
            self.load_video(path)

    def load_image(self, path):
        img = cv2.imread(path, cv2.IMREAD_COLOR)
        if img is None:
            QMessageBox.critical(self, "Error", "Failed to read image.")
            return

        self.orig_img = img
        self.proc_img = self.process_frame(img)

        self.lbl_orig.setPixmap(cv_bgr_to_qpixmap(self.orig_img))
        self.lbl_proc.setPixmap(cv_bgr_to_qpixmap(self.proc_img))

        self.btn_save_img.setEnabled(True)
        self.btn_save_vid.setEnabled(False)
        self.btn_play.setEnabled(False)
        self.btn_pause.setEnabled(False)

    def load_video(self, path):
        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            QMessageBox.critical(self, "Error", "Failed to open video.")
            return

        self.cap = cap
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps is None or fps <= 0:
            fps = 30.0
        self.video_fps = float(fps)

        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.video_size = (w, h)

        ok, frame = cap.read()
        if not ok or frame is None:
            QMessageBox.critical(self, "Error", "Failed to read first frame.")
            self.stop_video()
            return

        self.orig_img = frame
        self.proc_img = self.process_frame(frame)

        self.lbl_orig.setPixmap(cv_bgr_to_qpixmap(self.orig_img))
        self.lbl_proc.setPixmap(cv_bgr_to_qpixmap(self.proc_img))

        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

        self.btn_save_img.setEnabled(False)
        self.btn_save_vid.setEnabled(True)
        self.btn_play.setEnabled(True)
        self.btn_pause.setEnabled(True)

    # ----------------------------
    # Processing pipeline
    # ----------------------------
    def process_frame(self, bgr: np.ndarray) -> np.ndarray:
        out = bgr.copy()

        if self.chk_denoise.isChecked():
            k = int(self.sp_ksize.value())
            sigma = float(self.sp_sigma.value())
            out = gaussian_denoise(out, ksize=k, sigma=sigma)

        if self.chk_clahe.isChecked():
            gray = ensure_gray_u8(out)
            clip = float(self.sp_clip.value())
            tx = int(self.sp_tile_x.value())
            ty = int(self.sp_tile_y.value())
            gray = apply_clahe(gray, clipLimit=clip, tileGridSize=(tx, ty))
            out = gray_to_bgr(gray)

        if self.chk_sharpen.isChecked():
            w0 = float(self.sp_w0.value())
            w1 = float(self.sp_w1.value())
            w2 = float(self.sp_w2.value())
            out = laplacian_pyramid_sharpen(out, w0=w0, w1=w1, w2=w2)

        return out

    def refresh_once(self):
        if self.orig_img is None:
            return
        self.proc_img = self.process_frame(self.orig_img)
        self.lbl_proc.setPixmap(cv_bgr_to_qpixmap(self.proc_img))

    # ----------------------------
    # Video playback / saving
    # ----------------------------
    def play_video(self):
        if self.mode != "video" or self.cap is None:
            return
        interval = int(1000.0 / max(self.video_fps, 1.0))
        self.timer.start(interval)

    def pause_video(self):
        self.timer.stop()

    def stop_video(self):
        self.timer.stop()
        if self.cap is not None:
            self.cap.release()
            self.cap = None
        self._stop_video_writer_if_needed(reset_button=True)

    def _video_tick(self):
        if self.cap is None:
            return

        ok, frame = self.cap.read()
        if not ok or frame is None:
            self.timer.stop()
            self._stop_video_writer_if_needed(reset_button=True)
            return

        self.orig_img = frame
        proc = self.process_frame(frame)
        self.proc_img = proc

        self.lbl_orig.setPixmap(cv_bgr_to_qpixmap(frame))
        self.lbl_proc.setPixmap(cv_bgr_to_qpixmap(proc))

        if self.saving_video and self.video_writer is not None:
            self.video_writer.write(proc)

    def toggle_video_save(self):
        if self.mode != "video":
            return

        if not self.saving_video:
            out_path, _ = QFileDialog.getSaveFileName(
                self, "Save Processed Video As", "", "MP4 (*.mp4);;AVI (*.avi)"
            )
            if not out_path:
                return

            ext = os.path.splitext(out_path)[1].lower()
            if ext == ".mp4":
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            else:
                fourcc = cv2.VideoWriter_fourcc(*"XVID")

            self.video_writer = cv2.VideoWriter(
                out_path, fourcc, self.video_fps, self.video_size
            )
            if not self.video_writer.isOpened():
                self.video_writer = None
                QMessageBox.critical(self, "Error", "Failed to open VideoWriter.")
                return

            if self.cap is not None:
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

            self.saving_video = True
            self.btn_save_vid.setText("Stop Video Save")

            if not self.timer.isActive():
                self.play_video()
        else:
            self._stop_video_writer_if_needed(reset_button=True)

    def _stop_video_writer_if_needed(self, reset_button: bool):
        if self.video_writer is not None:
            self.video_writer.release()
            self.video_writer = None

        if self.saving_video:
            self.saving_video = False
            if reset_button:
                self.btn_save_vid.setText("Start Video Save")

    # ----------------------------
    # Save image
    # ----------------------------
    def save_image(self):

        if self.mode != "image" or self.proc_img is None:
            return

        out_path, selected_filter = QFileDialog.getSaveFileName(
            self,
            "Save Processed Image As",
            "",
            "PNG (*.png);;JPG (*.jpg *.jpeg);;BMP (*.bmp)"
        )
        if not out_path:
            return

        root, ext = os.path.splitext(out_path)
        if ext == "":
            if "PNG" in selected_filter:
                out_path = root + ".png"
            elif "JPG" in selected_filter:
                out_path = root + ".jpg"
            elif "BMP" in selected_filter:
                out_path = root + ".bmp"
            else:
                out_path = root + ".png"

        try:
            ok = cv2.imwrite(out_path, self.proc_img)

            if not ok:
                QMessageBox.critical(
                    self,
                    "Save Failed",
                    f"cv2.imwrite returned False.\nPath:\n{out_path}"
                )
                return

        except cv2.error as e:
            QMessageBox.critical(
                self,
                "Save Failed (OpenCV codec issue)",
                "OpenCV cannot write this image format in current environment.\n\n"
                f"{e}"
            )
            return

        QMessageBox.information(self, "Saved", f"Saved:\n{out_path}")

    def closeEvent(self, event):
        self.stop_video()
        event.accept()


def main():
    app = QApplication(sys.argv)
    w = MainWindow()
    w.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()