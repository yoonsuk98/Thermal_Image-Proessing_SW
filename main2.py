import sys
import os
import cv2
import numpy as np

from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QLabel, QPushButton, QFileDialog,
    QVBoxLayout, QHBoxLayout, QGridLayout, QGroupBox, QCheckBox,
    QDoubleSpinBox, QSpinBox, QMessageBox, QSlider, QSizePolicy
)

# ----------------------------
# Click-to-seek Slider
# ----------------------------
class ClickableSlider(QSlider):
    """슬라이더 바 클릭 위치로 즉시 이동 가능한 QSlider"""
    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton and self.orientation() == Qt.Horizontal:
            x = event.pos().x()
            w = max(1, self.width())
            v = self.minimum() + (self.maximum() - self.minimum()) * x / w
            self.setValue(int(round(v)))
            event.accept()
            return
        super().mousePressEvent(event)


# ----------------------------
# Slider + Spinbox helpers
# ----------------------------
def make_int_control(label_text, vmin, vmax, vinit, step=1):
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
    lbl = QLabel(label_text)

    scale = int(round(1.0 / step))
    imin = int(round(vmin * scale))
    imax = int(round(vmax * scale))
    iinit = int(round(vinit * scale))

    sld = QSlider(Qt.Horizontal)
    sld.setRange(imin, imax)
    sld.setSingleStep(1)
    sld.setPageStep(max(1, int(scale * step)))
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
# Processing functions
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

    L0 *= w0
    L1 *= w1
    L2 *= w2

    current = L3
    current = cv2.pyrUp(current, dstsize=(L2.shape[1], L2.shape[0])) + L2
    current = cv2.pyrUp(current, dstsize=(L1.shape[1], L1.shape[0])) + L1
    current = cv2.pyrUp(current, dstsize=(L0.shape[1], L0.shape[0])) + L0

    return np.clip(current, 0, 255).astype(np.uint8)


def gaussian_denoise(img, ksize=5, sigma=0, grayscale=False):
    if ksize % 2 == 0:
        ksize += 1
    return cv2.GaussianBlur(img, (ksize, ksize), sigmaX=sigma, sigmaY=sigma)


def apply_clahe(img, clipLimit=2.0, tileGridSize=(8, 8)):
    clahe = cv2.createCLAHE(clipLimit=float(clipLimit), tileGridSize=tileGridSize)
    return clahe.apply(img)


# ----------------------------
# Helpers
# ----------------------------
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


def bgr_to_qpixmap(bgr: np.ndarray) -> QPixmap:
    """스케일링 없이 QPixmap만 생성"""
    if bgr is None:
        return QPixmap()
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    h, w, ch = rgb.shape
    bytes_per_line = ch * w
    qimg = QImage(rgb.data, w, h, bytes_per_line, QImage.Format_RGB888)
    return QPixmap.fromImage(qimg)


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

        # Timeline state
        self.total_frames = 0
        self.duration_sec = 0.0
        self.seeking = False

        # UI
        central = QWidget()
        self.setCentralWidget(central)

        self.lbl_orig = QLabel("Original")
        self.lbl_orig.setAlignment(Qt.AlignCenter)
        self.lbl_orig.setStyleSheet("background:#222; color:#ddd; border:1px solid #444;")
        self.lbl_orig.setMinimumSize(640, 480)
        self.lbl_orig.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        self.lbl_proc = QLabel("Processed")
        self.lbl_proc.setAlignment(Qt.AlignCenter)
        self.lbl_proc.setStyleSheet("background:#222; color:#ddd; border:1px solid #444;")
        self.lbl_proc.setMinimumSize(640, 480)
        self.lbl_proc.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

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

        # Timeline widget (video only) - hidden by default
        self.timeline_widget = QWidget()
        tl = QHBoxLayout(self.timeline_widget)
        tl.setContentsMargins(0, 0, 0, 0)
        tl.setSpacing(8)

        self.lbl_video_info = QLabel("")  # no "Video:-"
        self.lbl_video_info.setAlignment(Qt.AlignLeft)
        self.lbl_video_info.setMinimumWidth(240)

        self.sld_timeline = ClickableSlider(Qt.Horizontal)
        self.sld_timeline.setEnabled(False)
        self.sld_timeline.setRange(0, 0)
        self.sld_timeline.setSingleStep(1)
        self.sld_timeline.setPageStep(10)
        self.sld_timeline.setFixedHeight(14)  #  더 얇게
        self.sld_timeline.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)

        self.lbl_time = QLabel("")
        self.lbl_time.setAlignment(Qt.AlignRight)
        self.lbl_time.setMinimumWidth(220)

        tl.addWidget(self.lbl_video_info, 0)
        tl.addWidget(self.sld_timeline, 1)
        tl.addWidget(self.lbl_time, 0)

        self.timeline_widget.setVisible(False)

        # Controls
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
        left.addLayout(top_btns, 0)
        left.addLayout(views, 1)              #  영상 영역이 항상 제일 크게 늘어남
        left.addWidget(self.timeline_widget, 0)

        root = QHBoxLayout()
        root.addLayout(left, 3)
        root.addWidget(controls, 1)

        central.setLayout(root)

        self._connect_param_signals()

        # Timeline signals
        self.sld_timeline.sliderPressed.connect(self._timeline_pressed)
        self.sld_timeline.sliderReleased.connect(self._timeline_released)
        self.sld_timeline.valueChanged.connect(self._timeline_changed)

    # ----------------------------
    # UI builders
    # ----------------------------
    def _build_controls(self) -> QWidget:
        box = QGroupBox("Processing Controls")
        layout = QVBoxLayout()

        self.chk_denoise = QCheckBox("Enable Gaussian Denoise")
        self.chk_denoise.setChecked(True)

        self.chk_clahe = QCheckBox("Enable CLAHE (on Gray)")
        self.chk_clahe.setChecked(True)

        self.chk_sharpen = QCheckBox("Enable Laplacian Pyramid Sharpen")
        self.chk_sharpen.setChecked(True)

        layout.addWidget(self.chk_denoise)
        layout.addWidget(self.chk_clahe)
        layout.addWidget(self.chk_sharpen)

        g_gauss = QGroupBox("Gaussian")
        gl = QGridLayout()
        lbl_ksize, self.sld_ksize, self.sp_ksize = make_int_control("ksize", 1, 99, 5, step=2)
        lbl_sigma, self.sld_sigma, self.sp_sigma = make_float_control("sigma", 0.0, 50.0, 0.0, step=0.25, decimals=2)
        gl.addWidget(lbl_ksize, 0, 0); gl.addWidget(self.sld_ksize, 0, 1); gl.addWidget(self.sp_ksize, 0, 2)
        gl.addWidget(lbl_sigma, 1, 0); gl.addWidget(self.sld_sigma, 1, 1); gl.addWidget(self.sp_sigma, 1, 2)
        g_gauss.setLayout(gl)

        g_clahe = QGroupBox("CLAHE")
        cl = QGridLayout()
        lbl_clip, self.sld_clip, self.sp_clip = make_float_control("clipLimit", 0.1, 20.0, 2.0, step=0.1, decimals=2)
        lbl_tx, self.sld_tile_x, self.sp_tile_x = make_int_control("tileGrid X", 2, 64, 8, step=1)
        lbl_ty, self.sld_tile_y, self.sp_tile_y = make_int_control("tileGrid Y", 2, 64, 8, step=1)
        cl.addWidget(lbl_clip, 0, 0); cl.addWidget(self.sld_clip, 0, 1); cl.addWidget(self.sp_clip, 0, 2)
        cl.addWidget(lbl_tx,   1, 0); cl.addWidget(self.sld_tile_x, 1, 1); cl.addWidget(self.sp_tile_x, 1, 2)
        cl.addWidget(lbl_ty,   2, 0); cl.addWidget(self.sld_tile_y, 2, 1); cl.addWidget(self.sp_tile_y, 2, 2)
        g_clahe.setLayout(cl)

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
    # Display helpers (fill labels)
    # ----------------------------
    def _set_label_bgr(self, label: QLabel, bgr: np.ndarray):
        if bgr is None:
            label.clear()
            return
        pix = bgr_to_qpixmap(bgr)
        # 라벨 크기에 맞춰 꽉 채우기(비율 유지)
        label.setPixmap(pix.scaled(label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))

    def _refresh_views(self):
        if self.orig_img is not None:
            self._set_label_bgr(self.lbl_orig, self.orig_img)
        if self.proc_img is not None:
            self._set_label_bgr(self.lbl_proc, self.proc_img)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        # 창 크기 바뀌면 현재 프레임을 라벨 크기에 맞춰 다시 스케일
        self._refresh_views()

    # ----------------------------
    # Timeline / seeking
    # ----------------------------
    def _timeline_pressed(self):
        if self.mode == "video":
            self.seeking = True

    def _timeline_released(self):
        if self.mode != "video":
            self.seeking = False
            return
        self.seeking = False
        self._seek_to_frame(self.sld_timeline.value())

    def _timeline_changed(self, v):
        if self.mode != "video":
            return
        v = int(v)
        if self.seeking:
            self._update_time_label(current_frame=v)
        else:
            if self.cap is not None:
                self._seek_to_frame(v)

    def _seek_to_frame(self, frame_idx: int):
        if self.cap is None or self.total_frames <= 0:
            return

        frame_idx = int(np.clip(frame_idx, 0, self.total_frames - 1))
        was_running = self.timer.isActive()
        if was_running:
            self.timer.stop()

        self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ok, frame = self.cap.read()
        if ok and frame is not None:
            self.orig_img = frame
            self.proc_img = self.process_frame(frame)
            self._refresh_views()

            pos = int(self.cap.get(cv2.CAP_PROP_POS_FRAMES)) - 1
            self._sync_timeline(pos)
        else:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            self._sync_timeline(0)

        if was_running:
            interval = int(1000.0 / max(self.video_fps, 1.0))
            self.timer.start(interval)

    def _sync_timeline(self, current_frame: int):
        if self.total_frames <= 0:
            return
        current_frame = int(np.clip(current_frame, 0, self.total_frames - 1))
        self.sld_timeline.blockSignals(True)
        self.sld_timeline.setValue(current_frame)
        self.sld_timeline.blockSignals(False)
        self._update_time_label(current_frame=current_frame)

    def _update_time_label(self, current_frame: int):
        if self.video_fps is None or self.video_fps <= 0:
            self.video_fps = 30.0

        cur_sec = current_frame / self.video_fps
        total_sec = self.duration_sec

        def fmt(t):
            t = max(0, int(round(t)))
            m = t // 60
            s = t % 60
            return f"{m:02d}:{s:02d}"

        self.lbl_time.setText(
            f"{fmt(cur_sec)} / {fmt(total_sec)}  (frame {current_frame+1}/{self.total_frames})"
        )

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
        self._refresh_views()

        self.btn_save_img.setEnabled(True)
        self.btn_save_vid.setEnabled(False)
        self.btn_play.setEnabled(False)
        self.btn_pause.setEnabled(False)

        # timeline hide
        self.timeline_widget.setVisible(False)
        self.lbl_video_info.setText("")
        self.lbl_time.setText("")
        self.sld_timeline.setEnabled(False)
        self.sld_timeline.setRange(0, 0)
        self.sld_timeline.setValue(0)
        self.total_frames = 0
        self.duration_sec = 0.0

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

        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if frame_count <= 0:
            frame_count = 1
        self.total_frames = frame_count
        self.duration_sec = self.total_frames / max(self.video_fps, 1.0)

        # show timeline only for video
        self.timeline_widget.setVisible(True)
        self.sld_timeline.setEnabled(True)
        self.sld_timeline.setRange(0, self.total_frames - 1)
        self.sld_timeline.setValue(0)

        self.lbl_video_info.setText(
            f"{os.path.basename(path)}  |  {w}x{h}  |  FPS {self.video_fps:.2f}  |  Frames {self.total_frames}"
        )
        self._update_time_label(current_frame=0)

        ok, frame = cap.read()
        if not ok or frame is None:
            QMessageBox.critical(self, "Error", "Failed to read first frame.")
            self.stop_video()
            return

        self.orig_img = frame
        self.proc_img = self.process_frame(frame)
        self._refresh_views()

        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        self._sync_timeline(0)

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
        self._set_label_bgr(self.lbl_proc, self.proc_img)

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
        self._stop_video_writer_if_needed(reset_button=True, show_saved_msg=False)

    def _video_tick(self):
        if self.cap is None:
            return

        ok, frame = self.cap.read()
        if not ok or frame is None:
            self.timer.stop()
            self._stop_video_writer_if_needed(reset_button=True, show_saved_msg=True)
            return

        self.orig_img = frame
        self.proc_img = self.process_frame(frame)
        self._refresh_views()

        pos = int(self.cap.get(cv2.CAP_PROP_POS_FRAMES)) - 1
        if not self.seeking:
            self._sync_timeline(pos)

        if self.saving_video and self.video_writer is not None:
            self.video_writer.write(self.proc_img)

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

            self.video_writer = cv2.VideoWriter(out_path, fourcc, self.video_fps, self.video_size)
            if not self.video_writer.isOpened():
                self.video_writer = None
                QMessageBox.critical(self, "Error", "Failed to open VideoWriter.")
                return

            self.video_out_path = out_path

            if self.cap is not None:
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                self._sync_timeline(0)

            self.saving_video = True
            self.btn_save_vid.setText("Stop Video Save")

            if not self.timer.isActive():
                self.play_video()
        else:
            self._stop_video_writer_if_needed(reset_button=True, show_saved_msg=True)

    def _stop_video_writer_if_needed(self, reset_button: bool, show_saved_msg: bool = False):
        out_path = self.video_out_path

        if self.video_writer is not None:
            self.video_writer.release()
            self.video_writer = None

        was_saving = self.saving_video
        if self.saving_video:
            self.saving_video = False
            if reset_button:
                self.btn_save_vid.setText("Start Video Save")

        self.video_out_path = None

        if show_saved_msg and was_saving and out_path:
            QMessageBox.information(self, "Saved", f"Saved processed video:\n{out_path}")

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
                QMessageBox.critical(self, "Save Failed", f"cv2.imwrite returned False.\nPath:\n{out_path}")
                return
        except cv2.error as e:
            QMessageBox.critical(
                self,
                "Save Failed (OpenCV codec issue)",
                "OpenCV cannot write this image format in current environment.\n\n"
                f"{e}"
            )
            return

        QMessageBox.information(self, "Saved", f"Saved processed image:\n{out_path}")

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
