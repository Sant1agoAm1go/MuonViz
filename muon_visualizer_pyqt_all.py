#!/usr/bin/env python3
"""
Muon Visualizer — PyQt5 + pyqtgraph.opengl
Версия: batch-отрисовка + "рисовать из всех детекторов"
Сохрани как muon_visualizer_pyqt.py и запусти: python muon_visualizer_pyqt.py
"""
import sys
import math
import numpy as np
import pandas as pd
from PyQt5 import QtWidgets, QtCore
import pyqtgraph as pg
import pyqtgraph.opengl as gl

# -------------------- Утилиты --------------------
def sph_to_dir(theta_deg, phi_deg):
    t = math.radians(float(theta_deg))
    p = math.radians(float(phi_deg))
    x = math.sin(t) * math.cos(p)
    y = math.sin(t) * math.sin(p)
    z = math.cos(t)
    return np.array([x, y, z], dtype=float)

def color_from_norm(norm):
    # piecewise blue->green->yellow->red
    if norm <= 0:
        return (0.0, 0.0, 1.0)
    if norm >= 1:
        return (1.0, 0.0, 0.0)
    if norm < 0.33:
        f = norm / 0.33
        r = 0.0
        g = f
        b = 1.0 - f
        return (r, g, b)
    if norm < 0.66:
        f = (norm - 0.33) / 0.33
        r = f
        g = 1.0
        b = 0.0
        return (r, g, b)
    else:
        f = (norm - 0.66) / 0.34
        r = 1.0
        g = 1.0 - f
        b = 0.0
        return (r, g, b)

# -------------------- Main Window --------------------
class MuonVisualizer(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Muon Visualizer")
        self.resize(1250, 840)

        # Data
        self.detectors = []   # [{'id','x','y','z'}]
        self.tracks = []      # [{'Tetta','Phi','Count','corr','Count_corr'}]
        self.eff_table = None

        # UI layout
        w = QtWidgets.QWidget()
        self.setCentralWidget(w)
        h = QtWidgets.QHBoxLayout(w)

        # Left controls
        left = QtWidgets.QFrame()
        left.setMaximumWidth(380)
        left.setMinimumWidth(320)
        left_layout = QtWidgets.QVBoxLayout(left)
        left_layout.setContentsMargins(8,8,8,8)

        # Load buttons
        self.btn_load_det = QtWidgets.QPushButton("Загрузить Detectors.txt")
        self.btn_load_tracks = QtWidgets.QPushButton("Загрузить Tracks(.dat/.csv)")
        self.btn_load_eff = QtWidgets.QPushButton("Загрузить EffCorFile_Tracks.dat (опц.)")
        left_layout.addWidget(self.btn_load_det)
        left_layout.addWidget(self.btn_load_tracks)
        left_layout.addWidget(self.btn_load_eff)

        # Detector selector
        left_layout.addSpacing(6)
        left_layout.addWidget(QtWidgets.QLabel("Выбрать детектор (начало лучей):"))
        self.det_combo = QtWidgets.QComboBox()
        left_layout.addWidget(self.det_combo)

        # Options
        left_layout.addSpacing(6)
        self.spin_max_rays = QtWidgets.QSpinBox(); self.spin_max_rays.setRange(1,5000000); self.spin_max_rays.setValue(5000)
        self.spin_len_scale = QtWidgets.QDoubleSpinBox(); self.spin_len_scale.setRange(0.01,1000.0); self.spin_len_scale.setDecimals(2); self.spin_len_scale.setValue(5.0)
        self.spin_min_count = QtWidgets.QDoubleSpinBox(); self.spin_min_count.setRange(0.0,1e9); self.spin_min_count.setDecimals(3); self.spin_min_count.setValue(1.0)
        self.check_log = QtWidgets.QCheckBox("Использовать лог-шкалу цвета")
        self.check_all_det = QtWidgets.QCheckBox("Рисовать из всех детекторов")
        form = QtWidgets.QFormLayout()
        form.addRow("Макс лучей (всего):", self.spin_max_rays)
        form.addRow("Масштаб длины:", self.spin_len_scale)
        form.addRow("Минимальный Count (исходный):", self.spin_min_count)
        form.addRow(self.check_log)
        form.addRow(self.check_all_det)
        left_layout.addLayout(form)

        # Action buttons
        self.btn_draw = QtWidgets.QPushButton("Отобразить")
        self.btn_clear = QtWidgets.QPushButton("Очистить сцену")
        left_layout.addWidget(self.btn_draw)
        left_layout.addWidget(self.btn_clear)
        left_layout.addStretch(1)

        # Status
        self.status = QtWidgets.QLabel("Ready")
        left_layout.addWidget(self.status)

        # 3D view
        self.view = gl.GLViewWidget()
        self.view.opts['distance'] = 400
        self.view.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        self.view.setCameraPosition(elevation=20, azimuth=-60)
        grid = gl.GLGridItem()
        grid.setSize(600,600)
        grid.setSpacing(20,20)
        self.view.addItem(grid)

        h.addWidget(left)
        h.addWidget(self.view, 1)

        # Signals
        self.btn_load_det.clicked.connect(self.load_detectors)
        self.btn_load_tracks.clicked.connect(self.load_tracks)
        self.btn_load_eff.clicked.connect(self.load_eff)
        self.btn_draw.clicked.connect(self.on_draw)
        self.btn_clear.clicked.connect(self.on_clear)

        # Drawn items
        self._drawn = []
        # chunk size (lines per batch) - tuneable
        self._chunk_lines = 20000

    # ---------------- Loaders ----------------
    def load_detectors(self):
        fn, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Open Detectors.txt", "", "Text files (*.txt *.dat);;All files (*)")
        if not fn:
            return
        try:
            with open(fn, 'r', encoding='utf-8') as f: text = f.read()
        except Exception:
            with open(fn, 'r', encoding='cp1251') as f: text = f.read()
        dets = []
        for line in text.splitlines():
            line = line.strip()
            if not line or line.startswith('#'): continue
            parts = line.split()
            if len(parts) >= 4:
                try:
                    idd = parts[0]
                    x = float(parts[1]); y = float(parts[2]); z = float(parts[3])
                    dets.append({'id': str(idd), 'x': x, 'y': y, 'z': z})
                except: pass
        if not dets:
            QtWidgets.QMessageBox.warning(self, "Ошибка", "Не удалось распознать Detectors.txt")
            return
        self.detectors = dets
        self.det_combo.clear()
        for d in self.detectors:
            self.det_combo.addItem(f"{d['id']} ({d['x']:.2f},{d['y']:.2f},{d['z']:.2f})")
        self.status.setText(f"Детекторов загружено: {len(self.detectors)}")
        self._draw_detectors()

    def load_tracks(self):
        fn, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Open Tracks", "", "Text files (*.txt *.dat *.csv);;All files (*)")
        if not fn: return
        try:
            df = pd.read_csv(fn, sep=r"\s+", header=None, engine="python")
        except Exception:
            try:
                df = pd.read_csv(fn, header=None)
            except Exception as e:
                QtWidgets.QMessageBox.warning(self, "Ошибка", f"Не удалось прочитать файл треков:\n{e}")
                return
        if df.shape[1] < 3:
            QtWidgets.QMessageBox.warning(self, "Ошибка", "Файл треков должен содержать минимум 3 колонки: Tetta Phi Count")
            return
        df = df.iloc[:, :5]
        df.columns = ['Tetta','Phi','Count','maybe_corr','maybe_Count_corr'][:df.shape[1]]
        df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=['Tetta','Phi','Count'], how='any')
        df['Tetta'] = df['Tetta'].astype(float)
        df['Phi'] = df['Phi'].astype(float)
        df['Count'] = df['Count'].astype(float)
        tracks = []
        for _, r in df.iterrows():
            obj = {'Tetta': float(r['Tetta']), 'Phi': float(r['Phi']), 'Count': float(r['Count'])}
            if 'maybe_corr' in df.columns and not pd.isna(r.get('maybe_corr')):
                try: obj['corr'] = float(r.get('maybe_corr'))
                except: pass
            if 'maybe_Count_corr' in df.columns and not pd.isna(r.get('maybe_Count_corr')):
                try: obj['Count_corr'] = float(r.get('maybe_Count_corr'))
                except: pass
            tracks.append(obj)
        self.tracks = tracks
        self.status.setText(f"Треков загружено: {len(self.tracks)}")

    def load_eff(self):
        fn, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Open EffCorFile_Tracks.dat", "", "Text files (*.txt *.dat *.csv);;All files (*)")
        if not fn: return
        try:
            eff = pd.read_csv(fn, sep=r"\s+", header=None, engine="python")
        except Exception as e:
            QtWidgets.QMessageBox.warning(self, "Ошибка", f"Не удалось прочитать EffCorFile_Tracks:\n{e}")
            return
        if eff.shape[1] < 4:
            QtWidgets.QMessageBox.warning(self, "Ошибка", "EffCorFile_Tracks должен иметь как минимум 4 колонки")
            return
        eff = eff.iloc[:, :4]
        eff.columns = ['T_deg','T_rad','calib','corr']
        eff = eff.sort_values('T_deg')
        self.eff_table = eff
        self.status.setText(f"Таблица эффективности загружена ({len(eff)} строк)")

    # ---------------- Drawing utils ----------------
    def _draw_detectors(self):
        # remove previous detectors among drawn items
        # keep lights & grid (we don't track them)
        # clear and redraw detectors
        self._remove_drawn()
        for d in self.detectors:
            md = gl.MeshData.sphere(rows=12, cols=12, radius=1.8)
            mesh = gl.GLMeshItem(meshdata=md, smooth=True, color=(1.0,0.6,0.2,1.0), shader='shaded', glOptions='opaque')
            mesh.translate(d['x'], d['y'], d['z'])
            self.view.addItem(mesh)
            self._drawn.append(mesh)
        self.status.setText(f"Отрисованы детекторы: {len(self.detectors)}")

    def _remove_drawn(self):
        for it in self._drawn:
            try: self.view.removeItem(it)
            except Exception: pass
        self._drawn = []

    def apply_eff_correction(self):
        if self.eff_table is None or len(self.eff_table)==0:
            for r in self.tracks:
                r['Count_corr'] = r.get('Count', 0.0)
            return
        t_vals = self.eff_table['T_deg'].values
        c_vals = self.eff_table['corr'].values
        for r in self.tracks:
            th = float(r['Tetta'])
            corr = float(np.interp(th, t_vals, c_vals, left=c_vals[0], right=c_vals[-1]))
            r['corr'] = corr
            r['Count_corr'] = r['Count'] * corr

    # ---------------- Main draw ----------------
    def on_draw(self):
        if not self.detectors:
            QtWidgets.QMessageBox.warning(self, "Ошибка", "Загрузите Detectors.txt прежде чем рисовать")
            return
        if not self.tracks:
            QtWidgets.QMessageBox.warning(self, "Ошибка", "Загрузите Tracks прежде чем рисовать")
            return

        # apply eff correction
        self.apply_eff_correction()

        # prepare counts (for normalization use corrected)
        all_vals_corr = np.array([r.get('Count_corr', r['Count']) for r in self.tracks], dtype=float)
        if len(all_vals_corr)==0:
            QtWidgets.QMessageBox.warning(self, "Ошибка", "Нет валидных треков для отображения")
            return
        vmin = float(np.nanmin(all_vals_corr))
        vmax = float(np.nanmax(all_vals_corr))

        # threshold based on original Count (not corrected)
        min_count = float(self.spin_min_count.value())

        # choose tracks that satisfy threshold (by original Count)
        filtered_tracks = [r for r in self.tracks if (r.get('Count', 0.0) >= min_count)]
        if len(filtered_tracks) == 0:
            QtWidgets.QMessageBox.information(self, "Пустой набор", f"Нет треков с Count >= {min_count}")
            return

        # sort by corrected value (desc)
        filtered_tracks.sort(key=lambda r: r.get('Count_corr', r['Count']), reverse=True)

        # origin setup
        all_detectors_mode = bool(self.check_all_det.isChecked())
        detector_list = self.detectors if all_detectors_mode else [self.detectors[max(0, self.det_combo.currentIndex())]]

        max_rays_total = int(self.spin_max_rays.value())
        len_scale = float(self.spin_len_scale.value())
        use_log = bool(self.check_log.isChecked())

        # we'll draw up to max_rays_total lines in total (summed across detectors if all_detectors_mode)
        # Prepare raysToDraw (top N by Count_corr)
        raysToDraw = filtered_tracks[:max_rays_total]  # choose top-N tracks (by Count_corr sorted before)
        if len(raysToDraw) == 0:
            QtWidgets.QMessageBox.information(self, "Пустой набор", "Нет треков после отбора.")
            return

        # Clear previous drawn items and redraw detectors
        self._remove_drawn()
        self._draw_detectors()

        # Build large buffers for GLLinePlotItem in chunks
        # For each detector, we'll add the same rays but starting at different detector positions.
        # To respect max_rays_total, we will duplicate rays across detectors until totalSegments reaches max_rays_total.
        total_allowed = max_rays_total
        total_added = 0
        chunk_lines = self._chunk_lines

        # We'll iterate detectors and rays, pushing segments into arrays, creating a GLLinePlotItem per chunk.
        pos_list = []   # list of arrays (Npoints x 3) for current chunk
        color_list = [] # list of arrays (Npoints x 4) for current chunk

        def flush_chunk():
            nonlocal pos_list, color_list, total_added
            if not pos_list:
                return
            P = np.vstack(pos_list).astype(np.float32)
            C = np.vstack(color_list).astype(np.float32)
            # create one GLLinePlotItem with mode='lines' (pairs)
            plt = gl.GLLinePlotItem(pos=P, color=C, width=1.2, antialias=True, mode='lines')
            self.view.addItem(plt)
            self._drawn.append(plt)
            pos_list = []
            color_list = []

        # iterate detectors and rays
        for det in detector_list:
            if total_added >= total_allowed:
                break
            ox, oy, oz = det['x'], det['y'], det['z']
            # iterate rays
            for r in raysToDraw:
                if total_added >= total_allowed:
                    break
                # compute parameters
                dirv = sph_to_dir(r['Tetta'], r['Phi'])
                val_corr = r.get('Count_corr', r['Count'])
                # normalization for color/length: use vmin/vmax from all_vals_corr
                if use_log:
                    norm = (math.log10(val_corr+1) - math.log10(vmin+1)) / (math.log10(vmax+1) - math.log10(vmin+1) + 1e-12)
                else:
                    norm = (val_corr - vmin) / (vmax - vmin + 1e-12)
                norm = float(np.clip(norm, 0.0, 1.0))
                color_rgb = color_from_norm(norm)
                length = len_scale * (1.0 + norm * 10.0)

                start = np.array([ox, oy, oz], dtype=np.float32)
                end = (start + dirv * length).astype(np.float32)

                # want to put [start, end] pair into pos_list and colors for both vertices
                pos_list.append(start)
                pos_list.append(end)
                rgba = (color_rgb[0], color_rgb[1], color_rgb[2], 1.0)
                color_list.append(rgba)
                color_list.append(rgba)

                total_added += 1

                # if chunk reached certain number of lines (pairs)
                if len(pos_list) >= (chunk_lines * 2):
                    # flush: create GLLinePlotItem
                    P = np.vstack(pos_list).astype(np.float32)
                    C = np.vstack(color_list).astype(np.float32)
                    plt = gl.GLLinePlotItem(pos=P, color=C, width=1.2, antialias=True, mode='lines')
                    self.view.addItem(plt)
                    self._drawn.append(plt)
                    pos_list = []
                    color_list = []

        # flush remaining
        if pos_list:
            P = np.vstack(pos_list).astype(np.float32)
            C = np.vstack(color_list).astype(np.float32)
            plt = gl.GLLinePlotItem(pos=P, color=C, width=1.2, antialias=True, mode='lines')
            self.view.addItem(plt)
            self._drawn.append(plt)

        # Optionally draw small markers at line ends (skip for huge batches for perf)
        # We skip per-end spheres for large total_added; draw them only if total_added <= 2000
        if total_added <= 2000:
            # draw small spheres at ends (one per drawn line)
            # Recompute quickly the ends for last chunked approach (costly); alternative: recompute from raysToDraw & detectors
            idx_drawn = 0
            for det in detector_list:
                if idx_drawn >= total_added: break
                ox, oy, oz = det['x'], det['y'], det['z']
                for r in raysToDraw:
                    if idx_drawn >= total_added: break
                    val_corr = r.get('Count_corr', r['Count'])
                    if use_log:
                        norm = (math.log10(val_corr+1) - math.log10(vmin+1)) / (math.log10(vmax+1) - math.log10(vmin+1) + 1e-12)
                    else:
                        norm = (val_corr - vmin) / (vmax - vmin + 1e-12)
                    norm = float(np.clip(norm, 0.0, 1.0))
                    color_rgb = color_from_norm(norm)
                    length = len_scale * (1.0 + norm * 10.0)
                    start = np.array([ox, oy, oz], dtype=float)
                    end = (start + sph_to_dir(r['Tetta'], r['Phi']) * length).astype(float)
                    md = gl.MeshData.sphere(rows=8, cols=8, radius=max(0.05, length*0.03))
                    mesh = gl.GLMeshItem(meshdata=md, smooth=True, color=(*color_rgb,1.0), shader='shaded', glOptions='opaque')
                    mesh.translate(float(end[0]), float(end[1]), float(end[2]))
                    self.view.addItem(mesh)
                    self._drawn.append(mesh)
                    idx_drawn += 1

        self.status.setText(f"Отрисовано линий: {total_added} (детекторов: {len(detector_list)})")

    def on_clear(self):
        self._remove_drawn()
        # reset grid
        self.view.clear()
        grid = gl.GLGridItem()
        grid.setSize(600,600)
        grid.setSpacing(20,20)
        self.view.addItem(grid)
        self.detectors = []
        self.tracks = []
        self.eff_table = None
        self.det_combo.clear()
        self.status.setText("Очищено")

# ---------------- main ----------------
def main():
    app = QtWidgets.QApplication(sys.argv)
    win = MuonVisualizer()
    win.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()
