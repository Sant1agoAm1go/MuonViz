#!/usr/bin/env python3
"""
Muon Visualizer — PyQt5 + pyqtgraph.opengl
Сохрани как muon_visualizer_pyqt.py и запусти: python muon_visualizer_pyqt.py
"""
import sys
import math
import numpy as np
import pandas as pd
from PyQt5 import QtWidgets, QtCore, QtGui
import pyqtgraph as pg
import pyqtgraph.opengl as gl

# -------------------- Утилиты --------------------
def sph_to_dir(theta_deg, phi_deg):
    # theta: zenith angle (deg), phi: azimuth (deg)
    t = math.radians(float(theta_deg))
    p = math.radians(float(phi_deg))
    x = math.sin(t) * math.cos(p)
    y = math.sin(t) * math.sin(p)
    z = math.cos(t)
    return np.array([x, y, z], dtype=float)

def color_from_norm(norm):
    # norm in [0,1] -> RGB tuple in 0..1
    # use a blue->green->yellow->red gradient
    # simple piecewise
    if norm <= 0:
        return (0.0, 0.0, 1.0)
    if norm >= 1:
        return (1.0, 0.0, 0.0)
    if norm < 0.33:
        # blue -> green
        f = norm / 0.33
        return (0.0 * (1-f) + 0.0*f, 0.0*(1-f) + 1.0*f, 1.0*(1-f) + 0.0*f)
    if norm < 0.66:
        # green -> yellow
        f = (norm - 0.33) / (0.33)
        return (f, 1.0, 0.0)
    else:
        # yellow -> red
        f = (norm - 0.66) / (0.34)
        return (1.0, 1.0 - f, 0.0)

# -------------------- Main Window --------------------
class MuonVisualizer(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Muon Visualizer — PyQt (offline)")
        self.resize(1200, 800)

        # Data holders
        self.detectors = []   # list of dict {id,x,y,z}
        self.tracks = []      # list of dict {Tetta,Phi,Count,corr?,Count_corr?}
        self.eff_table = None

        # UI
        w = QtWidgets.QWidget()
        self.setCentralWidget(w)
        h = QtWidgets.QHBoxLayout(w)

        # Left controls
        left = QtWidgets.QFrame()
        left.setMaximumWidth(360)
        left.setMinimumWidth(280)
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

        # Controls: max rays, len scale, threshold, log color
        left_layout.addSpacing(6)
        self.spin_max_rays = QtWidgets.QSpinBox(); self.spin_max_rays.setRange(1,100000); self.spin_max_rays.setValue(1000)
        self.spin_len_scale = QtWidgets.QDoubleSpinBox(); self.spin_len_scale.setRange(0.1,1000); self.spin_len_scale.setDecimals(2); self.spin_len_scale.setValue(5.0)
        self.spin_min_count = QtWidgets.QDoubleSpinBox(); self.spin_min_count.setRange(0.0,1e9); self.spin_min_count.setDecimals(3); self.spin_min_count.setValue(1.0)
        self.check_log = QtWidgets.QCheckBox("Использовать лог-шкалу цвета")
        form = QtWidgets.QFormLayout()
        form.addRow("Макс лучей:", self.spin_max_rays)
        form.addRow("Масштаб длины:", self.spin_len_scale)
        form.addRow("Порог Count (мин):", self.spin_min_count)
        form.addRow(self.check_log)
        left_layout.addLayout(form)

        # Action buttons
        self.btn_draw = QtWidgets.QPushButton("Отобразить")
        self.btn_clear = QtWidgets.QPushButton("Очистить сцены")
        left_layout.addWidget(self.btn_draw)
        left_layout.addWidget(self.btn_clear)

        left_layout.addStretch(1)

        # Status text
        self.status = QtWidgets.QLabel("Ready")
        left_layout.addWidget(self.status)

        # 3D area
        self.view = gl.GLViewWidget()
        self.view.opts['distance'] = 300
        self.view.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        self.view.setCameraPosition(elevation=20, azimuth=-60)
        grid = gl.GLGridItem()
        grid.setSize(400,400)
        grid.setSpacing(20,20)
        self.view.addItem(grid)

        h.addWidget(left)
        h.addWidget(self.view, 1)

        # Connect signals
        self.btn_load_det.clicked.connect(self.load_detectors)
        self.btn_load_tracks.clicked.connect(self.load_tracks)
        self.btn_load_eff.clicked.connect(self.load_eff)
        self.btn_draw.clicked.connect(self.on_draw)
        self.btn_clear.clicked.connect(self.on_clear)

        # Keep references to drawn items (to clear later)
        self._drawn = []

    # ----------------- Loaders -----------------
    def load_detectors(self):
        fn, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Open Detectors.txt", "", "Text files (*.txt *.dat);;All files (*)")
        if not fn:
            return
        try:
            with open(fn, 'r', encoding='utf-8') as f:
                text = f.read()
        except Exception:
            with open(fn, 'r', encoding='cp1251') as f:
                text = f.read()
        dets = []
        for line in text.splitlines():
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            parts = line.split()
            if len(parts) >= 4:
                try:
                    idd = parts[0]
                    x = float(parts[1]); y = float(parts[2]); z = float(parts[3])
                    dets.append({'id': str(idd), 'x': x, 'y': y, 'z': z})
                except:
                    pass
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
        fn, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Open Tracks_distr", "", "Text files (*.txt *.dat *.csv);;All files (*)")
        if not fn:
            return
        # Try pandas read_csv with whitespace or comma
        try:
            # try whitespace-separated
            df = pd.read_csv(fn, sep=r"\s+", header=None, engine="python")
        except Exception:
            try:
                df = pd.read_csv(fn, header=None)
            except Exception as e:
                QtWidgets.QMessageBox.warning(self, "Ошибка", f"Не удалось прочитать файл треков:\n{e}")
                return
        # Expect at least 3 columns: Tetta, Phi, Count
        if df.shape[1] < 3:
            QtWidgets.QMessageBox.warning(self, "Ошибка", "Файл треков должен содержать минимум 3 колонки: Tetta Phi Count")
            return
        df = df.iloc[:, :5]  # allow up to 5 cols
        df.columns = ['Tetta', 'Phi', 'Count', 'maybe_corr', 'maybe_Count_corr'][:df.shape[1]]
        df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=['Tetta','Phi','Count'], how='any')
        # convert types
        df['Tetta'] = df['Tetta'].astype(float)
        df['Phi'] = df['Phi'].astype(float)
        df['Count'] = df['Count'].astype(float)
        tracks = []
        for _, r in df.iterrows():
            obj = {'Tetta': float(r['Tetta']), 'Phi': float(r['Phi']), 'Count': float(r['Count'])}
            # optional columns if present
            if 'maybe_corr' in df.columns and not pd.isna(r.get('maybe_corr')):
                try:
                    obj['corr'] = float(r.get('maybe_corr'))
                except: pass
            if 'maybe_Count_corr' in df.columns and not pd.isna(r.get('maybe_Count_corr')):
                try:
                    obj['Count_corr'] = float(r.get('maybe_Count_corr'))
                except: pass
            tracks.append(obj)
        self.tracks = tracks
        self.status.setText(f"Треков загружено: {len(self.tracks)}")

    def load_eff(self):
        fn, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Open EffCorFile_Tracks.dat", "", "Text files (*.txt *.dat *.csv);;All files (*)")
        if not fn:
            return
        try:
            eff = pd.read_csv(fn, sep=r"\s+", header=None, engine="python")
        except Exception as e:
            QtWidgets.QMessageBox.warning(self, "Ошибка", f"Не удалось прочитать EffCorFile_Tracks:\n{e}")
            return
        # expect columns: degree, rad, calib_count, corr
        if eff.shape[1] < 4:
            QtWidgets.QMessageBox.warning(self, "Ошибка", "EffCorFile_Tracks должен иметь как минимум 4 колонки")
            return
        eff = eff.iloc[:, :4]
        eff.columns = ['T_deg','T_rad','calib','corr']
        eff = eff.sort_values('T_deg')
        self.eff_table = eff
        self.status.setText(f"Таблица эффективности загружена ({len(eff)} строк)")

    # ---------------- Drawing ----------------
    def _draw_detectors(self):
        # clear previous detector markers
        self._remove_drawn()
        for d in self.detectors:
            # sphere
            md = gl.MeshData.sphere(rows=12, cols=12, radius=1.8)
            mesh = gl.GLMeshItem(meshdata=md, smooth=True, color=(1.0,0.6,0.2,1.0), shader='shaded', glOptions='opaque')
            mesh.translate(d['x'], d['y'], d['z'])
            self.view.addItem(mesh)
            self._drawn.append(mesh)
            # label as small text overlay using GLText? fallback: simple 2D label not implemented; skip
        self.status.setText(f"Отрисованы детекторы: {len(self.detectors)}")

    def _remove_drawn(self):
        for it in self._drawn:
            try:
                self.view.removeItem(it)
            except Exception:
                pass
        self._drawn = []

    def apply_eff_correction(self):
        # if eff table present, interpolate corr by theta degrees
        if self.eff_table is None or len(self.eff_table)==0:
            # nothing to do — use Count as Count_corr
            for r in self.tracks:
                if 'Count_corr' not in r:
                    r['Count_corr'] = r['Count']
            return
        t_vals = self.eff_table['T_deg'].values
        c_vals = self.eff_table['corr'].values
        # intermittent: use numpy.interp
        for r in self.tracks:
            th = float(r['Tetta'])
            corr = float(np.interp(th, t_vals, c_vals, left=c_vals[0], right=c_vals[-1]))
            r['corr'] = corr
            r['Count_corr'] = r['Count'] * corr

    def on_draw(self):
        if not self.detectors:
            QtWidgets.QMessageBox.warning(self, "Ошибка", "Загрузите Detectors.txt прежде чем рисовать")
            return
        if not self.tracks:
            QtWidgets.QMessageBox.warning(self, "Ошибка", "Загрузите Tracks файл прежде чем рисовать")
            return
        # apply correction
        self.apply_eff_correction()

        # compute values
        counts = np.array([r.get('Count_corr', r['Count']) for r in self.tracks], dtype=float)
        if len(counts)==0:
            QtWidgets.QMessageBox.warning(self, "Ошибка", "Нет валидных треков для отображения")
            return
        vmin = float(counts.min()); vmax = float(counts.max())

        # get detector origin
        idx = max(0, self.det_combo.currentIndex())
        origin = self.detectors[idx] if idx < len(self.detectors) else {'x':0,'y':0,'z':0}
        ox, oy, oz = origin['x'], origin['y'], origin['z']

        max_rays = int(self.spin_max_rays.value())
        min_count = float(self.spin_min_count.value())
        len_scale = float(self.spin_len_scale.value())
        use_log = bool(self.check_log.isChecked())

        # sort by count desc
        sorted_tracks = sorted(self.tracks, key=lambda r: r.get('Count_corr', r['Count']), reverse=True)
        rendered = 0

        # clear non-light/model items
        self._remove_drawn()
        # draw detectors (so they remain)
        self._draw_detectors()

        # we will create arrays of line segments and colors to speed up drawing
        # but pyqtgraph GLLinePlotItem draws a single polyline; we draw many small lines (two points each).
        for r in sorted_tracks:
            val = r.get('Count_corr', r['Count'])
            if val < min_count:
                continue
            if rendered >= max_rays:
                break
            dirv = sph_to_dir(r['Tetta'], r['Phi'])  # x,y,z direction
            # length normalized
            if use_log:
                # log normalization
                norm = (math.log10(val+1) - math.log10(vmin+1)) / (math.log10(vmax+1)-math.log10(vmin+1)+1e-12)
            else:
                norm = (val - vmin) / (vmax - vmin + 1e-12)
            color_rgb = color_from_norm(max(0.0, min(1.0, norm)))
            length = len_scale * (1.0 + norm*10.0)

            start = np.array([ox, oy, oz], dtype=float)
            end = start + dirv * length

            pts = np.vstack([start, end])

            # draw line
            plt = gl.GLLinePlotItem(pos=pts, color=(*color_rgb, 1.0), width=2.0, antialias=True)
            self.view.addItem(plt)
            self._drawn.append(plt)

            # draw small sphere at end
            md = gl.MeshData.sphere(rows=8, cols=8, radius=length*0.05)
            mesh = gl.GLMeshItem(meshdata=md, smooth=True, color=(*color_rgb,1.0), shader='shaded', glOptions='opaque')
            mesh.translate(float(end[0]), float(end[1]), float(end[2]))
            self.view.addItem(mesh)
            self._drawn.append(mesh)

            rendered += 1

        self.status.setText(f"Отрисовано лучей: {rendered} (детектор {origin['id']})")

    def on_clear(self):
        self._remove_drawn()
        # re-add grid
        self.view.clear()
        grid = gl.GLGridItem()
        grid.setSize(400,400)
        grid.setSpacing(20,20)
        self.view.addItem(grid)
        self.detectors = []
        self.tracks = []
        self.eff_table = None
        self.det_combo.clear()
        self.status.setText("Очищено")

# ----------------- main -----------------
def main():
    app = QtWidgets.QApplication(sys.argv)
    m = MuonVisualizer()
    m.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()
