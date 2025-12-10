# src/poseapp/ui/right_panel.py
# ----------------------------------------------------------------------
# This module defines the “Right Panel” — a live plotting widget
# that displays rolling angle values (like joint angles) over time.
# It uses Matplotlib embedded inside a Qt widget to visualize
# pose-estimation data in real-time.
# ----------------------------------------------------------------------

from __future__ import annotations
import math
import time
from typing import Dict, Any, List
from collections import deque             # Efficient time-series buffer for rolling window

import numpy as np
from PySide6 import QtCore, QtWidgets     # Qt for UI components
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure      # Matplotlib for plotting inside Qt

# _now() alias for monotonic time (not affected by system clock changes)
_now = time.monotonic


# ---------------------- Embedded Matplotlib Canvas ----------------------
class _MatCanvas(FigureCanvas):
    """A simple wrapper around Matplotlib FigureCanvas to manage live lines."""

    def __init__(self):
        # Create a Matplotlib Figure (4x3 inches) with tight layout to avoid clipping labels
        self.fig = Figure(figsize=(4, 3), tight_layout=True)
        super().__init__(self.fig)               # Initialize the Qt canvas base class
        self.ax = self.fig.add_subplot(111)      # Create one subplot (1 row, 1 col, 1 plot)
        self.lines: Dict[str, Any] = {}          # Stores named Line2D objects: key → plotted line

    def ensure_line(self, name: str):
        """
        Ensure that a line exists for a given angle name.
        If missing, create one and register it in `self.lines`.
        """
        if name in self.lines:                   # If already exists, reuse the line
            return self.lines[name]
        (ln,) = self.ax.plot([], [], label=name) # Create new empty line with a label
        self.lines[name] = ln                    # Store in dict for quick access
        return ln

    def remove_missing(self, keep: List[str]):
        """
        Remove plot lines that are no longer active (not in 'keep' list).
        This keeps the plot clean when angles disappear from overlay.
        """
        for k in list(self.lines.keys()):        # Iterate over copy (avoid modifying while looping)
            if k not in keep:
                try:
                    self.lines[k].remove()       # Remove from the Axes if still present
                except Exception:
                    pass                         # Ignore if line already gone
                self.lines.pop(k, None)          # Delete from dictionary

    def reset_axes_labels(self):
        """Reset axis labels for time and angle units."""
        self.ax.set_xlabel("Time (s, last 10)")
        self.ax.set_ylabel("Angle (deg)")


# ---------------------- RightPanel Widget ----------------------
class RightPanel(QtWidgets.QWidget):
    """
    A QWidget showing live rolling plots of joint angles.

    Features:
    • Auto-pick mode — automatically plots all currently detected angles from overlay.
    • Manual mode — user can manually add specific angles via dropdown.
    • Maintains short rolling history (~10 s) using fixed-size deques.
    """

    # Signal sent to MainWindow to request re-selection of angles
    request_pick_angles = QtCore.Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumWidth(320)                # Ensure panel has enough space beside video feed

        # ---------------------- Buffers and State ----------------------
        self._buffer_s = 10.0                    # Rolling window in seconds (last 10s of data)
        self._maxlen = 2000                      # Max samples per series (controls deque size)
        self._series: Dict[str, deque] = {}      # angle_key → deque[(timestamp, value)]
        self._active_keys: List[str] = []        # currently plotted angles
        self._available_now: List[str] = []      # angles visible on overlay in current frame

        # ---------------------- Plot Canvas ----------------------
        self.canvas = _MatCanvas()               # Create custom Matplotlib canvas
        self.ax = self.canvas.ax                 # Shortcut for axis
        self.canvas.reset_axes_labels()          # Label axes

        # ---------------------- UI Controls ----------------------
        # Checkbox: toggles between auto-pick and manual modes
        self.chk_auto = QtWidgets.QCheckBox("Auto-pick (show all angles)")
        self.chk_auto.setChecked(True)           # Enabled by default
        self.chk_auto.stateChanged.connect(self._on_auto_toggle)

        # Dropdown: shows available angle keys detected on overlay
        self.cmb_keys = QtWidgets.QComboBox()
        self.cmb_keys.setEditable(True)          # Allow user to type custom key
        self.cmb_keys.setInsertPolicy(QtWidgets.QComboBox.NoInsert)
        self.cmb_keys.setSizeAdjustPolicy(QtWidgets.QComboBox.AdjustToContents)

        # Add and Clear buttons
        self.btn_add = QtWidgets.QPushButton("+ Add")
        self.btn_clear = QtWidgets.QPushButton("Clear")

        # ---------------------- Signals ----------------------
        # If user picks from dropdown index
        self.cmb_keys.currentIndexChanged.connect(self._on_pick_from_combo_idx)
        # If dropdown is editable, pressing Enter triggers add
        if self.cmb_keys.lineEdit():
            self.cmb_keys.lineEdit().returnPressed.connect(
                lambda: self._on_pick_from_combo(self.cmb_keys.currentText().strip())
            )
        # Button actions
        self.btn_add.clicked.connect(lambda: self._on_pick_from_combo(self.cmb_keys.currentText().strip()))
        self.btn_clear.clicked.connect(self._on_clear_clicked)

        # ---------------------- Layout ----------------------
        # Top row (auto-pick checkbox)
        row = QtWidgets.QHBoxLayout()
        row.addWidget(self.chk_auto)
        row.addStretch(1)                        # Pushes the checkbox to the left

        # Second row (dropdown and buttons)
        row2 = QtWidgets.QHBoxLayout()
        row2.addWidget(QtWidgets.QLabel("Angle:"))
        row2.addWidget(self.cmb_keys, 1)
        row2.addWidget(self.btn_add)
        row2.addWidget(self.btn_clear)

        # Main layout
        lay = QtWidgets.QVBoxLayout(self)
        lay.addLayout(row)                       # Add auto-pick toggle row
        lay.addLayout(row2)                      # Add angle selection row
        lay.addWidget(self.canvas, 1)            # Add plot canvas, stretch to fill

        # Initial draw (prepares empty plot)
        self.canvas.draw_idle()

    # ---------- Methods Called from MainWindow ----------
    def set_detected_from_main(self, shown_keys: List[str]) -> None:
        """
        Called each frame by MainWindow to tell which angles are visible on overlay.
        Updates dropdown items accordingly.
        """
        # Remove duplicates while preserving order
        seen = set()
        avail = [k for k in shown_keys if (k and (k not in seen) and not seen.add(k))]
        if avail == self._available_now:         # No change since last update → skip
            return

        self._available_now = avail

        # Refresh dropdown menu with current overlay keys
        cur = self.cmb_keys.currentText()
        self.cmb_keys.blockSignals(True)         # Prevent triggering signals during update
        self.cmb_keys.clear()
        self.cmb_keys.addItems(self._available_now)
        # Try to restore user’s selection if possible
        if cur:
            i = self.cmb_keys.findText(cur)
            if i >= 0:
                self.cmb_keys.setCurrentIndex(i)
            else:
                self.cmb_keys.setEditText(cur)
        self.cmb_keys.blockSignals(False)

        # Note: In auto mode we do NOT immediately remove missing lines here;
        # the pruning happens in update_angles() after sample expiration.

    # ---------- Live Data Update ----------
    @QtCore.Slot(dict)
    def update_angles(self, ang: dict):
        """Append new angle samples and redraw rolling plot."""
        if not isinstance(ang, dict):
            return
        t = _now()                               # Timestamp for current sample batch

        # Auto mode → build dynamic list of active angle keys
        if self.chk_auto.isChecked():
            # Gather recent keys with still-active buffers
            recent_keys = [k for k, buf in self._series.items() if buf]
            desired = []
            seen = set()
            # Combine old active, still-recent, and newly visible overlay keys
            for k in self._active_keys + recent_keys + self._available_now:
                if k and (k not in seen):
                    desired.append(k); seen.add(k)
            self._set_active_keys(desired)

        # Append valid numeric samples for each active key
        for k in self._active_keys:
            v = ang.get(k, None)
            if v is None or not isinstance(v, (int, float)) or not math.isfinite(v):
                continue
            buf = self._series.get(k)
            if buf is None:
                buf = deque(maxlen=self._maxlen) # Create new deque if this key is new
                self._series[k] = buf
            buf.append((t, float(v)))            # Append (timestamp, value)

        # Remove old samples beyond time window
        self._prune_old(t)
        # Update plot with latest data
        self._update_plot(t)

    # ---------- UI Event Handlers ----------
    def _on_auto_toggle(self, _state: int):
        """When auto-pick is toggled ON, re-enable all currently visible angles."""
        if self.chk_auto.isChecked():
            recent = [k for k, buf in self._series.items() if buf]
            self._set_active_keys(recent + self._available_now)

    def _on_pick_from_combo_idx(self, idx: int):
        """When user selects an angle from dropdown."""
        if idx >= 0:
            self._on_pick_from_combo(self.cmb_keys.itemText(idx))

    def _on_pick_from_combo(self, key: str):
        """Adds manually typed or selected key to active plot list."""
        key = (key or "").strip()
        if not key:
            return
        if key not in self._active_keys:
            self._set_active_keys(self._active_keys + [key])

    def _on_clear_clicked(self):
        """Clears all active plots and resets axes."""
        self._set_active_keys([])
        self.ax.cla()                            # Clear axis drawing area
        self.canvas.reset_axes_labels()          # Reset labels after clearing
        self.canvas.draw_idle()

    # ---------- Internal Helpers ----------
    def _set_active_keys(self, keys: List[str]):
        """Synchronize plotted lines with given list of active angle keys."""
        # Dedupe while preserving order
        seen = set()
        new_keys = [k for k in keys if (k and (k not in seen) and not seen.add(k))]

        # Remove lines that are no longer needed
        self.canvas.remove_missing(new_keys)

        # Ensure lines & buffers exist for each key
        for k in new_keys:
            self.canvas.ensure_line(k)
            if k not in self._series:
                self._series[k] = deque(maxlen=self._maxlen)

        # Update internal list
        self._active_keys = new_keys

        # Update legend display (if there are lines)
        try:
            if new_keys:
                self.ax.legend(loc="upper right", fontsize=8)
            else:
                if getattr(self.ax, "legend_", None):
                    self.ax.legend_.remove()
        except Exception:
            pass

        self.canvas.draw_idle()

    def _prune_old(self, now: float):
        """Drop samples older than rolling window (self._buffer_s seconds)."""
        cutoff = now - self._buffer_s
        for buf in self._series.values():
            while buf and buf[0][0] < cutoff:    # If sample timestamp < cutoff
                buf.popleft()                    # Remove oldest sample

    def _update_plot(self, now: float):
        """Redraw all active lines with updated data."""
        if not self._active_keys:
            # No active keys → clear time axis and refresh blank canvas
            self.ax.set_xlim(-self._buffer_s, 0.0)
            self.canvas.draw_idle()
            return

        ymins, ymaxs = [], []
        for k in self._active_keys:
            seq = self._series.get(k)
            if not seq:
                ln = self.canvas.lines.get(k)
                if ln:
                    ln.set_data([], [])
                continue
            # Convert timestamps into relative time (seconds ago)
            xs = [tt - now for (tt, _) in seq]   # X-axis: negative seconds (e.g. -9 → 1s ago)
            ys = [vv for (_, vv) in seq]         # Y-axis: angle values
            self.canvas.lines[k].set_data(xs, ys)
            if ys:
                ymins.append(min(ys)); ymaxs.append(max(ys))

        # Adjust axis limits dynamically based on visible data
        self.ax.set_xlim(-self._buffer_s, 0.0)
        if ymins and ymaxs:
            ymin, ymax = min(ymins), max(ymaxs)
            pad = 1.0 if ymin == ymax else 0.05 * (ymax - ymin)  # Add padding for readability
            self.ax.set_ylim(ymin - pad, ymax + pad)

        self.canvas.draw_idle()                 # Trigger non-blocking redraw

    def clear(self):
        """Completely reset the panel — used when session restarts."""
        self._series.clear()                    # Remove all data buffers
        self._active_keys.clear()               # Reset active key list
        self.canvas.lines.clear()               # Remove plotted line objects
        self.ax.cla()                           # Clear axes
        self.canvas.reset_axes_labels()         # Restore labels
        self.canvas.draw_idle()                 # Redraw empty canvas
