# src/poseapp/ui/mode_guided_panel.py
# ----------------------------------------------------------------------
# This module defines the Guided Mode side panel (dock) for PoseApp.
# It provides:
#   • An activity selector (drop-down)
#   • A “Start Guided Trial” button
#   • Real-time counters for reps and sets
#   • A built-in countdown display before a session starts
#
# The panel interacts closely with MainWindow, emitting a signal when
# the user initiates a guided trial and exposing helper functions to
# update counters dynamically.
# ----------------------------------------------------------------------

from __future__ import annotations
from PySide6 import QtCore, QtWidgets, QtGui  # Import Qt core, widgets, and GUI classes

# ---------------------- Guided Panel Definition ----------------------
class GuidedPanel(QtWidgets.QWidget):
    """
    Left dock panel for Guided Mode.
    - Emits `start_trial(key)` when the user presses the "Start Guided Trial" button.
    - Exposes `set_counters(reps_done, target_reps, set_idx)` for MainWindow updates.
    """

    # Define a Qt signal that carries a string (activity key)
    start_trial = QtCore.Signal(str)

    def __init__(self, parent: QtWidgets.QWidget | None = None):
        # Initialize QWidget
        super().__init__(parent)

        # ------------------ ACTIVITY SELECTOR ------------------
        self.activity_combo = QtWidgets.QComboBox()  # Dropdown for activity selection
        # The MainWindow populates this later using ACTIVITY_LIBRARY
        self.activity_combo.setMinimumWidth(160)     # Minimum size for consistent layout

        # "Start Guided Trial" button triggers a signal when pressed
        self.btn_start = QtWidgets.QPushButton("Start Guided Trial")
        self.btn_start.clicked.connect(self._emit_start)  # Connect to internal handler

        # ------------------ COUNTER LABELS ------------------
        self.lbl_set = QtWidgets.QLabel("Set: 1")           # Shows current set number
        self.lbl_reps = QtWidgets.QLabel("Reps: 0 / 0")     # Shows current rep count

        # Apply bold font to counters for better visibility
        f = self.lbl_set.font()
        f.setBold(True)
        self.lbl_set.setFont(f)
        self.lbl_reps.setFont(f)

        # ------------------ COUNTDOWN LABEL ------------------
        # Displayed inline (not as a popup dialog) before a guided session starts
        self.countdown_lbl = QtWidgets.QLabel("")
        self.countdown_lbl.setAlignment(QtCore.Qt.AlignCenter)
        # Styling for the countdown label
        self.countdown_lbl.setStyleSheet(
            "QLabel { background: #222; color: #ffd966; border-radius: 8px; padding: 8px; }"
        )
        self.countdown_lbl.hide()  # Hidden until countdown is triggered

        # ------------------ LAYOUT SETUP ------------------
        lay = QtWidgets.QVBoxLayout(self)     # Vertical layout for stacking elements

        # Form layout for labeled input (Activity:)
        form = QtWidgets.QFormLayout()
        form.addRow("Activity:", self.activity_combo)

        lay.addLayout(form)
        lay.addWidget(self.btn_start)
        lay.addSpacing(8)
        lay.addWidget(self.lbl_set)
        lay.addWidget(self.lbl_reps)
        lay.addStretch(1)                     # Push countdown to bottom
        lay.addWidget(self.countdown_lbl)

    # ---------------------- Public API (used by MainWindow) ----------------------

    def populate(self, items: list[tuple[str, str]]) -> None:
        """
        Populate dropdown with activities.
        items: list of (key, label)
        Example: [("squat", "Squat"), ("pushup", "Push-Up")]
        """
        self.activity_combo.clear()
        for k, label in items:
            self.activity_combo.addItem(label, k)   # label = visible text, k = hidden key data

    def select_index(self, idx: int):
        """Set the combo box index safely (ignore if invalid)."""
        try:
            self.activity_combo.setCurrentIndex(idx)
        except Exception:
            pass

    def current_key(self) -> str:
        """Return the currently selected activity key."""
        # Either from the data field or the visible text (fallback)
        return self.activity_combo.currentData() or self.activity_combo.currentText()

    def set_counters(self, reps_done: int, target_reps: int, set_idx: int) -> None:
        """
        Update the UI to reflect current progress (set number and reps).
        Called frequently during guided sessions.
        """
        self.lbl_set.setText(f"Set: {set_idx}")
        self.lbl_reps.setText(f"Reps: {reps_done} / {target_reps}")

    def run_countdown_blocking(self, seconds: int) -> None:
        """
        Show a countdown before starting a guided trial.

        This is "blocking" (execution halts until countdown finishes),
        but it keeps the UI responsive using an internal event loop.
        """
        self.countdown_lbl.show()       # Make label visible
        self.countdown_lbl.raise_()     # Bring to front (on top of other widgets)

        # Create a nested event loop and a timer guard
        loop = QtCore.QEventLoop(self)
        timer = QtCore.QTimer(self)
        timer.setSingleShot(True)

        remaining = seconds             # Remaining countdown time

        # Define nested function for tick updates
        def tick():
            nonlocal remaining
            if remaining <= 0:
                # Countdown finished → hide label and quit loop
                self.countdown_lbl.hide()
                loop.quit()
                return
            # Update label text with remaining seconds
            self.countdown_lbl.setText(f"Starting in {remaining}…")
            remaining -= 1
            # Schedule next tick after 1 second
            QtCore.QTimer.singleShot(1000, tick)

        # Start first tick immediately
        tick()

        # Safety timer to auto-quit loop if something goes wrong (seconds+1)
        timer.timeout.connect(loop.quit)
        timer.start((seconds + 1) * 1000)

        # Block execution here, but keep UI updating
        loop.exec()
        timer.stop()

    # ---------------------- Internal Handlers ----------------------
    @QtCore.Slot()
    def _emit_start(self):
        """
        Triggered when user clicks 'Start Guided Trial'.
        Validates selection and emits start_trial(key) signal.
        """
        key = self.current_key()
        if not key:
            # If user forgot to select activity, show friendly info popup
            QtWidgets.QMessageBox.information(self, "Guided", "Pick an activity to start.")
            return
        # Emit signal with selected activity key → handled by MainWindow
        self.start_trial.emit(key)
