# src/poseapp/ui/session_summary.py
# ---------------------------------------------------------------
# This module defines a Qt-based dialog window that displays a
# summary of a completed session — including activities, sets,
# repetitions, and performance percentages.
#
# It’s typically launched after completing a Guided or Freestyle
# workout, showing a summary table and allowing exports.
# ---------------------------------------------------------------

from __future__ import annotations          # Enables forward references for type hints (useful for Qt)
from typing import Dict, Any, List          # Typing support for structured data
import numpy as np                          # Used for calculating mean percentages
from PySide6 import QtCore, QtWidgets       # PySide6 provides the GUI components for Qt apps

# ------------------ SESSION SUMMARY DIALOG ------------------
class SessionSummaryDialog(QtWidgets.QDialog):
    # Custom Qt signal to notify parent widgets when the user clicks "Export"
    request_export = QtCore.Signal()  # The parent can connect this signal to trigger export logic elsewhere

    def __init__(self, parent=None, title: str = "Session Summary"):
        # Initialize the QDialog (modal popup window)
        super().__init__(parent)
        self.setWindowTitle(title)           # Set dialog title shown on top bar
        self.resize(560, 420)                # Define default window dimensions (width x height)

        # ------------------ TABLE SETUP ------------------
        # Create a 6-column table for displaying summary rows.
        # Start with 0 rows (rows will be dynamically inserted later).
        self.table = QtWidgets.QTableWidget(0, 6)

        # Define table headers to describe each column of session data.
        self.table.setHorizontalHeaderLabels(
            ["Activity", "Set", "Reps", "Target", "Mean %", "Final %"]
        )

        # Make sure the last column (Final %) stretches to fill the remaining space.
        self.table.horizontalHeader().setStretchLastSection(True)

        # ------------------ METADATA AND BUTTONS ------------------
        self.lbl_meta = QtWidgets.QLabel("")      # Displays meta-information (e.g., session time, user name)
        self.btn_export = QtWidgets.QPushButton("Export…")  # Button for exporting results (CSV/JSON)
        self.btn_close = QtWidgets.QPushButton("Close")     # Button for closing the dialog

        # When Export button is clicked, emit the `request_export` signal
        self.btn_export.clicked.connect(self.request_export.emit)

        # When Close button is clicked, accept() closes the dialog gracefully
        self.btn_close.clicked.connect(self.accept)

        # ------------------ BUTTON LAYOUT ------------------
        btns = QtWidgets.QHBoxLayout()       # Create a horizontal layout for buttons
        btns.addStretch(1)                   # Add stretch to push buttons to the right side
        btns.addWidget(self.btn_export)      # Add Export button to layout
        btns.addWidget(self.btn_close)       # Add Close button to layout

        # ------------------ MAIN LAYOUT ------------------
        lay = QtWidgets.QVBoxLayout(self)    # Create a vertical layout for the dialog
        lay.addWidget(self.lbl_meta)         # Add the metadata label at the top
        lay.addWidget(self.table, 1)         # Add the main table (1 = stretch factor)
        lay.addLayout(btns)                  # Add button layout at the bottom

    # ------------------ POPULATE TABLE ------------------
    def populate_from_payloads(self, rows: List[Dict[str, Any]], meta: str = ""):
        """
        Fills the summary table with performance data from session payloads.

        Parameters:
            rows: List of dictionaries (one per set/activity)
                Example entry:
                    {
                        "activity": "squat",
                        "label": "Squat",
                        "set_idx": 1,
                        "reps_counted": 10,
                        "target_reps": 12,
                        "rep_scores": [0.92, 0.93, 0.91],
                        "final_percent": 94.2
                    }

            meta: Optional text displayed above the table (e.g., "Session Duration: 12m 30s")
        """

        # Clear any existing data (reset table for new session summary)
        self.table.setRowCount(0)

        # Iterate through each session result payload
        for p in rows:
            # Determine which row to insert at (append to the end)
            r = self.table.rowCount()
            self.table.insertRow(r)

            # Compute mean of repetition scores (convert 0–1 range to 0–100%)
            mean_pct = float(np.mean(p.get("rep_scores", []) or [0.0]) * 100.0)

            # Prepare all columns as a list of strings
            items = [
                p.get("label", p.get("activity", "")),           # Activity name
                str(p.get("set_idx", 1)),                        # Set number
                str(p.get("reps_counted", 0)),                   # Actual repetitions completed
                str(p.get("target_reps", 0)),                    # Target repetitions (goal)
                f"{mean_pct:.1f}",                               # Mean % score (formatted with 1 decimal)
                f"{p.get('final_percent', 0.0):.1f}",            # Final overall % score
            ]

            # Add each item into the respective table cell
            for c, text in enumerate(items):
                self.table.setItem(r, c, QtWidgets.QTableWidgetItem(text))

        # If meta information (e.g., summary note) is provided, display it above the table
        if meta:
            self.lbl_meta.setText(meta)
