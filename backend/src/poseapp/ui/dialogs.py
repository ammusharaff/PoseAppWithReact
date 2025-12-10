# src/poseapp/ui/dialogs.py
import os  # filesystem utils (used to show folder names in the list)
from typing import List  # type hints for lists
from PySide6 import QtCore, QtWidgets  # Qt core types and widgets
from ..io.session_logger import SAVE_ROOT_TEMP  # base temp folder (used by caller; not directly here)

class SettingsDialog(QtWidgets.QDialog):
    def __init__(self, parent=None, current_backend_idx: int = 0):
        super().__init__(parent)  # initialize QDialog with optional parent
        self.setWindowTitle("Settings"); self.setModal(True)  # title + modal dialog (blocks parent)
        lab = QtWidgets.QLabel("Model backend:")  # static text label
        self.cmb_model = QtWidgets.QComboBox()  # dropdown for backend choice
        self.cmb_model.addItems(["MoveNet (auto)", "MediaPipe"])  # two options user can pick
        self.cmb_model.setCurrentIndex(current_backend_idx)  # preselect current backend index
        btn_ok = QtWidgets.QPushButton("OK")  # confirm button
        btn_cancel = QtWidgets.QPushButton("Cancel")  # cancel/close button
        btns = QtWidgets.QHBoxLayout()  # horizontal layout for buttons
        btns.addStretch(1)  # push buttons to the right
        btns.addWidget(btn_cancel)  # add Cancel
        btns.addWidget(btn_ok)  # add OK
        lay = QtWidgets.QVBoxLayout(self)  # main vertical layout for the dialog (parent = self)
        lay.addWidget(lab)  # row: label
        lay.addWidget(self.cmb_model)  # row: combo box
        lay.addStretch(1)  # spacer to push buttons down
        lay.addLayout(btns)  # row: buttons
        btn_ok.clicked.connect(self.accept)  # OK -> QDialog.accept() (returns Accepted)
        btn_cancel.clicked.connect(self.reject)  # Cancel -> QDialog.reject() (returns Rejected)
    def selected_backend_index(self) -> int:
        return self.cmb_model.currentIndex()  # helper for callers to read chosen index

class ExportDialog(QtWidgets.QDialog):
    def __init__(self, parent=None, session_paths: List[str] = None):
        super().__init__(parent)  # init QDialog with optional parent
        self.setWindowTitle("Export Sessions"); self.setModal(True)  # title + modal
        self.paths = session_paths or []  # incoming list of session directories (may be empty)
        self.list = QtWidgets.QListWidget()  # list widget to show selectable sessions
        for p in self.paths:  # populate the list
            item = QtWidgets.QListWidgetItem(os.path.basename(p))  # show only folder name
            item.setData(QtCore.Qt.UserRole, p)  # store the full path as item user data
            self.list.addItem(item)  # insert into widget
        self.btn_export = QtWidgets.QPushButton("Export Selected")  # primary action button
        self.btn_close  = QtWidgets.QPushButton("Close")  # close without exporting
        btns = QtWidgets.QHBoxLayout()  # horizontal layout for the bottom buttons
        btns.addStretch(1)  # push buttons to the right
        btns.addWidget(self.btn_close)  # add Close
        btns.addWidget(self.btn_export)  # add Export Selected
        lay = QtWidgets.QVBoxLayout(self)  # main vertical layout
        lay.addWidget(QtWidgets.QLabel("Temporary sessions available to export:"))  # header label above the list
        lay.addWidget(self.list)  # the list of sessions
        lay.addLayout(btns)  # the buttons row
        self.btn_close.clicked.connect(self.reject)  # Close -> cancel dialog
        self.btn_export.clicked.connect(self._on_export)  # Export -> run handler
        self.exported = []  # will hold the list of selected paths after Accept
    def _on_export(self):
        sel = self.list.selectedItems()  # gather selected rows
        if not sel:  # nothing chosen?
            QtWidgets.QMessageBox.information(self, "Export", "Select at least one session to export."); return
        self.exported = [it.data(QtCore.Qt.UserRole) for it in sel]  # collect selected paths
        self.accept()  # close dialog with Accepted status
