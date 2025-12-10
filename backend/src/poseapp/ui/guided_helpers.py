# src/poseapp/ui/guided_helpers.py
import os  # filesystem utilities (path joins, existence checks)
from typing import Optional  # for precise optional type hints
from PySide6 import QtCore, QtGui, QtWidgets  # Qt UI toolkit (signals, movie/GIF, widgets)
from ..activities.activity_defs import ACTIVITY_LIBRARY  # to map activity keys → human labels
from ..utils.resources import GUIDE_DIRS  # list of folders to search for guide GIFs

class GifPreviewMixin:
    """Attach a preview GIF below an activity combo in a panel."""
    def __init__(self):
        self._preview_gif_label: Optional[QtWidgets.QLabel] = None  # QLabel that will display the animated GIF
        self._gif_movie: Optional[QtGui.QMovie] = None  # QMovie object driving the animation

    def find_activity_gif(self, key: str) -> Optional[str]:
        filenames = [f"{key}.gif"]  # Candidate filenames: try key.gif first
        label = ACTIVITY_LIBRARY.get(key, {}).get("label", "")  # Get label for readable file naming
        if label:
            safe = "".join(c for c in label.lower().replace(" ", "_") if c.isalnum() or c in ("_","-"))  # Normalize name
            filenames.append(f"{safe}.gif")  # Also check for label-based GIF
        for base in GUIDE_DIRS:  # Search through each guide directory
            for fn in filenames:
                p = os.path.join(base, fn)  # Build full file path
                if os.path.exists(p):  # Return first found GIF path
                    return p
        return None  # No GIF found

    def set_activity_preview(self, guided_panel: QtWidgets.QWidget, key: str):
        if not self._preview_gif_label:  # If label not initialized
            self._preview_gif_label = QtWidgets.QLabel(guided_panel)  # Create QLabel to show GIF
            self._preview_gif_label.setAlignment(QtCore.Qt.AlignCenter)  # Center-align the image
            self._preview_gif_label.setStyleSheet("background: transparent; border: 0;")  # Transparent background
            self._preview_gif_label.setMinimumHeight(260)  # Reserve space for preview
            lay = guided_panel.layout() or QtWidgets.QVBoxLayout(guided_panel)  # Get or create layout
            if guided_panel.layout() is None:  # Apply layout if missing
                guided_panel.setLayout(lay)
            lay.addWidget(self._preview_gif_label)  # Add preview label to layout
        gif_path = self.find_activity_gif(key)  # Find GIF path for given key
        if not gif_path:  # If no GIF found, clear preview
            self._preview_gif_label.clear(); self._gif_movie = None; return
        movie = QtGui.QMovie(gif_path); movie.setCacheMode(QtGui.QMovie.CacheAll)  # Load GIF into QMovie with caching
        self._preview_gif_label.setMovie(movie)  # Attach movie to label
        if self._preview_gif_label.width() > 0 and self._preview_gif_label.height() > 0:  # Scale if size known
            movie.setScaledSize(self._preview_gif_label.size())
        self._gif_movie = movie; movie.start()  # Save movie ref and start animation
