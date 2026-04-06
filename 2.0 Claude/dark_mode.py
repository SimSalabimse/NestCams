"""
Dark Mode Stylesheet for Bird Motion Video Processor
Modern dark theme optimized for Windows
"""

DARK_STYLESHEET = """
/* Main Application */
QMainWindow {
    background-color: #1e1e1e;
    color: #e0e0e0;
}

QWidget {
    background-color: #1e1e1e;
    color: #e0e0e0;
    font-family: 'Segoe UI', Arial, sans-serif;
    font-size: 9pt;
}

/* Group Boxes */
QGroupBox {
    background-color: #2d2d2d;
    border: 1px solid #3d3d3d;
    border-radius: 5px;
    margin-top: 10px;
    padding-top: 10px;
    font-weight: bold;
    color: #ffffff;
}

QGroupBox::title {
    subcontrol-origin: margin;
    subcontrol-position: top left;
    padding: 0 5px;
    color: #60a5fa;
}

/* Labels */
QLabel {
    background-color: transparent;
    color: #e0e0e0;
}

/* Buttons */
QPushButton {
    background-color: #0d7377;
    color: #ffffff;
    border: none;
    border-radius: 4px;
    padding: 8px 16px;
    font-weight: bold;
}

QPushButton:hover {
    background-color: #14a085;
}

QPushButton:pressed {
    background-color: #0a5d61;
}

QPushButton:disabled {
    background-color: #3d3d3d;
    color: #808080;
}

/* Special Buttons */
QPushButton#start_btn {
    background-color: #16a34a;
    font-size: 11pt;
}

QPushButton#start_btn:hover {
    background-color: #22c55e;
}

QPushButton#cancel_btn {
    background-color: #dc2626;
    font-size: 11pt;
}

QPushButton#cancel_btn:hover {
    background-color: #ef4444;
}

/* Input Fields */
QLineEdit {
    background-color: #2d2d2d;
    border: 1px solid #3d3d3d;
    border-radius: 4px;
    padding: 6px;
    color: #e0e0e0;
}

QLineEdit:focus {
    border: 1px solid #60a5fa;
}

QLineEdit:disabled {
    background-color: #252525;
    color: #808080;
}

/* Text Edit */
QTextEdit {
    background-color: #1a1a1a;
    border: 1px solid #3d3d3d;
    border-radius: 4px;
    color: #e0e0e0;
    selection-background-color: #0d7377;
}

/* Combo Boxes */
QComboBox {
    background-color: #2d2d2d;
    border: 1px solid #3d3d3d;
    border-radius: 4px;
    padding: 6px;
    color: #e0e0e0;
}

QComboBox:hover {
    border: 1px solid #60a5fa;
}

QComboBox::drop-down {
    border: none;
    padding-right: 10px;
}

QComboBox::down-arrow {
    image: none;
    border-left: 4px solid transparent;
    border-right: 4px solid transparent;
    border-top: 6px solid #e0e0e0;
    margin-right: 5px;
}

QComboBox QAbstractItemView {
    background-color: #2d2d2d;
    border: 1px solid #3d3d3d;
    selection-background-color: #0d7377;
    color: #e0e0e0;
}

/* Spin Boxes */
QSpinBox, QDoubleSpinBox {
    background-color: #2d2d2d;
    border: 1px solid #3d3d3d;
    border-radius: 4px;
    padding: 6px;
    color: #e0e0e0;
}

QSpinBox:focus, QDoubleSpinBox:focus {
    border: 1px solid #60a5fa;
}

QSpinBox::up-button, QDoubleSpinBox::up-button {
    background-color: #3d3d3d;
    border-radius: 2px;
}

QSpinBox::down-button, QDoubleSpinBox::down-button {
    background-color: #3d3d3d;
    border-radius: 2px;
}

/* Sliders */
QSlider::groove:horizontal {
    background: #3d3d3d;
    height: 6px;
    border-radius: 3px;
}

QSlider::handle:horizontal {
    background: #60a5fa;
    width: 18px;
    margin: -6px 0;
    border-radius: 9px;
}

QSlider::handle:horizontal:hover {
    background: #93c5fd;
}

QSlider::sub-page:horizontal {
    background: #0d7377;
    border-radius: 3px;
}

/* Checkboxes */
QCheckBox {
    spacing: 8px;
    color: #e0e0e0;
}

QCheckBox::indicator {
    width: 18px;
    height: 18px;
    border: 2px solid #3d3d3d;
    border-radius: 4px;
    background-color: #2d2d2d;
}

QCheckBox::indicator:checked {
    background-color: #0d7377;
    border-color: #0d7377;
}

QCheckBox::indicator:checked:hover {
    background-color: #14a085;
}

QCheckBox::indicator:hover {
    border-color: #60a5fa;
}

/* Progress Bar */
QProgressBar {
    background-color: #2d2d2d;
    border: 1px solid #3d3d3d;
    border-radius: 4px;
    text-align: center;
    color: #ffffff;
    font-weight: bold;
}

QProgressBar::chunk {
    background-color: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                                     stop:0 #0d7377, stop:1 #14a085);
    border-radius: 3px;
}

/* Tab Widget */
QTabWidget::pane {
    border: 1px solid #3d3d3d;
    background-color: #1e1e1e;
    border-radius: 4px;
}

QTabBar::tab {
    background-color: #2d2d2d;
    border: 1px solid #3d3d3d;
    padding: 8px 16px;
    margin-right: 2px;
    border-top-left-radius: 4px;
    border-top-right-radius: 4px;
    color: #e0e0e0;
}

QTabBar::tab:selected {
    background-color: #0d7377;
    color: #ffffff;
    font-weight: bold;
}

QTabBar::tab:hover:!selected {
    background-color: #3d3d3d;
}

/* Table Widget */
QTableWidget {
    background-color: #2d2d2d;
    alternate-background-color: #252525;
    border: 1px solid #3d3d3d;
    border-radius: 4px;
    gridline-color: #3d3d3d;
    color: #e0e0e0;
}

QTableWidget::item {
    padding: 5px;
}

QTableWidget::item:selected {
    background-color: #0d7377;
}

QHeaderView::section {
    background-color: #3d3d3d;
    color: #ffffff;
    padding: 6px;
    border: none;
    font-weight: bold;
}

/* Scroll Bars */
QScrollBar:vertical {
    background-color: #2d2d2d;
    width: 12px;
    border-radius: 6px;
}

QScrollBar::handle:vertical {
    background-color: #3d3d3d;
    border-radius: 6px;
    min-height: 20px;
}

QScrollBar::handle:vertical:hover {
    background-color: #4d4d4d;
}

QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
    height: 0px;
}

QScrollBar:horizontal {
    background-color: #2d2d2d;
    height: 12px;
    border-radius: 6px;
}

QScrollBar::handle:horizontal {
    background-color: #3d3d3d;
    border-radius: 6px;
    min-width: 20px;
}

QScrollBar::handle:horizontal:hover {
    background-color: #4d4d4d;
}

QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal {
    width: 0px;
}

/* Status Bar */
QStatusBar {
    background-color: #252525;
    color: #e0e0e0;
    border-top: 1px solid #3d3d3d;
}

/* Menu Bar */
QMenuBar {
    background-color: #2d2d2d;
    color: #e0e0e0;
}

QMenuBar::item:selected {
    background-color: #0d7377;
}

QMenu {
    background-color: #2d2d2d;
    border: 1px solid #3d3d3d;
}

QMenu::item:selected {
    background-color: #0d7377;
}

/* Tool Tips */
QToolTip {
    background-color: #3d3d3d;
    color: #e0e0e0;
    border: 1px solid #4d4d4d;
    border-radius: 4px;
    padding: 4px;
}

/* Message Box */
QMessageBox {
    background-color: #1e1e1e;
}

QMessageBox QLabel {
    color: #e0e0e0;
}

QMessageBox QPushButton {
    min-width: 80px;
    min-height: 30px;
}
"""

def apply_dark_mode(app):
    """Apply dark mode to application"""
    app.setStyleSheet(DARK_STYLESHEET)
