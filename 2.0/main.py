def main():
    from ui.main_window import MainWindow
    import sys
    from PyQt5.QtWidgets import QApplication
    from PyQt5.QtCore import Qt
    from PyQt5.QtGui import QFont

    QApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)
    QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps, True)
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    font = QFont("Segoe UI", 10)
    font.setStyleStrategy(QFont.PreferAntialias)
    app.setFont(font)
    app.setPalette(app.style().standardPalette())

    window = MainWindow()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
