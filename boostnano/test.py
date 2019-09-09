import sys
import matplotlib
from PyQt5 import QtCore
import PyQt5.QtWidgets as QtW
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from matplotlib.widgets import RectangleSelector


class MainWindow(QtW.QMainWindow):

    def __init__(self):
        super().__init__()
        self.setWindowTitle('MyWindow')
        self._main = QtW.QWidget()
        self.setCentralWidget(self._main) 

        # Set canvas properties
        self.fig = matplotlib.figure.Figure(figsize=(5,5))
        self.canvas = FigureCanvasQTAgg(self.fig)
        self.ax = self.fig.add_subplot(1,1,1)
        self.canvas.draw()
        """
        self.rs = RectangleSelector(self.ax, self.line_select_callback,
                                                drawtype='box', useblit=True,
                                                button=[1, 3],  # don't use middle button
                                                minspanx=5, minspany=5,
                                                spancoords='pixels',
                                                interactive=True)
        """
        # set Qlayout properties and show window
        self.gridLayout = QtW.QGridLayout(self._main)
        self.gridLayout.addWidget(self.canvas)
        self.setLayout(self.gridLayout)
        self.show()

        # connect mouse events to canvas
        self.fig.canvas.mpl_connect('button_press_event', self.on_click)

    def on_click(self, event):
        #if event.button == 1 or event.button == 3 and not self.rs.active:
        #    self.rs.set_active(False)
        #else:
        #    self.rs.set_active(False)
        print("111")
    def line_select_callback(self, eclick, erelease):
        x1, y1 = eclick.xdata, eclick.ydata
        x2, y2 = erelease.xdata, erelease.ydata
        print("(%3.2f, %3.2f) --> (%3.2f, %3.2f)" % (x1, y1, x2, y2))
        print(" The button you used were: %s %s" % (eclick.button, erelease.button))

if __name__ == '__main__':
    app = QtCore.QCoreApplication.instance()
    if app is None: app = QtW.QApplication(sys.argv)
    win = MainWindow()
    app.aboutToQuit.connect(app.deleteLater)
    app.exec_()