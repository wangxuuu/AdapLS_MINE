
from __future__ import unicode_literals
import sys
import logging
import numbers
import numpy as np
import ast

from matplotlib.backends import qt_compat
use_pyside = qt_compat.QT_API == qt_compat.QT_API_PYSIDE
if use_pyside:
    from PySide import QtGui, QtCore
else:
    # from PyQt4 import QtGui, QtCore
    from PyQt5 import QtWidgets, QtGui, QtCore

from matplotlib.backends.backend_qt4agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

import torch

cuda = True if torch.cuda.is_available() else False

FloatTensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if cuda else torch.LongTensor

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


# 20200220 Thomas Huang Hing Pang 
# High-Dim Neural Network Heatmap viewer

# path of state_dict
PATH = "./results/MixedGaussian_MINE.pt"
# key of neural network in file
KEY = 'mine_state_list'
# dimension of Input of neural network, >2
DIM = 4
# size of hidden layer of NN
HIDDEN_SIZE = 100
# If more than one NN is in the file, pls give the index
INDEX = 0

# heatmap mesh array parameters
Xmax = 3.0
Xmin = -3.0
Xsample = 300
Ymax = 3.0
Ymin = -3.0
Ysample = 300 

# Usage
# 1. proper set upper para, and then run this code
# 2. select any x, y or center field, edit the number , and then hit "enter"
# 3. some errors will be report in "run.log"


class MineNet(nn.Module):
    def __init__(self, input_size=2, hidden_size=100):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, 1)
        nn.init.normal_(self.fc1.weight,std=0.02)
        nn.init.constant_(self.fc1.bias, 0)
        nn.init.normal_(self.fc2.weight,std=0.02)
        nn.init.constant_(self.fc2.bias, 0)
        nn.init.normal_(self.fc3.weight,std=0.02)
        nn.init.constant_(self.fc3.bias, 0)
        
    def forward(self, input):
        output = F.elu(self.fc1(input))
        output = F.elu(self.fc2(output))
        output = self.fc3(output)
        return output

class MplCanvas(FigureCanvas):
    """Ultimately, this is a QWidget (as well as a FigureCanvasAgg, etc.)."""

    def __init__(self, parent=None, width=5, height=4, dpi=100):
        self._width = width
        self._height = height
        self._dpi = dpi
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = self.fig.add_subplot(111)

        self.compute_initial_figure()

        FigureCanvas.__init__(self, self.fig)
        self.setParent(parent)

        FigureCanvas.setSizePolicy(self,
                                   QtWidgets.QSizePolicy.Expanding,
                                   QtWidgets.QSizePolicy.Expanding)
        FigureCanvas.updateGeometry(self)

    def compute_initial_figure(self):
        pass

class Heatmap(MplCanvas):
    """A canvas that updates itself every second with a new plot."""

    def __init__(self, *args, **kwargs):
        # state
        self.heatmap_update = False
        # para
        global Xmax
        global Xmin
        global Xsample
        global Ymax
        global Ymin
        global Ysample
        x = np.linspace(Xmin, Xmax, Xsample)
        y = np.linspace(Ymin, Ymax, Ysample)
        self.xs, self.ys = np.meshgrid(x,y)
        self.z = np.zeros((Xsample, Ysample))
        np_mesh = np.hstack((self.xs.flatten()[:,None],self.ys.flatten()[:,None]))

        self.x_mesh = np_mesh[:,0][:,None]
        self.y_mesh = np_mesh[:,1][:,None]

        MplCanvas.__init__(self, *args, **kwargs)
        timer = QtCore.QTimer(self)
        timer.timeout.connect(self.update_figure)
        timer.start(1000)

    def update_z(self, origin, x_dir, y_dir, net):
        plot_xy = origin + self.x_mesh*x_dir + self.y_mesh *y_dir
        heatmap_mesh = FloatTensor(plot_xy)
        ixy = net(heatmap_mesh).detach().cpu().numpy()
        self.z = ixy.reshape(self.xs.shape[1], self.ys.shape[0])
        self.heatmap_update = True

    def compute_initial_figure(self):
        z_min, z_max = self.z.min(), self.z.max()
        c = self.axes.pcolormesh(self.xs, self.ys, self.z, cmap="RdBu", vmin=z_min, vmax=z_max)
        self.cb = self.fig.colorbar(c, ax=self.axes)

    def update_figure(self):
        if self.heatmap_update:
            z_min, z_max = self.z.min(), self.z.max()
            self.axes.clear()
            self.cb.ax.clear()
            c = self.axes.pcolormesh(self.xs, self.ys, self.z, cmap="RdBu", vmin=z_min, vmax=z_max)
            self.cb = self.fig.colorbar(c, cax=self.cb.ax)
            self.draw()
            self.heatmap_update = False

class view_(QtWidgets.QWidget):
    def __init__(self):
        super(view_, self).__init__()
        # btn setting
        self.BTN_FONT_STYLE = "Arial"
        self.BTN_FONT_SIZE = 30
        self.BTN_W = 100  # button width
        self.BTN_H = 40  # button height

        # model
        global PATH
        global DIM
        global HIDDEN_SIZE
        self.dim = DIM
        self.hidden_size = HIDDEN_SIZE
        self.v_x_dir = np.pad([1, 0], (0, self.dim-2), 'constant', constant_values=0.)
        self.v_y_dir = np.pad([1, 0], (self.dim//2, self.dim-2-self.dim//2), 'constant', constant_values=0.)
        self.v_origin = np.pad([0, 0], (0, self.dim-2), 'constant', constant_values=0.)
        self.v_path = PATH

        state_dict = torch.load(self.v_path)

        self.net = MineNet(input_size=self.dim, hidden_size=self.hidden_size)

        global KEY
        global INDEX
        if INDEX:
            self.net.load_state_dict(state_dict[KEY][INDEX])
        else:
            self.net.load_state_dict(state_dict[KEY])

        # view
        # controller
        
        self.initGUI()

    def __del__(self):
        pass

    def initGUI(self):

        v0box = QtWidgets.QVBoxLayout()

        self.heatmap = Heatmap(self, width=50, height=40, dpi=100)
        v0box.addWidget(self.heatmap)

        # grid on the right
        grid0 = QtWidgets.QGridLayout()

        flo0 = QtWidgets.QFormLayout()

        # set x axis
        cur = QtWidgets.QLineEdit()
        cur.setEnabled(True)
        cur.setAlignment(QtCore.Qt.AlignCenter)
        cur.setFont(QtGui.QFont(self.BTN_FONT_STYLE, self.BTN_FONT_SIZE))
        cur.setMinimumWidth(3*self.BTN_W)
        cur.setFixedHeight(self.BTN_H)
        cur.editingFinished.connect(self.X_updated)
        cur.setText(str(self.v_x_dir.tolist()))
        flo0.addRow("X:", cur)
        self.x_dir = cur

        # set y axis
        cur = QtWidgets.QLineEdit()
        cur.setEnabled(True)
        cur.setAlignment(QtCore.Qt.AlignCenter)
        cur.setFont(QtGui.QFont(self.BTN_FONT_STYLE, self.BTN_FONT_SIZE))
        cur.setMinimumWidth(3*self.BTN_W)
        cur.setFixedHeight(self.BTN_H)
        cur.editingFinished.connect(self.Y_updated)
        cur.setText(str(self.v_y_dir.tolist()))
        flo0.addRow("Y:", cur)
        self.y_dir = cur

        # set origin
        cur = QtWidgets.QLineEdit()
        cur.setEnabled(True)
        cur.setAlignment(QtCore.Qt.AlignCenter)
        cur.setFont(QtGui.QFont(self.BTN_FONT_STYLE, self.BTN_FONT_SIZE))
        cur.setMinimumWidth(3*self.BTN_W)
        cur.setFixedHeight(self.BTN_H)
        cur.editingFinished.connect(self.Origin_updated)
        cur.setText(str(self.v_origin.tolist()))
        flo0.addRow("Window center:", cur)
        self.origin = cur

        # # set path of state_dict file
        # cur = QtWidgets.QLineEdit()
        # cur.setEnabled(True)
        # cur.setAlignment(QtCore.Qt.AlignCenter)
        # # cur.setFont(QtGui.QFont(self.BTN_FONT_STYLE, self.BTN_FONT_SIZE))
        # # cur.setMinimumWidth(3*self.BTN_W)
        # # cur.setFixedHeight(self.BTN_H)
        # cur.editingFinished.connect(self.inputPathDone)
        # cur.setText(self.v_path)
        # flo0.addRow("File path:", cur)
        # self.sd_path = cur

        grid0.addLayout(flo0, 0, 0, 4, 3)
        v0box.addLayout(grid0)

        self.setLayout(v0box)
        self.setGeometry(1000, 2000, 1000, 1000)
        self.setWindowTitle('DSP Storytelling GUI')
        self.show()

    def Origin_updated(self):
        if not str(self.v_origin) == self.origin.text():
            try:
                origin = ast.literal_eval(self.origin.text())
                if not type(origin) == list:
                    raise TypeError("{} is not list".format(type(origin)))
                elif len(origin) != self.dim:
                    raise ValueError("len(origin)=={}!=self.dim=={}".format(len(origin), self.dim))
                else:
                    for i, x in enumerate(origin):
                        if not isinstance(x, numbers.Number):
                            raise TypeError("type(origin[{}])=={}".format(i, type(x)))
                        else:
                            origin[i] = float(x)
                    
                    self.v_origin = origin
                    self.heatmap.update_z(self.v_origin, self.v_x_dir, self.v_y_dir, self.net)

            except Exception as e:
                except_msg = "{}:update failed:{}".format(e, self.origin.text())
                logging.debug(except_msg)
                self.origin.setText(str(self.v_origin))

    def X_updated(self):
        if not str(self.v_x_dir) == self.x_dir.text():
            try:
                x_dir = ast.literal_eval(self.x_dir.text())
                if not type(x_dir) == list:
                    raise TypeError("{} is not list".format(type(x_dir)))
                elif len(x_dir) != self.dim:
                    raise ValueError("len(x_dir)=={}!=self.dim=={}".format(len(x_dir), self.dim))
                else:
                    for i, x in enumerate(x_dir):
                        if not isinstance(x, numbers.Number):
                            raise TypeError("type(x_dir[{}])=={}".format(i, type(x)))
                        else:
                            x_dir[i] = float(x)
                    
                    # if  np.dot(x_dir, self.v_y_dir) != 0.:
                    #     raise ValueError("x_dir {} . self.v_y_dir {} != 0".format(x_dir, self.v_y_dir))
                    # else:
                    self.v_x_dir = x_dir
                    self.heatmap.update_z(self.v_origin, self.v_x_dir, self.v_y_dir, self.net)

            except Exception as e:
                except_msg = "{}:update failed:{}".format(e, self.x_dir.text())
                logging.debug(except_msg)
                self.x_dir.setText(str(self.v_x_dir))

    def Y_updated(self):
        if not str(self.v_y_dir) == self.y_dir.text():
            try:
                y_dir = ast.literal_eval(self.y_dir.text())
                if not type(y_dir) == list:
                    raise TypeError("{} is not list".format(type(y_dir)))
                elif len(y_dir) != self.dim:
                    raise ValueError("len(y_dir)=={}!=self.dim=={}".format(len(y_dir), self.dim))
                else:
                    for i, x in enumerate(y_dir):
                        if not isinstance(x, numbers.Number):
                            raise TypeError("type(y_dir[{}])=={}".format(i, type(x)))
                        else:
                            y_dir[i] = float(x)
                    
                    # if  np.dot(y_dir, self.v_y_dir) != 0.:
                    #     raise ValueError("y_dir {} . self.v_y_dir {} != 0".format(y_dir, self.v_x_dir))
                    # else:
                    self.v_y_dir = y_dir
                    self.heatmap.update_z(self.v_origin, self.v_x_dir, self.v_y_dir, self.net)

            except Exception as e:
                except_msg = "{}:update failed:{}".format(e, self.y_dir.text())
                logging.debug(except_msg)
                self.y_dir.setText(str(self.v_y_dir))

if __name__=="__main__":
    logging.basicConfig(filename='run.log', level=logging.DEBUG)
    app = QtWidgets.QApplication(sys.argv)
    v = view_()
    v.update()
    sys.exit(app.exec_())