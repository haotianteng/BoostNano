#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  9 14:48:15 2019
This script is for hand labelling the signal for training.
@author: heavens
"""
import matplotlib
matplotlib.use('TkAgg')
import os
import sys
import numpy as np
from matplotlib.backends.qt_compat import  QtWidgets, is_pyqt5
if is_pyqt5():
    from PyQt5.QtWidgets import QFileDialog, QGridLayout
    from PyQt5.QtGui import QMouseEvent
    from matplotlib.backends.backend_qt5agg import (
        FigureCanvasQTAgg, NavigationToolbar2QT as NavigationToolbar)
else:
    from PyQt4.QtGui import QFileDialog,QMouseEvent
    from matplotlib.backends.backend_qt4agg import (
        FigureCanvasQTAgg, NavigationToolbar2QT as NavigationToolbar)
from PyQt5.QtCore import Qt
from matplotlib.figure import Figure
import h5py
import semrnn
import numpy as np
from shutil import copyfile

def denoise(signal,max_std=5):
    signal = np.asarray(signal)
    mean = np.mean(signal)
    std = np.std(signal)
    return signal[np.logical_and(signal>mean-max_std*std,signal<mean+max_std*std)]

class SignalLabeler(QtWidgets.QMainWindow):
    def __init__(self,signal_folder=None,save_file=None):
        super().__init__()
        self._initUI()
        self.ax.text(0.5, 0.5,'Press S to start.',
             horizontalalignment='center',
             verticalalignment='center',
             transform = self.ax.transAxes)
        self.canvas.draw()
        self.line=None
        self.axvlines=[]
        self.cursor_line=None
        self.colors=['#990000','#ffa500','#0a0451']
        self.curr_color=self.colors[0]
        self.pos=[]
        self.signal_dir = signal_folder
        self.sig_iter=None
        self.signal=[]
        self.curr_file=None
        self.records=[]
        self.save=save_file
        self.start = False
        self.cache_dir=os.path.join(os.path.dirname(semrnn.__file__),'cache')
        self.cache_file=os.path.join(self.cache_dir,'cache.csv')
        self.cache_size=1 #Auto save the result to cache file after this number of labeled.
        self.cache_fns=[]
        if not os.path.isdir(self.cache_dir):    
            os.mkdir(self.cache_dir)

    def _initUI(self):
        self.dialog_p={'title':'PolyA label tool','left':350,'top':1200,'width':1280,'height':480}
        self.setWindowTitle(self.dialog_p['title'])
        self._main = QtWidgets.QWidget()
        self.setCentralWidget(self._main) 
        self.fig=Figure(figsize=(5, 3))
        # a figure instance to plot on
        self.canvas = FigureCanvasQTAgg(self.fig)
        self.ax=self.fig.add_subplot(111)
        self.setGeometry(self.dialog_p['left'], 
                         self.dialog_p['top'], 
                         self.dialog_p['width'], 
                         self.dialog_p['height'])
        self.gridLayout = QGridLayout(self._main)
        self.gridLayout.addWidget(self.canvas)
        self._main.setLayout(self.gridLayout)
        self.fig.canvas.mpl_connect('button_press_event', self.on_click)
        self.fig.canvas.mpl_connect('motion_notify_event', self.on_move)
        self.show()

    def _reinit(self):
        self.ax.clear()
        self.ax.text(0.5, 0.5,'Press S to start.',
             horizontalalignment='center',
             verticalalignment='center',
             transform = self.ax.transAxes)
        self.line=None
        self.curr_color=self.colors[0]
        self.canvas.draw_idle()
        self.pos=[]
        self.signal_dir = None
        self.sig_iter=None
        self.signal=[]
        self.curr_file=None
        self.records=[]
        self.axvlines=[]
        self.cursor_line=None
        self.save=None
        self.start=False
        self.cache_fns=[]

    def refresh(self,xpos):
        self.cursor_line.set_xdata(xpos)
        self.cursor_line.set_color(self.curr_color)
        self.ax.draw_artist(self.ax.patch)
        self.ax.draw_artist(self.line)
        self.ax.draw_artist(self.cursor_line)
        for axvline in self.axvlines:
            self.ax.draw_artist(axvline)
        self.canvas.update()
        self.canvas.flush_events()

    def redraw(self,cursor_x,xpos):
        self.ax.clear()
        self.fig.suptitle(os.path.basename(self.curr_file), fontsize=10)
        self.line=self.ax.plot(denoise(self.signal))[0]
        if cursor_x is not None:
            self.cursor_line=self.ax.axvline(x=cursor_x,color=self.curr_color)
        self.axvlines=[]
        for x,color in xpos:
            self.axvlines.append(self.ax.axvline(x=x,color=color))
        self.canvas.draw_idle()
        self.canvas.flush_events()

    def on_move(self,event):
        if not self.start:
            return
        x = event.xdata
        self.refresh(x)

    def on_click(self, event:QMouseEvent):
        if not self.start:
            return
        x = event.xdata
        self.pos.append(x)
        axv_pairs=[]
        for idx,p in enumerate(self.pos):
            axv_pairs.append((p,self.colors[idx]))
        if len(self.pos)>=3:
            self.records.append([self.curr_file]+self.pos)
            self.next_signal()
            self.pos=[]
            self.axvlines=[]
            self.curr_color=self.colors[0]
            axv_pairs=[]
        if len(self.records)>self.cache_size:
            self._cache()
        self.curr_color=self.colors[len(self.pos)]
        self.redraw(None,axv_pairs)

    def keyPressEvent(self,event):
        if event.key() == Qt.Key_Q:
            if not self.start:
                self._quit()
                return
            if self.save==None:
                self.save = str(QFileDialog.getExistingDirectory(self, "Choose directory to save."))
            self._save()
            self._reinit()
        elif event.key()==Qt.Key_S:
            if self.signal_dir==None:
                self.signal_dir = str(QFileDialog.getExistingDirectory(self, "Select Signal Directory"))
                if self.signal_dir=='':
                    self.signal_dir=None
                    return
            self._start()
        elif event.key()==Qt.Key_X:
            #skip current signal file
            self.pos=[]
            self.next_signal()
            self.curr_color=self.colors[0]
            axv_pairs=[]
            self.redraw(None,axv_pairs)

    def _iter_signals(self):
        file_list=os.listdir(self.signal_dir)
        for file in file_list:
            if file.endswith('.signal'):
                yield self._read_signal(os.path.join(self.signal_dir,file))
            elif file.endswith('.fast5'):
                yield self._read_fast5(os.path.join(self.signal_dir,file))

    def _read_signal(self,file):
        with open(file,'r') as f:
            for line in f:
                split_line = line.strip().split()
                return file,[int(x) for x in split_line]

    def _read_fast5(self,file):
        with h5py.File(file,mode='r') as root:
            raw_signal = np.asarray(list(root['/Raw/Reads'].values())[0][('Signal')])  
        return file,raw_signal[::-1]

    def _save(self):
        #self.canvas.mpl_disconnect(self.cidmotion)
        #self.canvas.mpl_disconnect(self.cidclick)
        self._cache()
        copyfile(self.cache_file,os.path.join(self.save,'result.csv'))

    def _cache(self):
        with open(self.cache_file,'a') as f:
            for record in self.records:
                f.write(','.join([str(x) for x in record]))
                f.write('\n')
        self.records=[]

    def _start(self):
        if self.start:
            print("Already start a job")
            return
        print("Begin reading signal file list")
        self.sig_iter=self._iter_signals()
        print("Try to read cache file.")
        if os.path.isfile(self.cache_file):
            with open(self.cache_file) as cache_f:
                for line in cache_f:
                    self.cache_fns.append(line.strip().split(',')[0])
            print("Sucessfully read cache file, load %d cache entries, delete the cache.csv under cache folder to not use cache record."%(len(self.cache_fns)))
        else:
            print("No cache file found.")
        self.next_signal()
        self.start=True
        self.redraw(0,[])
        print("Reading finished, press Q to save result, press X to skip current read.")

    def next_signal(self):
        self.curr_file,self.signal=next(self.sig_iter)
        while self.curr_file in self.cache_fns:
            self.curr_file,self.signal=next(self.sig_iter)
    
    def _quit(self):
        self.close()
        
if __name__ ==  "__main__":
    app = QtWidgets.QApplication(sys.argv)
    ui = SignalLabeler()
    ui.show()
    sys.exit(app.exec_())    

