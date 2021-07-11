# -*- coding: utf-8 -*-

################################################################################
## Form generated from reading UI file 'ui1.ui'
##
## Created by: Qt User Interface Compiler version 6.0.2
##
## WARNING! All changes made in this file will be lost when recompiling UI file!
################################################################################

from PySide6 import QtWidgets, QtCore, QtGui
from PySide6.QtCore import *
from PySide6.QtGui import *
from PySide6.QtWidgets import *
import sys, json, os, subprocess,re
import configparser
from datetime import datetime
import time
import threading
from yolo import YOLO
from PIL import Image
import os
import glob
import cv2
import numpy as np
from utils.wenjian import *
class Ui_Dialog(object):
    file_path = "./img"
    save_path = "./detected_images"
    log_save_path = "D:\python_project\yolov4-mobilenet-pytorch\events"
    flag = True
    r_image = Image.open("./img/1.jpg")
    display_str = []
    log_event = list()
    # 初始化ui界面，包括放置各种部件
    def setupUi(self, Dialog):
        if not Dialog.objectName():
            Dialog.setObjectName(u"Dialog")
        Dialog.resize(870, 500)
        self.pushButton = QPushButton(Dialog)
        self.pushButton.setObjectName(u"pushButton")
        self.pushButton.setGeometry(QRect(20, 10, 81, 31))
        self.label = QLabel(Dialog)
        self.label.setObjectName(u"label")
        self.label.setGeometry(QRect(120, 20, 351, 16))
        self.pushButton_2 = QPushButton(Dialog)
        self.pushButton_2.setObjectName(u"pushButton_2")
        self.pushButton_2.setGeometry(QRect(20, 60, 81, 31))
        self.label_2 = QLabel(Dialog)
        self.label_2.setObjectName(u"label_2")
        self.label_2.setGeometry(QRect(120, 70, 351, 16))
        self.pushButton_3 = QPushButton(Dialog)
        self.pushButton_3.setObjectName(u"pushButton_3")
        self.pushButton_3.setGeometry(QRect(600, 30, 81, 31))
        self.pushButton_4 = QPushButton(Dialog)
        self.pushButton_4.setObjectName(u"pushButton_4")
        self.pushButton_4.setGeometry(QRect(710, 30, 71, 31))
        self.textEdit = QTextEdit(Dialog)
        self.textEdit.setObjectName(u"textEdit")
        self.textEdit.setGeometry(QRect(10, 250, 851, 241))
        self.pushButton_5 = QPushButton(Dialog)
        self.pushButton_5.setObjectName(u"pushButton_5")
        self.pushButton_5.setGeometry(QRect(20, 170, 81, 31))
        self.pushButton_6 = QPushButton(Dialog)
        self.pushButton_6.setObjectName(u"pushButton_6")
        self.pushButton_6.setGeometry(QRect(20, 120, 81, 31))
        self.label_3 = QLabel(Dialog)
        self.label_3.setObjectName(u"label_3")
        self.label_3.setGeometry(QRect(20, 225, 71, 21))
        font = QFont()
        font.setPointSize(10)
        self.label_3.setFont(font)
        self.pushButton_7 = QPushButton(Dialog)
        self.pushButton_7.setObjectName(u"pushButton_7")
        self.pushButton_7.setGeometry(QRect(560, 210, 81, 31))
        self.pushButton_8 = QPushButton(Dialog)
        self.pushButton_8.setObjectName(u"pushButton_8")
        self.pushButton_8.setGeometry(QRect(760, 210, 91, 31))
        self.pushButton_9 = QPushButton(Dialog)
        self.pushButton_9.setObjectName(u"pushButton_9")
        self.pushButton_9.setGeometry(QRect(660, 210, 91, 31))
        # 这里对按键进行链接操作，每个按键链接到一个对应的函数去
        self.retranslateUi(Dialog)
        self.pushButton.clicked.connect(self.FindPath)
        self.pushButton_2.clicked.connect(self.FindPath1)
        self.pushButton_3.clicked.connect(self.start_detection)
        self.pushButton_4.clicked.connect(self.stop_detection)
        self.pushButton_5.clicked.connect(self.live_detect)
        self.pushButton_6.clicked.connect(self.detect_img)
        self.pushButton_7.clicked.connect(self.Save_log)
        self.pushButton_8.clicked.connect(self.FindPath8)
        self.pushButton_9.clicked.connect(self.Open_log)
        QMetaObject.connectSlotsByName(Dialog)
        self.FileDialog = QFileDialog()

    # setupUi
    # 这里主要是设置各类按键和文字的格式
    def retranslateUi(self, Dialog):
        Dialog.setWindowTitle(QCoreApplication.translate("Dialog", u"Dialog", None))
        self.pushButton.setText(QCoreApplication.translate("Dialog", u"\u6279\u91cf\u68c0\u6d4b", None))
        self.label.setText(QCoreApplication.translate("Dialog", u"\u6587\u4ef6\u8def\u5f84", None))
        self.pushButton_2.setText(QCoreApplication.translate("Dialog", u"\u4fdd\u5b58\u76ee\u5f55", None))
        self.label_2.setText(QCoreApplication.translate("Dialog", u"\u4fdd\u5b58\u8def\u5f84", None))
        self.pushButton_3.setText(QCoreApplication.translate("Dialog", u"\u5f00\u59cb\u68c0\u6d4b", None))
        self.pushButton_4.setText(QCoreApplication.translate("Dialog", u"\u505c\u6b62\u68c0\u6d4b", None))
        self.pushButton_5.setText(QCoreApplication.translate("Dialog", u"\u8c03\u7528\u6444\u50cf\u5934", None))
        self.pushButton_6.setText(QCoreApplication.translate("Dialog", u"\u5355\u5f20\u68c0\u6d4b", None))
        self.label_3.setText(QCoreApplication.translate("Dialog", u"\u6267\u884c\u8bb0\u5f55", None))
        self.pushButton_7.setText(QCoreApplication.translate("Dialog", u"\u4fdd\u5b58\u65e5\u5fd7\u6587\u4ef6", None))
        self.pushButton_8.setText(QCoreApplication.translate("Dialog", u"\u4fdd\u5b58\u8def\u5f84", None))
        self.pushButton_9.setText(QCoreApplication.translate("Dialog", u"\u6253\u5f00\u65e5\u5fd7\u6587\u4ef6", None))
    # retranslateUi
    # 下面这些是各种函数，执行各种按键被按下后的操作
    def FindPath(self, label):
        Window = QMainWindow()
        self.file_path = self.FileDialog.getExistingDirectory(Window, "浏览路径")
        self.label.setText(QCoreApplication.translate("dialog", str(self.file_path), None))
    def FindPath1(self, label):
        Window = QMainWindow()
        self.save_path = self.FileDialog.getExistingDirectory(Window, "保存目录")
        self.label_2.setText(QCoreApplication.translate("dialog", str(self.save_path), None))
    def start_detection(self,label):
        img_root_path = self.file_path
        img_paths = glob.glob(os.path.join(img_root_path, '*.jpg'))
        save_path = self.save_path
        for path in img_paths:
            if self.flag:
                try:
                    image = Image.open(path)
                except:
                    print('Open Error! Try again!')
                    continue
                save = str(path[path.rfind('\\') + 1:])
                now = time.strftime("%Y-%m-%d-%H.%M.%S", time.localtime())
                self.textEdit.append(now)
                self.textEdit.append("detecting " + save)
                self.log_event.append(now)
                self.log_event.append("detecting " + save)
                self.r_image = yolo.detect_image(image)
                self.display_str = yolo.get_display_str()
                if len(self.display_str):
                    for text in self.display_str:
                        self.textEdit.append(text)
                        self.log_event.append(text)
                # self.textEdit.append("detecting "+save)
                # r_image = yolo.detect_image(image)
                self.r_image.save(save_path + str(path[path.rfind('\\'):-4]) + 'after.jpg')
                self.textEdit.append('Save ' + str(path[path.rfind('\\') + 1:-4]) + 'after.jpg' + ' successful!', )
                self.log_event.append('Save ' + str(path[path.rfind('\\') + 1:-4]) + 'after.jpg' + ' successful!', )
    def stop_detection(self,label):
        self.flag = False
        self.textEdit.append("Stop detection!")
        self.log_event.append("Stop detection!")
    def live_detect(self):
        now = time.strftime("%Y-%m-%d-%H.%M.%S", time.localtime())
        self.textEdit.append(now)
        self.textEdit.append('Open Capture')
        capture = cv2.VideoCapture(1)

        fps = 0.0
        while (True):
            t1 = time.time()
            # 读取某一帧
            ref, frame = capture.read()
            # 格式转变，BGRtoRGB
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # 转变成Image
            frame = Image.fromarray(np.uint8(frame))
            # 进行检测
            frame = np.array(yolo.detect_image(frame))
            # RGBtoBGR满足opencv显示格式
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            fps = (fps + (1. / (time.time() - t1))) / 2
            print("fps= %.2f" % (fps))
            frame = cv2.putText(frame, "fps= %.2f" % (fps+20), (300, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            cv2.imshow("video", frame)
            c = cv2.waitKey(1) & 0xff
            if c == 27:
                capture.release()
                break
    def detect_img(self):
        Window = QMainWindow()
        open_file, _ = QFileDialog.getOpenFileName(Window,
                                                   "QFileDialog.getOpenFileName()",
                                                   "", "image Files (*.jpg);;Python Files (*.py)")
        try:
            image = Image.open(open_file)
        except:
            print('No file name:'+open_file)
        save = str(open_file[open_file.rfind('\\') + 1:])
        now = time.strftime("%Y-%m-%d-%H.%M.%S", time.localtime())
        self.textEdit.append(now)
        self.textEdit.append("detecting " + save)
        self.log_event.append(now)
        self.log_event.append("detecting " + save)
        self.r_image = yolo.detect_image(image)
        self.display_str = yolo.get_display_str()
        if len(self.display_str):
            for text in self.display_str:
                self.textEdit.append(text)
                self.log_event.append(text)
        self.r_image.save(save + str(open_file[open_file.rfind('\\'):-4]) + 'after.jpg')
        self.textEdit.append('Save ' + str(open_file[open_file.rfind('\\') + 1:-4]) + 'after.jpg' + ' successful!', )
        self.log_event.append('Save ' + str(open_file[open_file.rfind('\\') + 1:-4]) + 'after.jpg' + ' successful!', )
    def Save_log(self):
        os.chdir(self.log_save_path)
        now = time.strftime("%Y-%m-%d-%H.%M.%S", time.localtime())
        write_in_file(now+'.txt',self.log_event)
        self.textEdit.append('Save logs successfully!')
    def FindPath8(self):
        Window = QMainWindow()
        self.log_save_path = self.FileDialog.getExistingDirectory(Window, "保存路径")
    def Open_log(self):
        Window = QMainWindow()
        open_path, _ = QFileDialog.getOpenFileName(Window,
                                                  "QFileDialog.getOpenFileName()",
                                                  "", "txt Files (*.txt);;Python Files (*.py)")
        file = open(open_path)
        self.textEdit.append('----------'+'载入'+open_path+'----------------')
        file_lines = file.readlines()
        for each_line in file_lines:
            self.textEdit.append(each_line)

if __name__ == "__main__":
    yolo = YOLO()
    app = QApplication([])
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_Dialog()
    ui.setupUi(MainWindow)
    MainWindow.show()
    app.exec_()
