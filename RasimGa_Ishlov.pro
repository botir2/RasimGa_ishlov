#-------------------------------------------------
#
# Project created by QtCreator 2016-04-24T12:53:12
#
#-------------------------------------------------

QT       += core gui

greaterThan(QT_MAJOR_VERSION, 4): QT += widgets

TARGET = RasimGa_Ishlov
TEMPLATE = app


SOURCES += main.cpp\
    ResizeImage.cpp \
        mainwindow.cpp \
    graphicssceneex.cpp \
    graphicsviewex.cpp \
    text.cpp \
    io.cpp \
    wavelettrans.cpp

HEADERS  += mainwindow.h \
    ResizeImage.h \
    decimal.h \
    graphicssceneex.h \
    graphicsviewex.h \
    text.h \
    io.h \
    colordefs.h \
    wavelettrans.h

FORMS    += mainwindow.ui



INCLUDEPATH += D:\opencv\build\include

LIBS += D:\opencv\release\bin\libopencv_core411.dll
LIBS += D:\opencv\release\bin\libopencv_highgui411.dll
LIBS += D:\opencv\release\bin\libopencv_imgcodecs411.dll
LIBS += D:\opencv\release\bin\libopencv_imgproc411.dll
LIBS += D:\opencv\release\bin\libopencv_features2d411.dll
LIBS += D:\opencv\release\bin\libopencv_calib3d411.dll
