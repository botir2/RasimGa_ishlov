#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/opencv.hpp>


#include "mainwindow.h"
#include "ui_mainwindow.h"
#include <QtMath>
#include <QTimer>
#include<QDebug>
#include <math.h>
#include <stdio.h>
#include <algorithm>

#include "ResizeImage.h"
#include<iostream>


#define FIXED_SIZE 512
#define EDGE_THRESH 35
#define MIN_ZERO 0.05

#define SHOW_IMAGE(x) {cv::imshow(#x, x); cv::waitKey();}


typedef struct {
    int width;
    int height;
    int stride;
    int *data;
}HuMat;


using namespace std;
using namespace cv;


void create_humat(HuMat &mat, int width, int height)
{
    mat.width = width;
    mat.height = height;

    mat.stride = ((width + 3) >> 2) << 2;

    mat.data = (int*)malloc(sizeof(int) * mat.stride * mat.height);
}


void free_humat(HuMat &mat)
{
    free(mat.data);
    mat.data = NULL;
}


void show_humat(const char *winName, HuMat &src)
{
    int width = src.width;
    int height = src.height;

    cv::Mat res(height, width, CV_8UC1, cv::Scalar(0));

    int *ptrSrc = src.data;

    uchar *ptrRes = res.data;

    for(int y = 0; y < height; y++)
    {
        for(int x = 0; x < width; x++)
            ptrRes[x] = abs(ptrSrc[x]);

        ptrSrc += src.stride;
        ptrRes += res.step;
    }

    cv::imshow(winName, res);
    cv::waitKey();
}


void haar_wavelet_transform(HuMat &img, HuMat &res)
{
    int height = img.height;
    int width = img.width;

    HuMat centerMat;

    create_humat(centerMat, width, height);

    int *ptrSrc = img.data;
    int *ptrRes = centerMat.data;

    int stride1 = img.stride;
    int stride2 = centerMat.stride;

    int len = width/2;

    assert(height % 2 == 0 && width % 2 == 0);

    for(int y = 0; y < height; y++)
    {
        for(int x = 0; x < width; x += 2)
        {
            int a = ptrSrc[x];
            int b = ptrSrc[x+1];

            int idx = x >> 1;

            ptrRes[idx] = (a+b)/2;
            ptrRes[idx + len] = (a-b)/2;
        }

        ptrSrc += stride1;
        ptrRes += stride2;
    }

    create_humat(res, width, height);

    len = height/2;

    ptrSrc = centerMat.data;
    ptrRes = res.data;

    stride1 = centerMat.stride;
    stride2 = res.stride;

    for(int y = 0; y < height; y += 2)
    {
        for(int x = 0; x < width; x++)
        {
            int a = ptrSrc[x];
            int b = ptrSrc[stride1 + x];

            ptrRes[x] = (a+b)/2;
            ptrRes[len * stride2 + x] = (a-b)/2;
        }

        ptrSrc += 2 * stride1;
        ptrRes += stride2;
    }

    free_humat(centerMat);
}


void calc_emap(HuMat &src, HuMat &res)
{
    int height = src.height;
    int width = src.width;

    int halfH = height >> 1;
    int halfW = width >> 1;

    int *ptrSrc, *ptrRes;

    create_humat(res, halfW, halfH);

    int stride1 = src.stride;
    int stride2 = res.stride;

    ptrSrc = src.data;
    ptrRes = res.data;

    for(int y = 0; y < halfH; y++)
    {
        for(int x = 0; x < halfW; x++)
        {
            int HL = ptrSrc[y * stride1 + x + halfW];
            int LH = ptrSrc[(y + halfH) * stride1 + x];
            int HH = ptrSrc[(y + halfH) * stride2 + x + halfW];

            ptrRes[x] = sqrt(HL * HL + LH * LH + HH * HH);
        }

        ptrRes += stride2;
    }
}


void calc_emax(HuMat &emap, HuMat &emax, int winSize)
{
    int width = emap.width;
    int height = emap.height;

    int mH = height / winSize;
    int mW = width / winSize;

    create_humat(emax, mW, mH);

    int* ptrEmap = emap.data;
    int* ptrEmax = emax.data;

    int stride1 = emap.stride;
    int stride2 = emax.stride;

    for(int y = 0; y < height; y++)
    {
        int y_ = y / winSize;
        int idx = y_ * stride2;

        for(int x = 0; x < width; x++)
        {
            int x_ = x / winSize;

            if(ptrEmap[x] > ptrEmax[idx + x_])
                ptrEmax[idx + x_] = ptrEmap[x];
        }

        ptrEmap += stride1;
    }
}


int blur_detect(HuMat &img, float *conf, int waveMtd)
{
    HuMat haarRes[3];
    HuMat emap[3];
    HuMat emax[3];

    int width = img.width;
    int height = img.height;
    int stride = img.stride;

    int Nedge, Nda, Nrg, Nbrg;

    int *emax0, *emax1, *emax2;

    HuMat src = img;

    //printf("BEGIN ...\n");

    assert(width % 8 == 0 && height % 8 == 0);

    for(int i = 0; i < 3; i++)
    {

        switch (waveMtd) {

            case 0:
                show_humat("Test wevlet", src);
                haar_wavelet_transform(src, haarRes[i]);
                src.width = haarRes[i].width >> 1;
                src.height = haarRes[i].height >> 1;
                src.stride = haarRes[i].stride;
                src.data = haarRes[i].data;
                break;

            case 1:
                   haar_wavelet_transform(src, haarRes[i]);
                   show_humat("haar res", haarRes[i]);

                   calc_emap(haarRes[i], emap[i]);
                   //show_humat("edge map", emap[i]);

                   calc_emax(emap[i], emax[i], (1 << (3-i)));
                   //show_humat("edge max", emax[i]);

                   src.width = haarRes[i].width >> 1;
                   src.height = haarRes[i].height >> 1;
                   src.stride = haarRes[i].stride;
                   src.data = haarRes[i].data;
                break;

            case 2:
                haar_wavelet_transform(src, haarRes[i]);
                calc_emap(haarRes[i], emap[i]);
                show_humat("edge map", emap[i]);
                src.width = haarRes[i].width >> 1;
                src.height = haarRes[i].height >> 1;
                src.stride = haarRes[i].stride;
                src.data = haarRes[i].data;
                break;

            case 3:
                haar_wavelet_transform(src, haarRes[i]);
                calc_emap(haarRes[i], emap[i]);
                calc_emax(emap[i], emax[i], (1 << (3-i)));
                show_humat("edge max", emax[i]);
                src.width = haarRes[i].width >> 1;
                src.height = haarRes[i].height >> 1;
                src.stride = haarRes[i].stride;
                src.data = haarRes[i].data;
                break;

            default:
                show_humat("Test wevlet", src);
                haar_wavelet_transform(src, haarRes[i]);
                //show_humat("haar res", haarRes[i]);

                calc_emap(haarRes[i], emap[i]);
                //show_humat("edge map", emap[i]);

                //calc_emax(emap[i], emax[i], (1 << (3-i)));
                //show_humat("edge max", emax[i]);

                src.width = haarRes[i].width >> 1;
                src.height = haarRes[i].height >> 1;
                src.stride = haarRes[i].stride;
                src.data = haarRes[i].data;
                    break;

        }

    }

    for(int i = 0; i < 3; i++)
    {
        free_humat(haarRes[i]);
        free_humat(emap[i]);
    }

    Nedge = Nda = Nrg = Nbrg = 0;

    width  = emax[0].width;
    height = emax[0].height;
    stride = emax[0].stride;

    emax0 = emax[0].data;
    emax1 = emax[1].data;
    emax2 = emax[2].data;

    for(int l = 0; l < height; l++)
    {
        for(int k = 0; k < width; k++)
        {
            if(emax0[k] <= EDGE_THRESH &&
                emax1[k] <= EDGE_THRESH &&
                emax2[k] <= EDGE_THRESH)
                continue;

            Nedge ++;

            if(emax0[k] > emax1[k] && emax1[k] > emax2[k])
                Nda ++;

            else //if(emax0[k] < emax1[k] && emax1[k] < emax2[k])
            {
                Nrg++;
                if(emax0[k] < EDGE_THRESH)
                    Nbrg++;
            }
        }

        emax0 += stride;
        emax1 += stride;
        emax2 += stride;
    }


    float per = 1.0 * Nda / Nedge;

    *conf = 1.0 * Nbrg / Nrg;

    printf("%d %d %d %d\n", Nda, Nrg, Nbrg, Nedge);
    printf("%f %f\n", per, *conf);

    if(per < 0.01) return 1;

    return 0;
}



MainWindow::MainWindow(QWidget *parent) :
    QMainWindow(parent),
    ui(new Ui::MainWindow)
{
    ui->setupUi(this);
    dialog=new QFileDialog(this);
    dialog->setAcceptMode(QFileDialog::AcceptOpen);
    QStringList filters;
    filters<<"All images (*.jpg *.jpeg *.png *.gif *.bmp)"
           <<"JPEG images (*.jpg *.jpeg)"
           <<"PNG images (*.png)"
           <<"GIF images (*.gif)"
           <<"Bitmaps (*.bmp)";
    dialog->setNameFilters(filters);
    dialog->setDirectory(QStandardPaths::writableLocation(QStandardPaths::DocumentsLocation));
    connect(dialog,SIGNAL(fileSelected(QString)),this,SLOT(dialogFileSelected(QString)));
    connect(ui->browseBtn,SIGNAL(clicked(bool)),this,SLOT(browseBtnClicked()));
    connect(ui->loadBtn,SIGNAL(clicked(bool)),this,SLOT(loadBtnClicked()));
    connect(ui->fitToWindowBtn,SIGNAL(clicked(bool)),this,SLOT(fitToWindow()));
    connect(ui->resetZoomBtn,SIGNAL(clicked(bool)),this,SLOT(resetZoom()));
    image=0;
    originalImageData=0;
    scene=new QGraphicsScene();
    pixmapItem=new QGraphicsPixmapItem();
    scene->addItem(pixmapItem);
    ui->graphicsView->setScene(scene);

    connect(ui->resizeBtn,SIGNAL(clicked(bool)),this,SLOT(resizeBtnClicked()));
    connect(ui->saveAsBtn,SIGNAL(clicked(bool)),this,SLOT(saveAsBtnClicked()));
    connect(ui->resetBtn,SIGNAL(clicked(bool)),this,SLOT(resetBtnClicked()));

    ui->widthBtn->setChecked(true);


    // Quick example image loading code:
    QTimer *tmr=new QTimer(this);
    connect(tmr,&QTimer::timeout,[=]() {
        QString path="C:Desktop/link_here/example.jpg, .jpeg, .png, .gif, .bmp, .gif, .bmp";
        this->setWindowState(Qt::WindowMaximized);
        ui->pathBox->setText(path);
        ui->ratioBtn->setChecked(true);
        ui->ratioSpinBox->setValue(0.5);
        ui->methodBox->setCurrentIndex(0);
        ui->waveletBox->setCurrentIndex(0);
        //loadImageFromPath(path);
        ui->resetZoomBtn->click();
    });
    tmr->setSingleShot(true);
    tmr->start(100);


}

MainWindow::~MainWindow()
{
    free(originalImageData);
    delete ui;
}

void MainWindow::browseBtnClicked()
{
    dialog->exec();
}

void MainWindow::loadBtnClicked()
{
    QString path=ui->pathBox->text();
    if(path.length()==0)
    {
        QMessageBox::critical(this,"Error","No file selected.");
        return;
    }
    QFile f(path);
    if(!f.exists())
    {
        QMessageBox::critical(this,"Error","The selected file does not exist.");
        return;
    }
    image=new QImage(path);
    if(image->isNull())
    {
        QMessageBox::critical(this,"Error","The selected file has an unsupported format.");
        return;
    }

    int imgWidth = image->width(), imgHeight = image->height();
    originalImageWidth = imgWidth;
    originalImageHeight = imgHeight;
    scene->setSceneRect(0,0,imgWidth,imgHeight);
    pixmapItem->setPixmap(QPixmap::fromImage(*image));
    ui->graphicsView->viewport()->update();
    fitToWindow();
    ui->widthSpinBox->setValue(imgWidth);
    ui->heightSpinBox->setValue(imgHeight);
    free(originalImageData);
    originalImageData=qImageToBitmapData(image);
}

void MainWindow::fitToWindow()
{
    if(image==0||image->isNull())
        return;
    int width=image->width();
    int height=image->height();
    QRect rect=ui->graphicsView->contentsRect();
    int availableWidth=rect.width()-ui->graphicsView->verticalScrollBar()->width();
    int availableHeight=rect.height()-ui->graphicsView->horizontalScrollBar()->height();
    if((width-availableWidth)>(height-availableHeight))
        ui->graphicsView->setZoomFactor((decimal_t)((decimal_t)availableWidth)/((decimal_t)width));
    else if(height>availableHeight)
        ui->graphicsView->setZoomFactor((decimal_t)((decimal_t)availableHeight)/((decimal_t)height));
    else
        ui->graphicsView->setZoomFactor(1.0);
}

void MainWindow::resetZoom()
{
    ui->graphicsView->setZoomFactor(1.0);
}

void MainWindow::dialogFileSelected(QString path)
{
    ui->pathBox->setText(path);
    ui->loadBtn->click();
}

void MainWindow::resizeBtnClicked()
{
    if(image==0||image->isNull())
        return;

    bool useWidth = ui->widthBtn->isChecked();
    bool useHeight = ui->heightBtn->isChecked();

    int method = ui->methodBox->currentIndex();
    waveMtd = ui->waveletBox->currentIndex();
    qDebug() << "method____>Message";
    qDebug()<< method;

    if(method==-1)
        return;

    if(waveMtd == -1)
        return;

    if(method==0) // Nearest neighbor
    {
        int newWidth;
        int newHeight;
        decimal_t ratio;
        decimal_t rRatio;
        if(useWidth)
        {
            newWidth=ui->widthSpinBox->value();
            ratio=((decimal_t)newWidth)/((decimal_t)originalImageWidth);
            rRatio=1.0f/ratio;
            newHeight=(int)round(ratio*((decimal_t)originalImageHeight));
        }
        else if(useHeight)
        {
            newHeight=ui->heightSpinBox->value();
            ratio=((decimal_t)newHeight)/((decimal_t)originalImageHeight);
            rRatio=1.0f/ratio;
            newWidth=(int)round(ratio*((decimal_t)originalImageWidth));
        }
        else // if(useRatio)
        {
            ratio=ui->ratioSpinBox->value();
            rRatio=1.0f/ratio;
            newWidth=(int)round(ratio*((decimal_t)originalImageWidth));
            newHeight=(int)round(ratio*((decimal_t)originalImageHeight));
        }

        uint32_t *newImageData=(uint32_t*)malloc(newWidth * newHeight * sizeof(uint32_t));
        for(int y = 0; y < newHeight; y++)
        {
            int offset=y*newWidth;
            for(int x=0;x<newWidth;x++)
            {
                int oldX=(int)round(rRatio*x);
                int oldY=(int)round(rRatio*y);
                newImageData[offset+x] = originalImageData[oldY*originalImageWidth+oldX];
            }
        }
        delete image;
        image=new QImage((uchar*)newImageData,newWidth,newHeight,QImage::Format_ARGB32);
        scene->setSceneRect(0,0,newWidth,newHeight);
        pixmapItem->setPixmap(QPixmap::fromImage(*image));
        ui->graphicsView->viewport()->update();
        fitToWindow();
    }
    else if(method == 1) // Bilinear
    {
        int newWidth;
        int newHeight;
        decimal_t ratio;
        decimal_t rRatio;
        if(useWidth)
        {
            newWidth=ui->widthSpinBox->value();
            ratio=((decimal_t)newWidth)/((decimal_t)originalImageWidth);
            rRatio=1.0f/ratio;
            newHeight=(int)round(ratio*((decimal_t)originalImageHeight));
        }
        else if(useHeight)
        {
            newHeight=ui->heightSpinBox->value();
            ratio=((decimal_t)newHeight)/((decimal_t)originalImageHeight);
            rRatio=1.0f/ratio;
            newWidth=(int)round(ratio*((decimal_t)originalImageWidth));
        }
        else // if(useRatio)
        {
            ratio=ui->ratioSpinBox->value();
            rRatio=1.0f/ratio;
            newWidth=(int)round(ratio*((decimal_t)originalImageWidth));
            newHeight=(int)round(ratio*((decimal_t)originalImageHeight));
        }

        int xLim=originalImageWidth-1;
        int yLim=originalImageHeight-1;
        uint32_t *newImageData=(uint32_t*)malloc(newWidth*newHeight*sizeof(uint32_t));
        //const bool useAlpha=image->hasAlphaChannel();

        for(int y=0;y<newHeight;y++)
        {
            decimal_t oldYRF=rRatio*y;
            int oldY=(int)floor(oldYRF);
            decimal_t oldYF=(decimal_t)oldY;
            decimal_t yDiff=oldYRF-oldYF;
            decimal_t yDiffR=1.0-yDiff;
            int offset=y*newWidth;

            const bool checkVBounds=oldY<=1||oldY>=originalImageHeight-2;

            for(int x=0;x<newWidth;x++)
            {
                decimal_t oldXRF=rRatio*x;
                int oldX=(int)floor(oldXRF);
                decimal_t oldXF=(decimal_t)oldX;
                decimal_t xDiff=oldXRF-oldXF;
                decimal_t xDiffR=1.0f-xDiff;

                const bool checkBounds=checkVBounds||oldX<=1||oldX>=originalImageWidth-2;

                uint32_t c00,c01,c10,c11;

                if(checkBounds)
                {
                    c00=originalImageData[oldY*originalImageWidth+oldX];
                    c10=originalImageData[oldY*originalImageWidth+(oldX==xLim?oldX:oldX+1)];
                    c01=(oldY==yLim?c00:originalImageData[(oldY+1)*originalImageWidth+oldX]);
                    c11=(oldY==yLim?c10:(oldX==xLim?originalImageData[(oldY+1)*originalImageWidth+oldX]:originalImageData[(oldY+1)*originalImageWidth+(oldX+1)]));
                }
                else
                {
                    c00=originalImageData[oldY*originalImageWidth+oldX];
                    c10=originalImageData[oldY*originalImageWidth+oldX+1];
                    c01=originalImageData[(oldY+1)*originalImageWidth+oldX];
                    c11=originalImageData[(oldY+1)*originalImageWidth+(oldX+1)];
                }

                decimal_t w1=xDiffR*yDiffR;
                decimal_t w2=xDiff*yDiffR;
                decimal_t w3=xDiffR*yDiff;
                decimal_t w4=xDiff*yDiff;

                uint32_t newAlpha = bilinearInterpolate(getAlpha(c00),getAlpha(c01),getAlpha(c10),getAlpha(c11),w1,w2,w3,w4);
                uint32_t newRed=bilinearInterpolate(getRed(c00),getRed(c01),getRed(c10),getRed(c11),w1,w2,w3,w4);
                uint32_t newGreen=bilinearInterpolate(getGreen(c00),getGreen(c01),getGreen(c10),getGreen(c11),w1,w2,w3,w4);
                uint32_t newBlue=bilinearInterpolate(getBlue(c00),getBlue(c01),getBlue(c10),getBlue(c11),w1,w2,w3,w4);
                   
                 

                newImageData[offset+x]=getColor(newAlpha,newRed,newGreen,newBlue);
            }
        }
        delete image;
        image=new QImage((uchar*)newImageData,newWidth,newHeight,QImage::Format_ARGB32);
        scene->setSceneRect(0,0,newWidth,newHeight);
        pixmapItem->setPixmap(QPixmap::fromImage(*image));
        ui->graphicsView->viewport()->update();
        fitToWindow();
    }
    else if(method == 2) // Bicubic
    {
        int newWidth;
        int newHeight;
        decimal_t ratio;
        decimal_t rRatio;
        if(useWidth)
        {
            newWidth=ui->widthSpinBox->value();
            ratio=((decimal_t)newWidth)/((decimal_t)originalImageWidth);
            rRatio=1.0f/ratio;
            newHeight=(int)round(ratio*((decimal_t)originalImageHeight));
        }
        else if(useHeight)
        {
            newHeight=ui->heightSpinBox->value();
            ratio=((decimal_t)newHeight)/((decimal_t)originalImageHeight);
            rRatio=1.0f/ratio;
            newWidth=(int)round(ratio*((decimal_t)originalImageWidth));
        }
        else // if(useRatio)
        {
            ratio=ui->ratioSpinBox->value();
            rRatio=1.0f/ratio;
            newWidth=(int)round(ratio*((decimal_t)originalImageWidth));
            newHeight=(int)round(ratio*((decimal_t)originalImageHeight));
        }

        uint32_t *newImageData=(uint32_t*)malloc(newWidth*newHeight*sizeof(uint32_t));
        //const bool useAlpha=image->hasAlphaChannel();

        // Cache cubicWeigh (abbrevated as R) values for every row and column

        // Columns must be precached, rows can be cached on the run

        decimal_t *rColCache=(decimal_t*)malloc(4*newWidth*sizeof(decimal_t));

        for(int x=0;x<newWidth;x++)
        {
            decimal_t oldXRF=rRatio*x;
            int oldX=(int)oldXRF;
            decimal_t oldXF=(decimal_t)oldX;
            decimal_t xDiff=oldXRF-oldXF;
            for(int _x=-1;_x<=2;_x++)
            {
                size_t index=(_x+1)+4*x;
                rColCache[index]=cubicWeigh(((decimal_t)_x)-xDiff);
            }
        }

        for(int y=0;y<newHeight;y++)
        {
            int offset=y*newWidth;
            decimal_t oldYRF=rRatio*y;
            int oldY=(int)oldYRF;
            decimal_t oldYF=(decimal_t)oldY;
            decimal_t yDiff=oldYRF-oldYF;

            bool checkVBounds=oldY<=1||oldY>=originalImageHeight-2;

            decimal_t *rRowCache=(decimal_t*)malloc(4*sizeof(decimal_t));

            for(int _y=-1;_y<=2;_y++)
            {
                rRowCache[_y+1]=cubicWeigh(yDiff-((decimal_t)_y));
            }

            for(int x=0;x<newWidth;x++)
            {
                decimal_t oldXRF=rRatio*x;
                int oldX=(int)oldXRF;

                // This function uses optimized versions of bicubicInterpolate()

                uint32_t newAlpha;
                uint32_t newRed;
                uint32_t newGreen;
                uint32_t newBlue;

                decimal_t sumAlpha=0.0f;
                decimal_t sumRed=0.0f;
                decimal_t sumGreen=0.0f;
                decimal_t sumBlue=0.0f;

                const bool checkBounds=checkVBounds||oldX<=1||oldY>=originalImageHeight-2;

#define calculateColor \
                uint32_t color=originalImageData[ly*originalImageWidth+lx];\
                decimal_t orig=(decimal_t)((uint32_t)((color)>>24)&0xFF);\
                decimal_t prod=rColCache[4*x+_x+1]*rRowCache[_y+1];\
                sumAlpha+=orig*prod;\
                orig=(decimal_t)((uint32_t)((color)>>16)&0xFF);\
                sumRed+=orig*prod;\
                orig=(decimal_t)((uint32_t)((color)>>8)&0xFF);\
                sumGreen+=orig*prod;\
                orig=(decimal_t)((uint32_t)(color)&0xFF);\
                sumBlue+=orig*prod;\

                if(checkBounds)
                {
                    for(int _y=-1;_y<=2;_y++)
                    {
                        for(int _x=-1;_x<=2;_x++)
                        {
                            int lx=oldX+_x,ly=oldY+_y;
                            if(lx<0) lx=0;
                            if(lx>=originalImageWidth) lx=originalImageWidth-1;
                            if(ly<0) ly=0;
                            if(ly>=originalImageHeight) ly=originalImageHeight-1;

                            calculateColor;
                        }
                    }
                }
                else
                {
                    for(int _y=-1;_y<=2;_y++)
                    {
                        for(int _x=-1;_x<=2;_x++)
                        {
                            int lx=oldX+_x,ly=oldY+_y;
                            calculateColor;
                        }
                    }
                }

#undef calculateColor

                newAlpha=(uint32_t)round(sumAlpha);
                newRed=(uint32_t)round(sumRed);
                newGreen=(uint32_t)round(sumGreen);
                newBlue=(uint32_t)round(sumBlue);

                newImageData[offset+x]=getColor(newAlpha,newRed,newGreen,newBlue);
            }
            free(rRowCache);
        }

        free(rColCache);

        delete image;
        image=new QImage((uchar*)newImageData,newWidth,newHeight,QImage::Format_ARGB32);
        scene->setSceneRect(0,0,newWidth,newHeight);
        pixmapItem->setPixmap(QPixmap::fromImage(*image));
        ui->graphicsView->viewport()->update();
        fitToWindow();

    }

    else if(method == 3) //Lanczos method
    {

        int newWidth;
        int newHeight;
        decimal_t ratio;
        decimal_t rRatio;
        if(useWidth)
        {
            newWidth=ui->widthSpinBox->value();
            ratio=((decimal_t)newWidth)/((decimal_t)originalImageWidth);
            rRatio=1.0f/ratio;
            newHeight=(int)round(ratio*((decimal_t)originalImageHeight));
        }
        else if(useHeight)
        {
            newHeight=ui->heightSpinBox->value();
            ratio=((decimal_t)newHeight)/((decimal_t)originalImageHeight);
            rRatio=1.0f/ratio;
            newWidth=(int)round(ratio*((decimal_t)originalImageWidth));
        }
        else // if(useRatio)
        {
            ratio=ui->ratioSpinBox->value();
            rRatio=1.0f/ratio;
            newWidth=(int)round(ratio*((decimal_t)originalImageWidth));
            newHeight=(int)round(ratio*((decimal_t)originalImageHeight));
        }

        //uint32_t *newImageData=(uint32_t*)malloc(newWidth*newHeight*sizeof(uint32_t));

        Mat resized;
        cv::Mat cvImage(image->height(),image->width(),CV_8UC4, (uchar*)image->bits(),image->bytesPerLine());
        cv::Mat result;


        //cv::Mat cvImage(originalImageWidth, originalImageHeight, CV_8UC3, originalImageData);

         cv::resize(cvImage, resized, Size(newWidth, newHeight), INTER_LANCZOS4);

         imshow("Lanczos method display window", resized);


         delete image;
                 //image=new QImage((uchar*)newImageData,newWidth,newHeight,QImage::Format_ARGB32);
         //image = new QImage((const unsigned char *)(resized.data), resized.cols, resized.rows, QImage::Format_Indexed8);
         image = new QImage(resized.data, resized.cols, resized.rows, resized.step[0], QImage::Format_ARGB32);
         qDebug()<< *image;
         scene->setSceneRect(0,0,newWidth,newHeight);
         pixmapItem->setPixmap(QPixmap::fromImage(*image));
         ui->graphicsView->viewport()->update();
         fitToWindow();
    }

     else if(method == 4) //Wavelet filter
    {

        Mat im1, im2, im3, im4, im5, im6, temp, im11, im12, im13, im14, imi, imd, imd2, imd3, imd4, imr;
        float a, b, c, d;
        cv::Mat im(image->height(),image->width(),CV_8UC4, (uchar*)image->bits(),image->bytesPerLine());

        //imi = Mat::zeros(im.rows,im.cols,CV_8UC4);
        im.copyTo(imi);

        im.convertTo(im,CV_32F, 1.0,0.0);

        im1 = Mat::zeros(im.rows/2,im.cols,CV_32F);
        im2 = Mat::zeros(im.rows/2,im.cols,CV_32F);
        im3 = Mat::zeros(im.rows/2,im.cols/2,CV_32F);
        im4 = Mat::zeros(im.rows/2,im.cols/2,CV_32F);
        im5 = Mat::zeros(im.rows/2,im.cols/2,CV_32F);
        im6 = Mat::zeros(im.rows/2,im.cols/2,CV_32F);


        for(int rcnt=0;rcnt<im.rows;rcnt+=2)
                       {
                           for(int ccnt=0;ccnt<im.cols;ccnt++)
                           {

                               a=im.at<float>(rcnt,ccnt);
                               b=im.at<float>(rcnt+1,ccnt);
                               c=(a+b)*0.707;
                               d=(a-b)*0.707;
                               int _rcnt=rcnt/2;
                               im1.at<float>(_rcnt,ccnt)=c;
                               im2.at<float>(_rcnt,ccnt)=d;



                           }
                       }

                       for(int rcnt=0; rcnt<im.rows/2; rcnt++)
                       {
                           for(int ccnt=0; ccnt<im.cols; ccnt+=2)
                           {

                               a=im1.at<float>(rcnt,ccnt);
                               b=im1.at<float>(rcnt,ccnt+1);
                               c=(a+b)*0.707;
                               d=(a-b)*0.707;
                               int _ccnt=ccnt/2;
                               im3.at<float>(rcnt,_ccnt)=c;
                               im4.at<float>(rcnt,_ccnt)=d;
                           }
                       }

                       for(int rcnt=0; rcnt<im.rows/2; rcnt++)
                       {
                           for(int ccnt=0;ccnt<im.cols;ccnt+=2)
                           {

                               a=im2.at<float>(rcnt,ccnt);
                               b=im2.at<float>(rcnt,ccnt+1);
                               c=(a+b)*0.707;
                               d=(a-b)*0.707;
                               int _ccnt=ccnt/2;
                               im5.at<float>(rcnt,_ccnt)=c;
                               im6.at<float>(rcnt,_ccnt)=d;
                           }
                       }


         imr = cv::Mat::zeros(256,256,CV_32F);
         imd = cv::Mat::zeros(256,256,CV_32F);

         imd2 = cv::Mat::zeros(256,256,CV_32F);
         imd3 = cv::Mat::zeros(256,256,CV_32F);
         imd4 = cv::Mat::zeros(256,256,CV_32F);


         im11 = Mat::zeros(im.rows/2,im.cols,CV_32F);
         im12 = Mat::zeros(im.rows/2,im.cols,CV_32F);
         im13 = Mat::zeros(im.rows/2,im.cols,CV_32F);
         im14 = Mat::zeros(im.rows/2,im.cols,CV_32F);

         //imshow("Wavelet Decomposition", imd2);
          /*
          
          Mat dest;
          Mat src_f;
          im.convertTo(src_f, CV_32F);

          int kernel_size = 31;
          double sig = 1, th = 0, lm = 1.0, gm = 0.02, ps = 0;
          cv::Mat kernel = cv::getGaborKernel(cv::Size(kernel_size, kernel_size), sig, th, lm, gm, ps);
          cv::filter2D(src_f, dest, CV_32F, kernel);

          //cerr << dest(Rect(30,30,10,10)) << endl; // peek into the data

          Mat viz;
          dest.convertTo(viz,CV_8U,1.0/255.0);
         imshow("d",viz);
         */

         imd(Rect(0,0,128,128));
         im3.copyTo(imd);

         imd2(Rect(0,127,128,128));
         im4.copyTo(imd2);

         imd3(Rect(127,0,128,128));
         im5.copyTo(imd3);

         imd4(Rect(127,127,128,128));
         im6.copyTo(imd4);

         imd.convertTo(imd, CV_8UC4);
         imshow("Wavelet Decomposition",imd);

         imd2.convertTo(imd2, CV_8UC4);
         imshow("Wavelet Decomposition2",imd2);

         imd3.convertTo(imd3, CV_8UC4);
         imshow("Wavelet Decomposition3",imd3);

         imd4.convertTo(imd4, CV_8UC4);
         imshow("Wavelet Decomposition4",imd4);

         waitKey(0);

    }

    else if(method == 5) //INTER_AREA method
    {

        int newWidth;
        int newHeight;
        decimal_t ratio;
        decimal_t rRatio;
        if(useWidth)
        {
            newWidth=ui->widthSpinBox->value();
            ratio=((decimal_t)newWidth)/((decimal_t)originalImageWidth);
            rRatio=1.0f/ratio;
            newHeight=(int)round(ratio*((decimal_t)originalImageHeight));
        }
        else if(useHeight)
        {
            newHeight=ui->heightSpinBox->value();
            ratio=((decimal_t)newHeight)/((decimal_t)originalImageHeight);
            rRatio=1.0f/ratio;
            newWidth=(int)round(ratio*((decimal_t)originalImageWidth));
        }
        else // if(useRatio)
        {
            ratio=ui->ratioSpinBox->value();
            rRatio=1.0f/ratio;
            newWidth=(int)round(ratio*((decimal_t)originalImageWidth));
            newHeight=(int)round(ratio*((decimal_t)originalImageHeight));
        }


        Mat resized;
        cv::Mat cvImage(image->height(),image->width(),CV_8UC4, (uchar*)image->bits(),image->bytesPerLine());
        cv::Mat result;

        cv::resize(cvImage, resized, Size(newWidth, newHeight), INTER_AREA);

        delete image;
        image = new QImage(resized.data, resized.cols, resized.rows, resized.step[0], QImage::Format_ARGB32);
        qDebug()<< *image;
        scene->setSceneRect(0,0,newWidth,newHeight);
        pixmapItem->setPixmap(QPixmap::fromImage(*image));
        ui->graphicsView->viewport()->update();
        fitToWindow();
    }

    else if(method == 6){

        int newWidth;
        int newHeight;
        decimal_t ratio;
        decimal_t rRatio;
        if(useWidth)
        {
            newWidth=ui->widthSpinBox->value();
            ratio=((decimal_t)newWidth)/((decimal_t)originalImageWidth);
            rRatio=1.0f/ratio;
            newHeight=(int)round(ratio*((decimal_t)originalImageHeight));
        }
        else if(useHeight)
        {
            newHeight=ui->heightSpinBox->value();
            ratio=((decimal_t)newHeight)/((decimal_t)originalImageHeight);
            rRatio=1.0f/ratio;
            newWidth=(int)round(ratio*((decimal_t)originalImageWidth));
        }
        else // if(useRatio)
        {
            ratio=ui->ratioSpinBox->value();
            rRatio=1.0f/ratio;
            newWidth=(int)round(ratio*((decimal_t)originalImageWidth));
            newHeight=(int)round(ratio*((decimal_t)originalImageHeight));
        }

        cv::Mat img(image->height(),image->width(),CV_8UC4, (uchar*)image->bits(),image->bytesPerLine());
       // cv::Mat img = cv::imread("C:/Users/User/Documents/untitled2/registan.jpg", 0);
        float conf = 0;

        if(img.empty())
        {
            printf("Can't open image %s\n");
            return;
        }

        cv::resize(img, img, cv::Size(newWidth, newHeight));

        int rows = img.rows;
        int cols = img.cols;

        HuMat src;

        create_humat(src, cols, rows);

        uchar *ptrImg = img.data;
        int *ptrSrc = src.data;

        for(int y = 0; y < rows; y++)
        {
            for(int x = 0; x < cols; x++)
                ptrSrc[x] = ptrImg[x];

            ptrSrc += src.stride;
            ptrImg += img.step;
        }

        float confidence = 0;

        blur_detect(src, &confidence, waveMtd);

    }


}



void MainWindow::resetBtnClicked()
{
    if(image==0||image->isNull())
        return;

    delete image;
    image=new QImage((uchar*)originalImageData,originalImageWidth,originalImageHeight,QImage::Format_ARGB32);
    scene->setSceneRect(0,0,originalImageWidth,originalImageHeight);
    pixmapItem->setPixmap(QPixmap::fromImage(*image));
    ui->graphicsView->viewport()->update();
    fitToWindow();
}

void MainWindow::saveAsBtnClicked()
{
    if(image==0||image->isNull())
        return;

    QString path=QFileDialog::getSaveFileName(this,"Save as...",QStandardPaths::writableLocation(QStandardPaths::DocumentsLocation),"JPG image (*.jpg);;PNG image (*.png);;GIF image (*.gif);;Bitmap (*.bmp)");
    if(path=="")
        return;
    image->save(path,0,100);
}

uint32_t *MainWindow::qImageToBitmapData(QImage *image)
{
    int32_t width=image->width();
    int32_t height=image->height();
    uint32_t *out=(uint32_t*)malloc(width*height*sizeof(uint32_t));
    for(int32_t y=0;y<height;y++)
    {
        int32_t offset=y*width;
        QRgb *scanLine=(QRgb*)image->scanLine(y); // Do not free!
        for(int32_t x=0;x<width;x++)
        {
            QRgb color=scanLine[x];
            uint32_t alpha=qAlpha(color);
            uint32_t red=qRed(color);
            uint32_t green=qGreen(color);
            uint32_t blue=qBlue(color);
            out[offset+x]=(alpha<<24)|(red<<16)|(green<<8)|blue;
        }
        // Do not free "scanLine"!
    }
    return out;
}

decimal_t MainWindow::bilinearInterpolate(decimal_t c00, decimal_t c10, decimal_t c01, decimal_t c11, decimal_t w1, decimal_t w2, decimal_t w3, decimal_t w4)
{
    return w1*c00+w2*c01+w3*c10+w4*c11;
}

decimal_t MainWindow::bicubicInterpolate(int ix, int iy, decimal_t dx, decimal_t dy, size_t shift)
{
    decimal_t sum=0.0f;
    for(int _y=-1;_y<=2;_y++)
    {
        for(int _x=-1;_x<=2;_x++)
        {
            int x=ix+_x,y=iy+_y;
            if(x<0) x=0;
            if(x>=originalImageWidth) x=originalImageWidth-1;
            if(y<0) y=0;
            if(y>=originalImageHeight) y=originalImageHeight-1;
            decimal_t orig=(decimal_t)((uint32_t)((originalImageData[y*originalImageWidth+x])>>shift)&0xFF);
            sum+=orig*cubicWeigh(((decimal_t)_x)-dx)*cubicWeigh(dy-((decimal_t)_y));
        }
    }
    return sum;
}

decimal_t MainWindow::cubicWeigh(decimal_t x)
{
    //const float ratio=1.0f/6.0f;
    decimal_t ratio=(decimal_t)(1.0/6.0);
    return ratio*(ifGTZeroArg((x+2.0f),pow((x+2.0f),3.0f))-4.0f*ifGTZeroArg(x+1.0f,pow((x+1.0f),3.0f))+6.0f*ifGTZeroArg(x,pow(x,3.0f))-4.0f*ifGTZeroArg(x-1.0f,pow((x-1.0f),3.0f)));
}
