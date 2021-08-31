#include "mainwindow.h"
#include "ui_mainwindow.h"
#include <QMessageBox>
#include <QCoreApplication>
#include <QNetworkRequest>
#include <QNetworkReply>
#include <QNetworkAccessManager>
#include <QUuid>
#include <QJsonObject>
#include <QJsonDocument>
#include <QFile>
#include <QHttpPart>
#include <QUrlQuery>
#include <QFileDialog>

MainWindow::MainWindow(QWidget *parent) :
    QMainWindow(parent),
    ui(new Ui::MainWindow)
{
    ui->setupUi(this);
    connect(ui->choseFileBtn, SIGNAL(clicked()), this, SLOT(ChoseFile()));
    connect(ui->sendHttpBtn, SIGNAL(clicked()), this, SLOT(SendPost()));

}

void MainWindow::ChoseFile(){
    QString curPath=QDir::currentPath();//获取当前路径
    QString str=QFileDialog::getOpenFileName(this,"打开文件",curPath);
    // QString str=QFileDialog::getOpenFileName(this,"打开文件",".");//"." 代表程序运行目录
    // QString str=QFileDialog::getOpenFileName(this,"打开文件","/");//"/" 代表当前盘符的根目录
    qDebug()<<str;
    ui->showFilePath->setText(str.toUtf8());
}

void MainWindow::SendPost(){

    QHttpMultiPart *multiPart = new QHttpMultiPart(QHttpMultiPart::FormDataType);

    QHttpPart imageNamePart;
    QByteArray imageName="6500239266918274-1.jpg";
    imageNamePart.setHeader(QNetworkRequest::ContentDispositionHeader, QVariant("form-data; name=\"imageName\""));
    imageNamePart.setBody(imageName);
    multiPart->append(imageNamePart); // imageName

    QHttpPart originPart;
    QByteArray origin="Batch-Test";
    originPart.setHeader(QNetworkRequest::ContentDispositionHeader, QVariant("form-data; name=\"origin\""));
    originPart.setBody(origin);
    multiPart->append(originPart); // origin

    QHttpPart productTypePart;
    QByteArray productType="1";
    productTypePart.setHeader(QNetworkRequest::ContentDispositionHeader, QVariant("form-data; name=\"productType\""));
    productTypePart.setBody(productType);
    multiPart->append(productTypePart); // productType

    QHttpPart imagePart;
    QString filePath = ui->showFilePath->toPlainText(); //"C:\\Users\\zengxh\\Documents\\workspace\\qt-workspace\\opt\\6500239266918274-1.jpg";
    imagePart.setHeader(QNetworkRequest::ContentTypeHeader, QVariant("image/jpeg"));
    imagePart.setHeader(QNetworkRequest::ContentDispositionHeader, QVariant("form-data; name=\"file \"; filename=\"" + filePath + "\""));
    QFile *file = new QFile(filePath);
    file->open(QIODevice::ReadOnly);
    imagePart.setBodyDevice(file);
    file->setParent(multiPart); // Delete object with parent

    multiPart->append(imagePart); // attachment

    QUrl url("http://10.123.131.51:8001/pv/api/files/");
    QNetworkRequest request(url);

    QNetworkAccessManager *m_manager = new QNetworkAccessManager(this);

    //发送
    QNetworkReply *reply = m_manager->post(request, multiPart);
    multiPart->setParent(reply); // Delete object with parent
    QByteArray responseData;
    QEventLoop eventLoop;
    QObject::connect(m_manager, SIGNAL(finished(QNetworkReply *)), &eventLoop, SLOT(quit()));
    eventLoop.exec();

    //返回结果
    responseData = reply->readAll();

    qDebug() << responseData;
    QMessageBox::information(this, "My Tittle", responseData);
}

MainWindow::~MainWindow()
{
    delete ui;
}

