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

MainWindow::MainWindow(QWidget *parent) :
    QMainWindow(parent),
    ui(new Ui::MainWindow)
{
    ui->setupUi(this);
    connect(ui->sendHttpBtn, SIGNAL(clicked()), this, SLOT(SendPost()));
}

void MainWindow::SendPost(){
    QMessageBox::information(this, "My Tittle", "Hello World!");
//    ui->sendHttpBtn->setText(tr("(adgkl;kj)"));

    // 构建及发送请求
//    QNetworkAccessManager *manager = new QNetworkAccessManager();
//    QString url = "https://www.baidu.com";
//    QNetworkRequest request;
//    request.setUrl(QUrl(url));
//    QNetworkReply *pReply = manager->get(request);
//    // 开启一个局部的事件循环，等待页面响应结束
//    QEventLoop eventLoop;
//    QObject::connect(manager, &QNetworkAccessManager::finished, &eventLoop, &QEventLoop::quit);
//    eventLoop.exec();
//    // 获取网页Body中的内容
//    QByteArray bytes = pReply->readAll();
//    qDebug() << bytes;


    QNetworkAccessManager *m_manager = new QNetworkAccessManager(this);

    QString AppEn_Url = "http://10.123.131.51:8001/pv/api/files/";         //服务器地址
    QNetworkRequest netRequest;

    netRequest.setUrl(QUrl(AppEn_Url));
    netRequest.setHeader(QNetworkRequest::ContentTypeHeader, "application/x-www-form-urlencoded;charset=utf-8");

    QByteArray postArray;
    postArray.append("imageName=6500239266918274-1.jpg");
    postArray.append("&origin=Batch-Test");
    postArray.append("&productType=1");

//    file->open(QIODevice::ReadOnly);
//    netRequest.setBodyDevice(file);
//    netRequest.setHeader(QNetworkRequest::ContentLengthHeader,postArray.size());




    QHttpMultiPart *multiPart = new QHttpMultiPart(QHttpMultiPart::FormDataType);

    QHttpPart keyPart;
    keyPart.setHeader(QNetworkRequest::ContentDispositionHeader, QVariant("form-data; imageName=\"key\""));
    keyPart.setBody("6500239266918274-1.jpg");

    QString path = QString("C:\\Users\\zengxh\\Documents\\workspace\\qt-workspace\\opt\\6500239266918274-1.jpg");
    QHttpPart imagePart;
    imagePart.setHeader(QNetworkRequest::ContentTypeHeader, QVariant("image/jpeg"));
    QFile *imgFile = new QFile(path);
    imgFile->open(QIODevice::ReadOnly);
    QString dispositionHeader = QString("form-data; name=\"%1\";filename=\"%2\"")
            .arg("text_file")
            .arg(imgFile->fileName());
    imagePart.setHeader(QNetworkRequest::ContentDispositionHeader, dispositionHeader);
    imagePart.setBodyDevice(imgFile);
    imgFile->setParent(multiPart);
//    multiPart->append(imagePart);

    multiPart->setProperty("imageName","sdfadsf");

    //QByteArray filedata = imgFile->readAll();
    //postArray.append("&file=");
    //postArray.append(filedata);

//    QUrlQuery postData;
//    postData.addQueryItem("imageName", "6500239266918274-1.jpg");
//    postData.addQueryItem("origin", "Batch-Test");
//    postData.addQueryItem("productType","1");
//        QNetworkReply *reply = m_manager->post(netRequest,postData.toString(QUrl::FullyEncoded).toUtf8());

    //发送
    QNetworkReply *reply = m_manager->post(netRequest,postArray);
    QByteArray responseData;
    QEventLoop eventLoop;
    QObject::connect(m_manager, SIGNAL(finished(QNetworkReply *)), &eventLoop, SLOT(quit()));
    eventLoop.exec();

    //返回结果
    responseData = reply->readAll();

    qDebug() << responseData;
}

MainWindow::~MainWindow()
{
    delete ui;
}

