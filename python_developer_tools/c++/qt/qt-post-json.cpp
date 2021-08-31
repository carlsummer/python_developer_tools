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
    netRequest.setRawHeader("Host", "10.123.131.51:8001");
    netRequest.setRawHeader("Connection", "keep-alive");
    netRequest.setRawHeader("Cache-Control", "max-age=0");
    netRequest.setRawHeader("Origin", "http://10.123.131.51:8001");
    netRequest.setRawHeader("Upgrade-Insecure-Requests", "1");
    netRequest.setRawHeader("Content-Type", "application/x-www-form-urlencoded");
    netRequest.setRawHeader("User-Agent", "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/76.0.3809.132 Safari/537.36");
    netRequest.setRawHeader("Accept", "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3");
    netRequest.setRawHeader("Referer", "http://10.123.131.51:8001/");
    netRequest.setRawHeader("Accept-Encoding", "gzip, deflate");
    netRequest.setRawHeader("Accept-Language", "zh-CN,zh;q=0.9");

    QString strJSESSIONID = QUuid::createUuid().toString().replace("{", "").replace("}", "").replace("-", "");
    QByteArray JSESSIONID;
    JSESSIONID.append("Secure; JSESSIONID=");
    JSESSIONID.append(strJSESSIONID);

    netRequest.setRawHeader("Cookie", JSESSIONID);
    netRequest.setUrl(QUrl(AppEn_Url));

    //拼json内容
    QJsonObject obj;
    obj.insert("imageName", "6951739217000293.jpg");
    obj.insert("origin", "Batch-Test");
    obj.insert("productType", "1");
    QJsonDocument jsonDoc(obj);//QJsonObject转QJsonDocument
    
    //post内容    
    QString strPostdes =  jsonDoc.toJson();
    qDebug() << strPostdes;

    //发送
    QNetworkReply *reply = m_manager->post(netRequest, strPostdes.toUtf8());
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

