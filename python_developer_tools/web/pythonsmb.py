from smb.SMBConnection import *


class SMBClient(object):
    """
    smb连接客户端
    """
    prot = None
    status = False
    samba = None

    def __init__(self, user_name, passwd, ip, port=139):
        self.user_name = user_name
        self.passwd = passwd
        self.ip = ip
        self.port = port

    def connect(self):
        try:
            self.samba = SMBConnection(self.user_name, self.passwd, '', '', use_ntlm_v2=True)
            self.samba.connect(self.ip, self.port)
            self.status = self.samba.auth_result
        except:
            self.samba.close()

    def disconnect(self):
        if self.status:
            self.samba.close()

    def all_file_names_in_dir(self, service_name, dir_name):
        '''
        列出文件夹内所有文件名 smb.all_file_names_in_dir("chintAI", "/data/PVDefect/zengxiaohui/")
        :param service_name:服务器的文件夹路径 eg:chintAI
        :param dir_name: 相对于上面的路径 eg:/data/PVDefect
        :return:
        '''
        f_names = list()
        for e in self.samba.listPath(service_name, dir_name):
            # if len(e.filename) > 3: （会返回一些.的文件，需要过滤）
            if e.filename[0] != '.':
                f_names.append(e.filename)
        return f_names

    def upload(self, service_name, dir_name, file_name):
        '''
        上传文件
        localFile = open("readDataFromShare.py", "rb")
        smb.upload("chintAI", "/data/PVDefect/zengxiaohui/readDataFromShare.py", localFile)
        localFile.close()
        :param service_name:服务器的文件夹路径 eg:chintAI
        :param dir_name: 相对于上面的路径 eg:/data/PVDefect
        :param file_name: 要上传的文件对象
        :return:
        '''
        self.samba.storeFile(service_name, dir_name, file_name)

    def download(self, f_names, service_name, dir_name, local_dir):
        '''
        下载文件 smb.download(["2172438221500003[82].jpg"],"chintAI", "/data/PVDefect/zengxiaohui/","share")
        :param f_names:文件名
        :param service_name:服务器的文件夹路径 eg:chintAI
        :param dir_name: 相对于上面的路径 eg:/data/PVDefect
        :param local_dir: 本地文件夹
        :return:
        '''
        assert isinstance(f_names, list)
        for f_name in f_names:
            f = open(os.path.join(local_dir, f_name), 'wb')
            self.samba.retrieveFile(service_name, os.path.join(dir_name, f_name), f)
            f.close()


if __name__ == '__main__':
    smb = SMBClient('chintAI', 'chintAI', '10.20.200.170')
    smb.connect()
