### 插入语句
> INSERT INTO `pv_p_ai_agent` (origin,image_type) VALUES ("EL-3-1","danjing_9bb_bp_daojiao_6*12")

### inner join
```sql
SELECT
	ph.image_name AS "图片名字",
	air.ai_result_before_data AS "AI过滤前的结果",
	air.ai_result_data AS "AI过滤后的结果",
	air.it_filter_data AS "MES过滤后的结果"
FROM
	`pv_p_ai_result` AS air
	INNER JOIN pv_p_photovoltaic AS ph ON ph.ai_result_id = air.id 
LIMIT 1000
```

### like
```sql
SELECT	pvpbatch.id,
	pvpp.image_name,
	pvpai.id AS ai_resultId,
	concat(":8001/pv/back/ai_result_view/?id=",pvpp.id) AS photovoltaicId,
	CONCAT(":8001/pv/back/downloadOriginImg/?originPath=",pvpai.origin_image) as downloadOriginUrl
/*http://10.123.33.2:8001/pv/back/ai_result_view/?id=448571*/
FROM
	pv_p_batchforecast AS pvpbatch
	INNER JOIN pv_p_photovoltaic AS pvpp ON pvpbatch.id = pvpp.batch_forecast_id
	INNER JOIN pv_p_ai_result AS pvpai ON pvpp.ai_result_id = pvpai.id 
WHERE
	pvpp.image_name LIKE "%6507639277002748-1%"
	OR pvpp.image_name LIKE "%6506139277007066%"
	OR pvpp.image_name LIKE "%6507639277003224%"
	OR pvpp.image_name LIKE "%7521039277019682%"
	OR pvpp.image_name LIKE "%6507739277000611%"
	OR pvpp.image_name LIKE "%6507739277000607%"
	OR pvpp.image_name LIKE "%6507739277000599%"
```

show databases; 显示数据库

### 配置远程访问
```sql
use mysql;
SELECT `Host`,`User` FROM user;
UPDATE user SET `Host` = '%' WHERE `User` = 'root' LIMIT 1;
alter user 'root'@'%' identified with mysql_native_password by 'Zt@2020jt';
ALTER USER 'root'@'%' IDENTIFIED BY 'Zt@2020jt';
flush privileges;
SELECT `Host`,`User` FROM user;
```

### 创建数据库
```sql
CREATE database `ztpanels-haining` DEFAULT CHARACTER SET utf8 COLLATE utf8_general_ci;
```
### 导出数据库
```sql
mysqldump -u 用户名 -p 数据库名 > 导出的文件名
```
### 导入数据库
```sql
source D:/www/sql/back.sql;
```

### mysql-server安装
> 我把数据库的安装包文件放在了/home/admin/software/mysql这个，里面有一个tar包，
> 使用命令tar -xvf mysql-8.0.21-1.el7.x86_64.rpm-bundle.tar进行解压缩，
> 得到几个rpm安装包，在解压缩目录里使用yum -y localinstall *.rpm进行安装

### 启动mysql：
```shell script
#使用 service 启动
service mysqld restart 
systemctl start  mysqld.service
# 启动失败打开日志查看
cat /var/log/mysqld.log
```

```shell script
mysql -uroot -p -h10.123.32.49 -P3306

alter table pv_p_ai_result add column server_ip varchar(50) null; # 给表pv_p_ai_result插入一列server_ip
```

## update 和select结合使用
```sql
UPDATE pv_p_photovoltaic as A INNER JOIN(
    SELECT	
			pvpp.id,pvpp.server_ip
		FROM
			pv_p_batchforecast AS pvpbatch
			INNER JOIN pv_p_photovoltaic AS pvpp ON pvpbatch.id = pvpp.batch_forecast_id
			INNER JOIN pv_p_ai_result AS pvpai ON pvpp.ai_result_id = pvpai.id 
		WHERE
		pvpbatch.id IN (509,510,511,512,513)
) c ON A.id = c.id SET A.server_ip = "10.123.32.49"; 
```