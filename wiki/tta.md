### tta分类预测
- x:torch输入模型预测的变量
model:分类的模型
返回经过水平翻转，垂直翻转预测后的结果
https://github.com/qubvel/ttach
- tta.py
```py
def tta_Classification(x, model):
    transforms = Compose(
        [
            ttach.HorizontalFlip(),  # 水平翻转
            ttach.VerticalFlip(),  # 垂直翻转
        ]
    )
    tta_model = ttach.ClassificationTTAWrapper(model, transforms,merge_mode="mean")
    pre_batch = tta_model(x)
    return pre_batch
```