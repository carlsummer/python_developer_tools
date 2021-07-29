### 发布docker
1. [我编写的](https://gitee.com/zengxiaohui/tianchi-logic-object/tree/master/remote_sensing_code)
2. [直接用c++编译](https://github.com/carlsummer/tensorrtCV)
2. [我写的torch转c++的demo](https://github.com/carlsummer/libtorch-demo)

### onnx转tensorrt
```shell script
/TensorRT-7.2.2.3/bin/trtexec --onnx=/user_data/model_data/checkpoint-best.onnx --tacticSources=-cublasLt,+cublas --workspace=2048 --fp16 --saveEngine=/user_data/model_data/checkpoint-best.engine
```