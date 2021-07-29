```shell script
git clone https://github.com/onnx/onnx-tensorrt.git
cd onnx-tensorrt
git submodule update --init --recursive
mkdir build && cd build
cmake .. -DCUDA_INCLUDE_DIRS=/usr/local/cuda-10.2/include -DTENSORRT_ROOT=/TensorRT-7.2.2.3 -DCMAKE_CUDA_COMPILER=/usr/local/cuda-10.2/bin/nvcc -DGPU_ARCHS="61"
make -j8
make install
cd .. && /opt/conda/envs/net3/bin/python setup.py build && /opt/conda/envs/net3/bin/python setup.py install
onnx2trt /user_data/model_data/checkpoint-best.onnx -o /user_data/model_data/checkpoint-best.trt
/TensorRT-7.2.2.3/bin/trtexec --onnx=/user_data/model_data/checkpoint-best.onnx --tacticSources=-cublasLt,+cublas --workspace=2048 --fp16 --saveEngine=/user_data/model_data/checkpoint-best.engine

```