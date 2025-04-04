# PCB实例分割
本项目是一个PCB元件实例分割项目, 使用了pybind11,使得其可以在cpp环境下运行. 
模型使用onnx格式

## 环境配置
本项目的环境配置和金线分割一致
### 1. CUDA & Cudnn 测试可运行版本
| CUDA Version | Python Version | Cudnn Version | GPU      |
| ------------ | -------------- | ------------- | -------- |
| CUDA 11.8    | 3.8.10         | 8.x           | RTX 3080 |


### 2. Python 依赖安装
```bash
cd PCB_Seg
pip install onnx==1.17.0
pip install onnxruntime-gpu==1.18.1
pip install torch==2.0.0+cu118
pip install opencv-python
pip install -e .
```

### 3. 环境验证

##### 3.1 编译项目
```bash
cd PCB_Seg
rm -rf build  
mkdir build && cd build
cmake ..
make -j
```

##### 3.2 运行项目
```bash
./PCB_Seg
```
以上步骤运行了models/synpcbseg.onnx模型, 分割了图片input/demopcb.png, 分割的可视化结果保存在build/runs/predict/exp/visuals/1.jpg之中.

### 4. 其他资源
[接口文档](docs/api.md)

[大图切分说明文档](docs/slice.md)

[模型权重文件](models/)
models/synpcbseg.pt,models/synpcbseg.onnx

[示例用pcb图片](input/demopcb.png)
input/demopcb.png

[使用demo](source/main.cpp)
请参考source/main.cpp的main()

[纯python demo mytest/mainapi.py](mytest/mainapi.py)
在项目根目录运行pip install -e .
然后使用下面代码，即可调用python的api infer()



## License

This project is licensed under the Creative Commons Attribution-NonCommercial 4.0 International License - see the [LICENSE](LICENSE) file for details.

This means:
- ✅ You can freely use this code for non-commercial purposes
- ✅ You can modify and share this code
- ❌ You cannot use this code for commercial purposes
- ❌ You cannot sublicense or sell this code

