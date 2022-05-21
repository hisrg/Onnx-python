# 模型部署入门教程一
### 源代码:
请参考文件ONNX_RunTime_G1.py
# 模型部署入门教程二
## 提纲
### 一、本节目的：部署一个支持动态放大倍数的模型
### 二、模型部署遇到的困难：
                1.模型的动态化
                2.新算子的实现 
                3.中间表示与推理引擎的兼容问题
### 三、pytorch版本实现：
请参考文件Onnx_Runtime_G2.py

1.nn.interpolate 代替nn.Upsample
2.torch.tensor(3) 代替 3
### 四、Onnx 版本实现：
请参考文件ONNX_Runtime_G2_1.py

1.自定义一个算子，修改staticmethod 方法，将nn.interpolate映射到onnx.resize()算子中

### github.com/hisrg