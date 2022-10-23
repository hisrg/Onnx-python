#first you need to pip install pyHessian 
from re import L
import numpy as np
import torch 
import torch.onnx
import onnx
from torchvision import datasets, transforms
from utils import * # get the dataset
from density_plot import get_esd_plot # ESD plot
from pytorchcv.model_provider import get_model as ptcv_get_model # model

import matplotlib.pyplot as plt

# get the model 
model = ptcv_get_model("resnet20_cifar10", pretrained=True)
# change the model to eval mode to disable running stats upate
model.eval()

# create loss function
criterion = torch.nn.CrossEntropyLoss()

# get dataset 
train_loader, test_loader = getData()
# print(type(train_loader))

# for illustrate, we only use one batch to do the tutorial
calibrate_images=[]
num_max=100
i=0
for inputs, targets in train_loader:
    calibrate_images.append(inputs)
    i=i+1
    if i>=num_max:
        break
print("number of images to calibrate:", len(calibrate_images))
# we use cuda to make the computation fast
model = model.cpu()
inputs, targets = inputs.cpu(), targets.cpu()

# for name , module in model.named_modules():
#     print('--',name,'>>',module)
from torch.fx import symbolic_trace
# Symbolic tracing frontend - captures the semantics of the module
symbolic_traced : torch.fx.GraphModule = symbolic_trace(model)

# High-level intermediate representation (IR) - Graph representation
# layer_indx=1
# for n in symbolic_traced.graph.nodes:
#     print(f'{n.name}={n.op} target={n.target} args={n.args}')
#     layer_indx=layer_indx+1
# print("fx_layer:", layer_indx)
# 打印查看FX的IR
print(symbolic_traced.graph)

from torch.quantization import get_default_qat_qconfig, quantize_jit
from torch.quantization.quantize_fx import prepare_fx, convert_fx
from torch.quantization import default_dynamic_qconfig, float_qparams_weight_only_qconfig
qconfig=get_default_qat_qconfig("fbgem") #indicates the backend as sever
# model.qconfig=get_default_qat_qconfig("fbgem") #indicates the backend as sever
qconfig_dict={"":qconfig} #enable all quantization
model_prepared=prepare_fx(model, qconfig_dict)
calbration_loader=calibrate_images
#Calibration
def calibrate(model, data_loader):
    model.eval()
    with torch.no_grad():
        for image in data_loader:
            model(image)
calibrate(model_prepared,calbration_loader)
model_int8=convert_fx(model_prepared)

print(f'quantized int8 model name:{model_int8.named_modules}')
test(model_int8,test_loader,cuda=False)
