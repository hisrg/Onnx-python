import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import os
import torch
from torch import nn
import onnx
import onnxruntime

onnx_model=onnx.load("/mnt/d/WSL2_Ubuntu/workspace/resnet18_train.onnx")
try: 
    onnx.checker.check_model(onnx_model) 
except Exception: 
    print("Model incorrect") 
else: 
    print("Model correct")
print(type(onnx_model))

deptmap=np.load("/mnt/d/WSL2_Ubuntu/workspace/layer2_0_conv1_weight_resnet18.npy", allow_pickle=True)
print(type(deptmap))
print(len(deptmap))
ch=deptmap[0][0]
ls_max=np.array([0])
ls_min=np.array([0])
for ch in range(len(deptmap)):
    max=np.amax(deptmap[ch])
    min=np.amin(deptmap[ch])
    # print("max:",np.amax(deptmap[ch]))
    # print("mix:",np.amin(deptmap[ch]))
    ls_max=np.append(ls_max,max)
    ls_min=np.append(ls_min,min)
    # for i in range(len(deptmap[ch])):
    #     # print(deptmap[ch][i])
    #     max=deptmap[ch][i]
    #     min=deptmap[ch][i]
    #     max=np.amax(max)
    #     min=np.amin(min)
    #     # print (np.amin(a,0))
    #     print(max)
    #     print(min)
    #     # print("max"+str(ch)+str(i)+str(max))
    #     # print("max"+str(ch)+str(i)+str(min))
npa=np.array([ls_max,ls_min])
print(npa)
plt.scatter(ls_max, ls_min)
# plt.imshow(npa)
plt.savefig("ch.jpg")
