#Ref:https://nenadmarkus.com/p/fusing-batchnorm-and-conv/


import torch
import torchvision
import time
import onnxruntime as rt

import conv_bn_fusion
torch.set_grad_enabled(False)
x = torch.randn(1, 3, 256, 256)
#1.Get original resnet model from pytorch with pretrained
rn18 = torchvision.models.resnet18(pretrained=True)
rn18.eval()
torch.save(rn18,'rn18_org_pth.pth')
print(rn18)
#2.Get the Submodel which contains Conv and BN layer 
net = torch.nn.Sequential(
	rn18.conv1,
	rn18.bn1
)
#3.Test Original Conv-BN layer time consuming in pytorch Runtime
start=time.time()
y1 = net.forward(x)
end=time.time()
time_consuming=start-end
print("Original Conv-BN layer time consuming in pytorch Runtime:", time_consuming)
#4.Merge Conv Layer and BN Layer
fusedconvbn = conv_bn_fusion.fuse_conv_and_bn(net[0], net[1])
# print(net[0])
# print(net[1])
#5.Test merged Conv-BN layer time in pytorch Runtime
start=time.time()
y2 = fusedconvbn.forward(x)
end=time.time()
time_consuming_merge=start-end
print("conv_bn_fused Layer time consuming in pytorch Runtime:", time_consuming_merge)
#6.Generate the fused conv-bn model as onnx formal 
with torch.no_grad(): 
    torch.onnx.export( 
        fusedconvbn, 
        x, 
        "fusedconvbn.onnx", 
        opset_version=11, 
        input_names=['input'], 
        output_names=['output'])
print("Saved conv_bn_fused mode using onnx format!")

with torch.no_grad(): 
    torch.onnx.export( 
        net, 
        x, 
        "original_submodel.onnx", 
        do_constant_folding=True,
        opset_version=11, 
        input_names=['input'], 
        output_names=['output'])
print("Saved original submodel mode using onnx format!")

#check model
import onnx 
print("Conv_BN Fused Model check using Onnx")
onnx_model = onnx.load("fusedconvbn.onnx") 
try: 
    onnx.checker.check_model(onnx_model) 
except Exception: 
    print("	Model incorrect") 
else: 
    print("	Model correct")

print("original_submodel check using Onnx")
onnx_model = onnx.load("original_submodel.onnx") 
try: 
    onnx.checker.check_model(onnx_model) 
except Exception: 
    print("	Model incorrect") 
else: 
    print("	Model correct")
# test onnx model

ort_session = rt.InferenceSession("fusedconvbn.onnx",providers=['CPUExecutionProvider']) #notice
ort_inputs = {'input': x.numpy()}
start=time.time()
ort_output = ort_session.run(['output'], ort_inputs)[0]
y3=ort_output
end=time.time()
time_consuming_merge_onnx=start-end

ort_session = rt.InferenceSession("original_submodel.onnx",providers=['CPUExecutionProvider']) #notice
ort_inputs = {'input': x.numpy()}
start=time.time()
ort_output = ort_session.run(['output'], ort_inputs)[0]
y4=ort_output
end=time.time()
time_consuming_org_onnx=start-end
print("-"*100)
print("conv_bn_fused Layer time consuming in onnx Runtime:", time_consuming_merge_onnx)
print("original Layer time consuming in onnx Runtime:", time_consuming_org_onnx)
d1 = (y1 - y2).norm().div(y1.norm()).item()
print("error between original and conv_bn_fused in pytorch runtime: %.8f" % d1)
d2=(y1-y3).norm().div(y1.norm()).item()
print("error between original and conv_bn_fused in Onnx runtime: %.8f" % d2)
d3=(y1-y4).norm().div(y1.norm()).item()
print("error between pytorch original and  Onnx original in Onnx runtime: %.8f" % d3)







