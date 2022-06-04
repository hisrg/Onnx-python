import torch 
 
class Model(torch.nn.Module): 
    def __init__(self): 
        super().__init__() 
 
    def forward(self, x): 
        return torch.asinh(x) 
 
from torch.onnx.symbolic_registry import register_op 
 
def asinh_symbolic(g, input, *, out=None): 
    return g.op("Asinh", input) 
 
register_op('asinh', asinh_symbolic, '', 9) 
 
model = Model() 
input = torch.rand(1, 3, 10, 10) 
torch.onnx.export(model, input, 'asinh.onnx',opset_version=9)