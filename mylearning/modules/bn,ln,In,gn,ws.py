from typing import Union
import torch
import torch.nn as nn
from pathlib import Path

from torch.nn.common_types import _size_2_t
class BN2d(nn.Module):
    def __init__(self,num_features,eps=1e-08,
                 momentum=0.9,affine=True,dtype=None) -> None:
        super(BN2d,self).__init__()
        self.num_features = num_features
        self.affine = affine
        self.dtype = dtype
        self.momentum = momentum
        self.eps=eps
        self.gamma = nn.Parameter(torch.ones((1,num_features,1,1)),requires_grad=affine)
        self.beta = nn.Parameter(torch.zeros((1,num_features,1,1)),requires_grad=affine)
        # register_buffer: 将属性当成parameter进行处理，唯一的区别就是不参与反向传播的梯度求解
        self.register_buffer("running_mean",torch.zeros(1,num_features,1,1))
        self.register_buffer("running_var",torch.zeros(1,num_features,1,1))

    def forward(self,x):
        if len(x.shape) !=4:
            raise "error:input shape error!"
            return
        if self.training:
            _mean=torch.mean(x,dim=(0,2,3),keepdim=True)
            _var = torch.var(x,dim=(0,2,3),keepdim=True)
            self.running_mean = (1-self.momentum)*self.running_mean+self.momentum*_mean
            self.running_var = (1-self.momentum)*self.running_var+self.momentum*_var
        else:   
            _mean = self.running_mean
            _var = self.running_var
        _x = (x-_mean)/torch.sqrt(_var+self.eps)*self.gamma+self.beta
        return _x
class LN2d(nn.Module):
    def __init__(self) -> None:
        super(LN2d,self).__init__()
        pass
    def forward(self,x):
        pass
class IN2d(nn.Module):
    def __init__(self) -> None:
        super(IN2d,self).__init__()
        pass
    def forward(self,x):
        pass
class GN2d(nn.Module):
    def __init__(self) -> None:
        super(GN2d,self).__init__()
        pass
    def forward(self,x):
        pass
#不是实现ws操作，而是实现参数标准化的卷积操作
class WSCond2d(nn.Conv2d):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: _size_2_t, 
                 stride: _size_2_t = 1, padding= 0, 
                 dilation: _size_2_t = 1, groups: int = 1, bias: bool = True, 
                 padding_mode: str = 'zeros', device=None, 
                 dtype=None,eps=1e-05) -> None:
        super().__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias, padding_mode, device, dtype)
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(out_channels,1,1,1))
        self.beta = nn.Parameter(torch.zeros(out_channels,1,1,1))
    def forward(self, input):
        w=self.weight#[OC,IC,kh,kw]
        # 参数标准化
        w_mean = torch.mean(w,dim=(1,2,3),keepdim=True)
        w_var=torch.var(w,dim=(1,2,3),keepdim=True)
        w = self.gamma*(w-w_mean)/torch.sqrt(w_var+self.eps)+self.beta
        #self.weight带梯度必须要使用data
        self.weight.data = w
        return super(WSCond2d, self).forward(input)

if __name__=="__main__":
    net=WSCond2d(8,64,(3,3),(1,1),padding=1)
    x=torch.rand((7,8,224,224))
    print(net(x))
# if __name__ == '__main__':
#     torch.manual_seed(28)
#     path_dir = Path("./output/models")
#     path_dir.mkdir(parents=True, exist_ok=True)
#     device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')
#     bn = BN2d(num_features=12)
#     bn.to(device)  # 只针对子模块或者参数进行转换

#     # 模拟训练过程
#     bn.train()
#     xs = [torch.randn(8, 12, 32, 32).to(device) for _ in range(10)]
#     for _x in xs:
#         bn(_x)

#     print(bn.running_mean.view(-1))
#     print(bn.running_var.view(-1))

#     # 模拟推理过程
#     bn.eval()
#     _r = bn(xs[0])
#     print(_r.shape)

#     bn = bn.cpu()
#     # 模拟模型保存
#     torch.save(bn, str(path_dir / "bn_model.pkl"))
#     # state_dict: 获取当前模块的所有参数(Parameter + register_buffer)
#     torch.save(bn.state_dict(), str(path_dir / "bn_params.pkl"))
#     # pt结构的保存
#     traced_script_module = torch.jit.trace(bn.eval(), xs[0].cpu())
#     traced_script_module.save('./output/models/bn_model.pt')