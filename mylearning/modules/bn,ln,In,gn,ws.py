import torch
import torch.nn as nn
from pathlib import Path
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
class WS2d(nn.Module):
    
if __name__ == '__main__':
    torch.manual_seed(28)
    path_dir = Path("./output/models")
    path_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')
    bn = BN2d(num_features=12)
    bn.to(device)  # 只针对子模块或者参数进行转换

    # 模拟训练过程
    bn.train()
    xs = [torch.randn(8, 12, 32, 32).to(device) for _ in range(10)]
    for _x in xs:
        bn(_x)

    print(bn.running_mean.view(-1))
    print(bn.running_var.view(-1))

    # 模拟推理过程
    bn.eval()
    _r = bn(xs[0])
    print(_r.shape)

    bn = bn.cpu()
    # 模拟模型保存
    torch.save(bn, str(path_dir / "bn_model.pkl"))
    # state_dict: 获取当前模块的所有参数(Parameter + register_buffer)
    torch.save(bn.state_dict(), str(path_dir / "bn_params.pkl"))
    # pt结构的保存
    traced_script_module = torch.jit.trace(bn.eval(), xs[0].cpu())
    traced_script_module.save('./output/models/bn_model.pt')