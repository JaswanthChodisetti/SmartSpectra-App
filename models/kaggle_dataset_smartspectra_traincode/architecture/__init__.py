import torch
from .edsr import EDSR
from .HDNet import HDNet
from .hinet import HINet
from .hrnet import SGN
from .HSCNN_Plus import HSCNN_Plus
from .MIRNet import MIRNet
from .MPRNet import MPRNet
from .MST import MST
from .MST_Plus_Plus import MST_Plus_Plus
from .Restormer import Restormer
from .AWAN import AWAN

def model_generator(method, pretrained_model_path=None):
    # SmartSpectra patch: .cuda() was hardcoded on every branch — broke
    # any CPU run (including the patched train.py when run without GPU).
    # Guard each branch: build the module, then move to CUDA only if a
    # GPU is actually available. CPU training now "just works".
    def _build(m):
        return m.cuda() if torch.cuda.is_available() else m

    if method == 'mirnet':
        model = _build(MIRNet(n_RRG=3, n_MSRB=1, height=3, width=1))
    elif method == 'mst_plus_plus':
        model = _build(MST_Plus_Plus())
    elif method == 'mst':
        model = _build(MST(dim=31, stage=2, num_blocks=[4, 7, 5]))
    elif method == 'hinet':
        model = _build(HINet(depth=4))
    elif method == 'mprnet':
        model = _build(MPRNet(num_cab=4))
    elif method == 'restormer':
        model = _build(Restormer())
    elif method == 'edsr':
        model = _build(EDSR())
    elif method == 'hdnet':
        model = _build(HDNet())
    elif method == 'hrnet':
        model = _build(SGN())
    elif method == 'hscnn_plus':
        model = _build(HSCNN_Plus())
    elif method == 'awan':
        model = _build(AWAN())
    else:
        print(f'Method {method} is not defined !!!!')
    if pretrained_model_path is not None:
        print(f'load model from {pretrained_model_path}')
        checkpoint = torch.load(pretrained_model_path)
        model.load_state_dict({k.replace('module.', ''): v for k, v in checkpoint['state_dict'].items()},
                              strict=True)
    return model
