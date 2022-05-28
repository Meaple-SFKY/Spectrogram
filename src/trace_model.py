import torch
import torchvision
from models.BiSpecNet import BinSpecCNN
from models.SpecNet import SpecCNN
from torch.utils.mobile_optimizer import optimize_for_mobile

model = BinSpecCNN(10)
model = torch.load('./model/bin,94.87,180,10.pt', map_location='cpu')
model.eval()
example = torch.rand(1, 3, 201, 201)
traced_script_module = torch.jit.trace(model, example)
torchscript_model_optimized = optimize_for_mobile(traced_script_module)
torchscript_model_optimized._save_for_lite_interpreter("HelloWorld/HelloWorld/model/model_bin.pt")
