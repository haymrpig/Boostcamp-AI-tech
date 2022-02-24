import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
from tqdm import tqdm

class _BaseWrapper():
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.handlers = []

    def forward(self, images):
        self.image_shape = images.shape[2:]
        print(self.image_shape)
        self.logits = self.model(images)
        self.probs = F.softmax(self.logits, dim=1)
        return self.probs.sort(dim=1, descending=True)

    def backward(self, ids):
        one_hot = F.one_hot(ids, self.logits.shape[-1])
        one_hot = one_hot.squeeze()
        self.model.zero_grad()
        self.logits.backward(gradient=one_hot, retain_graph=True)
        # gradient는 해당 index에 대해서만 미분을 통한 backpropagation을 하겠다는 의미이다. 
        # 즉, 내가 확인하고 싶은 class에 대해서 featuremap이 얼마나 영향을 미쳤는지 확인할 수 있다. 

    def generate(self):
        raise NotImplementedError


class GradCAM(_BaseWrapper):
    def __init__(self, model, layers=None):
        super().__init__(model)
        self.feature_map = {}
        self.grad_map = {}
        self.layers = layers

        def save_fmaps(key):
            def forward_hook(module, input, output):
                self.feature_map[key]=output.detach()

            return forward_hook

        def save_grads(key):
            def backward_hook(modeul, grad_in, grad_out):
                self.grad_map[key] = grad_out[0].detach()

            return backward_hook

        for name, module in self.model.named_modules():
            if self.layers is None or name in self.layers:
                self.handlers.append(module.register_forward_hook(save_fmaps(name)))
                self.handlers.append(module.register_backward_hook(save_grads(name)))

    def findLayers(self, layers, target_layer):
        if target_layer in layers.keys():
            return layers[target_layer]
        else:
            raise ValueError(f"{target_layer} not exists")

    def generate(self, target_layer):
        feature_maps = self.findLayers(self.feature_map, target_layer)
        grad_maps = self.findLayers(self.grad_map, target_layer)
        weights = F.adaptive_avg_pool2d(grad_maps, 1)
        grad_cam = torch.mul(feature_maps, weights).sum(dim=1, keepdim=True)
        grad_cam = F.relu(grad_cam)
        grad_cam = F.interpolate(grad_cam, self.image_shape, mode="bilinear", align_corners=False)
        B, C, H, W = grad_cam.shape
        # C는 1인듯?

        grad_cam = grad_cam.view(B, -1)
        grad_cam -= grad_cam.min(dim=1, keepdim=True)[0]
        # 양수 만들어주려고 하는듯
        grad_cam /= grad_cam.max(dim=1, keepdim=True)[0]
        grad_cam = grad_cam.view(B, C, H, W)

        return grad_cam

