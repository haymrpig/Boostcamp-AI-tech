import glob
import os
from PIL import Image
import torch
from gradcan import *

def readImages(img_path):
    return glob.glob(os.path.join(img_path, "*.*"))
    

def load_images(img_path, transform):
    img_list = readImages(img_path)

    images = []
    raw_images = []
    
    for fname in img_list:
        img = np.array(Image.open(fname))
        img_tf = transform(img)
        
        images.append(img_tf)
        raw_images.append(img)

    if len(images)>1:
        images = torch.stack(images)
    else:
        images = images[0].unsqueeze(0)

    return images, raw_images

def gradCam(model, img_path, dest_path, transform=None, device="cuda", target_class=0, target_layers :list =[]):
    model.to(device)
    model.eval()
    
    images, raw_images = load_images(img_path, transform)
    images = images.to(device)
    # images는 B,C,W,H로 된 tensor, raw_images는 리스트로 된 numpy 이미지 배열

    grad_cam = GradCAM(model)
    probs, ids = grad_cam.forward(images)
    target_ids = torch.LongTensor([[target_class]]*len(images)).to(device)
    grad_cam.backward(target_ids)

    check = []
    for target_layer in target_layers:
        print(f"generating Grad-CAM : {target_layer}")

        regions = grad_cam.generate(target_layer)
        check.append(regions)

        
    return check, raw_images
