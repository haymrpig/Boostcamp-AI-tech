import torch
from ipywidgets import interact
import torchvision.transforms as transforms


def gauss(x,a,b,c):
    return torch.exp(-torch.pow(torch.add(x,-b),2).div(2*c*c)).mul(a)

def colorize(x):
    if x.dim() == 2:
        x = torch.unsqueeze(x, 0)
    if x.dim() == 3:
        cl = torch.zeros([3, x.size(1), x.size(2)])
        cl[0] = gauss(x,.5,.6,.2) + gauss(x,1,.8,.3)
        cl[1] = gauss(x,1,.5,.3)
        cl[2] = gauss(x,1,.2,.3)
        cl[cl.gt(1)] = 1
    elif x.dim() == 4:
        cl = torch.zeros([x.size(0), 3, x.size(2), x.size(3)])
        cl[:,0,:,:] = gauss(x,.5,.6,.2) + gauss(x,1,.8,.3)
        cl[:,1,:,:] = gauss(x,1,.5,.3)
        cl[:,2,:,:] = gauss(x,1,.2,.3)
    return cl

def showImg(idx):
    mask = check[0][idx].squeeze().cpu().detach()
    mask = colorize(mask)
    H, W = mask.shape[-2], mask.shape[-1]
    fig, axes = plt.subplots(1, 3, figsize=(18,6))
    original_img = torch.tensor(original[idx]).permute(2, 0, 1)
    
    original_img = transforms.Resize((H,W))(original_img).permute(1,2,0)

    axes[0].imshow(original_img)
    axes[1].imshow(mask.permute(1,2,0))
    axes[2].imshow(original_img)
    axes[2].imshow(mask.permute(1,2,0), alpha=0.5)
    
    plt.show()

def interactive_show(check):
    return interact(showimg, idx=(0, check[0].shape[0]-1)

