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


model = torchvision.models.resnet18(pretrained=True).to(device)
model = modelTuning(model, 3)
model.load_state_dict(torch.load("/opt/ml/model/age_group/epoch_2_loss_0.18800862965581472_acc_94.15189203492987.pt"))
transform_ = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((512,384)),
        transforms.Normalize((0.56, 0.524, 0.501), (0.258, 0.265, 0.267)),
    ])

check, original = gradCam(model, "/opt/ml/input/data/train/images/000002_female_Asian_52", "/opt/ml",transform_,"cuda", target_class=1)

@interact(idx = (0, check[0].shape[0]-1))
def showImg(idx):
    sample = check[0][idx].squeeze().cpu().detach()
    sample = colorize(sample)
    fig, axes = plt.subplots(1, 3, figsize=(18,6))

    axes[0].imshow(original[idx])
    axes[1].imshow(sample.permute(1,2,0))
    axes[2].imshow(original[idx])
    axes[2].imshow(sample.permute(1,2,0), alpha=0.5)

    plt.show()
