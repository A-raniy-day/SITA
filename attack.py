from DWT import *
import torch.nn as nn
from torchvision import transforms
import torch
import torch.optim as optim
from torch.autograd import Variable
from tqdm import *
from util import *
import kornia
import torch.nn.functional as F
import torch.nn as nn
from torch.autograd import Variable
device = "cuda" if torch.cuda.is_available() else "cpu"


def get_content(imgs):
    imgs = rgb2gary(imgs)
    imgs = torch.cat((imgs, imgs, imgs), dim = 1)
    imgs = kornia.filters.gaussian_blur2d(imgs, (5,5),(2,2))
    return imgs


def get_destyle_loss(imgs, adv, opt, model):
    input_ids = torch.tensor([[49406, 320, 49407]]).cuda()
    attention_mask = torch.tensor([[1, 1, 1]]).cuda()
    contents = get_content(imgs)
    if opt.size != 224:
        imgs = F.interpolate(imgs, size=(224, 224), mode='bilinear', align_corners=False)
        contents = get_content(imgs)
        adv = F.interpolate(adv, size=(224, 224), mode='bilinear', align_corners=False)


    z = model(input_ids, imgs, attention_mask).image_embeds
    z_c = model(input_ids, contents, attention_mask).image_embeds
    z_a = model(input_ids, adv, attention_mask).image_embeds

    term1 = z - z_c
    term2 = z_a - z_c

    return (1-F.cosine_similarity(term1, term2))


def get_perception_loss(inputs_homo, adv_homo, inputs_stru, adv_stru, inputs_stru_gray, adv_stru_gray):
    loss = nn.SmoothL1Loss(reduction='sum')
    loss_homo = loss(inputs_homo, adv_homo)
    loss_stru = loss(inputs_stru, adv_stru)
    loss_stru_gray = loss(inputs_stru_gray, adv_stru_gray)
    return loss_homo + loss_stru - loss_stru_gray


def attack(train_loader, opt, model):
    mean = torch.tensor([0.491, 0.482, 0.446]).cuda()
    std = torch.tensor([0.202, 0.199, 0.201]).cuda()
    normalizer, denomalizer = get_normalizer(mean, std)

    wave = 'haar'
    DWT = DWT_2D_tiny(wavename=wave)
    IDWT = IDWT_2D_tiny(wavename=wave)

    for idx, (images, lables) in enumerate(train_loader):
        
        if torch.cuda.is_available():
            images = images.cuda(non_blocking=True)
        
            
        eps = 3e-7
        modifier = torch.arctanh(images * (2 - eps * 2) - 1 + eps)
        modifier = Variable(modifier, requires_grad=True)
        modifier = modifier.to(device)

        optimizer = optim.Adam([modifier], lr=opt.learning_rate)

        inputs_homo = DWT(images)
        inputs_homo = IDWT(inputs_homo)
        inputs_stru_gray = rgb2gary(images)-rgb2gary(inputs_homo)
        inputs_stru = images - inputs_homo


        for step in tqdm(range(opt.steps)):
            adv = 0.5 * (torch.tanh(modifier) + 1)
            adv_homo = DWT(adv)
            adv_homo = IDWT(adv_homo)
            adv_stru_gray = rgb2gary(adv) - rgb2gary(adv_homo)
            adv_stru = adv - adv_homo

            perception_loss = get_perception_loss(inputs_homo, adv_homo, inputs_stru, adv_stru, inputs_stru_gray, adv_stru_gray)

            destyle_loss = get_destyle_loss(normalizer(images), normalizer(adv), opt, model)

            loss =   perception_loss - 100 * destyle_loss

            optimizer.zero_grad()
            loss.backward() 
            optimizer.step()
        
        adv = 0.5 * (torch.tanh(modifier.detach()) + 1)
        output_dir = opt.output
        save_tenor_image(adv, output_dir, lables)

