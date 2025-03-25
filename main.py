# encoding:utf-8
from __future__ import print_function
import argparse
from data import set_loader
from attack import *
from transformers import CLIPModel

device = "cuda" if torch.cuda.is_available() else "cpu"


def parse_option():
    parser = argparse.ArgumentParser('argument for attack')

    # data
    parser.add_argument('--batch_size', type=int, default=1,
                        help='batch_size')
    parser.add_argument('--num_workers', type=int, default=1,                       
                        help='num of workers to use')
    parser.add_argument('--data_folder', type=str, default='data', help='path to dataset')    
    parser.add_argument('--size', type=int, default=224, help='parameter for RandomResizedCrop')
    parser.add_argument('--output', type=str, default='output', help='path to output folder') 
    
    # optimization
    parser.add_argument('--learning_rate', type=float, default=0.005,
                        help='learning rate')
    parser.add_argument('--steps', type=int, default=50,
                        help='number of attacking steps')
    
    opt = parser.parse_args()
    return opt


def set_model():
    clip = CLIPModel.from_pretrained('openai/clip-vit-large-patch14').to(device)
    model = clip
    return model.eval()


def main():
    opt = parse_option()

    # build data loader
    train_loader = set_loader(opt)

    # build target encoder
    model = set_model()
    
    # attack
    attack(train_loader, opt, model)


if __name__ == '__main__':
    main()
