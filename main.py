"""
Author: Haoran Chen
Date: 2022.08.15
"""
import argparse
import torch
from clip import clip
import os
from torch import nn
from model import Custom_Clip, PromptGenerator
from train_prompt import train_Prompt
from train_msf import train_MSF
from dataloader import load_pseudo_label_data, load_data
import numpy as np

torch.manual_seed(1)
np.random.seed(1)
torch.cuda.manual_seed(1)
torch.cuda.manual_seed_all(1)
torch.backends.cudnn.deterministic = True

def arg_parse():
    parser = argparse.ArgumentParser('Training and Evaluation Script', add_help=False)

    # for config
    parser.add_argument('--file_root', type=str, default=r'/vhome/chenhaoran/hrchen/MPA/',
                        help='model output path')
    parser.add_argument('--data_root', type=str, default=r'/share/test/hrchen/', help='data file path')
    parser.add_argument('--backbone', type=str, default='RN101', help='')
    parser.add_argument('--dataset', type=str, default='ImageCLEF', help='')
    parser.add_argument('--device', type=str, default='cuda', help='')


    # for dataloader
    parser.add_argument('--batch_size', type=int, default=32, help='')
    parser.add_argument('--num_workers', type=int, default=8, help='')
    parser.add_argument('--pin_memory', type=bool, default=True, help='')
    parser.add_argument('--threshold', type=float, default=0.4, help='threshold tau for generating pseudo labels')

    # for prompt settings
    parser.add_argument('--M1', type=int, default=12, help='number of classification tokens')
    parser.add_argument('--M2', type=int, default=12, help='number of domain tokens')

    # for encoder settings
    parser.add_argument('--mid_dim', type=int, default=384, help='dimension for the first feed forward layer')
    parser.add_argument('--out_dim', type=int, default=250, help='dimension for the intrinsic subspace')

    # for training settings
    parser.add_argument('--prompt_iteration', type=int, default=5000, help='')
    parser.add_argument('--msf_iteration', type=int, default=5000, help='')
    parser.add_argument('--prompt_learning_rate', type=float, default=0.01, help='')
    parser.add_argument('--prompt_momentum', type=float, default=0.9, help='')
    parser.add_argument('--prompt_weight_decay', type=float, default=0.0001, help='')
    parser.add_argument('--msf_learning_rate', type=float, default=0.001, help='')
    parser.add_argument('--msf_alpha', type=int, default=500, help='')
    parser.add_argument('--output_folder', type=str, default='', help='')
    parser.add_argument('--n_cls', type=int, default=0, help='number of classes in dataset')
    parser.add_argument('--AE_domain', type=bool, default=False, help='number of classes in dataset')


    return parser


def args_update(args):
    if args.dataset == 'ImageCLEF':
        args.backbone = 'RN50'
        args.out_dim = 150
        args.prompt_iteration = 400
        args.msf_iteration = 250

    if args.dataset == 'DomainNet':
        args.backbone = 'RN101'
        args.prompt_iteration = 4000
        args.msf_iteration = 2000
        args.out_dim = 300

    if args.dataset == 'OfficeHome':
        args.backbone = 'RN50'
        args.out_dim = 150
        args.prompt_iteration = 1000
        args.msf_iteration = 500


def train(domain_list, classnames, clip_model, preprocess, args):
    custom_clip_model = Custom_Clip(clip_model)
    custom_clip_model = nn.DataParallel(custom_clip_model)
    custom_clip_model = custom_clip_model.module

    for name, param in custom_clip_model.named_parameters():
        param.requires_grad_(False)

    for target_name in domain_list:
        source_name_list = domain_list.copy()
        source_name_list.remove(target_name)

        if not os.path.exists(os.path.join(args.output_folder, target_name)):
            os.makedirs(os.path.join(args.output_folder, target_name))

        target_path = os.path.join(args.data_root, args.dataset, target_name)
        target_train_loader = load_pseudo_label_data(target_name, target_path, preprocess, clip_model, args)
        target_test_loader = load_data(target_path, preprocess, args)

        prompt_name = []
        for source_name in domain_list:
            if source_name != target_name:
                name = source_name + '2' + target_name + '.pkl'
                prompt_name.append(name)

                if os.path.exists(args.output_folder + '/' + target_name + '/' + name):
                    continue

                source_path = os.path.join(args.data_root, args.dataset, source_name)
                source_train_loader = load_data(source_path, preprocess, args)
                
                print("Start training {} to {} prompt".format(source_name, target_name))
                train_Prompt(target_train_loader, target_test_loader, source_train_loader, classnames, clip_model, 
                                custom_clip_model, source_name, target_name, args)
                print("===========================================================================================")
        
        prompt_cls_list = []
        prompt_domain_list = []
        
        for i in range(len(prompt_name)):
            name = prompt_name[i]
            source_name = source_name_list[i]
          
            prompt_learner = PromptGenerator(classnames, clip_model, source_name, target_name, args)
            prompt_learner.load_state_dict(torch.load(args.output_folder + '/' + target_name + '/' + name))

            ctx_cls = prompt_learner.ctx_cls.float()
            ctx_source = prompt_learner.ctx_source.float()
            ctx_target = prompt_learner.ctx_target.float()
            prompt_cls_list.append(ctx_cls)
            prompt_domain_list.append(ctx_target)

        print("Start aligning {} prompts".format(target_name))
        train_MSF(target_name, target_train_loader, target_test_loader, prompt_cls_list, prompt_domain_list, custom_clip_model, clip_model, classnames, args)
        print("===========================================================================================")


def main(args):
    args_update(args)

    model_path = args.file_root + args.backbone + '.pt'
    model, preprocess = clip.load(args.backbone, device=args.device, model_path=model_path)

    domain_list = os.listdir(args.data_root + args.dataset)
    domain_list = [x for x in domain_list if '.txt' not in x]

    classnames_path = os.path.join(args.data_root, args.dataset, domain_list[0])
    classnames = os.listdir(classnames_path)
    n_cls = len(classnames)
    classnames.sort()

    args.output_folder = os.path.join(args.file_root, args.dataset, args.backbone, 'MPA_FINAL')

    if not os.path.exists(args.output_folder):
        os.makedirs(args.output_folder)

    args.n_cls = n_cls

    print(args)

    train(domain_list, classnames, model, preprocess, args)



if __name__ == '__main__':
    parser = argparse.ArgumentParser('Training and Evaluation Script', parents=[arg_parse()])
    args = parser.parse_args()

    main(args)
