"""
Author: Haoran Chen
Date: 2022.08.15
"""
import torch
from torch import nn
import torch.nn.functional as F
import os
from model import Custom_Clip, AutoEncoder
from torch.optim.lr_scheduler import CosineAnnealingLR
from utils import Prompt, l1
import tqdm


def test_MSF(target_test_loader, custom_clip_model, prompt_list, tokenized_prompts, args):
    scale = custom_clip_model.logit_scale.exp()

    correct = 0
    tot = 0
    with torch.no_grad():
        for data, label in target_test_loader:
            tot += args.batch_size
            data = data.to(args.device)
            label = label.to(args.device)

            tot_logits = 0

            for prompt in prompt_list:
                img_feature, txt_feature = custom_clip_model(data, prompt, tokenized_prompts)
                logits = scale * img_feature @ txt_feature.t()
                logits = logits.softmax(dim=-1)
                tot_logits += logits

            tot_logits /= len(prompt_list)
            output = torch.argmax(tot_logits, dim=1)

            correct += (output == label).sum().item()

        print("accuracy is: {} with a total of {} data".format(correct / tot, tot))

    return correct / tot

def train_MSF(target_name, target_train_loader, target_test_loader, cls_list, domain_list, custom_clip_model, clip_model, classnames, args):
    output_root = os.path.join(args.output_folder, target_name)

    AE_cls = AutoEncoder(512, args.mid_dim, args.out_dim).to(args.device)
    AE_cls = nn.DataParallel(AE_cls)
    AE_cls = AE_cls.module

    AE_domain = AutoEncoder(512, args.mid_dim, args.out_dim).to(args.device)
    AE_domain = nn.DataParallel(AE_domain)
    AE_domain = AE_domain.module

    scale = custom_clip_model.logit_scale.exp()

    optimizer = torch.optim.AdamW(list(AE_cls.parameters()) + list(AE_domain.parameters()), lr=args.msf_learning_rate)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.msf_iteration)

    target_iter = iter(target_train_loader)
    source_num = len(cls_list)

    file_path = output_root + '/result.txt'
    
    for i in tqdm.tqdm(range(1, args.msf_iteration + 1)):
        try:
            target_data, target_label = next(target_iter)
        except Exception as err:
            target_iter = iter(target_train_loader)
            target_data, target_label = next(target_iter)

        target_data = target_data.to(args.device)
        target_label = target_label.to(args.device)

        AE_loss = 0
        CLS_loss = 0

        logits_list = []
        new_prompt_list = []
        
        for j in range(source_num):
            target_cls = cls_list[j]
            target_domain = domain_list[j]

            target_prompt = torch.cat([target_domain.repeat(args.n_cls, 1, 1), target_cls], dim=1)
            target_new_cls = AE_cls(target_cls)
            target_new_domain = AE_domain(target_domain)
            target_new_prompt = torch.cat([target_domain.repeat(args.n_cls, 1, 1), target_new_cls], dim=1)

            AE_loss += torch.pow(torch.linalg.norm(target_new_prompt - target_prompt) / 12, 2)
            target_new_prompt, tokenized_prompts = Prompt(classnames, clip_model, target_new_prompt, args)
            new_prompt_list.append(target_new_prompt)

            img_feature, txt_feature = custom_clip_model(target_data, target_new_prompt, tokenized_prompts)
            logits = scale * img_feature @ txt_feature.t()
            logits_list.append(logits)

            CLS_loss += F.cross_entropy(logits, target_label)

        AE_loss /= source_num
        while AE_loss > 5:
            AE_loss = AE_loss / 10

        CLS_loss /= source_num
        L1_loss = l1(logits_list)

        optimizer.zero_grad()
        loss = AE_loss + CLS_loss + L1_loss * args.msf_alpha
        loss.backward()

        optimizer.step()
        if i % (args.msf_iteration / 10) == 0:
            scheduler.step()
    
    acc = test_MSF(target_test_loader, custom_clip_model, new_prompt_list, tokenized_prompts, args)

    with open(file_path, 'a') as f:
        print("Accuracy is {}".format(acc), file=f)
