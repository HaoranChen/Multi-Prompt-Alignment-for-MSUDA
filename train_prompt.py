"""
Author: Haoran Chen
Date: 2022.08.15
"""
import torch
import os
import torch.nn.functional as F
from dataloader import load_data
from torch.optim.lr_scheduler import CosineAnnealingLR
from model import PromptGenerator
from torchvision import datasets
from utils import target_text
import tqdm


def test_Prompt(target_test_loader, prompts, tokenized_prompts, custom_clip_model, args):
    scale = custom_clip_model.logit_scale.exp()

    tot_acc = 0
    tot = 0
    with torch.no_grad():
        for data, label in target_test_loader:
            tot += args.batch_size
            data = data.to(args.device)
            label = label.to(args.device)
            label += args.n_cls

            img_features, txt_features = custom_clip_model(data, prompts, tokenized_prompts)
            logits = scale * img_features @ txt_features.t()

            output = torch.argmax(logits.softmax(dim=-1), dim=1)
            tot_acc += (output == label).sum().item()

    print("Accuracy is {}".format(tot_acc / tot))
    return tot_acc / tot


def train_Prompt(target_train_loader, target_test_loader, source_train_loader, classnames, clip_model, 
                                custom_clip_model, source_name, target_name, args):
    output_root = os.path.join(args.output_folder, target_name)
    if not os.path.exists(output_root):
        os.makedirs(output_root)

    source_iter = iter(source_train_loader)
    target_iter = iter(target_train_loader)

    scale = custom_clip_model.logit_scale.exp()
    prompt_learner = PromptGenerator(classnames, clip_model, source_name, target_name, args)
    tokenized_prompts = torch.cat([prompt_learner.tokenized_prompts, prompt_learner.tokenized_prompts], dim=0)

    optimizer = torch.optim.AdamW(list(prompt_learner.parameters()) + list(custom_clip_model.parameters()), lr=args.prompt_learning_rate)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.prompt_iteration)

    output_file_path = os.path.join(output_root, '{}2{}.txt'.format(source_name, target_name))
    output_model_path = os.path.join(output_root, '{}2{}.pkl'.format(source_name, target_name))

    for i in tqdm.tqdm(range(1, args.prompt_iteration + 1)):
        prompts = prompt_learner()
        try:
            source_data, source_label = next(source_iter)
        except Exception as err:
            source_iter = iter(source_train_loader)
            source_data, source_label = next(source_iter)

        try:
            target_data, target_label = next(target_iter)
        except Exception as err:
            target_iter = iter(target_train_loader)
            target_data, target_label = next(target_iter)

        source_data = source_data.to(args.device)
        source_label = source_label.to(args.device)
        target_data = target_data.to(args.device)
        target_label = target_label.to(args.device)
        target_label += args.n_cls

        optimizer.zero_grad()

        source_img_features, source_txt_features = custom_clip_model(source_data, prompts, tokenized_prompts)
        source_logits = scale * source_img_features @ source_txt_features.t()
        source_loss = F.cross_entropy(source_logits, source_label)

        target_img_features, target_txt_features = custom_clip_model(target_data, prompts, tokenized_prompts)
        target_logits = scale * target_img_features @ target_txt_features.t()
        target_loss = F.cross_entropy(target_logits, target_label)

        loss = source_loss + target_loss
        loss.backward()
        optimizer.step()

        if i % (args.prompt_iteration / 20) == 0:
            scheduler.step()

    acc = test_Prompt(target_test_loader, prompts, tokenized_prompts, custom_clip_model, args)
    with open(output_file_path, 'a') as f:
        print("Accuracy is {}".format(acc), file=f)
    torch.save(prompt_learner.state_dict(), output_model_path)