"""
Author: Haoran Chen
Date: 2022.06.10
"""
import torch
from torch import nn
import clip
import os
from torchvision import datasets
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR

N_epoch = 25
M1 = 16
M2 = 16


# creating learnable parameters for prompt
def ParamGenerator(model, classnames, M1, M2):
    n_cls = len(classnames)
    dtype = model.dtype
    embedding_dim = model.ln_final.weight.shape[0]

    ctx_cls_vectors = torch.empty(n_cls, M1, embedding_dim, requires_grad=True, dtype=dtype, device=device)
    ctx_source_vectors = torch.empty(1, M2, embedding_dim, requires_grad=True, dtype=dtype, device=device)
    ctx_target_vectors = torch.empty(1, M2, embedding_dim, requires_grad=True, dtype=dtype, device=device)

    nn.init.normal_(ctx_cls_vectors, std=0.02)
    nn.init.normal_(ctx_source_vectors, std=0.02)
    nn.init.normal_(ctx_target_vectors, std=0.02)

    return ctx_cls_vectors, ctx_source_vectors, ctx_target_vectors

# creating prompts
def PromptGenerator(classnames, M1, M2, ctx_cls_vectors, ctx_source_vectors, ctx_target_vectors):
    n_cls = len(classnames)
    dtype = model.dtype
    prompt_prefix = " ".join(["X"] * (M1 + M2))

    classnames = [name.replace("_", " ") for name in classnames]
    prompts = [prompt_prefix + " " + name + "." for name in classnames]

    tokenized_prompts = torch.cat([clip.tokenize(p) for p in prompts]).to(device)

    embedding = model.token_embedding(tokenized_prompts).type(dtype)
    prefix = embedding[:, :1, :]
    suffix = embedding[:, 1 + M1 + M2:, :]

    source_prompts = torch.cat(
        [prefix,  # (n_cls, 1, dim)
         ctx_cls_vectors,  # (n_cls, M1, dim)
         ctx_source_vectors.repeat(n_cls, 1, 1),  # (n_cls, M2, dim)
         suffix,  # (n_cls, *, dim)
         ],
        dim=1)
    target_prompts = torch.cat(
        [prefix,  # (n_cls, 1, dim)
         ctx_cls_vectors,  # (n_cls, M1, dim)
         ctx_target_vectors.repeat(n_cls, 1, 1),  # (n_cls, M2, dim)
         suffix,  # (n_cls, *, dim)
         ],
        dim=1)
    prompts = torch.cat([source_prompts, target_prompts], dim=0)
    return prompts, tokenized_prompts

#re-defining clip's textencoder
class TextEncoder(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = clip_model.dtype

    def forward(self, prompts, tokenized_prompts):
        x = prompts + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)

        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)] @ self.text_projection

        return x

#custom clip for training learnable prompts
class CustomCLIP(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.image_encoder = clip_model.visual
        self.text_encoder = TextEncoder(clip_model)
        self.logit_scale = clip_model.logit_scale
        self.dtype = clip_model.dtype

    def forward(self, image, prompts, tokenized_prompts):
        image_features = self.image_encoder(image.type(self.dtype))

        tokenized_prompts = torch.cat([tokenized_prompts, tokenized_prompts], dim=0)
        text_features = self.text_encoder(prompts, tokenized_prompts)

        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        logit_scale = self.logit_scale.exp()
        logits = logit_scale * image_features @ text_features.t()
        return logits.softmax(dim=-1)


def load_train(source_path, target_path, batch_size, preprocess):
    source_data = datasets.ImageFolder(root=source_path, transform=preprocess)
    target_data = datasets.ImageFolder(root=target_path, transform=preprocess)
    source_loader = torch.utils.data.DataLoader(source_data, batch_size=batch_size, shuffle=True, drop_last=True)
    target_loader = torch.utils.data.DataLoader(target_data, batch_size=batch_size, shuffle=True, drop_last=True)
    return source_loader, target_loader


def target_text(target_path):
    target_classes = os.listdir(target_path)
    for i in range(len(target_classes)):
        target_classes[i] = 'A photo of a ' + target_classes[i]
    return target_classes


def train(source_path, target_path, batch_size, classnames, clip_model, custom_clip, preprocess):
    ctx_cls_vectors, ctx_source_vectors, ctx_target_vectors = ParamGenerator(clip_model, classnames, M1, M2)
    optimizer = torch.optim.Adam([ctx_cls_vectors, ctx_source_vectors, ctx_target_vectors], lr=0.003)
    scheduler = CosineAnnealingLR(optimizer, T_max=N_epoch)

    source_loader, target_loader = load_train(source_path, target_path, batch_size, preprocess)

    text = target_text(target_path)
    for epoch in range(N_epoch):
        source_loss = 0
        target_loss = 0
        tot_acc = 0
        source_len = 0
        target_len = 0
        for i, (data, label) in enumerate(source_loader):
            data = data.to(device)
            label = label.to(device)

            prompts, tokenized_prompts = PromptGenerator(classnames, M1, M2, ctx_cls_vectors, ctx_source_vectors, ctx_target_vectors)

            optimizer.zero_grad()
            output = custom_clip(data, prompts, tokenized_prompts)
            loss = F.cross_entropy(output, label)
            loss.backward()
            optimizer.step()
            scheduler.step()

            source_loss += loss
            tot_acc += (output.argmax(1) == label).sum().item()
            source_len += 1

        print("source loss is " + str(source_loss/source_len))
        for i, (data, label) in enumerate(target_loader):
            data = data.to(device)
            label = label.to(device)
            label += 65 #target class labels

            target_token = clip.tokenize(text).to(device)
            logits_per_image, logits_per_text = model(data, target_token)
            pseudo_label = torch.argmax(logits_per_image.softmax(dim=-1),dim=1) #target class pseudo labels

            prompts, tokenized_prompts = PromptGenerator(classnames, M1, M2, ctx_cls_vectors, ctx_source_vectors, ctx_target_vectors)

            optimizer.zero_grad()
            output = custom_clip(data, prompts, tokenized_prompts)
            loss = F.cross_entropy(output, pseudo_label)
            loss.backward()
            optimizer.step()
            target_loss += loss
            tot_acc += (output.argmax(1) == label).sum().item()
            target_len += 1
        print("target loss is " + str(target_loss/target_len))
    return ctx_cls_vectors, ctx_source_vectors, ctx_target_vectors



if __name__ == '__main__':
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load("RN101", device=device)

    source_name = 'Art'
    target_name = 'Clipart'

    source_root_path = r'D:/学习/课题1/OfficeHome/'
    target_root_path = r'D:/学习/课题1/OfficeHome_target/'

    source_path = source_root_path + source_name
    target_path = target_root_path + target_name

    classnames = os.listdir(source_path)

    custom_clip = CustomCLIP(model)
    train(source_path, target_path, 16, classnames, model, custom_clip, preprocess)





