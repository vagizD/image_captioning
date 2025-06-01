import warnings
warnings.filterwarnings("ignore")

import torch
from torch.utils.data import Dataset, DataLoader
from torch import nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple
import torchvision.transforms as tr
from torchvision import models
from einops import rearrange

import cv2
import re
import os
import numpy as np
import pandas as pd
from os.path import join

channel_mean = np.array([0.485, 0.456, 0.406])  # efficient_net_b0
channel_std = np.array([0.229, 0.224, 0.225])

image_prepare = tr.Compose([
    tr.ToPILImage(),
    #   https://pytorch.org/vision/stable/transforms.html
    tr.RandomRotation(15),
    tr.RandomHorizontalFlip(0.2),
    tr.ColorJitter(brightness=0.5, contrast=0.1),
    tr.ToTensor(),
    tr.Resize(320, interpolation=tr.InterpolationMode.BICUBIC),
    tr.CenterCrop(300),
    tr.Normalize(mean=channel_mean, std=channel_std),
])

image_prepare_val = tr.Compose([
    tr.ToPILImage(),
    tr.ToTensor(),
    tr.Resize(320, interpolation=tr.InterpolationMode.BICUBIC),
    tr.CenterCrop(300),
    tr.Normalize(mean=channel_mean, std=channel_std),
])

def get_vocab(unzip_root: str) -> Tuple[Dict[str, int], Dict[int, str]]:
    vocab = pd.read_csv(join(unzip_root, 'vocab.tsv'), header=None, sep='\t').iloc[0, :].values
    tok_to_ind, ind_to_tok = {}, {}
    for i, v in enumerate(vocab):
        tok_to_ind[str(v)] = i
        ind_to_tok[i] = str(v)

    return tok_to_ind, ind_to_tok


N_CAPTIONS = 1

tok_to_ind, ind_to_tok = get_vocab("chkp")

def tokenize(t):
    t = t.lower()
    t = re.sub(r"[,.;@#?!&$]+\ *", " ", t, flags=re.VERBOSE)
    t = ' '.join(t.split()).split(' ')
    t = ["<BOS>"] + t + ["<EOS>"]
    return t

def to_ids(text):
    return [tok_to_ind.get(t, tok_to_ind['<UNK>']) for t in tokenize(text)]


vocab_size = len(tok_to_ind)
img_out_features = 1536  # en3

## INITS
init_kwargs = {"emb_dim": 300, "hidden_size": 1024, "img_out_features": img_out_features,
                "n_captions": N_CAPTIONS, "num_layers": 3, "inter_dim": None}
optimizer_params = {"lr": 1e-4}


class img_fe_class(nn.Module):
    def __init__(self):
        super(img_fe_class, self).__init__()
        model = models.efficientnet_b3()
        # weights = models.EfficientNet_B3_Weights.DEFAULT
        # model = models.efficientnet_b3(weights=weights, progress=True)

        model.classifier = nn.Identity()
        self.model = model

    def forward(self, imgs):
        feature_map = self.model(imgs)
        return feature_map


class text_fe_class(nn.Module):
    def __init__(self, emb_dim, hidden_size, img_out_features, n_captions, num_layers):
        super(text_fe_class, self).__init__()
        self.emb_dim = emb_dim
        self.hidden_size = hidden_size
        self.n_captions = n_captions
        self.num_layers = num_layers

        self.img_to_hidden_mapper = nn.Linear(img_out_features, hidden_size)

        self.embed = nn.Embedding(num_embeddings=vocab_size, embedding_dim=emb_dim, padding_idx=tok_to_ind['<PAD>'])
        # self.embed.weight = nn.Parameter(
        #     torch.from_numpy(glove_weights).to(dtype=self.embed.weight.dtype),
        #     requires_grad=True,
        # )

        # self.rnn = nn.RNN(batch_first=True, input_size=emb_dim, hidden_size=hidden_size,
        #                   num_layers=num_layers)
        self.lstm = nn.LSTM(batch_first=True, input_size=emb_dim, hidden_size=hidden_size,
                            num_layers=num_layers)

    def forward(self, texts, img_features):
        texts = self.embed(texts)
        h_0 = self.img_to_hidden_mapper(img_features)
        h_0 = h_0.repeat((self.num_layers, self.n_captions, 1))
        texts = rearrange(texts, "bs cap seq emb -> (bs cap) seq emb")
        # outputs, h_n = self.rnn(texts, h_0)
        h_0, c_0 = (torch.zeros(h_0.shape).to(h_0.device), h_0)
        outputs, (h_n, c_n) = self.lstm(texts, (h_0, c_0))
        outputs = rearrange(outputs, "(bs cap) seq emb -> bs cap seq emb", cap=self.n_captions)
        return outputs


class image_captioning_model(nn.Module):
    def __init__(self, emb_dim, hidden_size, img_out_features, n_captions, num_layers, inter_dim):
        super(image_captioning_model, self).__init__()
        self.img_fe  = img_fe_class()
        self.text_fe = text_fe_class(emb_dim=emb_dim, hidden_size=hidden_size, img_out_features=img_out_features,
                                     n_captions=n_captions, num_layers=num_layers)

        for param_name, param in list(self.img_fe.named_parameters())[:-29]:
            param.requires_grad = False
        for param in list(self.img_fe.model.children())[0][-2:]:
            param.requires_grad = True
        # self.to_logits = nn.Sequential(
        #     nn.Linear(hidden_size, inter_dim),
        #     nn.ReLU(inplace=True),
        #     nn.Linear(inter_dim, vocab_size)
        # )
        self.to_logits = nn.Linear(hidden_size, vocab_size)

    def forward(self, img_batch, texts_batch):
        img_feature_map = self.img_fe(img_batch)
        outputs = self.text_fe(texts_batch, img_feature_map)
        logits = self.to_logits(outputs)
        return logits


def get_model(unzip_root: str, model_name: str, device):
    # model_name = "en3FR_rnn2_aug1_ncap1#0_weights"
    # checkpoint = torch.load(os.path.join(unzip_root, f"{model_name}.pt"), weights_only=True)
    checkpoint = torch.load(os.path.join(unzip_root, f"{model_name}.pt"), map_location=device)

    model = image_captioning_model(**init_kwargs)

    model.load_state_dict(checkpoint['model_state_dict'])
    return model


global_max_seq_len = 199
