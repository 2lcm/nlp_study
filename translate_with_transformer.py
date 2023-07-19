import torch
import torch.nn as nn

import os
import tqdm
import wandb
import glob
import json
import time
import argparse
import math
from multiprocessing import Pool

from gensim.models.fasttext import FastText, load_facebook_model
from icu_tokenizer import Tokenizer

from torch.utils.data import Dataset, DataLoader
from torchtext.data.utils import get_tokenizer
from torch.optim import Adam
from torchtext.data.metrics import bleu_score

def print_tensor(x, desc=""):
    if desc:
        print(f"{desc}: {x.shape} {x.dtype} {x.min()} {x.max()}")
    else:
        print(f"{x.shape} {x.dtype} {x.min()} {x.max()}")


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len: int = 5000):
        super().__init__()
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(1, max_len, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return x

class Model(nn.Module):
    def __init__(self, y_dim, d_model, nhead, nlayers, d_hid, dropout):
        super().__init__()
        self.pos_encoder = PositionalEncoding(d_model)
        self.transformer = nn.Transformer(d_model, nhead, nlayers, nlayers, d_hid, dropout, batch_first=True)
        self.fc_out = nn.Linear(d_model, y_dim)

    def forward(self, src, tgt):
        src = self.pos_encoder(src)
        tgt = self.pos_encoder(tgt)
        tgt_mask = nn.Transformer.generate_square_subsequent_mask(tgt.size(1), device=tgt.device)
        out = self.transformer(src, tgt, tgt_mask=tgt_mask)
        out = self.fc_out(out)
        return out
    
class ENKO_TS_Dataset(Dataset):
    def __init__(self, tsv_fpath, en_tokenizer, ko_tokenizer, ft_en, ft_ko, maxlen):
        super().__init__()
        print("Making dataset")
        self.vector_dim = 256
        self.maxlen = maxlen

        with open(tsv_fpath, 'r') as f:
            lines = f.read().strip().split("\n")

        self.data = []
        

        # Do tokenize
        print("Tokenize...")
        for line in tqdm.tqdm(lines):
            en, ko = line.split("\t")
            en_tokens = en_tokenizer.tokenize(en)
            ko_tokens = ko_tokenizer.tokenize(ko)
            en_tokens.insert(0, "<sos>")
            en_tokens.append("<eos>")
            ko_tokens.insert(0, "<sos>")
            ko_tokens.append("<eos>")
            en_vector = []
            ko_vector = []
            for token in en_tokens:
                en_vec = ft_en.wv[token]
                en_vector.append(torch.Tensor(en_vec))
            en_vector = torch.stack(en_vector, dim = 0)
            for token in ko_tokens:
                ko_vec = ft_ko.wv[token]
                ko_vector.append(torch.Tensor(ko_vec))
            ko_vector = torch.stack(ko_vector, dim = 0)
            
            self.data.append([en_vector, ko_vector])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        en, ko = self.data[index]
        en_vector = torch.zeros(self.maxlen, self.vector_dim, dtype=torch.float32)
        en_vector[:en.size(0)] = en
        ko_vector = torch.zeros(self.maxlen, self.vector_dim, dtype=torch.float32)        
        ko_vector[:ko.size(0)] = ko
        return en_vector, ko_vector

def vector_to_sentences(vectors, ft_ko):
    sentences = []
    for vector in vectors:
        sentence = []
        for token in vector:
            word = ft_ko.wv.similar_by_vector(token, )
            sentence.append(word[0][0])
        sentences.append(sentence)
    
    return sentences

def helper_func(fname):
        try:
            with open(fname, 'r') as f:
                val = json.load(f)
            en = val["원문"]
            ko = val["최종번역문"]
            return en, ko
        except Exception as e:
            print(e, fname)
            return None

def preprocessing_data():
    pass
    # ================================================#
    # data preprocessing
    # ================================================#
    # fnames = glob.glob("/data/aihub/032.korean-english-translation-corpus/01.data/Training/02.label_data/enko/*")
    # ret = []
    # pool = Pool(8)
    # for en, ko in tqdm.tqdm(pool.imap_unordered(helper_func, fnames), total=len(fnames)):
    #     ret.append([en, ko])

    # with open("enko_ts_corpus_train.tsv", 'wt') as f:
    #     for val in ret:
    #         if val is not None:
    #             en, ko = val
    #             f.write(f"{en}\t{ko}\n")

    # fnames = glob.glob("/data/aihub/032.korean-english-translation-corpus/01.data/Validation/02.label_data/enko/*")
    # ret = []
    # pool = Pool(8)
    # for en, ko in tqdm.tqdm(pool.imap_unordered(helper_func, fnames), total=len(fnames)):
    #     ret.append([en, ko])

    # with open("enko_ts_corpus_validate.tsv", 'wt') as f:
    #     for val in ret:
    #         if val is not None:
    #             en, ko = val
    #             f.write(f"{en}\t{ko}\n")
    
    # ================================================#
    # en_tokenizer = Tokenizer('en')
    # ko_tokenizer = Tokenizer('ko')

    # with open("data/enko_ts_corpus_train.tsv", 'r') as f:
    #     lines = f.read().strip().split("\n")
    
    # en_data = []
    # ko_data = []
    # for line in tqdm.tqdm(lines):
    #     en_sentence, ko_sentence = line.split("\t")
    #     en_tokens = en_tokenizer.tokenize(en_sentence)
    #     ko_tokens = ko_tokenizer.tokenize(ko_sentence)
    #     en_tokens.insert(0, "<sos>")
    #     en_tokens.append("<eos>")
    #     ko_tokens.insert(0, "<sos>")
    #     ko_tokens.append("<eos>")
    #     en_data.append(en_tokens)
    #     ko_data.append(ko_tokens)
    
    # en_model = FastText.load('cc.en.256.bin')
    # ko_model = FastText.load('cc.ko.256.bin')

    # en_model.build_vocab(en_data, update=True)
    # ko_model.build_vocab(ko_data, update=True)

    # en_model.train(en_data, total_examples=len(en_data), epochs=5)
    # ko_model.train(ko_data, total_examples=len(ko_data), epochs=5)

    # en_model.save("en_model.bin")
    # ko_model.save("ko_model.bin")
    # ================================================#


def train(device, record_wandb):
    def sample_data(loader):
        while True:
            for batch in loader:
                yield batch

    if record_wandb:
        wandb.init(project="en_ko_ts")

    en_tokenizer = Tokenizer('en')
    ko_tokenizer = Tokenizer('ko')

    print("Loading fasttext en...")
    ft_en = FastText.load("en_model.bin")
    print("Loading fasttext ko...")
    ft_ko = FastText.load("ko_model.bin")
    print("Loading finished")

    model = Model(y_dim=256, d_model=256, nhead=8, nlayers=6, d_hid=1024, dropout=0.1)
    model = model.to(device)


    train_dataset = ENKO_TS_Dataset("data/enko_ts_corpus_train.tsv", en_tokenizer, ko_tokenizer, ft_en, ft_ko, maxlen=256)
    val_dataset = ENKO_TS_Dataset("data/enko_ts_corpus_validate.tsv", en_tokenizer, ko_tokenizer, ft_en, ft_ko, maxlen=256)

    train_loader = DataLoader(
        train_dataset, batch_size=32, shuffle=True, num_workers=0, drop_last=True
    )

    val_loader = DataLoader(
        val_dataset, batch_size=32, shuffle=True, num_workers=0, drop_last=True
    )

    train_loader = sample_data(train_loader)
    val_loader = sample_data(val_loader)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    pbar = tqdm.trange(600000, dynamic_ncols=True)
    for step in pbar:
        model.train()
        phrase_en, phrase_kor = next(train_loader)

        phrase_en = phrase_en.to(device)
        phrase_kor = phrase_kor.to(device)

        pred_kor = model(phrase_en, phrase_kor)
        loss = criterion(pred_kor, phrase_kor)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            pbar.set_description(
                f"loss:{loss.item():.6f} "
            )
            if record_wandb:
                wandb.log({
                    "loss" : loss.item(),
                })

            if step % 1000 == 0:
                ckpt_path = f"checkpoints/en_ko_ts/{step:07}.pt"
                torch.save({
                    "model" : model.state_dict(),
                    "optimizer" : optimizer.state_dict(),
                    "step" : step
                }, ckpt_path)
    
@ torch.no_grad()
def test(device, ckpt_path):
    model = Model(y_dim=256, d_model=256, nhead=8, nlayers=6, d_hid=1024, dropout=0.1)
    model = model.to(device)

    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model"])

    print("Loading fasttext en...")
    ft_en = FastText.load("en_model.bin")
    print("Loading fasttext ko...")
    ft_ko = FastText.load("ko_model.bin")
    print("Loading finished")

    en_tokenizer = Tokenizer('en')
    ko_tokenizer = Tokenizer('ko')

    val_dataset = ENKO_TS_Dataset("data/enko_ts_corpus_validate.tsv", en_tokenizer, ko_tokenizer, ft_en, ft_ko, maxlen=256)
    val_loader = DataLoader(
        val_dataset, batch_size=1, shuffle=True, num_workers=0, drop_last=True
    )
    
    model.eval()

    for phrase_en, phrase_kor in tqdm.tqdm(val_loader, total=len(val_loader)):
        phrase_en = phrase_en.to(device)
        phrase_kor = phrase_kor.to(device)

        pred_kor = model(phrase_en, phrase_kor)

        pred_sentences = vector_to_sentences(pred_kor, ft_ko)
        gt_sentences = vector_to_sentences(phrase_kor, ft_ko)
        print(pred_sentences)
        print(gt_sentences)
        break


if __name__ == "__main__":
    argparser = argparse.ArgumentParser()

    argparser.add_argument("--preprocessing", action="store_true")
    
    argparser.add_argument("--train", action="store_true")
    argparser.add_argument("--wandb", action="store_true")

    argparser.add_argument("--test", action="store_true")
    argparser.add_argument("--ckpt_path", default=None)
    
    args = argparser.parse_args()

    if args.preprocessing:
        preprocessing_data()
    elif args.train:
        train(device='cuda', record_wandb=args.wandb)
    elif args.test:
        test(device='cpu', ckpt_path=args.ckpt_path)
    else:
        raise NotImplementedError

    