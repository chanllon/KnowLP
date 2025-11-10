import os

import numpy as np
import torch

from torch.nn import Module, Embedding, LSTM, Linear, Dropout
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

class hgnn(Module):
    # def __init__(self, num_l, num_p, ddp, dp, ddl, dl):
    def __init__(self, num_p , num_c , dp , ddp , dl , ddl ,diff_lever):
        super().__init__()
        self.num_p = num_p
        self.diff_lever = diff_lever
        self.dp = dp
        self.ddp = ddp
        self.dl = dl
        self.ddl = ddl
        self.num_c = num_c
        self.Edp_emb = Embedding(self.diff_lever, self.dp).to(device)
        self.Eddp_emb = Embedding(self.num_p+1, self.ddp).to(device)
        self.Edl_emb = Embedding(self.diff_lever, self.dl).to(device)
        self.Eddl_emb = Embedding(self.num_c+1, self.ddl).to(device)


    def forward(self, dp, p, dl, l):

        # print(f"q.shape is {q.shape}"
        dp = torch.tensor(dp)
        p = torch.tensor(p)
        dl = torch.tensor(dl)
        l = torch.tensor(l)


        dp = dp.unsqueeze(0)
        p = p.unsqueeze(0)
        dl = dl.unsqueeze(0)
        l = l.unsqueeze(0)

        dp = dp.to(device).long()
        p = p.to(device).long()
        dl = dl.to(device).long()
        l = l.to(device)

        Edp = self.Edp_emb(dp)
        Eddp = self.Eddp_emb(p)
        Edl = self.Edl_emb(dl)
        Eddl = self.Eddl_emb(l)
        xp = torch.cat((Edp, Eddp, Edl, Eddl), dim=2)




        return xp