# coding: utf-8
import torch
import torch.nn as nn

device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

class Agent_KT(nn.Module):
    def __init__(self, trained_model_file = "./trained_model.pt"):
        super(Agent_KT, self).__init__()
        model = torch.load(trained_model_file)
        self.trained_model = model.to(device)
        self.know = None

    def forward_state(self, ques, ans):
        ques = torch.tensor(ques)
        ans = torch.tensor(ans)
        ques = ques.unsqueeze(0)
        ans = ans.unsqueeze(0)
        ques = ques.to(device)
        ans = ans.to(device)
        model = self.trained_model
        know = model(ques,ans)
        self.know = know
        return know
