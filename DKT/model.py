import torch
import torch.nn as nn
from Utils.data_loader import load_problem_skill_mapping


class DKT(nn.Module):
    def __init__(self, args):
        super(DKT, self).__init__()
        self.dict, self.embedding = load_problem_skill_mapping(args)  # 问题对应技能的multi-hot编码
        self.embed_dim = len(self.embedding[0])
        self.fusion = Fusion_Module(self.embed_dim, args.device)

        self.device = args.device
        self.hidden_size = self.embed_dim
        self.num_layers = 1

        self.lstm = nn.LSTM(2 * self.embed_dim, self.hidden_size, num_layers=self.num_layers, dropout=0,
                            batch_first=True)
        self.hidden = nn.Linear(2 * self.embed_dim, self.embed_dim)
        self.predict = nn.Linear(self.embed_dim, 1)
        self.dropout = nn.Dropout(p=0.2)

    def forward(self, q, a, next_q, _=None, __=None, ___=None):
        # 获取skill multi-hot编码
        s = nn.functional.embedding(q, self.embedding)
        next_s = nn.functional.embedding(next_q, self.embedding)

        # 融合技能编码与答案
        x = self.fusion(s, a)
        # LSTM知识状态更新层
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(self.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(self.device)
        stu_state, _ = self.lstm(x, (h0, c0))
        # 预测层
        y = torch.concat((stu_state, next_s), dim=-1)
        y = self.hidden(y)
        y = torch.relu(y)
        y = self.predict(y)
        y = self.dropout(y)
        y = torch.sigmoid(y).squeeze(-1)
        return y


class Fusion_Module(nn.Module):
    def __init__(self, emb_dim, device):
        super(Fusion_Module, self).__init__()
        self.transform_matrix = torch.zeros(2, emb_dim * 2).to(device)
        self.transform_matrix[0][emb_dim:] = 1.0
        self.transform_matrix[1][:emb_dim] = 1.0

    def forward(self, ques_emb, pad_answer):
        ques_emb = torch.cat((ques_emb, ques_emb), -1)
        answer_emb = nn.functional.embedding(pad_answer, self.transform_matrix)
        input_emb = ques_emb * answer_emb
        return input_emb
