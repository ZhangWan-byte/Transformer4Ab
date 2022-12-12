import torch
import torch.nn as nn
import torch.optim as optim

from dataset import *
from utils import *


class InteractTransformer(nn.Module):
    def __init__(self, embed_size=64, para_seq_length=128, epi_seq_length=400, hidden=128):
        super(InteractTransformer, self).__init__()
        
        self.embedding = nn.Embedding(len(vocab), embed_size)

        self.Linear_para = nn.Sequential(nn.Linear(para_seq_length, hidden), nn.LeakyReLU(), nn.Dropout(0.1), \
                                         nn.Linear(hidden, hidden), nn.LeakyReLU())
        self.Linear_epi = nn.Sequential(nn.Linear(epi_seq_length, hidden), nn.LeakyReLU(), nn.Dropout(0.1), \
                                        nn.Linear(hidden, hidden), nn.LeakyReLU())

        self.transformer_para = nn.Transformer(d_model=embed_size, nhead=2, num_encoder_layers=6, num_decoder_layers=6, dim_feedforward=512, dropout=0.1)
        self.transformer_epi = nn.Transformer(d_model=embed_size, nhead=2, num_encoder_layers=6, num_decoder_layers=6, dim_feedforward=512, dropout=0.1)

        self.MLP_para = nn.Sequential(nn.Linear(embed_size, embed_size//2), nn.LeakyReLU(), nn.Dropout(0.1), \
                                      nn.Linear(embed_size//2, 1), nn.LeakyReLU())
        self.MLP_epi = nn.Sequential(nn.Linear(embed_size, embed_size//2), nn.LeakyReLU(), nn.Dropout(0.1), \
                                     nn.Linear(embed_size//2, 1), nn.LeakyReLU())
        
        self.output_layer = nn.Sequential(nn.Linear(hidden, hidden//2), nn.LeakyReLU(), nn.Dropout(0.1), \
                                          nn.Linear(hidden//2, 1), nn.Sigmoid())
    
    def forward(self, para, epi):
        
        # paratope
        para = self.embedding(para)
        # (batch, para_seq_length, embed_size)
        para = para.permute(0,2,1)
        para = self.Linear_para(para)
        para = para.permute(0,2,1)
        # (batch, hidden, embed_size)
        para = self.transformer_para(para, para)        
        # (batch, hidden, embed_size)
        para = self.MLP_para(para)
        # (batch, hidden, 1)
        para = para.squeeze(2)
        # (batch, hidden)
        
        # epitope
        epi = self.embedding(epi)
        # (batch, epi_seq_length, embed_size)
        epi = epi.permute(0,2,1)
        epi = self.Linear_epi(epi)
        epi = epi.permute(0,2,1)
        # (batch, hidden, embed_size)
        epi = self.transformer_epi(epi, epi)        
        # (batch, hidden, embed_size)
        epi = self.MLP_epi(epi)
        # (batch, hidden, 1)
        epi = epi.squeeze(2)
        # (batch, hidden)

        x = para * epi
        
        x = self.output_layer(x)
        
        return x


class CoAttention(nn.Module):
    def __init__(self, embed_size, output_size, dropout=None):
        super(CoAttention, self).__init__()

        self.dropout = dropout
        self.embed_size = embed_size
        self.linear_a = nn.Linear(embed_size, output_size)
        self.linear_b = nn.Linear(embed_size, output_size)
        self.W = nn.Linear(output_size, output_size)
        self._init_weight()
    
    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
    
    def forward(self, input_a, input_b):
        orig_a = input_a
        orig_b = input_b
        seq_len = orig_a.size()[1]

        input_a = input_a.view(-1, input_a.size()[1], input_a.size()[2])
        input_b = input_b.view(-1, input_b.size()[1], input_b.size()[2])
        
        a_len = input_a.size()[1]
        b_len = input_b.size()[1]
        input_dim = input_a.size()[2]
        max_len = a_len

        input_a = self.linear_a(input_a)
        input_a = nn.ReLU()(input_a)
        input_b = self.linear_b(input_b)
        input_b = nn.ReLU()(input_b)

        # print("input_a ", input_a.shape)

        dim = input_a.size()[2]

        _b = input_b.permute(0, 2, 1)
        zz = self.W(input_a)
        z = torch.matmul(zz, _b)

        # print("_b ", _b.shape)
        # print("zz ", zz.shape)
        # print("z ", z.shape)

        att_row = torch.mean(z, 1)
        att_col = torch.mean(z, 2)

        # print("att_row, att_col", att_row.shape, att_col.shape)

        a = orig_a * att_row.unsqueeze(2)
        b = orig_b * att_col.unsqueeze(2)

        # print(att_row.unsqueeze(2).shape)
        # print(a.shape, b.shape)

        return a, b


class InteractCoattnTransformer(nn.Module):
    def __init__(self, embed_size=64, seq_length=128):
        super(InteractCoattnTransformer, self).__init__()
        
        self.embedding = nn.Embedding(len(vocab), embed_size)
        self.transformer_para = nn.Transformer(d_model=embed_size, nhead=4, num_encoder_layers=6, num_decoder_layers=6, dim_feedforward=1024, dropout=0.1)
        self.transformer_epi = nn.Transformer(d_model=embed_size, nhead=4, num_encoder_layers=6, num_decoder_layers=6, dim_feedforward=1024, dropout=0.1)
                
        self.co_attn = CoAttention(embed_size=embed_size, output_size=embed_size)

        self.MLP_para = nn.Sequential(nn.Linear(embed_size, embed_size//2), nn.LeakyReLU(), nn.Dropout(0.1), \
                                 nn.Linear(embed_size//2, 1))
        self.MLP_epi = nn.Sequential(nn.Linear(embed_size, embed_size//2), nn.LeakyReLU(), nn.Dropout(0.1), \
                                 nn.Linear(embed_size//2, 1))
        
        self.output_layer = nn.Sequential(nn.Linear(seq_length, seq_length//2), nn.LeakyReLU(), nn.Dropout(0.1), \
                                          nn.Linear(seq_length//2, 1), nn.Sigmoid())
    
    def forward(self, para, epi):
        
        # paratope
        para = self.embedding(para)
        # (batch, seq_length, embed_size)

        # epitope
        epi = self.embedding(epi)
        # (batch, seq_length, embed_size)


        # co-attention
        # print(para.shape, epi.shape)
        para, epi = self.co_attn(para, epi)


        # paratope
        para = self.transformer_para(para, para)
        # (batch, seq_length, embed_size)
        para = self.MLP_para(para)
        # (batch, seq_length, 1)
        para = para.squeeze(2)
        # (batch, seq_length)


        # epitope
        epi = self.transformer_epi(epi, epi)        
        # (batch, seq_length, embed_size)
        epi = self.MLP_epi(epi)
        # (batch, seq_length, 1)
        epi = epi.squeeze(2)
        # (batch, seq_length)

        x = para * epi
        
        x = self.output_layer(x)
        
        return x


class InteractCoattn_noTransformer(nn.Module):
    def __init__(self, embed_size=64, seq_length=128):
        super(InteractCoattn_noTransformer, self).__init__()
        
        self.embedding = nn.Embedding(len(vocab), embed_size)
        self.transformer_para = nn.Transformer(d_model=embed_size, nhead=4, num_encoder_layers=6, num_decoder_layers=6, dim_feedforward=1024, dropout=0.1)
        self.transformer_epi = nn.Transformer(d_model=embed_size, nhead=4, num_encoder_layers=6, num_decoder_layers=6, dim_feedforward=1024, dropout=0.1)
                
        self.co_attn = CoAttention(embed_size=embed_size, output_size=embed_size)

        self.MLP_para = nn.Sequential(nn.Linear(embed_size, embed_size//2), nn.LeakyReLU(), nn.Dropout(0.1), \
                                 nn.Linear(embed_size//2, 1))
        self.MLP_epi = nn.Sequential(nn.Linear(embed_size, embed_size//2), nn.LeakyReLU(), nn.Dropout(0.1), \
                                 nn.Linear(embed_size//2, 1))
        
        self.output_layer = nn.Sequential(nn.Linear(seq_length, seq_length//2), nn.LeakyReLU(), nn.Dropout(0.1), \
                                          nn.Linear(seq_length//2, 1), nn.Sigmoid())
    
    def forward(self, para, epi):
        
        # paratope
        para = self.embedding(para)
        # (batch, seq_length, embed_size)

        # epitope
        epi = self.embedding(epi)
        # (batch, seq_length, embed_size)


        # co-attention
        # print(para.shape, epi.shape)
        para, epi = self.co_attn(para, epi)


        # paratope
        para = self.transformer_para(para, para)
        # (batch, seq_length, embed_size)
        para = self.MLP_para(para)
        # (batch, seq_length, 1)
        para = para.squeeze(2)
        # (batch, seq_length)


        # epitope
        epi = self.transformer_epi(epi, epi)        
        # (batch, seq_length, embed_size)
        epi = self.MLP_epi(epi)
        # (batch, seq_length, 1)
        epi = epi.squeeze(2)
        # (batch, seq_length)

        x = para * epi
        
        x = self.output_layer(x)
        
        return x


if __name__ == "__main__":
    k_iter = 0

    train_dataset = SeqDataset(data_path="../data/SARS-SAbDab_Shaun/CoV-AbDab_extract.csv", seq_length=128, \
                    kfold=10, holdout_fold=k_iter, is_train=True)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=16, shuffle=False)

    test_dataset = SeqDataset(data_path="../data/SARS-SAbDab_Shaun/CoV-AbDab_extract.csv", seq_length=128, \
                    kfold=10, holdout_fold=k_iter, is_train=False)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False)

    model = InteractCoattnTransformer(embed_size=64, seq_length=128)

    print(model)

    for i, (para, epi, label) in enumerate(train_loader):
        pred = model(para, epi)
        break

    print(para.shape, epi.shape, label.shape, pred.shape)