import time
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, matthews_corrcoef
from imblearn.metrics import geometric_mean_score
from metrics import *
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim

from dataset import *
from utils import *
from models import *


def prepare_lstm(config):

    if config["use_fine_tune"]==True:
        config["model_name"] += "_ft"

        if config["use_pair"]==True:
            config["model_name"] += "_pairPreTrain"

    if config["model_name"]=="lstm":
        config["model"] = BiLSTM(embed_size=32, 
                                 hidden=64, 
                                 num_layers=1, 
                                 dropout=0.5, 
                                 use_pretrain=False).cuda()

        config["epochs"] = 300
        config["lr"] = 1e-4
        config["l2_coef"] = 5e-4

    elif config["model_name"]=="lstm_ft":
        config["model"] = torch.load("./results/SAbDab/full/seq1_neg0/lstm/model_best.pth")
        config["model"].train()
        
        if config["fix_FE"]==True:
            for name, param in config["model"].LSTM_para.named_parameters():
                param.requires_grad = False
            for name, param in config["model"].LSTM_epi.named_parameters():
                param.requires_grad = False
        
        config["epochs"] = 500
        config["lr"] = 1e-4
        config["l2_coef"] = 5e-4
        
    elif config["model_name"]=="lstm_ft_pairPreTrain":

        encoder = torch.load("./results/SAbDab/full/seq1_neg0/lstm_encoder/model_best.pth")
        encoder.train()
        config["model"] = TowerBaseModel(embed_size=64, hidden=128, encoder=encoder, 
                                         use_two_towers=False, use_coattn=False, fusion=1).cuda()
        
        if config["fix_FE"]==True:
            for name, param in config["model"].encoder.named_parameters():
                param.requires_grad = False
        
        config["epochs"] = 500
        config["lr"] = 1e-4
        config["l2_coef"] = 5e-4

    else:
        print("Error Model Name")
        exit()

    return config

def prepare_textcnn(config):
    pass

def prepare_masonscnn(config):

    if config["use_fine_tune"]==True:
        config["model_name"] += "_ft"

        if config["use_pair"]==True:
            config["model_name"] += "_pairPreTrain"

    if config["model_name"]=="masonscnn":
        config["model"] = MasonsCNN(amino_ft_dim=len(vocab), 
                                    max_antibody_len=100, 
                                    max_virus_len=100, 
                                    h_dim=512, 
                                    dropout=0.1).cuda()
        config["epochs"] = 100
        config["lr"] = 1e-4
        config["l2_coef"] = 5e-4
        
    elif config["model_name"]=="masonscnn_ft":
        config["model"] = torch.load("./results/SAbDab/full/seq1_neg0/masonscnn/model_best.pth")

        if config["fix_FE"]==True:
            for name, param in model.cnnmodule.named_parameters():
                param.requires_grad = False
            for name, param in model.cnnmodule2.named_parameters():
                param.requires_grad = False


        config["epochs"] = 500
        config["lr"] = 1e-4
        config["l2_coef"] = 5e-4
        
    elif config["model_name"]=="masonscnn_ft_pairPreTrain":
        
        encoder = torch.load("./results/SAbDab/full/seq1_neg0/masonscnn_encoder/model_best.pth")
        config["model"] = TowerBaseModel(embed_size=32, hidden=128, encoder=encoder, 
                                         use_two_towers=False, use_coattn=False, fusion=0).cuda()
        
        if config["fix_FE"]==True:
            for name, param in model.encoder.named_parameters():
                param.requires_grad = False
        
        config["epochs"] = 500
        config["lr"] = 1e-4
        config["l2_coef"] = 5e-4

    else:
        print("Error Model Name")
        exit()

    return config

def prepare_ag_fast_parapred(config):
    pass

def prepare_pipr(config):
    pass

def prepare_resppi(config):
    pass

def prepare_deepaai(config):
    pass

def prepare_pesi(config):
    pass

def cov_train(config):

    # if config["use_fine_tune"]==True:
    #     config["model_name"] += "_ft"

    #     if config["use_pair"]==True:
    #         config["model_name"] += "_pairPreTrain"

    print("make folder ./results/CoV-AbDab/{}/".format(config["model_name"]))
    os.makedirs("./results/CoV-AbDab/{}/".format(config["model_name"]), exist_ok=True)

    print("model name: {}".format(config["model_name"]))

    kfold_labels = []
    kfold_preds = []

    for k_iter in range(config["kfold"]):
        
        print("=========================================================")
        print("fold {} as val set".format(k_iter))

        # model name
        if model_name=="lstm":
            config = prepare_lstm(config)
        elif model_name=="textcnn":
            config = prepare_textcnn(config)
        elif model_name=="masonscnn":
            config = prepare_masonscnn(config)
        elif model_name=="ag_fast_parapred":
            config = prepare_ag_fast_parapred(config)
        elif model_name=="pipr":
            config = prepare_pipr(config)
        elif model_name=="resppi":
            config = prepare_resppi(config)
        elif model_name=="deepaai":
            config = prepare_deepaai(config)
        elif model_name=="pesi":
            config = prepare_pesi(config)
        
        train_dataset = SeqDataset(data_path=config["data_path"], \
                                kfold=config["kfold"], holdout_fold=k_iter, is_train_test_full="train", \
                                use_pair=config["use_pair"], balance_samples=False)
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=False, \
                                                collate_fn=collate_fn)

        test_dataset = SeqDataset(data_path=config["data_path"], \
                                kfold=config["kfold"], holdout_fold=k_iter, is_train_test_full="test", \
                                use_pair=config["use_pair"], balance_samples=False)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False, \
                                                collate_fn=collate_fn)

    #     if model_name=="demo":
    #         model = BiLSTM_demo(embed_size=32, hidden=64, num_layers=1, dropout=0.5, use_pretrain=False).cuda()
            
    #         epochs = 100
    #         lr = 6e-5
        
        
    #     elif model_name=="InteractTransformer":
    #         model = InteractTransformer(embed_size=32, 
    #                                     num_encoder_layers=1, 
    #                                     nhead=2, 
    #                                     dropout=0.3, 
    #                                     use_coattn=False).cuda()
    #         epochs = 200
    #         lr = 3e-5
            
    #     elif model_name=="InteractTransformer_ft":
    #         model = torch.load("./results/SAbDab/full/seq1_neg0/InteractTransformer/model_best.pth")
    #         model.train()
            
    #         if config["fix_FE"]==True:
    #             for name, param in model.transformer_para.named_parameters():
    #                 param.requires_grad = False
    #             for name, param in model.transformer_epi.named_parameters():
    #                 param.requires_grad = False

    #         epochs = 1000
    #         lr = 1e-4
    #         l2_coef = 5e-4
                    
    #     elif model_name=="InteractCoAttnTransformer":
    #         model = InteractTransformer(embed_size=32, 
    #                                     num_encoder_layers=1, 
    #                                     nhead=2, 
    #                                     dropout=0.5, 
    #                                     use_coattn=True).cuda()
    #         epochs = 200
    #         lr = 3e-5
            
    #     elif model_name=="InteractCoAttnTransformer_ft":
    #         model = torch.load("./results/SAbDab/full/seq1_neg0/InteractCoAttnTransformer/model_best.pth")
    #         model.train()
        
    #         if config["fix_FE"]==True:
    #             for name, param in model.transformer_para.named_parameters():
    #                 param.requires_grad = False
    #             for name, param in model.transformer_epi.named_parameters():
    #                 param.requires_grad = False
            
    #         epochs = 1500
    #         lr = 6e-5
    #         l2_coef = 5e-4
            
    #     elif model_name=="InteractCoAttnTransformer_ft_pairPreTrain":
    #         encoder = torch.load("./results/SAbDab/full/seq1_neg0/InteractTransformer_encoder/model_best.pth")
    #         encoder.train()
            
    #         model = TowerBaseModel(embed_size=32, hidden=128, encoder=encoder, 
    #                             use_two_towers=False, use_coattn=True, fusion=1).cuda()
        
    #         if config["fix_FE"]==True:
    #             for name, param in model.encoder.named_parameters():
    #                 param.requires_grad = False
            
    #         epochs = 1500
    #         lr = 6e-5
    #         l2_coef = 5e-4

    #     elif model_name=="InteractTransformerLSTM":
    #         model = InteractTransformerLSTM(embed_size=32, 
    #                                         hidden=64, 
    #                                         num_encoder_layers=1, 
    #                                         num_lstm_layers=1, 
    #                                         nhead=2, 
    #                                         dropout=0.5, 
    #                                         use_coattn=True).cuda()
    #         epochs = 200
    #         lr = 6e-5

    #     elif model_name=="InteractTransformerLSTM_ft":
    #         model = torch.load("./results/SAbDab/full/seq1_neg0/InteractTransformerLSTM/model_best.pth")
    #         model.train()
            
    #         epochs = 200
    #         lr = 6e-5
                    
    #     elif model_name=="SetTransformer":
    #         model = SetTransformer(dim_input=32, 
    #                             num_outputs=32, 
    #                             dim_output=32, 
    #                             dim_hidden=64, 
    #                             num_inds=6, 
    #                             num_heads=4, 
    #                             ln=True, 
    #                             dropout=0.5, 
    #                             use_coattn=False, 
    #                             share=False).cuda()
            
    #         epochs = 500
    #         lr = 1e-4
    #         l2_coef = 5e-4
            
    #     elif model_name=="SetTransformer_ft":

    #         model = torch.load("./results/SAbDab/full/seq1_neg0/SetTransformer/model_best.pth")
    #         model.train()

    #         if config["fix_FE"]==True:
    #             for name, param in model.para_enc.named_parameters():
    #                 param.requires_grad = False
    #             for name, param in model.para_dec.named_parameters():
    #                 param.requires_grad = False
    #             for name, param in model.epi_enc.named_parameters():
    #                 param.requires_grad = False
    #             for name, param in model.epi_dec.named_parameters():
    #                 param.requires_grad = False

    #         epochs = 500
    #         lr = 1e-4
    #         l2_coef = 5e-4
            
            
    #     elif model_name=="SetCoAttnTransformer":
    #         model = SetTransformer(dim_input=32, 
    #                             num_outputs=32, 
    #                             dim_output=32, 
    #                             dim_hidden=64, 
    #                             num_inds=6, 
    #                             num_heads=4, 
    #                             ln=True, 
    #                             dropout=0.5, 
    #                             use_coattn=True).cuda()
    #         epochs = 500
    #         lr = 6e-5
    #         l2_coef = 5e-4
            
    #     elif model_name=="SetCoAttnTransformer_ft":
    #         if config["use_BSS"]==False:
    # #             model = torch.load("./results/SAbDab/full/seq1_neg0/SetCoAttnTransformer/model_best.pth")
    # #             model.train()

    # #             if fix_FE==True:
    # #                 for name, param in model.para_enc.named_parameters():
    # #                     param.requires_grad = False
    # #                 for name, param in model.para_dec.named_parameters():
    # #                     param.requires_grad = False
    # #                 for name, param in model.epi_enc.named_parameters():
    # #                     param.requires_grad = False
    # #                 for name, param in model.epi_dec.named_parameters():
    # #                     param.requires_grad = False
                
    # #             epochs = 500
    # #             lr = 3e-5
    # #             l2_coef = 6e-4
    #             model = SetTransformer(dim_input=32, 
    #                                 num_outputs=32, 
    #                                 dim_output=32, 
    #                                 dim_hidden=64, 
    #                                 num_inds=6, 
    #                                 num_heads=4, 
    #                                 ln=True, 
    #                                 dropout=0.5, 
    #                                 use_coattn=False, 
    #                                 share=False, 
    #                                 use_BSS=False).cuda()
            
    #             pt_model = torch.load("./results/SAbDab/full/seq1_neg0/SetCoAttnTransformer/model_best.pth")
            
    #             model.para_enc = pt_model.para_enc
    #             model.para_dec = pt_model.para_dec
    #             model.epi_enc = pt_model.epi_enc
    #             model.epi_dec = pt_model.epi_dec
    #             model.train()
            
    #             epochs = 500
    #             lr = 6e-5
    #             l2_coef = 5e-4

    #         elif config["use_BSS"]==True:
    #             print(model_name, config["use_BSS"])
    #             model = SetTransformer(dim_input=32, 
    #                                 num_outputs=32, 
    #                                 dim_output=32, 
    #                                 dim_hidden=64, 
    #                                 num_inds=6, 
    #                                 num_heads=4, 
    #                                 ln=True, 
    #                                 dropout=0.5, 
    #                                 use_coattn=False, 
    #                                 share=False, 
    #                                 use_BSS=True).cuda()
            
    #             # load pre-trained weights
    #             pt_model = torch.load("./results/SAbDab/full/seq1_neg0/SetCoAttnTransformer/model_best.pth")
            
    #             model.embedding = pt_model.embedding
                
    #             model.para_enc = pt_model.para_enc
    #             model.epi_enc = pt_model.epi_enc
                
    #             model.co_attn = pt_model.co_attn
                
    #             model.para_dec = pt_model.para_dec
    #             model.epi_dec = pt_model.epi_dec
                
    #             model.output_layer = pt_model.output_layer
                
    #             model.train()
            
    #             # params
    #             epochs = 500
    #             lr = 6e-5
    #             l2_coef = 5e-4
    #         else:
    #             print("wrong use_BSS!")
    #             quit()
            
            
    #     elif model_name=="SetCoAttnTransformer_ft_pairPreTrain":
            
    #         encoder = torch.load("./results/SAbDab/full/seq1_neg0/SetTransformer_encoder/model_best.pth")
    #         encoder.train()
    #         model = TowerBaseModel(embed_size=32, hidden=128, encoder=encoder, use_two_towers=False, mid_coattn=True, use_coattn=True, fusion=1).cuda()
            
    #         if config["fix_FE"]==True:
    #             for name, param in model.encoder.named_parameters():
    #                 param.requires_grad = False
            
    #         epochs = 1500
    #         lr = 1e-4 #6e-5
    #         l2_coef = 3e-4 #5e-4
            
            
    #     elif model_name=="SetModel":
    #         model = SetModel(embed_size=32, 
    #                         hidden=64, 
    #                         num_layers=1, 
    #                         dropout=0.3, 
    #                         k4kmer=3, 
    #                         use_pretrain=False, 
    #                         use_coattn=False, 
    #                         seq_encoder_type="transformer", 
    #                         num_heads=2, 
    #                         num_inds=6, 
    #                         num_outputs=6, 
    #                         ln=True).cuda()
            
    #         epochs = 200
    #         lr = 3e-5
    #         l2_coef = 5e-4
            
    #     elif model_name=="SetModel_ft":
    #         model = torch.load("./results/SAbDab/full/seq1_neg0/SetModel/model_best.pth")
    #         model.train()
            
    #         if config["fix_FE"]==True:
    #             for name, param in model.para_enc.named_parameters():
    #                 param.requires_grad = False
    #             for name, param in model.para_dec.named_parameters():
    #                 param.requires_grad = False
    #             for name, param in model.epi_enc.named_parameters():
    #                 param.requires_grad = False
    #             for name, param in model.epi_dec.named_parameters():
    #                 param.requires_grad = False
            
    #         epochs = 200
    #         lr = 3e-5
    #         l2_coef = 5e-4
        
    #     elif model_name=="SetCoAttnModel":
    #         model = SetModel(embed_size=32, 
    #                         hidden=64, 
    #                         num_layers=1, 
    #                         dropout=0.3, 
    #                         k4kmer=3, 
    #                         use_pretrain=False, 
    #                         use_coattn=True, 
    #                         seq_encoder_type="transformer", 
    #                         num_heads=2, 
    #                         num_inds=6, 
    #                         num_outputs=6, 
    #                         ln=True).cuda()
    #         epochs = 200
    #         lr = 3e-5
        
    #     elif model_name=="SetCoAttnModel_ft":
    #         model = torch.load("./results/SAbDab/full/seq1_neg0/SetCoAttnModel/model_best.pth")
    #         model.train()
            
    #         if config["fix_FE"]==True:
    #             for name, param in model.para_enc.named_parameters():
    #                 param.requires_grad = False
    #             for name, param in model.para_dec.named_parameters():
    #                 param.requires_grad = False
    #             for name, param in model.epi_enc.named_parameters():
    #                 param.requires_grad = False
    #             for name, param in model.epi_dec.named_parameters():
    #                 param.requires_grad = False
            
    #         epochs = 200
    #         lr = 3e-5
    #         l2_coef = 5e-4
            
    #     elif model_name=="SetModel_ablation":
    #         model = SetModel_ablation(embed_size=32, 
    #                         hidden=64, 
    #                         num_layers=2, 
    #                         dropout=0.5, 
    #                         k4kmer=7, 
    #                         use_pretrain=False, 
    #                         use_coattn=False, 
    #                         use_kmer_embed=True, 
    #                         use_seq_encoder=False, 
    #                         seq_encoder_type="lstm", 
    #                         num_heads=4, 
    #                         num_inds=6, 
    #                         num_outputs=6, 
    #                         ln=True).cuda()
    #         epochs = 150
    #         lr = 6e-5
            
    #     elif model_name=="FTransformer":
    #         model = FTransformer(embed_size=32, 
    #                             hidden=64, 
    #                             num_layers=2, 
    #                             dropout=0.5, 
    #                             k4kmer=3, 
    #                             use_pretrain=False, 
    #                             use_coattn=True, 
    #                             seq_encoder_type="transformer", 
    #                             num_heads=2).cuda()
            
    #         epochs = 100
    #         lr = 3e-5
            
    #     elif model_name=="EnsembleModel":
    #         model = EnsembleModel(embed_size=16, 
    #                     hidden=64, 
    #                     max_len=100, 
    #                     num_encoder_layers=1, 
    #                     num_heads=2, 
    #                     num_inds=6, 
    #                     num_outputs=6, 
    #                     ln=True, 
    #                     dropout=0.5, 
    #                     use_coattn=True).cuda()
            
    #         epochs = 500
    #         lr = 1e-5
            
    #     elif model_name=="EnsembleModel_ft":
    #         model = torch.load("./results/SAbDab/full/seq1_neg0/EnsembleModel/model_best.pth")
    #         model.train()
            
    #         epochs = 500
    #         lr = 1e-4
            
    #     elif model_name=="PESI":
    # #         model = PESI(embed_size=7, 
    # #                      hidden=512, 
    # #                      max_len=100, 
    # #                      num_heads=2, 
    # #                      num_inds=6, 
    # #                      num_outputs=6, 
    # #                      ln=True, 
    # #                      dropout=0.5, 
    # #                      use_coattn=True).cuda()
    #         model = PESI(embed_size=8, 
    #                     hidden=64, 
    #                     max_len=100, 
    #                     num_heads=2, 
    #                     num_inds=6, 
    #                     num_outputs=6, 
    #                     ln=True, 
    #                     dropout=0.5, 
    #                     use_coattn=True).cuda()
            
    #         epochs = 200
    #         lr = 5e-5
    # #         wd = 3e-4
    #         l2_coef = 5e-4
            
    #     elif model_name=="PESI_ft":
    #         model = torch.load("./results/SAbDab/full/seq1_neg0/PESI/model_best.pth")
    #         model.train()
            
    #         # freeze frame feature extractor
    #         for name, param in model.Frame_para.named_parameters():
    #             param.requires_grad = False
    #         for name, param in model.Frame_epi.named_parameters():
    #             param.requires_grad = False
                
    #         # freeze frame feature extractor        
    #         for name, param in model.Set_para.named_parameters():
    #             param.requires_grad = False
    #         for name, param in model.Set_epi.named_parameters():
    #             param.requires_grad = False
            
    #         epochs = 500
    #         lr = 3e-5
    #         l2_coef = 5e-4

    #     else:
    #         print("wrong model name!!!")
    #         break

        print("model_name: {}".format(config["model_name"]))

        print("model parameters: ", sum(p.numel() for p in config["model"].parameters() if p.requires_grad))
        
        criterion = nn.BCELoss()
        optimizer = optim.Adam(config["model"].parameters(), lr=config["lr"])#, weight_decay=wd)
        # scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=5, eta_min=1e-6, last_epoch=-1)

        loss_buf = []
        val_loss_buf = []
        val_acc_buf = []
        val_f1_buf = []
        val_auc_buf = []
        val_gmean_buf = []
        val_mcc_buf = []
        best_val_loss = float("inf")
        
        for epoch in range(config["epochs"]):

            config["model"].train()

            loss_tmp = []
            for i, (para, epi, label) in enumerate(tqdm(train_loader)):
                optimizer.zero_grad()

                if config["use_BSS"]==False:
                    pred = config["model"](para, epi)
                elif config["use_BSS"]==True:
                    pred, BSS = config["model"](para, epi)
                else:
                    pass
                    
                loss = criterion(pred.view(-1), label.view(-1).cuda())
                
                if config["use_reg"]==0:
                    param_l2_loss = 0
                    for name, param in config["model"].named_parameters():
                        if 'bias' not in name:
                            param_l2_loss += torch.norm(param, p=2)
                    param_l2_loss = config["l2_coef"] * param_l2_loss
                    loss += param_l2_loss
                elif config["use_reg"]==1:
                    param_l1_loss = 0
                    for name, param in config["model"].named_parameters():
                        if 'bias' not in name:
                            param_l1_loss += torch.norm(param, p=1)
                    param_l1_loss = config["l1_coef"] * param_l1_loss
                    loss += param_l1_loss
                else:
                    print("wrong use_reg! only 0 or 1!")
                    exit()
                
                if config["use_BSS"]==True:
                    loss += 0.001*BSS

                loss.backward()
                
                torch.nn.utils.clip_grad_norm_(config["model"].parameters(), config["clip_norm"])

                optimizer.step()

                loss_tmp.append(loss.item())
            
            loss_buf.append(np.mean(loss_tmp))

        #     scheduler.step()
            print("lr: ", optimizer.param_groups[0]['lr'])

            with torch.no_grad():

                config["model"].eval()

                preds = []
                labels = []
                val_loss_tmp = []
                for i, (para, epi, label) in enumerate(test_loader):
                    if config["use_BSS"]==False:
                        pred = config["model"](para, epi)
                    elif config["use_BSS"]==True:
                        pred, BSS = config["model"](para, epi)
                    
                    val_loss = criterion(pred.view(-1), label.view(-1).cuda())
                    
                    if config["use_BSS"]==True:
                        val_loss += 0.001*BSS
                    
                    preds.append(pred.detach().cpu().view(-1))
                    labels.append(label.view(-1))
                    val_loss_tmp.append(val_loss.item())
                
                preds = torch.stack(preds, axis=1).view(-1)
                labels = torch.stack(labels, axis=1).view(-1)

                acc, f1, auc, gmean, mcc = evaluate_metrics(pred_proba=preds, label=labels)

                val_acc_buf.append(acc)
                val_f1_buf.append(f1)
                val_auc_buf.append(auc)
                val_gmean_buf.append(gmean)
                val_mcc_buf.append(mcc)
                val_loss_buf.append(np.mean(val_loss_tmp))

                print("Epoch {}: \n Train Loss\t{:.4f} \n Val Loss\t{:.4f} \n Val Acc\t{:.4f} \n Val F1\t\t{:.4f} \n Val AUC\t{:.4f} \n Val GMean\t{:.4f} \n Val MCC\t{:.4f}".format(epoch, np.mean(loss_buf), np.mean(val_loss_buf), acc, f1, auc, gmean, mcc))
                
                if np.mean(val_loss_tmp)<best_val_loss:
                    best_val_loss = np.mean(val_loss_tmp)
                    torch.save(config["model"], "./results/CoV-AbDab/{}/model_{}_best.pth".format(config["model_name"], k_iter))
                    np.save("./results/CoV-AbDab/{}/val_acc_{}_best.npy".format(config["model_name"], k_iter), acc)
                    np.save("./results/CoV-AbDab/{}/val_f1_{}_best.npy".format(config["model_name"], k_iter), f1)
                    np.save("./results/CoV-AbDab/{}/val_auc_{}_best.npy".format(config["model_name"], k_iter), auc)
                    np.save("./results/CoV-AbDab/{}/val_gmean_{}_best.npy".format(config["model_name"], k_iter), gmean)
                    np.save("./results/CoV-AbDab/{}/val_mcc_{}_best.npy".format(config["model_name"], k_iter), mcc)

            config["model"].train()
        
        torch.save(config["model"], "./results/CoV-AbDab/{}/model_{}.pth".format(config["model_name"], k_iter))
        np.save("./results/CoV-AbDab/{}/loss_buf_{}.npy".format(config["model_name"], k_iter), np.array(loss_buf))
        np.save("./results/CoV-AbDab/{}/val_loss_buf_{}.npy".format(config["model_name"], k_iter), np.array(val_loss_buf))
        np.save("./results/CoV-AbDab/{}/val_acc_buf_{}.npy".format(config["model_name"], k_iter), np.array(val_acc_buf))
        np.save("./results/CoV-AbDab/{}/val_f1_buf_{}.npy".format(config["model_name"], k_iter), np.array(val_f1_buf))
        np.save("./results/CoV-AbDab/{}/val_auc_buf_{}.npy".format(config["model_name"], k_iter), np.array(val_auc_buf))
        np.save("./results/CoV-AbDab/{}/val_gmean_buf_{}.npy".format(config["model_name"], k_iter), np.array(val_gmean_buf))
        np.save("./results/CoV-AbDab/{}/val_mcc_buf_{}.npy".format(config["model_name"], k_iter), np.array(val_mcc_buf))
        
        
        kfold_labels.append(labels)
        kfold_preds.append(preds)
        
    #     break

    res = evaluate(model_name=config["model_name"], kfold=config["kfold"])

    return res


if __name__=='__main__':

    # data = pd.read_csv("../SARS-SAbDab_Shaun/CoV-AbDab_extract.csv")

    # model_name = ["lstm", "textcnn", "masonscnn", "ag_fast_parapred", "pipr", "resppi", "deepaai"]

    # set_seed(seed=3407)
    set_seed(seed=42)
    # model_name = "lstm"
    model_name = "masonscnn"

    config = {
        # data type
        "clip_norm": 1, 
        "data_type": "seq1_neg0", 
        "data_path": "../SARS-SAbDab_Shaun/CoV-AbDab_extract.csv", 

        # fine-tuning params
        "use_fine_tune": False,                 # load pre-trained weights as initialisation
        "fix_FE": False,                        # only load pre-trained feature extractor weights
        "use_pair": False,                      # whether using pairwise pre-training or not

        # training params
        "use_reg": 0,                           # regularisation type: 0 - L2; 1 - L1
        "use_BSS": False,                       # Batch Spectral Shrinkage regularisation

        # experiment params
        "ntimes": 3,                            # repeat ntimes of kfold
        "kfold": 10,                            # kfold cross validation
        "batch_size": 16,                       # batch size

        # model_params
        "model_name":model_name

    }


    print(config)

    # training
    for i in range(config["ntimes"]):
        print("Run {} times of {}fold".format(config["ntimes"], config["kfold"]))
        result = cov_train(config=config)
        # current_time = time.strftime('%Y-%m-%d-%H-%M', time.localtime())
        print("Results dump to: ")
        print("./results/CoV-AbDab/{}/result_{}.pkl".format(config["model_name"], i))
        pickle.dump(result, open("./results/CoV-AbDab/{}/result_{}.pkl".format(config["model_name"], i), "wb"))