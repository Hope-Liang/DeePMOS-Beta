import argparse
import copy
import math
import os
import random
import typing

import numpy as np
import pandas as pd
import scipy
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

import scipy.stats as stats
from scipy.special import beta

from dataset import get_dataloader, get_dataset
from EMA import WeightExponentialMovingAverage
from model import DeePMOS_Beta


parser = argparse.ArgumentParser(description='Training DeePMOS-Beta model.')
parser.add_argument('--num_epochs', type=int, help='Number of epochs.', default=60)
parser.add_argument('--lamb_c', type=float, help='Weight of consistency loss lambda_c.', default=1.0)
parser.add_argument('--log_valid', type=int, help='Logging valid score each log_valid epochs.', default=1)
parser.add_argument('--log_epoch', type=int, help='Logging training during a global run.', default=1)
parser.add_argument('--dataset', type=str, help='Dataset.', default='vcc2018')
parser.add_argument('--data_path', type=str, help='Path to data.', default='../VCC2018/testVCC2/')
parser.add_argument('--id_table', type=str, help='Path to ID of judges.', default='../VCC2018/id_table/')
parser.add_argument('--save_path', type=str, help='Path to save the model.', default='')
args = parser.parse_args()

def beta_nll_loss(mos_alpha, target, mos_beta, eps = 1e-6):
    #mos_alpha = torch.clamp(mos_alpha, min=eps)
    #mos_beta = torch.clamp(mos_beta, min=eps)

    log_likelihood = (mos_alpha - 1) * torch.log(target) + (mos_beta - 1) * torch.log(1 - target)
    log_likelihood -= torch.lgamma(mos_alpha) + torch.lgamma(mos_beta) - torch.lgamma(mos_alpha + mos_beta)
    
    # Take the negative of the log likelihood since we want to minimize it
    neg_log_likelihood = -torch.mean(log_likelihood)
    
    return neg_log_likelihood


def valid(model,
          dataset,
          dataloader, 
          systems,
          steps,
          prefix,
          device,
          MSE_list,
          LCC_list,
          SRCC_list):
    model.eval()

    mos_alphas = []
    mos_betas = []
    mos_means = []
    mos_targets = []
    mos_vars = []
    mos_means_sys = {system:[] for system in systems}
    mos_vars_sys = {system:[] for system in systems}
    true_sys_mean_scores = {system:[] for system in systems}

    for i, batch in enumerate(tqdm(dataloader, ncols=0, desc=prefix, unit=" step")):
        if dataset == 'vcc2018':
            wav, filename, _, mos, _ = batch
            sys_names = list(set([name.split("_")[0] for name in filename])) # system name, e.g. 'D03'
        elif dataset == 'bvcc':
            wav, mos, sys_names = batch
        wav = wav.to(device)
        wav = wav.unsqueeze(1) # shape (batch, 1, seq_len, 257)

        with torch.no_grad():
            try:
                mos_alpha, mos_beta = model(speech_spectrum = wav) # shape (batch, seq_len, 1)
                mos_alpha = mos_alpha.squeeze(-1) # shape (batch, seq_len)
                mos_beta = mos_beta.squeeze(-1)
                mos_alpha = torch.mean(mos_alpha, dim = -1) # torch.Size([1])
                mos_beta = torch.mean(mos_beta, dim = -1)

                mos_mean = mos_alpha/(mos_alpha+mos_beta)
                mos_var = (mos_alpha*mos_beta)/((mos_alpha+mos_beta)*(mos_alpha+mos_beta)*(mos_alpha+mos_beta+1))
                
                mos_mean = mos_mean*4+1 # from range [0,1] to range [1,5]
                mos_var = mos_var*16 # times 4^2

                mos_alpha = mos_alpha.cpu().detach().numpy()
                mos_beta = mos_beta.cpu().detach().numpy()
                mos_mean = mos_mean.cpu().detach().numpy()
                mos_var = mos_var.cpu().detach().numpy()

                mos_alphas.extend(mos_alpha.tolist())
                mos_betas.extend(mos_beta.tolist())
                mos_means.extend(mos_mean.tolist())
                mos_vars.extend(mos_var.tolist())

                mos_targets.extend(mos.tolist())

                for j, sys_name in enumerate(sys_names):
                    mos_means_sys[sys_name].append(mos_mean[j])
                    mos_vars_sys[sys_name].append(mos_var[j])
                    true_sys_mean_scores[sys_name].append(mos.tolist()[j])

            except RuntimeError as e:
                if "CUDA out of memory" in str(e):
                    print(f"[Runner] - CUDA out of memory at step {global_step}")
                    with torch.cuda.device(device):
                        torch.cuda.empty_cache()
                    continue
                else:
                    raise
    
    mos_alphas = np.array(mos_alphas)
    mos_betas = np.array(mos_betas)
    mos_means = np.array(mos_means)
    mos_vars = np.array(mos_vars)
    mos_targets = np.array(mos_targets)

    mos_means_sys = np.array([np.mean(scores) for scores in mos_means_sys.values()])
    true_sys_mean_scores = np.array([np.mean(scores) for scores in true_sys_mean_scores.values()])
    
    utt_MSE=np.mean((mos_targets-mos_means)**2)
    utt_LCC=np.corrcoef(mos_targets, mos_means)[0][1]
    utt_SRCC=scipy.stats.spearmanr(mos_targets, mos_means)[0]
    
    sys_MSE=np.mean((true_sys_mean_scores-mos_means_sys)**2)
    sys_LCC=np.corrcoef(true_sys_mean_scores, mos_means_sys)[0][1]
    sys_SRCC=scipy.stats.spearmanr(true_sys_mean_scores, mos_means_sys)[0]

    Likelihoods = []
    for i in range(len(mos_targets)):
        Likelihoods.append(stats.beta.pdf((mos_means[i]-1)/4, mos_alphas[i], mos_betas[i]))
    
    utt_AML=np.mean(Likelihoods)/4 # arithemic mean of likelihood
    utt_MoV=np.mean(mos_vars) # mean of variance
    utt_VoV=np.var(mos_vars) # variance of variance

    MSE_list.append(utt_MSE)
    LCC_list.append(utt_LCC) 
    SRCC_list.append(utt_SRCC)

    print(
        f"\n[{prefix}][{steps}][UTT][ MSE = {utt_MSE:.4f} | LCC = {utt_LCC:.4f} | SRCC = {utt_SRCC:.4f} ] [SYS][ MSE = {sys_MSE:.4f} | LCC = {sys_LCC:.4f} | SRCC = {sys_SRCC:.4f} ]"
    )
    print(f"[{prefix}][{steps}][UTT][ AML = {utt_AML:.6f} | MoV = {utt_MoV:.6f} | VoV = {utt_VoV:.6f} ]" )

    model.train()
    return MSE_list, LCC_list, SRCC_list, sys_SRCC


def train(num_epochs,
          lamb_c,
          lamb_t,
          log_valid,
          log_epoch,
          dataset,
          train_set,
          valid_set,
          test_set,
          train_loader,
          valid_loader,
          test_loader):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = DeePMOS_Beta().to(device)
    momentum_model = DeePMOS_Beta().to(device) 
    for param in momentum_model.parameters(): 
        param.detach_()
    momentum_model.load_state_dict(model.state_dict())

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5) #lr=1e-4, weight_decay=1e-5
    optimizer_momentum = WeightExponentialMovingAverage(model, momentum_model) 
    optimizer.zero_grad()
    criterion1 = beta_nll_loss # Beta Negative Log Likelihood (BNLLloss)
    criterion2 = F.mse_loss

    backward_steps = 0
    all_loss = []

    best_LCC = -1
    best_LCC_teacher = -1
    best_sys_SRCC = -1
    
    MSE_list = []
    LCC_list = []
    SRCC_list = []
    train_loss = []
    MSE_teacher, LCC_teacher, SRCC_teacher = [], [], []
    
    model.train()
    epoch = 0
    while epoch <= num_epochs:
        if epoch == 5:
            optimizer_momentum.set_alpha(alpha = 0.999) 
        
        for i, batch in enumerate(tqdm(train_loader, ncols=0, desc="Train", unit=" step")):
            try:
                wavs, _, _, mos, _ = batch
                wavs = wavs.to(device)
                wavs = wavs.unsqueeze(1) # shape (batch, 1, seq_len, 257[dim feature])
                mos = mos.to(device) # shape (batch)

                # Stochastic Gradient Noise (SGN)
                label_noise = torch.randn(mos.size(), device=device) # standard normal distribution
                mos += 0.1*label_noise
                mos = torch.clamp(mos, min=1, max=5)

                mos = 0.98*(mos-1)/4+0.01 # from range [1,5] to range [0.01,0.99]

                # Forward
                mos_alpha, mos_beta = model(speech_spectrum=wavs) # (batch, seq_len, 1), (batch, seq_len, 1)
                mos_alpha_mom, mos_beta_mom = momentum_model(speech_spectrum=wavs)
                mos_alpha = mos_alpha.squeeze() # (batch, seq_len)
                mos_beta = mos_beta.squeeze() # (batch, seq_len)
                mos_alpha_mom = mos_alpha_mom.squeeze()
                mos_beta_mom = mos_beta_mom.squeeze() 
                seq_len = mos_alpha.shape[1]
                bsz = mos_alpha.shape[0]

                mos = mos.unsqueeze(1).repeat(1, seq_len) # (batch, seq_len) by repeat seq_len times

                # Loss
                loss = criterion1(mos_alpha, mos, mos_beta) # torch.Size([])
                cost_alpha = criterion2(mos_alpha, mos_alpha_mom)
                cost_beta = criterion2(mos_beta, mos_beta_mom)
                loss_teacher = criterion1(mos_alpha_mom, mos, mos_beta_mom)
                loss = loss + lamb_c*(cost_alpha+cost_beta) +lamb_t*loss_teacher
                
                # Backwards
                loss.backward()

                all_loss.append(loss.item())
                del loss

                # Gradient clipping
                grad_norm = nn.utils.clip_grad_norm_(model.parameters(), max_norm=5) #max_norm=5
                optimizer.step()
                optimizer_momentum.step() 
                optimizer.zero_grad()
    
            except Exception as e:
                print(e)

        if epoch % log_epoch == 0:
            average_loss = torch.FloatTensor(all_loss).mean().item()
            train_loss.append(average_loss)
            print(f"Average loss={average_loss}")
            all_loss = []

        if epoch % log_valid == 0:
            MSE_teacher, LCC_teacher, SRCC_teacher, sys_SRCC_teacher = valid(momentum_model,
                                                                     dataset,
                                                                     valid_loader,
                                                                     valid_set.systems,
                                                                     epoch,
                                                                     'Valid(teacher)',
                                                                     device,
                                                                     MSE_teacher,
                                                                     LCC_teacher,
                                                                     SRCC_teacher)

            if LCC_teacher[-1] > best_LCC_teacher:
                best_LCC_teacher = LCC_teacher[-1]
                best_model = copy.deepcopy(momentum_model)

        epoch += 1

    print('Best model performance test:')
    _, _, _, _ = valid(best_model, dataset, test_loader, test_set.systems, epoch, 'Test(best)', device, MSE_teacher, LCC_teacher, SRCC_teacher)
    return best_model, train_loss, MSE_list, LCC_list, SRCC_list, LCC_teacher

def main():
    data_path = args.data_path
    id_table = args.id_table
    dataset = args.dataset

    if dataset == 'vcc2018':
        train_set = get_dataset(data_path, "training_data.csv", vcc18=True, idtable=os.path.join(id_table, 'idtable.pkl'))
        valid_set = get_dataset(data_path, "valid_data.csv", vcc18=True, valid=True, idtable=os.path.join(id_table, 'idtable.pkl'))
        test_set = get_dataset(data_path, "testing_data.csv", vcc18=True, valid=True, idtable=os.path.join(id_table, 'idtable.pkl'))
    elif dataset == 'bvcc':
        train_set = get_dataset(data_path, "train", bvcc=True, idtable=os.path.join(id_table, 'idtable.pkl'))
        valid_set = get_dataset(data_path, "valid", bvcc=True, valid=True, idtable=os.path.join(id_table, 'idtable.pkl'))
        test_set = get_dataset(data_path, "test", bvcc=True, valid=True, idtable=os.path.join(id_table, 'idtable.pkl'))

    train_loader = get_dataloader(train_set, batch_size=64, num_workers=1)
    valid_loader = get_dataloader(valid_set, batch_size=1, num_workers=1)
    test_loader = get_dataloader(test_set, batch_size=1, num_workers=1)
    
    best_model, train_loss, MSE_list, LCC_list, SRCC_list, LCC_teacher = train(
        args.num_epochs, args.lamb_c, args.lamb_t, args.log_valid, args.log_epoch, 
        dataset, train_set, valid_set, test_set, train_loader, valid_loader, test_loader)

    model_scripted = torch.jit.script(best_model) # Export to TorchScript
    model_scripted.save(args.save_path+'best.pt')

main()
