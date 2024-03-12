'''
python msfuda_target_adaptation_evi_domainnet126.py --dset domainnet126 --gpu_id 1 --output_src ckps/source/ --output ckps/adapt --batch_size 17 --worker 0
'''
import argparse
from concurrent.futures import thread

import os, sys
import time

import os.path as osp

from torch.cuda import amp
import torchvision
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from numpy import argmax, linalg as LA
from torchvision import transforms
import network as network
import loss
from torch.utils.data import DataLoader, Dataset
from data_list import ImageList, ImageList_idx,Listset,Listset3,ImageList_aug
import random, pdb, math, copy
from tqdm import tqdm
from randaug import RandAugmentMC
from scipy.spatial.distance import cdist
from sklearn.metrics import confusion_matrix

from robustbench.data import load_imagenetc
from robustbench.utils import load_model
from robustbench.model_zoo.enums import ThreatModel

from domainnet126 import get_domainnet126

from PIL import Image


def op_copy(optimizer):
    for param_group in optimizer.param_groups:
        param_group['lr0'] = param_group['lr']
    return optimizer

def lr_scheduler(optimizer, iter_num, max_iter, gamma=10, power=0.75):
    decay = (1 + gamma * iter_num / max_iter) ** (-power)
    for param_group in optimizer.param_groups:
        param_group['lr'] = param_group['lr0'] * decay
        param_group['weight_decay'] = 1e-3
        param_group['momentum'] = 0.9
        param_group['nesterov'] = True
    return optimizer

def image_train(resize_size=256, crop_size=224, alexnet=False):
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    return transforms.Compose([
        transforms.Resize((resize_size, resize_size)),
        transforms.RandomCrop(crop_size),
        #transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize
    ])

def positive_aug(resize_size=256, crop_size=224, alexnet=False):
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    return transforms.Compose([
        transforms.Resize((resize_size, resize_size)),
        transforms.RandomCrop(crop_size),
        #transforms.RandomHorizontalFlip(),
        RandAugmentMC(n=2, m=10),
        transforms.ToTensor(),
        normalize
    ])

def image_test(resize_size=256, crop_size=224, alexnet=False):
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    return transforms.Compose([
        transforms.Resize((resize_size, resize_size)),
        transforms.CenterCrop(crop_size),
        transforms.ToTensor(),
        normalize
    ])

def data_load(args):
    ## prepare data

    class TempSet(Dataset):
        def __init__(
            self,
            image_root: str,
            label_file: str,
            transform=None, transform1=None,
            batch_idx=None,
            pseudo_item_list=None,
        ):
            self.image_root = image_root
            self._label_file = label_file
            self.transform = transform
            self.transform1=transform1

            assert (
                label_file or pseudo_item_list
            ), f"Must provide either label file or pseudo labels."
            self.item_list = (
                self.build_index(label_file) if label_file else pseudo_item_list
            )
            if batch_idx != None: 
                self.item_list = self.item_list[batch_idx*50 : (batch_idx+1)*50]

        def build_index(self, label_file):
            # read in items; each item takes one line
            with open(label_file, "r") as fd:
                lines = fd.readlines()
            lines = [line.strip() for line in lines if line]

            item_list = []
            for item in lines:
                img_file, label = item.split()
                img_path = os.path.join(self.image_root, img_file)
                label = int(label)
                item_list.append((img_path, label, img_file))

            return item_list

        def __getitem__(self, idx):
            img_path, label, _ = self.item_list[idx]
            img = Image.open(img_path)
            img = img.convert("RGB")

            if self.transform is not None:
                img1 = self.transform(img)
            if self.transform1 is not None:
                img2 = self.transform1(img)

            if self.transform1 is not None:
                return [img1, img2], label, idx
            else:
                return img1, label, idx

        def __len__(self):
            return len(self.item_list)

    dsets = {}
    dset_loaders = {}
    image_root = '/home/yxue/datasets/DomainNet-126'
    label_file = os.path.join(image_root, f"{args.name_tar}_list.txt")

    dsets["target"] = TempSet(image_root, label_file, transform=image_train(), batch_idx=args.batch_idx)
    dset_loaders["target"] = DataLoader(dsets["target"], batch_size=args.batch_size, shuffle=True, pin_memory=True, drop_last=False, num_workers=args.worker)
    dsets['target_'] = TempSet(image_root, label_file,  transform=image_train(), transform1=positive_aug(), batch_idx=args.batch_idx)
    dset_loaders['target_'] = DataLoader(dsets['target_'], batch_size=args.batch_size, shuffle=True, pin_memory=True, drop_last=False, num_workers=args.worker)
    dsets["test"] = TempSet(image_root, label_file, transform=image_test(), batch_idx=args.batch_idx)
    dset_loaders["test"] = DataLoader(dsets["test"], batch_size=args.batch_size * 10, shuffle=False, pin_memory=True, drop_last=False, num_workers=args.worker)

    return dset_loaders


def sigma3(ten_sor,k=0.5):
    u=torch.mean(ten_sor)
    v=torch.std(ten_sor)
    thread_=u+k*v
    l_id=torch.nonzero(ten_sor<thread_).squeeze().cpu().numpy()
    h_id=torch.nonzero(ten_sor>thread_).squeeze().cpu().numpy()
    return l_id,h_id


def train_target_primary(args,dset_loaders,mddn_F,mddn_C1,mddn_C2,mddn_E1,mddn_E2,primary_idx,unc_list,evi_list,out_list,pse_list,fea_list):
    def DS_Combin_two(alpha1, alpha2,):
        """
        :param alpha1: Dirichlet distribution parameters of view 1
        :param alpha2: Dirichlet distribution parameters of view 2
        :return: Combined Dirichlet distribution parameters
        """
        alpha = dict()
        alpha[0], alpha[1] = alpha1, alpha2

        M,Q, b, S, E, u = dict(), dict(), dict(), dict(), dict(), dict()
        for v in range(2):
            S[v] = torch.sum(alpha[v], dim=1, keepdim=True)
            M[v], _ = torch.max(alpha[v], dim=1, keepdim=True)
            Q[v]=M[v]/S[v]
            alpha[v]=alpha[v]*Q[v]
        for v in range(2):
            S[v] = torch.sum(alpha[v], dim=1, keepdim=True)
            E[v] = alpha[v]-1
            b[v] = E[v]/(S[v].expand(E[v].shape))
            u[v] = args.class_num/S[v]

        # b^0 @ b^(0+1)
        bb = torch.bmm(b[0].view(-1, args.class_num, 1), b[1].view(-1, 1, args.class_num))
        # b^0 * u^1
        uv1_expand = u[1].expand(b[0].shape)
        bu = torch.mul(b[0], uv1_expand)
        # b^1 * u^0
        uv_expand = u[0].expand(b[0].shape)
        ub = torch.mul(b[1], uv_expand)
        # calculate C
        bb_sum = torch.sum(bb, dim=(1, 2), out=None)
        bb_diag = torch.diagonal(bb, dim1=-2, dim2=-1).sum(-1)
        C = bb_sum - bb_diag
        # calculate b^a
        b_a = (torch.mul(b[0], b[1]) + bu + ub)/((1-C).view(-1, 1).expand(b[0].shape))
        # calculate u^a
        u_a = torch.mul(u[0], u[1])/((1-C).view(-1, 1).expand(u[0].shape))
        # calculate new S
        S_a = args.class_num / u_a
        # calculate new e_k
        e_a = torch.mul(b_a, S_a.expand(b_a.shape))
        alpha_a = e_a + 1
        S_a= torch.sum(alpha_a, dim=1, keepdim=True)
        M_a, _ = torch.max(alpha_a, dim=1, keepdim=True)
        alpha_a=alpha_a/M_a*S_a
        return alpha_a
    def initial(args):
        param_group = []
   
        for k, v in mddn_F.named_parameters():
            if args.lr_decay1 > 0:
                param_group += [{'params': v, 'lr': args.lr * args.lr_decay1}]
            else:
                v.requires_grad = False
        for k, v in mddn_C1.named_parameters():
            if args.lr_decay2 > 0:
                param_group += [{'params': v, 'lr': args.lr * args.lr_decay2}]
            else:
                v.requires_grad = False
        for k, v in mddn_E1.named_parameters():
            if args.lr_decay2 > 0:
                param_group += [{'params': v, 'lr': args.lr * args.lr_decay2}]
            else:
                v.requires_grad = False
        optimizer = optim.SGD(param_group)
        optimizer = op_copy(optimizer)
        for k, v in mddn_C2.named_parameters():
            v.requires_grad = False
            #param_group_c += [{"params": v, "lr": args.lr * 1}] 
        for k, v in mddn_E2.named_parameters():
            v.requires_grad = False  

        optimizer = optim.SGD(param_group)
        optimizer = op_copy(optimizer)

      
        return mddn_F,mddn_C1,mddn_C2,mddn_E1,mddn_E2,optimizer

    mddn_F,mddn_C1,mddn_C2,mddn_E1,mddn_E2,optimizer=initial(args)
    max_iter =len(dset_loaders["target"])
   
    iter_num = 0

    
    mddn_F.eval()
    mddn_C1.eval()
    mddn_C2.eval()
    num_sample = len(dset_loaders["target"].dataset)
    fea_bank=dict()
  
    fea_bank = torch.randn(num_sample, 256*len(args.src))
    score_bank = torch.randn(num_sample, args.class_num).cuda()
    
   
    fea_list_norm=fea_list
    for k in fea_list_norm.keys():
        fea_list_norm[k]=F.normalize(fea_list_norm[k])
    for i in range(len(args.src)):
        if i==primary_idx:
            with torch.no_grad():
                iter_test = iter(dset_loaders["target"])
                for i in range(len(dset_loaders["target"])):
                    data = iter_test.next()
                    inputs = data[0]
                    indx = data[-1]
                    # labels = data[1]
                    inputs = inputs.cuda()
                    fea=mddn_F(inputs)
                    fea = mddn_C1(fea)
                    fea_norm = F.normalize(fea)
                    for i in range(len(args.src)):
                        if i==primary_idx:
                            continue
                        fea_norm=torch.cat([fea_norm,fea_list_norm[i][indx].cuda()],1)
                    outputs = mddn_C2(fea)
                    outputs = nn.Softmax(-1)(outputs)

                    fea_bank[indx] = fea_norm.detach().clone().cpu()
                    score_bank[indx] = outputs.detach().clone()  # .cpu()
    epoch=0
    
    fuse_epoch=args.max_epoch
    while epoch < fuse_epoch:
        epoch+=1
        iter_num=0
        lr_scheduler(optimizer, iter_num=epoch, max_iter=fuse_epoch)
        mddn_F.eval()
        mddn_C1.eval()
        mddn_E1.eval()
        mddn_C2.eval()
        mddn_E2.eval()
       
        mem_label,mem_sec_label,all_evidence,all_label = obtain_label(dset_loaders['test'], mddn_F, mddn_C1, mddn_C2,mddn_E1, mddn_E2,primary_idx,out_list, fea_list,evi_list,args)
        #pdb.set_trace()
        unc_list[primary_idx]=args.class_num/(torch.max(all_evidence+1,1)[0])
       
        mem_label = torch.from_numpy(mem_label)

        pse_list[primary_idx]=mem_label
        mddn_F.train()
        mddn_C1.train()
        mddn_E1.train()
        

        if epoch<=1:
            source_pse=torch.zeros(mem_label.shape[0], len(args.src))
            source_unc=torch.zeros(mem_label.shape[0], len(args.src))
            for i in range(len(args.src)):
                if i!=primary_idx:
                    source_pse[:,i]=pse_list[i]
                    source_unc[:,i]=unc_list[i]
                   
                else:
                    source_pse[:,i]=pse_list[i]
                    source_unc[:,i]=unc_list[i]*0.6
                   
        else:       
            source_pse[:,primary_idx]=pse_list[primary_idx]
            source_unc[:,primary_idx]=unc_list[primary_idx]*0.6
        sorts=torch.argmin(source_unc,1)
        
        PLE=source_pse[:,sorts]


        mem_sec_label = torch.from_numpy(mem_sec_label)
        threshold=torch.ones(args.class_num)
        threshold_unc=torch.ones(args.class_num)

        local_inconsistency_mask=torch.zeros(mem_label.shape[0])
        pse_smooth_mask=torch.zeros(mem_label.shape[0])
        all_alpha=all_evidence+1
        
        EAU_ini=torch.log(1+(torch.sum(all_alpha)-torch.max(all_alpha,1)[0])/(torch.max(all_alpha,1)[0]))
        E_inter_pre=all_alpha/(torch.sum(all_alpha,1,keepdims=True))
        distance = fea_bank@ fea_bank.T / len(args.src)
        dis_near, idx_near = torch.topk(distance, dim=-1, largest=True, k=args.r + 2)
        idx_near = idx_near[:, 1:]  # batch x K
        dis_near = dis_near[:, 1:]
        
        E_inter=torch.sum(np.abs(E_inter_pre[idx_near[:,1]]-E_inter_pre))
        for i in range(2,args.r + 1):
            E_inter=E_inter+torch.sum(np.abs(E_inter_pre[idx_near[:,i]]-E_inter_pre))
       
        EAU=EAU_ini*E_inter


        EPU=args.class_num/(torch.max(all_alpha,1)[0]+args.class_num)
        for i in range(args.class_num):
            eau_class_i=EAU[mem_label==i]
            threshold[i]=torch.mean(eau_class_i)+2*torch.std(eau_class_i)

        for i in range(args.class_num):
            unc_class_i=EPU[mem_label==i]
            threshold_unc[i]=torch.mean(unc_class_i)+2*torch.std(unc_class_i)

        for i in range(mem_label.shape[0]):
            p=mem_label[i]
            if EAU[i]>threshold[p]:
                local_inconsistency_mask[i]=1
            else:
                local_inconsistency_mask[i]=0

        for i in range(mem_label.shape[0]):
            p=mem_label[i]
            if EPU[i]>threshold_unc[p]:
                pse_smooth_mask[i]=1
            else:
                pse_smooth_mask[i]=0
        

        
        while iter_num < max_iter:
            try:
                inputs_test, label, tar_idx = iter_test.next()
            except:
                iter_test = iter(dset_loaders["target"])
                inputs_test, label, tar_idx = iter_test.next()

            if inputs_test.size(0) == 1:
                continue
            iter_num += 1
            agg_loss = torch.tensor(0.0).cuda()
            inputs_test = inputs_test.cuda()
            fea=mddn_F(inputs_test)
            evidence = mddn_E2(mddn_E1(fea))

            alpha = evidence+1
            S=torch.sum(alpha,1,keepdim=True)
            features_test = mddn_C1(fea)
            outputs_test = mddn_C2(features_test)
            #pred=mem_label[tar_idx].cuda()
            #label=label.cuda()
            with torch.no_grad():
                f=mddn_F(inputs_test)
                f1 = mddn_C1(f)
                o1 = mddn_C2(f1).cpu()
                out_list[primary_idx][tar_idx] =o1.cpu()
                primary_evidence = mddn_E2(mddn_E1(f)).detach()
            agg_loss_mul= torch.tensor(0.0).cuda()
            agg_loss_evi= torch.tensor(0.0).cuda()
            for i in range(len(args.src)):
                if i==primary_idx:
                    pred_u=PLE[i][tar_idx].long().cuda()
                    weak_evidence=evi_list[i][tar_idx].cuda()
                    fused_alpha=DS_Combin_two(primary_evidence+1,weak_evidence+1)
                    
                    combined_S=torch.sum(fused_alpha,1,keepdim=True)
                    #pdb.set_trace()
                    pse_loss_ele=loss.ce_loss2(pred_u,alpha,args.class_num,iter_num+(epoch-1)*max_iter,( fuse_epoch-1)*max_iter)+torch.mean( nn.CrossEntropyLoss(reduction='none')(outputs_test,pred_u))
                if i!=primary_idx:
                    mask=(unc_list[i][tar_idx]<unc_list[primary_idx][tar_idx]+0).unsqueeze(1).cuda()
                    pred_u=out_list[i][tar_idx]+out_list[primary_idx][tar_idx]
                    pred_u=nn.Softmax(1)(pred_u)
                    _,pred_u=torch.max(pred_u,1)
                    pred_u=pred_u.cuda()
                    weak_evidence=evi_list[i][tar_idx].cuda()
                    fused_alpha=DS_Combin_two(primary_evidence+1,weak_evidence+1)
                    
                    combined_S=torch.sum(fused_alpha,1,keepdim=True)
                    #pdb.set_trace()
                    agg_loss_evi+=1/2*torch.mean( nn.KLDivLoss(reduction='none')((fused_alpha/combined_S).log(),alpha/S)*mask)
                    agg_loss_mul+=1/2*torch.mean( nn.CrossEntropyLoss(reduction='none')(outputs_test,pred_u)*mask)

            if args.ent:
                softmax_out = nn.Softmax(dim=1)(outputs_test)
                entropy_loss = torch.mean(loss.Entropy(softmax_out))

                if args.gent:
                    msoftmax = softmax_out.mean(dim=0)
                
                    gentropy_loss = torch.sum(-msoftmax * torch.log(msoftmax + args.epsilon))


                    entropy_loss -=gentropy_loss
                im_loss = entropy_loss * args.ent_par
            agg_loss= 0.1*pse_loss_ele+agg_loss_evi+agg_loss_mul+im_loss
            #agg_loss+=nn.CrossEntropyLoss()(outputs_test,pred)
            optimizer.zero_grad()
            agg_loss.backward()
            optimizer.step()
            if epoch<=5 and args.t==0:
                continue
            try:
                [inputs_test,inputs_test1], _, tar_idx = iter_test_.next()
            except:
                iter_test_ = iter(dset_loaders["target_"])
                [inputs_test,inputs_test1], _, tar_idx = iter_test_.next()

            if inputs_test.size(0) == 1:
                continue

            inputs_test = inputs_test.cuda()
            inputs_test1 = inputs_test.cuda()

            pred_u1=mem_label[tar_idx]
            targets_1 = torch.zeros((pred_u1.size(0),args.class_num)).scatter_(1, pred_u1.unsqueeze(1).cpu(), 1)
            pred_u2=mem_sec_label[tar_idx]
            targets_2 = torch.zeros((pred_u2.size(0),args.class_num)).scatter_(1, pred_u2.unsqueeze(1).cpu(), 1)
            unc=EPU[tar_idx].unsqueeze(1)
            targets=targets_1*(1-0.5*unc)+targets_2*(0.5*unc)
            smoothed=(pse_smooth_mask[tar_idx]==1)
            
            targets_1[smoothed,:]=targets[smoothed,:]
            targets_smooth=targets_1.cuda()

            all_inputs = torch.cat([inputs_test, inputs_test1], dim=0)
            #all_targets = torch.cat([targets, targets], dim=0).cuda()

            
            lc_mask=local_inconsistency_mask[tar_idx].cuda()
            features_test = mddn_C1(mddn_F(all_inputs))
            outputs_test = mddn_C2(features_test)
            softmax_out = nn.Softmax(dim=1)(outputs_test)
            # output_re = softmax_out.unsqueeze(1)

            # print(all_inputs.size(), features_test.size(), outputs_test.size(), softmax_out.size())

            with torch.no_grad():
                output_f_norm = F.normalize(features_test[:features_test.shape[0]//2])
                output_f_ = output_f_norm.cpu().detach().clone()

                
                fea_bank[tar_idx][:,:256] = output_f_.detach().clone().cpu()
                score_bank[tar_idx] = softmax_out[:features_test.shape[0]//2].detach().clone()

                distance = fea_bank[tar_idx] @ fea_bank.T / len(args.src)
                dis_near, idx_near = torch.topk(distance, dim=-1, largest=True, k=args.r + 1)
                idx_near = idx_near[:, 1:]  # batch x K
                dis_near = dis_near[:, 1:]
                score_near = score_bank[idx_near]  # batch x K x C

            # nn
            softmax_out_un = softmax_out[:features_test.shape[0]//2].unsqueeze(1).expand(
                -1, args.r, -1
            )  # batch x K x C
            #pdb.set_trace()
            local_loss_sec = torch.mean((F.kl_div(softmax_out_un, score_near, reduction="none").sum(-1)).sum(1)) # Equal to - dot product
            local_loss_risk= -1*torch.mean(lc_mask*((dis_near.cuda()*(F.kl_div(softmax_out_un, score_near, reduction="none").sum(-1))).sum(1)))
            local_loss=local_loss_sec+local_loss_risk
            local_loss+=torch.mean(F.kl_div(softmax_out[:features_test.shape[0]//2],softmax_out[features_test.shape[0]//2:], reduction="none"))
            pse_loss_smoooth=loss.CrossEntropy1(args.class_num)(outputs_test[:features_test.shape[0]//2],targets_smooth)
            
            mask = torch.ones((inputs_test.shape[0], inputs_test.shape[0]))
            diag_num = torch.diag(mask)
            mask_diag = torch.diag_embed(diag_num)
            mask = mask - mask_diag
            copy_soft = softmax_out[:features_test.shape[0]//2].T  # .detach().clone()#

            dot_neg = softmax_out[:features_test.shape[0]//2] @ copy_soft  # batch x batch

            dot_neg = (dot_neg * mask.cuda()).sum(-1)  # batch
            neg_pred = torch.mean(dot_neg)
            local_loss += neg_pred 
            adapt_loss=0.1*pse_loss_smoooth+local_loss
            optimizer.zero_grad()
            #optimizer_c.zero_grad()
            adapt_loss.backward()
            optimizer.step()

    
        mddn_F.eval()
        mddn_C1.eval()
        mddn_E1.eval()
        mddn_E2.eval()
        if args.dset=='VISDA-C':
            acc_s_te, acc_list = cal_acc(dset_loaders['test'], mddn_F, mddn_C1, mddn_C2, True)
            log_str = 'Task: {}, Iter:{}/{}; Accuracy = {:.2f}%'.format(args.name_tar, epoch, fuse_epoch, acc_s_te) + '\n' + acc_list
        else:
            acc_s_te, _ = cal_acc(dset_loaders['test'], mddn_F, mddn_C1, mddn_C2, False)
            log_str = 'Task: {}, Iter:{}/{}; Accuracy = {:.2f}%'.format(args.name_tar, epoch,  fuse_epoch, acc_s_te)

        
        print(log_str+'\n')
        mddn_F.train()
        mddn_C1.train()
        mddn_E1.train()
        mddn_E2.train()
            
        #optimizer_c.step()
            

        
        

        # while iter_num < max_iter:
        #     try:
        #         inputs_test, label, tar_idx = iter_test.next()
        #     except:
        #         iter_test = iter(dset_loaders["target"])
        #         inputs_test, label, tar_idx = iter_test.next()

        #     if inputs_test.size(0) == 1:
        #         continue
        #     iter_num += 1
        #     lr_scheduler(optimizer, iter_num=iter_num, max_iter=max_iter)
        #     lr_scheduler(optimizer_c, iter_num=iter_num, max_iter=max_iter)
        #     inputs_test = inputs_test.cuda()
        #     features_test = mddn_C1(mddn_F(inputs_test))
        #     outputs_test = mddn_C2(features_test)
        #     if args.cls_par > 0:
        #         pred = mem_label[tar_idx]
        #         agg_loss = nn.CrossEntropyLoss()(outputs_test, pred)
        #         agg_loss *= args.cls_par
        #         if iter_num < interval_iter and args.dset == "VISDA-C":
        #             agg_loss *= 0
        #     else:
        #         agg_loss = torch.tensor(0.0).cuda()

        #     if args.ent:
        #         softmax_out = nn.Softmax(dim=1)(outputs_test)
        #         entropy_loss = torch.mean(loss.Entropy(softmax_out))

        #         if args.gent:
        #             msoftmax = softmax_out.mean(dim=0)
        #             gentropy_loss = torch.sum(-msoftmax * torch.log(msoftmax + args.epsilon))


        #             entropy_loss -=gentropy_loss
        #         im_loss = entropy_loss * args.ent_par
        #         agg_loss += im_loss

        #     optimizer.zero_grad()
        #     agg_loss.backward()
        #     optimizer.step()
            
            
        

    

    # if args.issave:   
    #     torch.save(mddn_F.state_dict(), osp.join(args.output_dir, "target_F"  + ".pt"))
    #     torch.save(mddn_C1.state_dict(), osp.join(args.output_dir, "target_B"  + ".pt"))
    #     torch.save(mddn_C2.state_dict(), osp.join(args.output_dir, "target_C"  + ".pt"))
        
    return mddn_F, mddn_C1, mddn_C2, acc_s_te


def cal_acc(loader, mddn_F, mddn_C1, mddn_C2, flag=False):
    start_test = True
    with torch.no_grad():
        iter_test = iter(loader)
        for i in range(len(loader)):
            data = iter_test.next()
            inputs = data[0]
            labels = data[1]
            inputs = inputs.cuda()
            outputs = mddn_C2(mddn_C1(mddn_F(inputs)))
            if start_test:
                all_output = outputs.float().cpu()
                all_label = labels.float()
                start_test = False
            else:
                all_output = torch.cat((all_output, outputs.float().cpu()), 0)
                all_label = torch.cat((all_label, labels.float()), 0)
    _, predict = torch.max(all_output, 1)
    accuracy = torch.sum(torch.squeeze(predict).float() == all_label).item() / float(all_label.size()[0])
    mean_ent = torch.mean(loss.Entropy(nn.Softmax(dim=1)(all_output))).cpu().data.item()

    if flag:
        matrix = confusion_matrix(all_label, torch.squeeze(predict).float())
        acc = matrix.diagonal()/matrix.sum(axis=1) * 100
        aacc = acc.mean()
        aa = [str(np.round(i, 2)) for i in acc]
        acc = ' '.join(aa)
        return aacc, acc
    else:
        return accuracy*100, mean_ent
    

def obtain_label(loader, mddn_F, mddn_C1, mddn_C2,mddn_E1, mddn_E2,primary_idx,all_o_list, all_f_list,all_e_list,args):
    start_test = True
    with torch.no_grad():
        iter_test = iter(loader)
        for _ in range(len(loader)):
            data = iter_test.next()
            inputs = data[0]
            labels = data[1]
            inputs = inputs.cuda()
            fea = (mddn_F(inputs))
            #pdb.set_trace()
            evidence = mddn_E2(mddn_E1(fea))
            feas=mddn_C1(fea)
            
            outputs = mddn_C2(feas)
            if start_test:
                all_fea = feas.float().cpu()
                all_output = outputs.float().cpu()
                all_evidence= evidence.float().cpu()
                all_label = labels.float()
                start_test = False
            else:
                all_fea = torch.cat((all_fea, feas.float().cpu()), 0)
                all_output = torch.cat((all_output, outputs.float().cpu()), 0)
                all_label = torch.cat((all_label, labels.float()), 0)
                all_evidence = torch.cat((all_evidence, evidence.float().cpu()), 0)
    #for i in range(len(args.src)):
    all_o_list[primary_idx]=all_output
    all_f_list[primary_idx]=all_fea
    all_e_list[primary_idx]=all_evidence
    #all_output = nn.Softmax(dim=1)(all_output)
    #ent = torch.sum(-all_output * torch.log(all_output + args.epsilon), dim=1)
    #unknown_weight = 1 - ent / np.log(args.class_num)
    #all_output=#all_o_list[0]*weights[0]
    #for i in range(1,3):
    #    all_output+=all_o_list[i]*weights[i]

    all_output = nn.Softmax(dim=1)(all_output)
    _, predict = torch.max(all_output, 1)

    accuracy = torch.sum(torch.squeeze(predict).float() == all_label).item() / float(all_label.size()[0]) 
    if primary_idx==0:
        all_fea=all_f_list[0]
    else:
        all_fea=all_f_list[0]*0.5
    for i in range(1, len(args.src)):
        if i==primary_idx:
            all_fea=torch.cat((all_fea,all_f_list[i]),1)
        else:
            all_fea=torch.cat((all_fea,all_f_list[i]*0.5),1)

    if args.distance == 'cosine':
        all_fea = torch.cat((all_fea, torch.ones(all_fea.size(0), 1)), 1)
        all_fea = (all_fea.t() / torch.norm(all_fea, p=2, dim=1)).t()
   

    all_fea = all_fea.float().cpu().numpy()
    all_fea = all_fea
    K = all_output.size(1)
    aff = all_output.float().cpu().numpy()
    initc = aff.transpose().dot(all_fea)
    initc = initc / (1e-8 + aff.sum(axis=0)[:,None])
    cls_count = np.eye(K)[predict].sum(axis=0)
    labelset = np.where(cls_count>args.threshold)
    labelset = labelset[0]
    # print(labelset)

    dd = cdist(all_fea, initc[labelset], args.distance)
    pred_label = dd.argmin(axis=1)
    pred_label = labelset[pred_label]

    for round in range(1):
        aff = np.eye(K)[pred_label]
        initc = aff.transpose().dot(all_fea)
        initc = initc / (1e-8 + aff.sum(axis=0)[:,None])
        dd = cdist(all_fea, initc[labelset], args.distance)
        # print(all_fea.size())
        # print(initc[labelset].shape)
        # print(np.argsort(dd).shape)
        # print(np.argsort(dd))

        pred_label = np.argsort(dd)[:,0]
        pred_label2 = np.argsort(dd)[:,1]
        pred_label = labelset[pred_label]
        pred_label2 = labelset[pred_label2]

    acc = np.sum(pred_label2 == all_label.float().numpy()) / len(all_fea)
    log_str = 'Accuracy2 = {:.2f}%'.format( acc * 100)
    print(log_str+'\n')
    acc = np.sum(pred_label == all_label.float().numpy()) / len(all_fea)
    log_str = 'Accuracy = {:.2f}% -> {:.2f}%'.format(accuracy * 100, acc * 100)
   
        
    
    print(log_str+'\n')

    return pred_label.astype('int'),pred_label2.astype('int'),all_evidence,all_label


def initial(net,args):
    param_group = []
    for i in range(len(args.src)):
        modelpath = args.output_dir_src[i] + '/source_F.pt'
        #print(modelpath)
        net.mddn_f[i].load_state_dict(torch.load(modelpath))
        net.mddn_f[i].eval()
        for k, v in net.mddn_f[i].named_parameters():
            param_group += [{'params': v, 'lr': args.lr * args.lr_decay1}]

        modelpath = args.output_dir_src[i] + '/source_B.pt'
        #print(modelpath)
        net.mddn_C1[i].load_state_dict(torch.load(modelpath))
        net.mddn_C1[i].eval()

        for k, v in net.mddn_C1[i].named_parameters():
            param_group += [{'params': v, 'lr': args.lr * args.lr_decay2}]

        modelpath = args.output_dir_src[i] + '/source_C.pt'
        #print(modelpath)
        net.mddn_C2[i].load_state_dict(torch.load(modelpath))
        net.mddn_C2[i].eval()
        for k, v in net.mddn_C2[i].named_parameters():
            v.requires_grad = False
        modelpath = args.output_dir_src[i] + '/source_E.pt'
        #print(modelpath)
        net.mddn_E1[i].load_state_dict(torch.load(modelpath))
        net.mddn_E1[i].eval()
        for k, v in net.mddn_E1[i].named_parameters():
            param_group += [{'params': v, 'lr': args.lr * args.lr_decay2}]
        modelpath = args.output_dir_src[i] + '/source_EC.pt'
        #print(modelpath)
        net.mddn_E2[i].load_state_dict(torch.load(modelpath))
        net.mddn_E2[i].eval()    
        for k, v in net.mddn_E2[i].named_parameters():
            v.requires_grad = False
    optimizer = optim.SGD(param_group)
    optimizer = op_copy(optimizer)
    return net,optimizer   

def train_target(args):
    dset_loaders = data_load(args)
    ## set base network
    if args.net[0:3] == 'res':
        mddn_F_list = [network.ResBase(res_name=args.net).cuda() for i in range(len(args.src))]
      
    elif args.net[0:3] == 'vgg':
        mddn_F_list = [network.VGGBase(vgg_name=args.net).cuda() for i in range(len(args.src))]

    mddn_C1_list = [network.feat_bottleneck(type=args.classifier, feature_dim=mddn_F_list[i].in_features,
                                         bottleneck_dim=args.bottleneck).cuda() for i in range(len(args.src))]
                                    
    mddn_C2_list = [network.feat_classifier(type=args.layer, class_num=args.class_num, bottleneck_dim=args.bottleneck).cuda() for i in range(len(args.src))]
    
    mddn_E1_list = [network.feat_bottleneck(type=args.classifier, feature_dim=mddn_F_list[i].in_features,
                                         bottleneck_dim=args.bottleneck).cuda() for i in range(len(args.src))]
    mddn_E2_list = [network.evidence_classifier(type='linear', class_num=args.class_num, bottleneck_dim=args.bottleneck).cuda() for i in range(len(args.src))]
   
    
    #netQ = network.MyLinear(256,256).cuda()
    net=network.MDDN(mddn_F_list,mddn_C1_list,mddn_C2_list,mddn_E1_list,mddn_E2_list,args.class_num,len(args.src), args.max_epoch * len(dset_loaders["target"]) // args.interval)

    net, optimizer = initial(net,args)
    
    out_list,evi_list,pse_list,all_f_list,all_label = pre_infer(dset_loaders['test'],net)
    pred=dict()
    consistency=[]
    unc=dict()
    prev=dict()
    for i in range(net.source):
        prev[i], pred[i]= torch.max(out_list[i], 1)
        unc[i]=args.class_num/(torch.max(evi_list[i]+1,1)[0])
        consistency.append(torch.sum(torch.max(evi_list[i],1)[0] ))
        accuracy = torch.sum(torch.squeeze(pred[i]).float() == all_label).item() / float(all_label.size()[0])
        print(accuracy)
    max_model=np.argmax(np.array(consistency))
    print('max_model',max_model)
    _, _, _, acc = train_target_primary(args,dset_loaders,net.mddn_f[max_model],net.mddn_C1[max_model],net.mddn_C2[max_model],net.mddn_E1[max_model],net.mddn_E2[max_model],max_model,unc,evi_list,out_list,pse_list,all_f_list)
    
    return acc
                        

def pre_infer(loader, net):
    start_test = True
    with torch.no_grad():
        iter_test = iter(loader)
        for _ in range(len(loader)):
            data = iter_test.next()
            inputs = data[0]
            labels = data[1]

            inputs = inputs.cuda()

            evi_list, out_list, f_list = net.forward(inputs,'test',-1)
            features=f_list[0]

            for i in range(1,net.source):
                features=torch.cat((features, f_list[i]), 1)

            for i in range(1,net.source):
                features=torch.cat((features, f_list[i]), 1)

            if start_test:
                
                all_features = features.float().cpu()
               
                all_label = labels.float()
                all_f_list=dict()
                all_o_list=dict()
                all_evi_list=dict()
                for i in range(net.source):
                    all_f_list[i]=f_list[i].float().cpu()
                    all_o_list[i]=out_list[i].float().cpu()
                    all_evi_list[i]=evi_list[i].float().cpu()

                start_test = False

            else:
                all_features = torch.cat((all_features, features.float().cpu()), 0)
               
                all_label = torch.cat((all_label, labels.float()), 0)
               
                for i in range(net.source):
                    all_o_list[i] = torch.cat((all_o_list[i], out_list[i].float().cpu()), 0)

                    all_evi_list[i] = torch.cat((all_evi_list[i], evi_list[i].float().cpu()), 0)
                    all_f_list[i] = torch.cat((all_f_list[i], f_list[i].float().cpu()), 0)
    pred=dict()
    prev=dict()
    #consistency=[]
    for i in range(net.source):
        prev[i], pred[i]= torch.max(all_o_list[i], 1)
        #consistency.append(torch.sum(torch.max(all_evi_list[i],1)[0] ))
        accuracy = torch.sum(torch.squeeze(pred[i]).float() == all_label).item() / float(all_label.size()[0])
        print('model:',i,'acc:' ,accuracy,' evi:',torch.max(all_evi_list[i]))
        x=sigma3(torch.sum(all_evi_list[i],1),3)
        #x=sigma3(torch.max(all_evi_list[i],1)[0],3)
       
    all_pse_list=dict()
    dd=dict()
    initc=dict()
    for i in range(net.source):
        _, pred[i]= torch.max(all_o_list[i], 1)
        all_pse_list[i],dd[i],initc[i]=cluster(nn.Softmax(1)(all_o_list[i]),all_f_list[i],all_label)
      
        acc = np.sum(all_pse_list[i] == all_label.float().numpy()) /(all_label.shape[0])
        # #pred_label_list[i]=pred_label_list[i].detach()
        # #initc[i]=initc[i].detach()
        # all_f_list[i]=all_f_list[i].detach()
        all_pse_list[i]=torch.from_numpy(all_pse_list[i].astype('int'))
    
        all_f_list[i]=all_f_list[i].float().cpu()#.numpy()
    return  all_o_list,all_evi_list,all_pse_list,all_f_list,all_label

  
def cluster(all_output,all_features,all_label):
    
    
    K = all_output.size(1)
    all_fea=all_features.cpu()
    
    # all_fea = torch.cat((all_fea, torch.ones(all_fea.size(0), 1)), 1)
    # all_fea = (all_fea.t() / torch.norm(all_fea, p=2, dim=1)).t()

    aff = (all_output).float().cpu().numpy()
    
    initc = aff.transpose().dot(all_fea)
    initc = initc / (1e-8 + aff.sum(axis=0)[:,None])
   
    # print(labelset)

    dd = cdist(all_fea, initc, args.distance)
    pred_label= dd.argmin(axis=1)
   
    
    for round in range(1):
        aff = np.eye(K)[pred_label]
        initc = aff.transpose().dot(all_features)
        initc = initc / (1e-8 + aff.sum(axis=0)[:,None])
        dd_fuse = cdist(all_features, initc, args.distance)

        pred_label = dd_fuse.argmin(axis=1)

    return pred_label,dd,initc



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='CAiDA')
    parser.add_argument('--gpu_id', type=str, nargs='?', default='0', help="device id to run")
    parser.add_argument('--t', type=int, default=0,
                        help="target")  ## Choose which domain to set as target {0 to len(names)-1}
    parser.add_argument('--max_epoch', type=int, default=10, help="max iterations")
    parser.add_argument('--interval', type=int, default=1)
    parser.add_argument('--batch_size', type=int, default=64, help="batch_size")
    parser.add_argument('--worker', type=int, default=4, help="number of workers")
    parser.add_argument('--dset', type=str, default='office-home', choices=['office31', 'office-home', 'office-caltech','domainnet', 'imagenetc', 'domainnet126'])
    parser.add_argument('--lr', type=float, default=1e-2, help="learning rate")
    parser.add_argument('--net', type=str, default='resnet50', help="vgg16, resnet50, res101")
    parser.add_argument('--seed', type=int, default=2023, help="random seed")
    parser.add_argument("--r", type=int, default=3)
    parser.add_argument('--gent', type=bool, default=True)
    parser.add_argument('--ent', type=bool, default=True)
    parser.add_argument('--threshold', type=int, default=0)
    parser.add_argument('--cls_par', type=float, default=0.7)
    parser.add_argument('--ent_par', type=float, default=1.0)
    parser.add_argument('--crc_par', type=float, default=1e-2)
    parser.add_argument('--lr_decay1', type=float, default=0.1)
    parser.add_argument('--lr_decay2', type=float, default=1.0)
    parser.add_argument('--ema', type=float, default=0.6)
    parser.add_argument('--bottleneck', type=int, default=256)
    parser.add_argument('--epsilon', type=float, default=1e-5)
    parser.add_argument('--layer', type=str, default="wn", choices=["linear", "wn"])
    parser.add_argument('--classifier', type=str, default="bn", choices=["ori", "bn"])
    parser.add_argument('--distance', type=str, default='cosine', choices=["euclidean", "cosine"])
    parser.add_argument('--output', type=str, default='ckps/v32')
    parser.add_argument('--output_src', type=str, default='ckps/source_mul_evi')
    parser.add_argument('--output_tar', type=str, default='ckps/target')
    args = parser.parse_args()

    args.dset = 'domainnet126'
    names = 'clipart, painting, real, sketch'.split(', ')
    print(names)
    args.class_num = 126

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    args.t_dset_path = '/home/yxue/datasets'

    args.src = ['painting', 'sketch']
    args.output_dir_src = []  # 源模型的位置
    for i in range(len(args.src)):
        args.output_dir_src.append(osp.join(args.output_src, args.dset, args.src[i]))
    print(args.output_dir_src)
    
    for k in [0]:
        args.t = k
        args.name_tar = names[args.t]
       
        args.output_dir_tar = []
        args.output_dir_src = []
        for i in range(len(args.src)):
            args.output_dir_tar.append(osp.join(args.output_tar, args.dset, args.src[i]+names[args.t]))
            args.output_dir_src.append(osp.join(args.output_src, args.dset, args.src[i]))
        print(args.output_dir_tar)
        print(args.output_dir_src)
        args.output_dir = osp.join(args.output, args.dset, names[args.t])

        if not osp.exists(args.output_dir):
            os.system('mkdir -p ' + args.output_dir)
        if not osp.exists(args.output_dir):
            os.mkdir(args.output_dir)

        args.savename = 'par_' + str(args.cls_par) + '_' + str(args.crc_par)

        LEN_SET_DomainNet126 = {
            'clipart':18523,
            'painting':30042,
            'real':69622,
            'sketch':24147,
        }

        for i in range(math.ceil(LEN_SET_DomainNet126[args.name_tar] / 50)):
            t1 = time.time()
            args.batch_idx = i
            acc = train_target(args)
            f = open(f'DomainNet126_{args.src}_target-{args.name_tar}_bs{args.batch_size}.txt', 'a')
            f.write(f'{str(acc)}\n')
            f.close()
            t2 = time.time()
            print(f'batch time: {t2-t1}')
