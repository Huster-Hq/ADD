import os
import torch
import sys
import argparse
import logging
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
from dataloader import CPCDataset
from tqdm import tqdm
from torch import Tensor
from torch.utils.tensorboard import SummaryWriter
from lib import resnet50, resnet50_w, embed_layer, PSR, refine_cams_with_bkg, SR_generation


def load_pretrained(teacher, student, scratch):
    stu_path ='./pretrained/' + str(opt.fold) + '_student.pth'
    tea_path = './pretrained/' + str(opt.fold) + '_teacher.pth'

    if not scratch:
        if os.path.exists(stu_path):
            student.load_state_dict(torch.load(stu_path, map_location=opt.device), strict=False)
            print('load stu_path compeleted')
        if os.path.exists(tea_path):
            teacher.load_state_dict(torch.load(tea_path, map_location=opt.device), strict=False)
            print('load teacher compeleted')
    return teacher, student


def get_ce_loss(img, label, network):
    pred,f_embed, f4= network(img)
    CEloss = ce_loss(pred, label)
    pred_label = torch.argmax(pred, dim=-1)
    acc = (pred_label == label).sum().float()
    return CEloss, acc, pred, f4, f_embed


def save_model(epoch):
    print('update model..')
    stu_path = opt.train_save+'/weights/student_model-{}.pth'.format(epoch)
    torch.save(student.state_dict(), stu_path)


def cam(fmaps,model_predict,cls_idx):
    score =torch.ones(size=(model_predict.shape[0],1))
    for pp in range(cls_idx.shape[0]):
        score[pp]=model_predict[pp, cls_idx[pp]]
    score = score.sum()
    weights = torch.autograd.grad(outputs=score, inputs=fmaps,create_graph=True)[0]
    weights = weights.mean(dim=(2, 3))
    grad_cam = (weights.view(*weights.shape, 1, 1) * fmaps.squeeze(0)).sum(1)

    def _normalize(cams: Tensor) -> Tensor:
        """CAM normalization"""
        cams.sub_(cams.flatten(start_dim=-2).min(-1).values.unsqueeze(-1).unsqueeze(-1))
        cams.div_(cams.flatten(start_dim=-2).max(-1).values.unsqueeze(-1).unsqueeze(-1))
        return cams
    
    B, H, W = grad_cam.shape
    grad_cam = _normalize(F.relu(grad_cam, inplace=True))
    grad_cam = grad_cam.view(B, 1, H, W)
    return grad_cam


# The key component: Alignment-free Dense Distillation (ADD) module (in Eq. 2.1)
def ADD(pseudo_label1, pseudo_label2, q, kv):
    # Get Semantic Relation Map (in Eq. 2.2)
    aff_mask = SR_generation(pseudo_label1, pseudo_label2, ignore_index=opt.ignore_index, confuse_value=0)
    k = kv
    v = kv.transpose(2,1)
    matmul_qk = torch.matmul(q.transpose(2,1), k)
    matmul_qk_= matmul_qk * aff_mask
    matmul_qk_= matmul_qk_.masked_fill(aff_mask == 0, -1e9)
    attmap = F.softmax(matmul_qk_, dim=-1)
    f_cam_att = torch.matmul(attmap, v)            
    loss_align=sim_loss(f_cam_att.transpose(1,2).flatten(1), q.flatten(1), torch.ones(1).to(opt.device))
    return loss_align


def train(teacher, student,embed_layer_, epochs=1000, is_test=True):
    optimizer_stu = torch.optim.Adam(student.parameters(), lr=1e-4, weight_decay=1e-8)
    optimizer_embed_layer=torch.optim.Adam(embed_layer_.parameters(), lr=1e-4, weight_decay=1e-8)
    psr = PSR(num_iter=10, dilations=[1,2,4,8])
    psr.to(opt.device)
    
    if is_test:
        phases = ('test',)
    else:
        phases = ('train', 'test')

    val_acc_best = 0
    space_loss = torch.zeros(1).to(opt.device)
    loss_ADD_1=torch.zeros(1).to(opt.device)
    loss_ADD_2=torch.zeros(1).to(opt.device)
    for epoch in range(1, epochs):
        for phase in iter(phases):
            if phase == 'train':
                teacher.eval()
                student.train()
                ldr = loader
                train_acc_num = 0
            else:
                teacher.eval()
                student.eval()
                ldr = val_loader
                val_acc_num = 0
            summary = []
            sample_num = 0
            for batch in tqdm(ldr):
                wli_img, nbi_img, label= \
                    batch[0].to(opt.device).float(), batch[1].to(opt.device).float(), batch[2].to(opt.device).long()
                pred_nbi,f4_tea = teacher(nbi_img)
                CE_loss, acc_wli, pred_wli, f4_stu,f_stu_att = get_ce_loss(wli_img, label, student)
                if phase == 'train':
                    # train student
                    optimizer_stu.zero_grad()
                    optimizer_embed_layer.zero_grad()
                    f_tea_att = embed_layer_(f4_tea.detach())
                    logit_loss = sim_loss(pred_nbi.detach(), pred_wli, torch.ones(1).to(opt.device))

                    if epoch>15:
                        cam1 = cam(f4_stu, pred_wli, label)
                        cam2 = cam(f4_tea, pred_nbi, label)

                        pseudo_label1 = refine_cams_with_bkg(psr, wli_img, cams=cam1, cfg=opt)
                        pseudo_label2 = refine_cams_with_bkg(psr, nbi_img, cams=cam2, cfg=opt)

                        loss_ADD_1 = ADD(pseudo_label1, pseudo_label2, f_tea_att, f_stu_att)
                        loss_ADD_2 = ADD(pseudo_label2, pseudo_label1, f_stu_att, f_tea_att)
                        space_loss = loss_ADD_1+loss_ADD_2

                    loss=CE_loss + space_loss +logit_loss
                    if not torch.isfinite(loss):
                        print('WARNING: non-finite loss, ending training ', loss)
                        sys.exit(1)

                    loss.backward()
                    optimizer_embed_layer.step()
                    optimizer_stu.step()
                    summary.append((loss.item(), CE_loss.item(), loss_ADD_1.item(),loss_ADD_2.item()))
                    # train_acc
                    train_acc_num = train_acc_num+acc_wli
                    sample_num += wli_img.shape[0]
                else:
                    sample_num += wli_img.shape[0]
                    val_acc_num = val_acc_num+acc_wli
  
            summary = np.array(summary).mean(axis=0)
            
            if phase == 'train':
                train_acc=train_acc_num / sample_num
                tags = ['acc_wli','train_total_loss','CE_loss', 'loss_ADD_1','loss_ADD_2']
                tb_writer.add_scalar(tags[0], train_acc, epoch)
                tb_writer.add_scalar(tags[1], summary[0], epoch)
                tb_writer.add_scalar(tags[2], summary[1], epoch)
                tb_writer.add_scalar(tags[3], summary[2], epoch)
                tb_writer.add_scalar(tags[4], summary[3], epoch)
                print('Epoch %d' % epoch,'acc: %0.2f, Total_loss: %0.2f, CE_loss: %0.2f, loss_ADD_1: %0.2f,loss_ADD_2: %0.2f' % (train_acc, summary[0], summary[1], summary[2], summary[3]))#
                save_model(epoch)
            else:
                val_acc=val_acc_num / sample_num
                logging.info('epoch: {},  test_acc: {}'.format(epoch, val_acc))
                if val_acc > val_acc_best and val_acc < train_acc:
                    val_acc_best = val_acc
                    print('##############################################################################best', val_acc_best)
                    logging.info('##############################################################################best:{}'.format(val_acc_best))
                print('[EVAL] Epoch %d' % epoch, 'val_acc: %0.2f, best_acc: %0.3f' % (val_acc, val_acc_best))
        

if __name__ == '__main__':
    for i in range(5):
        is_test = False
        parser = argparse.ArgumentParser()
        parser.add_argument('--train_save', type = str, default = './log/ADD/' + str(i))
        parser.add_argument('--fold', type = int, default = i)
        parser.add_argument('--batch_size', type = int, default = 16)   
        parser.add_argument('--epochs', type = int, default = 200)      
        parser.add_argument('--device', default = 'cuda:3', help = 'device id (i.e. 0 or 0,1 or cpu)')
        parser.add_argument("--high_thre", default = 0.7, type = float, help = "high_bkg_score")
        parser.add_argument("--low_thre", default = 0.3, type = float, help = "low_bkg_score")
        parser.add_argument("--bkg_thre", default = 0.5, type = float, help = "bkg_score")
        parser.add_argument("--ignore_index", default = 255, type = int, help = "random index")
        opt = parser.parse_args()
        
        if os.path.exists(opt.train_save + '/run/') is False:
            os.makedirs(opt.train_save + '/run/')
        if os.path.exists(opt.train_save + '/weights/') is False:
            os.makedirs(opt.train_save + '/weights/')    
        logging.basicConfig(filename=opt.train_save + '/train_log.log',
                    format='[%(asctime)s-%(filename)s-%(levelname)s:%(message)s]',
                    level=logging.INFO, filemode='a', datefmt='%Y-%m-%d %I:%M:%S %p')

        dataset = CPCDataset(is_train=True,split_id=opt.fold)
        val_dataset = CPCDataset(is_train=False,split_id=opt.fold)
        loader = DataLoader(dataset, batch_size=opt.batch_size, num_workers=opt.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=1, num_workers=opt.batch_size, shuffle=False)

        ce_loss = nn.CrossEntropyLoss()
        sim_loss = torch.nn.CosineEmbeddingLoss()
        
        student = resnet50_w(pretrained=False, num_classes=2).to(opt.device)
        teacher = resnet50(pretrained=False, num_classes=2).to(opt.device)
        embed_layer_=embed_layer().to(opt.device)
        teacher, student = load_pretrained(teacher, student, scratch=False)
        tb_writer = SummaryWriter(opt.train_save + '/run/')
        train(teacher, student,embed_layer_, is_test = is_test,epochs = opt.epochs)
