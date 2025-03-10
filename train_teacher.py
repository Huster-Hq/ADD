import os
import torch
import argparse
import logging
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
from dataloader import CPCDataset
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from lib import resnet50


def get_ce_loss(img, label, network):
    pred,_ = network(img)
    errCE = ce_loss(pred, label)
    pred_label = torch.argmax(pred, dim=-1)
    acc = (pred_label == label).sum().float()
    return errCE, acc


def save_model(epoch):
    print('update model..')
    tea_path = opt.train_save+'/weights/teacher_model-{}.pth'.format(epoch)
    torch.save(teacher.state_dict(), tea_path)


def train(teacher,  epochs=400, is_test=True):
    optimizer_tea = torch.optim.Adam(teacher.parameters(), lr=1e-4, weight_decay=1e-8)
    val_acc_best = 0

    if is_test:
        phases = ('test',)
    else:
        phases = ('train', 'test')

    for epoch in range(1, epochs):
        for phase in iter(phases):
            if phase == 'train':
                teacher.train()
                ldr = loader
                train_acc_num=0
            else:
                teacher.eval()
                ldr = val_loader
                val_acc_num=0
            summary = []
            sample_num = 0
            for batch in tqdm(ldr):
                nbi_img, label= \
                     batch[1].to(opt.device).float(), batch[2].to(opt.device).long()
                optimizer_tea.zero_grad()

                CE_loss, acc = get_ce_loss(nbi_img, label, teacher)
                if phase == 'train':
                    CE_loss.backward()
                    optimizer_tea.step()
                    train_acc_num=train_acc_num+acc
                    sample_num += nbi_img.shape[0]
                else:
                    sample_num += nbi_img.shape[0]
                    val_acc_num=val_acc_num+acc
                summary.append((CE_loss.item()))
            summary = np.array(summary).mean(axis=0)

            if phase == 'train':
                train_acc=train_acc_num / sample_num
                print('Epoch %d' % epoch, 'CE_loss: %0.2f, acc: %0.2f' % (summary, train_acc))
                tags = ['train_total_loss',  'acc']
                tb_writer.add_scalar(tags[0], summary, epoch)
                tb_writer.add_scalar(tags[1], train_acc, epoch)
                save_model(epoch)
            else:
                val_acc=val_acc_num / sample_num
                logging.info('epoch: {},  test_acc: {}'.format(epoch, val_acc))
                if val_acc > val_acc_best and val_acc<=train_acc:
                    val_acc_best = val_acc
                    print('##############################################################################best', val_acc_best)
                    logging.info('##############################################################################best:{}'.format(val_acc_best))
                print('[EVAL] Epoch %d' % epoch, 'val_acc: %0.2f, best_acc: %0.3f' % (val_acc, val_acc_best))


if __name__ == '__main__':
    for i in range(5):
        is_test = False
        parser = argparse.ArgumentParser()
        parser.add_argument('--train_save', type=str, default='./log/teacher/'+str(i))
        parser.add_argument('--fold', type=int, default=i)
        parser.add_argument('--batch_size', type=int, default=16)      
        parser.add_argument('--epochs', type=int, default=200)               
        parser.add_argument('--device', default='cuda:3', help='device id (i.e. 0 or 0,1 or cpu)')
        opt = parser.parse_args()
        
        dataset = CPCDataset(is_train=True, split_id=opt.fold)
        val_dataset = CPCDataset(is_train=False, split_id=opt.fold)
        loader = DataLoader(dataset, batch_size = opt.batch_size, num_workers = opt.batch_size, shuffle = True)
        val_loader = DataLoader(val_dataset, batch_size = 1, num_workers = opt.batch_size, shuffle = False)

        ce_loss = nn.CrossEntropyLoss()

        if os.path.exists(opt.train_save + '/run/') is False:
            os.makedirs(opt.train_save + '/run/')
        if os.path.exists(opt.train_save + '/weights/') is False:
            os.makedirs(opt.train_save + '/weights/')

        logging.basicConfig(filename=opt.train_save+'/train_log.log',
            format='[%(asctime)s-%(filename)s-%(levelname)s:%(message)s]',
            level=logging.INFO, filemode='a', datefmt='%Y-%m-%d %I:%M:%S %p')
        tb_writer = SummaryWriter(opt.train_save+'/run/')

        teacher = resnet50(pretrained=True, num_classes=2).to(opt.device)
        train(teacher, is_test=is_test, epochs=opt.epochs)

