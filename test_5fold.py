import os
import csv
import torch
import argparse
import numpy as np
import torch.nn.functional as F
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from lib import resnet50_w
from dataloader import CPCDataset
from tqdm import tqdm
from prettytable import PrettyTable


class ConfusionMatrix(object):
    def __init__(self, num_classes: int, labels: list):
        self.matrix = np.zeros((num_classes, num_classes))
        self.num_classes = num_classes
        self.labels = labels

    def update(self, preds, labels):
        for p, t in zip(preds, labels):
            self.matrix[p, t] += 1

    def summary(self):
        # calculate accuracy
        sum_TP = 0
        n = np.sum(self.matrix)
        for i in range(self.num_classes):
            sum_TP += self.matrix[i, i]
        acc = sum_TP / n
        print("the model accuracy is ", acc)
		
        sum_po = 0
        sum_pe = 0
        eval_list=[]
        eval_list.append(["label", "Precision", "Recall", "Specificity","F1-score"])
        for i in range(len(self.matrix[0])):
            sum_po += self.matrix[i][i]
            row = np.sum(self.matrix[i, :])
            col = np.sum(self.matrix[:, i])
            sum_pe += row * col
        
        # precision, recall, specificity
        table = PrettyTable()
        table.field_names = [ "Precision", "Recall", "Specificity","F1-score"]
        for i in range(self.num_classes):
            TP = self.matrix[i, i]
            FP = np.sum(self.matrix[i, :]) - TP
            FN = np.sum(self.matrix[:, i]) - TP
            TN = np.sum(self.matrix) - TP - FP - FN

            Precision = round(TP / (TP + FP), 3) if TP + FP != 0 else 0.
            Recall = round(TP / (TP + FN), 3) if TP + FN != 0 else 0.
            Specificity = round(TN / (TN + FP), 3) if TN + FP != 0 else 0.
            F1_score=round(2*Precision*Recall / (Precision+Recall), 3)
            eval_list.append([self.labels[i], Precision, Recall, Specificity,F1_score])
            table.add_row([Precision, Recall, Specificity,F1_score])
        print(table)
        eval_list.append(['accuracy', acc])
        return str(acc), eval_list


def load_model(i, model, model_num):
    pth_path = os.path.join(opt.pth_path,str(i), 'weights', model_num)
    print(model.load_state_dict(torch.load(pth_path, map_location = opt.device), strict = False))
    return model

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_classes', type=int, default=2)
    parser.add_argument('--device', default='cuda:1', help='device id (i.e. 0 or 0,1 or cpu)')
    parser.add_argument('--pth_path', type=str, default='')
    parser.add_argument('--save_path', type=str, default='', help='save_path')
    opt = parser.parse_args()

    if not os.path.exists(opt.save_path):
        os.makedirs(opt.save_path)

    model = resnet50_w(pretrained = False, num_classes = 2).to(opt.device)

    all_predict_list = []
    all_label_list = []

    for i in range(5):
        if i == 0:
            model_num = 'poly_student_model-0.pth'
        elif i == 1:
            model_num = 'poly_student_model-0.pth'
        elif i == 2:
            model_num = 'poly_student_model-0.pth'
        elif i == 3:
            model_num = 'poly_student_model-0.pth'
        elif i == 4:
            model_num = 'poly_student_model-0.pth'

        model = load_model(i, model, model_num)
        val_dataset = CPCDataset(is_train = False,split_id = i)
        val_loader = torch.utils.data.DataLoader(val_dataset,
                                                batch_size = 1,
                                                shuffle = False,
                                                pin_memory = True,
                                                #  num_workers=nw
                                                )
        model.eval()
        for batch in tqdm(val_loader):
            wli_img, labels = batch[0].to(opt.device).float(), batch[2].to(opt.device).long()
            pred, _, _ = model(wli_img)
            all_label_list.extend(labels)
            pred_sigmoid = F.softmax(pred, dim = 1)
            all_predict_list.extend(pred_sigmoid[:,1].cpu().detach().numpy())

    all_label_list = torch.tensor(all_label_list)
    fpr, tpr, _ = roc_curve(all_label_list,all_predict_list, pos_label=1,drop_intermediate=False)#
    threshold = 0.5
    pred_classes = [1 if x >= threshold else 0 for x in all_predict_list]
    pred_classes = torch.tensor(pred_classes)
    confusion = ConfusionMatrix(num_classes = opt.num_classes, labels = ['0','1'])
    confusion.update(pred_classes, all_label_list)
    _,eval_list = confusion.summary()    
    
    # compute AUC
    AUC = auc(fpr, tpr)
    print('AUC:', AUC)
    eval_list.append(['AUC'])
    eval_list.append([AUC])   
    with open(opt.save_path + '/table.csv','w') as f3:
        writer = csv.writer(f3)
        writer.writerows(eval_list)

    # draw figure
    plt.figure(figsize=(10, 8))
    plt.plot(fpr, tpr, lw = 2,color = 'purple', label = 'ROC(AUC = %0.3f)' % (AUC))
    plt.plot([0, 1], [0, 1], '--', color = (0.6, 0.6, 0.6), label = 'Random guessing')
    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('False Positive Rate',fontsize = 15)
    plt.ylabel('True Positive Rate',fontsize = 15)
    plt.title('Receiver operating characteristic example')
    plt.legend(loc = "lower right",fontsize = 15)
    name = opt.save_path + '/average_ROC.png'
    plt.savefig(name)
    plt.close()