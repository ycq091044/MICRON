import torch
import torch.nn as nn
from sklearn.metrics import jaccard_score, roc_auc_score, precision_score, f1_score, average_precision_score
import numpy as np
import dill
import time
import argparse
from torch.nn import CrossEntropyLoss
from torch.optim import Adam
import os
import torch.nn.functional as F
from collections import defaultdict

import sys
sys.path.append("..")
from models import Retain
from util import llprint, multi_label_metric, ddi_rate_score, get_n_params

# os.environ["CUDA_VISIBLE_DEVICES"] = "1"

torch.manual_seed(1203)
model_name = 'Retain'
resume_path = 'Epoch_39_JA_0.4792_DDI_0.09041.model'

if not os.path.exists(os.path.join("saved", model_name)):
        os.makedirs(os.path.join("saved", model_name))

# Training settings
parser = argparse.ArgumentParser()
parser.add_argument('--Test', action='store_true', default=False, help="test mode")
parser.add_argument('--model_name', type=str, default=model_name, help="model name")
parser.add_argument('--resume_path', type=str, default=resume_path, help='resume path')
parser.add_argument('--lr', type=float, default=5e-4, help='learning rate')
parser.add_argument('--target_ddi', type=float, default=0.06, help='target ddi')
parser.add_argument('--kp', type=float, default=0.05, help='coefficient of P signal')
parser.add_argument('--dim', type=int, default=64, help='dimension')
parser.add_argument('--cuda', type=int, default=1, help='which cuda')
args = parser.parse_args()

# evaluate
def eval(model, data_eval, voc_size, epoch):
    model.eval()

    smm_record = []
    ja, prauc, avg_p, avg_r, avg_f1 = [[] for _ in range(5)]
    med_cnt, visit_cnt = 0, 0
    add_list, delete_list = [], []

    for step, input in enumerate(data_eval):
        y_gt, y_pred, y_pred_prob, y_pred_label = [], [], [], []
        if len(input) < 2: continue
        add_temp_list, delete_temp_list = [], []

        previous_set = input[0][2]
        for i in range(1, len(input)):
            target_output = model(input[:i])

            y_gt_tmp = np.zeros(voc_size[2])
            y_gt_tmp[input[i][2]] = 1
            y_gt.append(y_gt_tmp)

            # prediction prob
            target_output = F.sigmoid(target_output).detach().cpu().numpy()[0]
            y_pred_prob.append(target_output)

            # prediction med set
            y_pred_tmp = target_output.copy()
            y_pred_tmp[y_pred_tmp >= 0.3] = 1
            y_pred_tmp[y_pred_tmp < 0.3] = 0
            y_pred.append(y_pred_tmp)

            # prediction label
            y_pred_label_tmp = np.where(y_pred_tmp == 1)[0]
            y_pred_label.append(y_pred_label_tmp)
            med_cnt += len(y_pred_label_tmp)
            visit_cnt += 1

            #### add or delete
            add_gt = set(np.where(y_gt_tmp == 1)[0]) - set(previous_set)
            delete_gt = set(previous_set) - set(np.where(y_gt_tmp == 1)[0])

            add_pre = set(np.where(y_pred_tmp == 1)[0]) - set(previous_set)
            delete_pre = set(previous_set) - set(np.where(y_pred_tmp == 1)[0])
            
            add_distance = len(set(add_pre) - set(add_gt)) + len(set(add_gt) - set(add_pre))
            delete_distance = len(set(delete_pre) - set(delete_gt)) + len(set(delete_gt) - set(delete_pre))
            ####

            add_temp_list.append(add_distance )
            delete_temp_list.append(delete_distance)

            previous_set = y_pred_label_tmp

        if len(add_temp_list) > 1:
            add_list.append(np.mean(add_temp_list))
            delete_list.append(np.mean(delete_temp_list))
        elif len(add_temp_list) == 1:
            add_list.append(add_temp_list[0])
            delete_list.append(delete_temp_list[0])

        smm_record.append(y_pred_label)
        adm_ja, adm_prauc, adm_avg_p, adm_avg_r, adm_avg_f1 = multi_label_metric(np.array(y_gt), np.array(y_pred), np.array(y_pred_prob))
        
        ja.append(adm_ja)
        prauc.append(adm_prauc)
        avg_p.append(adm_avg_p)
        avg_r.append(adm_avg_r)
        avg_f1.append(adm_avg_f1)
        llprint('\rtest step: {} / {}'.format(step, len(data_eval)))

    # ddi rate
    ddi_rate = ddi_rate_score(smm_record, path='../data/output/ddi_A_final.pkl')

    llprint('\nDDI Rate: {:.4}, Jaccard: {:.4},  AVG_F1: {:.4}, Add: {:.4}, Delete: {:.4}, AVG_MED: {:.4}\n'.format(
        np.float(ddi_rate), np.mean(ja), np.mean(avg_f1), np.mean(add_list), np.mean(delete_list), med_cnt / visit_cnt
    ))

    return np.float(ddi_rate), np.mean(ja), np.mean(prauc), np.mean(avg_p), np.mean(avg_r), np.mean(avg_f1), np.mean(add_list), np.mean(delete_list), med_cnt / visit_cnt


def main():

    # load data
    data_path = '../data/output/records_final.pkl'
    voc_path = '../data/output/voc_final.pkl'
    device = torch.device('cuda:{}'.format(args.cuda))

    data = dill.load(open(data_path, 'rb'))
    voc = dill.load(open(voc_path, 'rb'))
    diag_voc, pro_voc, med_voc = voc['diag_voc'], voc['pro_voc'], voc['med_voc']

    np.random.seed(1203)
    np.random.shuffle(data)
    split_point = int(len(data) * 3 / 5)

    data_train = data[:split_point]
    eval_len = int(len(data[split_point:]) / 2)
    data_test = data[split_point:split_point + eval_len]
    data_eval = data[split_point+eval_len:]
    voc_size = (len(diag_voc.idx2word), len(pro_voc.idx2word), len(med_voc.idx2word))

    model = Retain(voc_size, device=device)
    # model.load_state_dict(torch.load(open(os.path.join("saved", args.model_name, args.resume_path), 'rb')))

    if args.Test:
        model.load_state_dict(torch.load(open(os.path.join("saved", args.model_name, args.resume_path), 'rb')))
        model.to(device=device)
        tic = time.time()
        eval(model, data_test, voc_size, 0)
        print ('test time: {}'.format(time.time() - tic))
        return 

    print('parameters', get_n_params(model))
    optimizer = Adam(model.parameters(), args.lr)
    model.to(device=device)

    history = defaultdict(list)
    best_epoch, best_ja = 0, 0

    EPOCH = 40
    for epoch in range(EPOCH):
        tic = time.time()
        print ('\nepoch {} --------------------------'.format(epoch + 1))
        model.train()
        for step, input in enumerate(data_train):
            if len(input) < 2: continue
            loss = 0
            for i in range(1, len(input)):
                target = np.zeros((1, voc_size[2]))
                target[:, input[i][2]] = 1

                output_logits = model(input[:i])
                loss += F.binary_cross_entropy_with_logits(output_logits, torch.FloatTensor(target).to(device))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            llprint('\rtraining step: {} / {}'.format(step, len(data_train)))

        print ()
        tic2 = time.time() 
        ddi_rate, ja, prauc, avg_p, avg_r, avg_f1, avg_add, avg_delete, avg_med = eval(model, data_eval, voc_size, epoch)
        print ('training time: {}, test time: {}'.format(time.time() - tic, time.time() - tic2))

        history['ja'].append(ja)
        history['ddi_rate'].append(ddi_rate)
        history['avg_p'].append(avg_p)
        history['avg_r'].append(avg_r)
        history['avg_f1'].append(avg_f1)
        history['avg_add'].append(avg_add)
        history['avg_delete'].append(avg_delete)
        history['prauc'].append(prauc)
        history['med'].append(avg_med)

        if epoch >= 5:
            print ('ddi: {}, Med: {}, Ja: {}, F1: {}, Add: {}, Delete: {}'.format(
                np.mean(history['ddi_rate'][-5:]),
                np.mean(history['med'][-5:]),
                np.mean(history['ja'][-5:]),
                np.mean(history['avg_f1'][-5:]),
                np.mean(history['avg_add'][-5:]),
                np.mean(history['avg_delete'][-5:]),
                np.mean(history['prauc'][-5:])
                ))

        torch.save(model.state_dict(), open(os.path.join('saved', args.model_name, \
            'Epoch_{}_JA_{:.4}_DDI_{:.4}.model'.format(epoch, ja, ddi_rate)), 'wb'))

        if epoch != 0 and best_ja < ja:
            best_epoch = epoch
            best_ja = ja

        print ('best_epoch: {}'.format(best_epoch))

    dill.dump(history, open(os.path.join('saved', args.model_name, 'history_{}.pkl'.format(args.model_name)), 'wb'))


if __name__ == '__main__':
    main()
