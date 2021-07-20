import torch
import torch.nn as nn
import argparse
from sklearn.metrics import jaccard_score, roc_auc_score, precision_score, f1_score, average_precision_score
import numpy as np
import dill
import time
from torch.nn import CrossEntropyLoss
from torch.optim import Adam
import os
import torch.nn.functional as F
import random
from collections import defaultdict

import sys
sys.path.append("..")
from models import Leap
from util import llprint, sequence_metric, sequence_output_process, ddi_rate_score, get_n_params

torch.manual_seed(1203)
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"

model_name = 'Leap'
resume_path = 'xxx'

if not os.path.exists(os.path.join("saved", model_name)):
        os.makedirs(os.path.join("saved", model_name))

# Training settings
parser = argparse.ArgumentParser()
parser.add_argument('--Test', action='store_true', default=False, help="test mode")
parser.add_argument('--model_name', type=str, default=model_name, help="model name")
parser.add_argument('--resume_path', type=str, default=resume_path, help='resume path')
parser.add_argument('--lr', type=float, default=0.0005, help='learning rate')

args = parser.parse_args()

# evaluate
def eval(model, data_eval, voc_size, epoch):
    model.eval()

    ja, prauc, avg_p, avg_r, avg_f1 = [[] for _ in range(5)]
    smm_record = []
    med_cnt, visit_cnt = 0, 0
    add_list, delete_list = [], []

    for step, input in enumerate(data_eval):
        y_gt, y_pred, y_pred_prob, y_pred_label = [], [], [], []
        if len(input) < 2: continue
        add_temp_list, delete_temp_list = [], []

        for adm_idx, adm in enumerate(input):
            if adm_idx == 0: 
                previous_set = adm[2] 
                continue
            output_logits = model(adm)

            y_gt_tmp = np.zeros(voc_size[2])
            y_gt_tmp[adm[2]] = 1
            y_gt.append(y_gt_tmp)

            # prediction prod
            output_logits = output_logits.detach().cpu().numpy()

            # prediction med set
            out_list, sorted_predict = sequence_output_process(output_logits, [voc_size[2], voc_size[2]+1])
            y_pred_label.append(sorted(sorted_predict))
            y_pred_prob.append(np.mean(output_logits[:, :-2], axis=0))

            # prediction label
            y_pred_tmp = np.zeros(voc_size[2])
            y_pred_tmp[out_list] = 1
            y_pred.append(y_pred_tmp)
            visit_cnt += 1
            med_cnt += len(sorted_predict)

            #### add or delete
            add_gt = set(np.where(y_gt_tmp == 1)[0]) - set(previous_set)
            delete_gt = set(previous_set) - set(np.where(y_gt_tmp == 1)[0])

            add_pre = set(np.where(y_pred_tmp == 1)[0]) - set(previous_set)
            delete_pre = set(previous_set) - set(np.where(y_pred_tmp == 1)[0])
            
            add_distance = len(set(add_pre) - set(add_gt)) + len(set(add_gt) - set(add_pre))
            delete_distance = len(set(delete_pre) - set(delete_gt)) + len(set(delete_gt) - set(delete_pre))
            ####

            add_temp_list.append(add_distance)
            delete_temp_list.append(delete_distance)

            previous_temp_set = out_list

        if len(add_temp_list) > 1:
            add_list.append(np.mean(add_temp_list))
            delete_list.append(np.mean(delete_temp_list))
        else:
            add_list.append(add_temp_list[0])
            delete_list.append(delete_temp_list[0])

        smm_record.append(y_pred_label)

        adm_ja, adm_prauc, adm_avg_p, adm_avg_r, adm_avg_f1 = \
                sequence_metric(np.array(y_gt), np.array(y_pred), np.array(y_pred_prob), np.array(y_pred_label))
        ja.append(adm_ja)
        prauc.append(adm_prauc)
        avg_p.append(adm_avg_p)
        avg_r.append(adm_avg_r)
        avg_f1.append(adm_avg_f1)
        llprint('\rtest step: {} / {}'.format(step, len(data_eval)))

    # ddi rate
    ddi_rate = ddi_rate_score(smm_record, path='../data/ddi_A_final.pkl')

    llprint('\nDDI Rate: {:.4}, Jaccard: {:.4},  AVG_F1: {:.4}, Add: {:.4}, Delete: {:.4}, AVG_MED: {:.4}\n'.format(
        np.float(ddi_rate), np.mean(ja), np.mean(avg_f1), np.mean(add_list), np.mean(delete_list), med_cnt / visit_cnt
    ))

    return np.float(ddi_rate), np.mean(ja), np.mean(prauc), np.mean(avg_p), np.mean(avg_r), np.mean(avg_f1), np.mean(add_list), np.mean(delete_list), med_cnt / visit_cnt

def main():

    # load data
    data_path = '../data/records_final.pkl'
    voc_path = '../data/voc_final.pkl'

    ddi_adj_path = '../data/ddi_A_final.pkl'
    device = torch.device('cuda')

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

    END_TOKEN = voc_size[2] + 1

    model = Leap(voc_size, device=device)

    if args.Test:
        model.load_state_dict(torch.load(open(args.resume_path, 'rb')))
        model.to(device=device)
        tic = time.time()
        eval(model, data_test, voc_size, 0)
        print ('test time: {}'.format(time.time() - tic))
        return 

    model.to(device=device)
    print('parameters', get_n_params(model))
    optimizer = Adam(model.parameters(), lr=args.lr)

    history = defaultdict(list)
    best_epoch, best_ja = 0, 0

    EPOCH = 40
    for epoch in range(EPOCH):
        tic = time.time()
        print ('\nepoch {} --------------------------'.format(epoch + 1))

        model.train()
        for step, input in enumerate(data_train):
            loss = 0
            if len(input) < 2: continue
            for idx, adm in enumerate(input):
                if idx == 0: continue
                loss_target = adm[2] + [END_TOKEN]
                output_logits = model(adm)
                loss += F.cross_entropy(output_logits, torch.LongTensor(loss_target).to(device))
            optimizer.zero_grad()
            loss.backward(retain_graph=True)
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
            print ('ddi: {}, Med: {}, Ja: {}, F1: {}'.format(
                np.mean(history['ddi_rate'][-5:]),
                np.mean(history['med'][-5:]),
                np.mean(history['ja'][-5:]),
                np.mean(history['avg_f1'][-5:]),
                np.mean(history['avg_add'][-5:]),
                np.mean(history['avg_delete'][-5:])
                ))

        torch.save(model.state_dict(), open(os.path.join('saved', args.model_name, \
            'Epoch_{}_JA_{:.4}_DDI_{:.4}.model'.format(epoch, ja, ddi_rate)), 'wb'))

        if epoch != 0 and best_ja < ja:
            best_epoch = epoch
            best_ja = ja

        print ('best_epoch: {}'.format(best_epoch))

        dill.dump(history, open(os.path.join('saved', args.model_name, 'history_{}.pkl'.format(args.model_name)), 'wb'))


def fine_tune(fine_tune_name=''):

    # load data
    data_path = '../data/records_final.pkl'
    voc_path = '../data/voc_final.pkl'
    device = torch.device('cpu:0')

    data = dill.load(open(data_path, 'rb'))
    voc = dill.load(open(voc_path, 'rb'))
    diag_voc, pro_voc, med_voc = voc['diag_voc'], voc['pro_voc'], voc['med_voc']
    ddi_A = dill.load(open('../data/ddi_A_final.pkl', 'rb'))

    split_point = int(len(data) * 2 / 3)
    data_train = data[:split_point]
    eval_len = int(len(data[split_point:]) / 2)
    data_test = data[split_point:split_point + eval_len]
    # data_eval = data[split_point+eval_len:]
    voc_size = (len(diag_voc.idx2word), len(pro_voc.idx2word), len(med_voc.idx2word))

    model = Leap(voc_size, device=device)
    model.load_state_dict(torch.load(open(os.path.join("saved", args.model_name, fine_tune_name), 'rb')))
    model.to(device)

    END_TOKEN = voc_size[2] + 1

    optimizer = Adam(model.parameters(), lr=args.lr)
    ddi_rate_record = []

    EPOCH = 40
    for epoch in range(EPOCH):
        loss_record = []
        start_time = time.time()
        random_train_set = [random.choice(data_train) for i in range(len(data_train))]
        for step, input in enumerate(random_train_set):
            if len(input) < 3: continue
            model.train()
            K_flag = False
            for idx, adm in enumerate(input):
                if idx == 0: continue
                target = adm[2]
                output_logits = model(adm)
                out_list, sorted_predict = sequence_output_process(output_logits.detach().cpu().numpy(), [voc_size[2], voc_size[2] + 1])

                inter = set(out_list) & set(target)
                union = set(out_list) | set(target)
                jaccard = 0 if union == 0 else len(inter) / len(union)
                K = 0
                for i in out_list:
                    if K == 1:
                        K_flag = True
                        break
                    for j in out_list:
                        if ddi_A[i][j] == 1:
                            K = 1
                            break

                loss = -jaccard * K * torch.mean(F.log_softmax(output_logits, dim=-1))
                loss_record.append(loss.item())
                optimizer.zero_grad()
                loss.backward(retain_graph=True)
                optimizer.step()

            llprint('\rtraining step: {} / {}'.format(step, len(random_train_set)))

        if K_flag:
            print ()
            ddi_rate, ja, prauc, avg_p, avg_r, avg_f1, avg_med = eval(model, data_test, voc_size, epoch)

    # test
    torch.save(model.state_dict(), open(
        os.path.join('saved', args.model_name, 'final.model'), 'wb'))


if __name__ == '__main__':
    main()
    # fine_tune(fine_tune_name='Epoch_1_JA_0.2765_DDI_0.1158.model')
