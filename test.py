"""
This code is extended from Hengyuan Hu's repository.
https://github.com/hengyuan-hu/bottom-up-attention-vqa
"""
import argparse
import json
import progressbar
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.autograd import Variable
from print_result import print_result

from dataset_VQA import Dictionary, VQAFeatureDataset
import dataset_TDIUC
import base_model
import utils
import _pickle as pickle
import numpy as np
import os

def compute_score_with_logits(logits, labels):
    logits = torch.max(logits, 1)[1].data
    one_hots = torch.zeros(*labels.size()).to(logits.device)
    one_hots.scatter_(1, logits.view(-1, 1), 1)
    scores = (one_hots * labels)
    return scores

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ensemble', type=bool, default=False,
                        help='ensemble flag. If True, generate a logit file which is used in the ensemble part')
    # MODIFIABLE MILQT HYPER-PARAMETERS--------------------------------------------------------------------------------
    # Model loading/saving
    parser.add_argument('--split', type=str, default='test2015')
    parser.add_argument('--input', type=str, default='saved_models/MILQT',
                        help='input file directory for loading a model')
    parser.add_argument('--output', type=str, default='results/MILQT',
                        help='output file directory for saving VQA answer prediction file')
    # Utilities
    parser.add_argument('--epoch', type=int, default=12,
                        help='the best epoch')

    # Gradient accumulation
    parser.add_argument('--batch_size', type=int, default=64,
                        help='batch size')

    # Choices of models
    parser.add_argument('--model', type=str, default='MILQT', choices=['MILQT'],
                        help='the model we use')
    parser.add_argument('--comp_attns', type=str, default='BAN_COUNTER,BAN,SAN',
                        help='list of attention components. Note that, component attentions are seperated by commas, e.g. <BAN_COUNTER,BAN,SAN>.')

    # INTERACTION LEARNING COMPONENTS HYPER-PARAMETERS------------------------------------------------------------------
    # BAN
    parser.add_argument('--gamma', type=int, default=2,
                        help='glimpse in Bilinear Attention Networks')
    parser.add_argument('--use_counter', action='store_true', default=False,
                        help='use counter module')

    # Stacked Attention Networks
    parser.add_argument('--num_stacks', default=2, type=int,
                        help='num of stacks in Stack Attention Networks')

    #CONSTANT HYPER-PARAMETERS (Advanced hyper-params for testing, experimenting or fine-tuning)------------------------
    # Utilities - gpu
    parser.add_argument('--debug', type=bool, default=False)
    parser.add_argument('--gpu', type=int, default=0,
                        help='specify index of GPU using for training, to use CPU: -1')

    #Bounding box set
    parser.add_argument('--max_boxes', default=50, type=int, metavar='N',
                        help='number of maximum bounding boxes for K-adaptive')

    # Question embedding
    parser.add_argument('--op', type=str, default='c',
                        help='concatenated 600-D word embedding')

    # Joint representation C dimension
    parser.add_argument('--num_hid', type=int, default=1024,
                        help='dim of joint semantic features')

    # MILQT hyper-params
    parser.add_argument('--combination_operator', type=str, default='mul', choices=['add', 'mul'],
                        help='multi-level multi-model operation')
    parser.add_argument('--question_type_mapping', type=str, default='question_type_mapping_VQA.txt',
                        help='the path of question type mapping file')
    parser.add_argument('--counter_act', type=str, default='zhang', choices=['zhang'],
                        help='the counter activation')
    parser.add_argument('--activation', type=str, default='swish', choices=['relu', 'swish'],
                        help='the activation to use for final classifier')
    parser.add_argument('--dropout', default=0.45, type=float, metavar='dropout',
                        help='dropout of rate of final classifier')

    # Use MoD features
    parser.add_argument('--use_MoD', action='store_true', default=False,
                        help='Using MoD features')
    parser.add_argument('--MoD_dir', type=str,
                        help='MoD features dir')

    # Train with TDIUC
    parser.add_argument('--use_TDIUC', action='store_true', default=False,
                        help='Using TDIUC dataset to train')
    parser.add_argument('--TDIUC_dir', type=str,
                        help='TDIUC dir')

    # Return args
    args = parser.parse_args()
    return args

# Load questions
def get_question(q, dataloader):
    str = []
    dictionary = dataloader.dataset.dictionary
    for i in range(q.size(0)):
        str.append(dictionary.idx2word[q[i]] if q[i] < len(dictionary.idx2word) else '_')
    return ' '.join(str)

# Load answers
def get_answer(p, dataloader):
    _m, idx = p.max(0)
    return dataloader.dataset.label2ans[idx.item()]

# Logit computation (for train, test or evaluate)
def get_logits(model, dataloader, device, args):
    N = len(dataloader.dataset)
    M = dataloader.dataset.num_ans_candidates
    pred = torch.FloatTensor(N, M).zero_()
    if args.use_TDIUC:
        qt_pred = torch.IntTensor(N).zero_()
    qIds = torch.IntTensor(N).zero_()
    idx = 0
    bar = progressbar.ProgressBar(maxval=N)
    bar.start()
    with torch.no_grad():
        for v, b, q, i, qt in iter(dataloader):
            bar.update(idx)
            batch_size = v.size(0)
            v = v.to(device)
            b = b.to(device)
            q = q.to(device)
            if args.model == "MILQT":
                _, _, logits, question_type_preds, _ = model(v, b, q)
            qt_pre = torch.max(question_type_preds, 1)[1].data  # argmax
            if args.use_TDIUC:
                qt_pred[idx:idx+batch_size].copy_(qt_pre.data)
            pred[idx:idx+batch_size,:].copy_(logits.data)
            qIds[idx:idx+batch_size].copy_(i)
            idx += batch_size
            if args.debug:
                print(get_question(q.data[0], dataloader))
                print(get_answer(logits.data[0], dataloader))
    bar.update(idx)
    if args.use_TDIUC:
        return pred, qIds, qt_pred
    return pred, qIds

# Return results with three fields: question_id, question_type and answer
def make_json_with_qt(logits, qIds, qt, dataloader):
    utils.assert_eq(logits.size(0), len(qIds))
    results = []
    for i in range(logits.size(0)):
        result = {}
        result['question_id'] = int(qIds[i])
        result['question_type'] = int(qt[i])
        result['answer'] = get_answer(logits[i], dataloader)
        results.append(result)
    return results

# Return results with two fields: question_id and answer
def make_json(logits, qIds, dataloader):
    utils.assert_eq(logits.size(0), len(qIds))
    results = []
    for i in range(logits.size(0)):
        result = {}
        result['question_id'] = int(qIds[i])
        result['answer'] = get_answer(logits[i], dataloader)
        results.append(result)
    return results

# Test phase
if __name__ == '__main__':
    args = parse_args()
    print(args)
    torch.backends.cudnn.benchmark = True
    args.device = torch.device("cuda:" + str(args.gpu) if args.gpu >= 0 else "cpu")

    # Check if evaluating on TDIUC dataset or VQA dataset
    if args.use_TDIUC:
        dictionary = dataset_TDIUC.Dictionary.load_from_file(os.path.join(args.TDIUC_dir , 'dictionary.pkl'))
        eval_dset = dataset_TDIUC.VQAFeatureDataset(args.split, args, dictionary, adaptive=True)
    else:
        dictionary = Dictionary.load_from_file('data/dictionary.pkl')
        eval_dset = VQAFeatureDataset(args.split, args, dictionary, adaptive=True)

    n_device = torch.cuda.device_count()
    batch_size = args.batch_size * n_device

    constructor = 'build_%s' % args.model
    model = getattr(base_model, constructor)(eval_dset, args)
    print(model)
    eval_loader = DataLoader(eval_dset, batch_size, shuffle=False, num_workers=0, pin_memory=True, collate_fn=utils.trim_collate)

    # Testing process
    def process(args, model, eval_loader):
        model_path = args.input + '/model_epoch%s.pth' % args.epoch
        print('loading %s' % model_path)
        model_data = torch.load(model_path)

        # Comment because do not use multi gpu
        # model = nn.DataParallel(model)
        model = model.to(args.device)
        model.load_state_dict(model_data.get('model_state', model_data))

        model.train(False)
        if not os.path.exists(args.output):
            os.makedirs(args.output)
        out_file = args.output + '/' + args.input.split('/')[-1] + '.json'
        if args.use_TDIUC:
            logits, qIds, qt_pred = get_logits(model, eval_loader, args.device, args)
            results = make_json_with_qt(logits, qIds, qt_pred, eval_loader)
        else:
            logits, qIds = get_logits(model, eval_loader, args.device, args)
            if args.ensemble == True:
                pkl_res_file = args.output + '/' + args.input.split('/')[-1] + '.pkl'
                ans_dict = pickle.load(open('data/cache/trainval_label2ans.pkl','rb'))
                print_result(qIds.data.numpy(), np.float16(logits.data.numpy()), ans_dict, out_file, False, pkl_res_file)
                print("Done!")
                return
            else:
                results = make_json(logits, qIds, eval_loader)
        with open(out_file, 'w') as f:
            json.dump(results, f)
        print("Done!")
    process(args, model, eval_loader)
