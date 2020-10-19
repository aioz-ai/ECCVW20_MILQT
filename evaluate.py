"""
This code is modified from Hengyuan Hu's repository.
https://github.com/hengyuan-hu/bottom-up-attention-vqa
"""
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import os

from dataset_VQA import Dictionary, VQAFeatureDataset
import dataset_TDIUC
import base_model
from train import evaluate
import utils

def parse_args():
    parser = argparse.ArgumentParser()
    # MODIFIABLE MILQT HYPER-PARAMETERS--------------------------------------------------------------------------------
    # Model loading/saving
    parser.add_argument('--split', type=str, default='val')
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
    parser.add_argument('--question_type_mapping', type=str, default='question_type_mapping.txt',
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

if __name__ == '__main__':
    print('Evaluate a given model optimized by training split using validation split.')
    args = parse_args()
    print(args)
    torch.backends.cudnn.benchmark = True
    args.device = torch.device("cuda:" + str(args.gpu) if args.gpu >= 0 else "cpu")

    if args.use_TDIUC:
        dictionary = dataset_TDIUC.Dictionary.load_from_file(os.path.join(args.TDIUC_dir, 'dictionary.pkl'))
        eval_dset = dataset_TDIUC.VQAFeatureDataset(args.split, args, dictionary, adaptive=True)
    else:
        dictionary = Dictionary.load_from_file('data/dictionary.pkl')
        eval_dset = VQAFeatureDataset(args.split, args, dictionary, adaptive=True)

    n_device = torch.cuda.device_count()
    batch_size = args.batch_size * n_device

    constructor = 'build_%s' % args.model
    model = getattr(base_model, constructor)(eval_dset, args.num_hid, args.op, args.gamma)
    print(model)
    eval_loader = DataLoader(eval_dset, batch_size, shuffle=False, num_workers=1, collate_fn=utils.trim_collate)

    model_path = args.input + '/model_epoch%s.pth' % args.epoch
    print('loading %s' % model_path)
    model_data = torch.load(model_path)

    # Comment because do not use multi gpu
    # model = nn.DataParallel(model)
    model = model.to(args.device)
    model.load_state_dict(model_data.get('model_state', model_data))

    print("Evaluating...")
    model.train(False)
    eval_score, bound, eval_question_type_score, eval_question_type_upper_bound = evaluate(model, eval_loader, args)
    print('\teval score: %.2f (%.2f)' % (100 * eval_score, 100 * bound))
    print('\tqt_eval score: %.2f (%.2f)' % (100 * eval_question_type_score, 100 * eval_question_type_upper_bound))