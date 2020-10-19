"""
This code is modified from Hengyuan Hu's repository.
https://github.com/hengyuan-hu/bottom-up-attention-vqa
"""
import os
import argparse
import torch
from torch.utils.data import DataLoader, ConcatDataset

from dataset_VQA import Dictionary, VQAFeatureDataset, VisualGenomeFeatureDataset

import dataset_TDIUC
import base_model
from train import train
import utils

try:
    import _pickle as pickle
except:
    import pickle

def parse_args():
    parser = argparse.ArgumentParser()
    # MODIFIABLE MILQT HYPER-PARAMETERS--------------------------------------------------------------------------------
    # Model loading/saving
    parser.add_argument('--input', type=str, default=None,
                        help='input file directory for continue training from stop one')
    parser.add_argument('--output', type=str, default='saved_models/MILQT',
                        help='save file directory')

    # Utilities
    parser.add_argument('--seed', type=int, default=1204,
                        help='random seed')
    parser.add_argument('--epochs', type=int, default=13,
                        help='the number of epoches')
    parser.add_argument('--lr', default=7e-4, type=float, metavar='lr',
                        help='initial learning rate')

    # Gradient accumulation
    parser.add_argument('--batch_size', type=int, default=64,
                        help='batch size')
    parser.add_argument('--update_freq', default='4', metavar='N',
                        help='update parameters every n batches in an epoch')

    # Data
    parser.add_argument('--use_both', action='store_true',
                        help='use both train/val datasets to train?')
    parser.add_argument('--use_vg', action='store_true',
                        help='use visual genome dataset to train?')

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

    # CONSTANT HYPER-PARAMETERS (Advanced hyper-params for testing, experimenting or fine-tuning)------------------------
    # Utilities - support testing, gpu training or sampling
    parser.add_argument('--testing', action='store_true', default=False,
                        help='for fast testing 1 epoch')
    parser.add_argument('--print_interval', default=200, type=int, metavar='N',
                        help='print per certain number of steps')
    parser.add_argument('--gpu', type=int, default=0,
                        help='specify index of GPU using for training, to use CPU: -1')
    parser.add_argument('--clip_norm', default=.25, type=float, metavar='NORM',
                        help='clip threshold of gradients')
    parser.add_argument('--weight_init', type=str, default='none', choices=['none', 'kaiming_normal'],
                        help='dynamic weighting with Kaiming normalization')

    # Bounding box set
    parser.add_argument('--max_boxes', default=50, type=int, metavar='N',
                        help='number of maximum bounding boxes for K-adaptive')

    # Question embedding
    parser.add_argument('--question_len', default=12, type=int, metavar='N',
                        help='maximum length of input question')
    parser.add_argument('--tfidf', type=bool, default=True,
                        help='tfidf word embedding?')
    parser.add_argument('--op', type=str, default='c',
                        help='concatenated 600-D word embedding')

    # Joint representation C dimension
    parser.add_argument('--num_hid', type=int, default=1024,
                        help='dim of joint semantic features')

    # MILQT hyper-params
    parser.add_argument('--combination_operator', type=str, default='mul', choices=['add', 'mul'],
                        help='multi-level multi-model operation')
    parser.add_argument('--question_type_mapping', type=str, default='question_type_mapping_VQA.txt',
                        help='the path of question type mapping file .e.g. <question_type_mapping_VQA.txt> '
                             'or <question_type_mapping_TDIUC.txt>')
    parser.add_argument('--counter_act', type=str, default='zhang', choices=['zhang'],
                        help='the counter activation')
    parser.add_argument('--activation', type=str, default='swish', choices=['relu', 'swish'],
                        help='the activation to use for final classifier')
    parser.add_argument('--dropout', default=0.45, type=float, metavar='dropout',
                        help='dropout of rate of final classifier')

    # Weight of losses
    parser.add_argument('--g_ratio', default=.01, type=float,
                        help='influence the contribution of awareness weighting matrix')
    parser.add_argument('--q_ratio', default=.01, type=float,
                        help='influence of question-type loss')

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
    args = parse_args()

    utils.create_dir(args.output)
    logger = utils.Logger(os.path.join(args.output, 'log.txt'))
    logger.write(args.__repr__())

    device = torch.device("cuda:" + str(args.gpu) if args.gpu >= 0 else "cpu")
    args.device = device

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.backends.cudnn.benchmark = True

    if args.use_TDIUC:
        dictionary = dataset_TDIUC.Dictionary.load_from_file(os.path.join(args.TDIUC_dir, 'dictionary.pkl'))
        train_dset = dataset_TDIUC.VQAFeatureDataset('train', args, dictionary, adaptive=True, max_boxes=args.max_boxes,
                                       question_len=args.question_len)
        val_dset = dataset_TDIUC.VQAFeatureDataset('val', args, dictionary, adaptive=True, max_boxes=args.max_boxes, question_len=args.question_len)
    else:
        dictionary = Dictionary.load_from_file('data/dictionary.pkl')
        train_dset = VQAFeatureDataset('train', args, dictionary, adaptive=True, max_boxes=args.max_boxes, question_len=args.question_len)
        val_dset = VQAFeatureDataset('val', args, dictionary, adaptive=True, max_boxes=args.max_boxes, question_len=args.question_len)

    batch_size = args.batch_size

    constructor = 'build_%s' % args.model
    model = getattr(base_model, constructor)(train_dset, args)

    # Comment because do not use multi gpu
    # model = nn.DataParallel(model)
    optim = None
    epoch = 0
    # load snapshot
    if args.input is not None:
        print('loading %s' % args.input)
        model_data = torch.load(args.input)
        model.load_state_dict(model_data.get('model_state', model_data))
        model.to(device)
        optim = torch.optim.Adamax(filter(lambda p: p.requires_grad, model.parameters()))
        optim.load_state_dict(model_data.get('optimizer_state', model_data))
        epoch = model_data['epoch'] + 1

    if args.use_both: # use train & val splits to optimize
        if args.use_vg: # use a portion of Visual Genome dataset
            vg_dsets = [
                VisualGenomeFeatureDataset('train', train_dset.features, train_dset.spatials, dictionary, adaptive=True, pos_boxes=train_dset.pos_boxes, max_boxes=args.max_boxes, question_len=args.question_len),
                VisualGenomeFeatureDataset('val', val_dset.features, val_dset.spatials, dictionary, adaptive=True, pos_boxes=val_dset.pos_boxes, max_boxes=args.max_boxes, question_len=args.question_len)
            ]
            trainval_dset = ConcatDataset([train_dset, val_dset]+vg_dsets)
        else:
            trainval_dset = ConcatDataset([train_dset, val_dset])
        train_loader = DataLoader(trainval_dset, batch_size, shuffle=True, num_workers=0, collate_fn=utils.trim_collate, pin_memory=True)
        eval_loader = None
    else:
        if args.use_vg:
            vg_dsets = [
                VisualGenomeFeatureDataset('train', train_dset.features, train_dset.spatials, dictionary, adaptive=True, pos_boxes=train_dset.pos_boxes, max_boxes=args.max_boxes, question_len=args.question_len),
                VisualGenomeFeatureDataset('val', val_dset.features, val_dset.spatials, dictionary, adaptive=True, pos_boxes=val_dset.pos_boxes, max_boxes=args.max_boxes, question_len=args.question_len)
            ]
            train_dset = train_dset + vg_dsets[0]
            val_dset = val_dset + vg_dsets[1]
        train_loader = DataLoader(train_dset, batch_size, shuffle=True, num_workers=0, collate_fn=utils.trim_collate, pin_memory=True)
        eval_loader = DataLoader(val_dset, batch_size, shuffle=False, num_workers=0, collate_fn=utils.trim_collate, pin_memory=False)

    train(args, model, train_loader, eval_loader, args.epochs, args.output, optim, epoch)
