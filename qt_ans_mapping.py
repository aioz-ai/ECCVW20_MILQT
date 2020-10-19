"""
MILQT - "Multiple interaction learning with question-type prior knowledge for constraining answer search space in visual question answering."
Do, Tuong, Binh X. Nguyen, Huy Tran, Erman Tjiputra, Quang D. Tran, and Thanh-Toan Do.
Our arxiv link: https://arxiv.org/abs/2009.11118

This code is written by Huy Tran.
"""


import torch
from dataset_TDIUC import Dictionary, VQAFeatureDataset
# from dataset_VQA import Dictionary, VQAFeatureDataset
import _pickle as pickle

def statisticAns(masks, dset, field):
    st_masks = masks
    for i in dset.entries:
        if i['answer']['labels'] is None:
            continue
        for j in i['answer']['labels']:
            for k in range(0, len(i[field])):
                if i[field][k] == 1.0:
                    st_masks[k][int(j)] += 1
    return st_masks


def mapping(nums_qt, nums_ans, train_dset, val_dset, ithOther):
    init_masks = torch.zeros(nums_qt, nums_ans)
    train_masks = statisticAns(init_masks, train_dset, 'answer_type')
    # val_masks = statisticAns(init_masks, val_dset, 'answer_type')
    masks = train_masks #+ val_masks
    mapping = torch.argmax(masks, 0)
    check_visible = (0==(0==torch.sum(masks, 0)))
    for i in range(0,len(check_visible)):
        if (0==check_visible[i]):
            mapping[i] = ithOther
    return mapping


if __name__ == '__main__':
    dictionary = Dictionary.load_from_file('data/dictionary.pkl')
    train_dset = VQAFeatureDataset('train', dictionary, adaptive=True)
    val_dset = None#VQAFeatureDataset('val', dictionary, adaptive=True)
    mapping = mapping(12, 1479, train_dset, val_dset, 2)
    with open('question_type_mapping.txt', 'w') as outfile:
        for i in range(0, len(mapping)):
            outfile.write(str(str(i) + " " + str(int(mapping[i]))) + "\n")
    outfile.close()