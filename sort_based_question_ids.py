"""
MILQT - "Multiple interaction learning with question-type prior knowledge for constraining answer search space in visual question answering."
Do, Tuong, Binh X. Nguyen, Huy Tran, Erman Tjiputra, Quang D. Tran, and Thanh-Toan Do.
Our arxiv link: https://arxiv.org/abs/2009.11118

This code is written by Vuong Pham and Tuong Do.
"""

import argparse
import numpy as np
import _pickle as pickle
import glob
import os
import torch
import torch.nn.functional as F

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out",
                        type=str,
                        required=True,
                        help="output file name")
    parser.add_argument("--res_dirs", nargs='+',
                        help="directories for results, NOTE:"
                             "all *.pkl file under these dirs will be ensembled",
                        default=None)
    parser.add_argument('--action', type=str, default='sort', choices=['sort', 'sort_softmax'], required=True,
                        help='the model we use')
    argments = parser.parse_args()

    return argments
# Softmax and non softmax prediction mapping accompanying with question id, answer list order sorted
if __name__ == '__main__':

    args = parse_args()
    result_dirs = args.res_dirs
    cnt = 0
    # Load models
    for res_dir in result_dirs:
        for file in glob.glob(res_dir + "/**/*.pkl", recursive=True):
            with open(file, 'rb') as f:
                print(file)

                cnt += 1
                sm = pickle.load(f)
                question_ids = np.array(pickle.load(f))
                ans_dic = pickle.load(f)
                # if the predictions only need to sort question id and answer list order
                if args.action == "sort":
                    # Sort based on question ids
                    print("\tSorting based on question ids...")
                    sorted_positions = question_ids.argsort() # Sort based on question ids and get new position of question ids in sorted array
                    sm = sm[sorted_positions]
                    question_ids = question_ids[sorted_positions]
                # if the predictions need softmax, apply the option below
                elif args.action == "sort_softmax":
                    # Sort based on dict
                    print("\tSorting based on dict...")
                    sorted_positions = np.argsort(ans_dic)
                    sm = sm[:, sorted_positions]
                    print("\tSoftmax...")
                    sm = torch.from_numpy(np.float32(sm))
                    sm = F.softmax(sm, 1)
                    sm = sm.numpy()
                    sm = np.float16(sm)
                print("\tDone.")
                # Save file
                out_file = args.out
                pkl_file = out_file + '/' + file.split('/')[-1].split('.')[0] + '.' + args.action + '.pkl'
                print("\tSaving to pickle file: " + pkl_file)
                if not os.path.isdir(out_file):
                    os.makedirs(out_file)
                with open(pkl_file, 'wb') as writeFile:
                    pickle.dump(sm, writeFile, protocol=4)
                    pickle.dump(question_ids, writeFile, protocol=4)
                    pickle.dump(np.sort(ans_dic), writeFile, protocol=4)

    print("Total %d models" % cnt)
    print("Done!")

