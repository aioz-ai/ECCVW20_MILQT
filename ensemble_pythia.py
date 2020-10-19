"""

This code is modified from Pythia Facebook: https://github.com/facebookresearch/pythia
"""

import argparse
import numpy as np
import _pickle as pickle
import glob
from print_result import print_result

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
    argments = parser.parse_args()

    return argments

class answer_json:
    def __init__(self):
        self.answers = []

    def add(self, ques_id, ans):
        res = {
            "question_id": ques_id,
            "answer": ans
        }
        self.answers.append(res)

if __name__ == '__main__':
    f = open('question_type_mapping_VQA.txt','r')
    question_type_mapping = f.readlines()
    for i in range(len(question_type_mapping)):
        question_type_mapping[i] = int(question_type_mapping[i].split()[1])
    binary_mapping = np.zeros((max(question_type_mapping) + 1, len(question_type_mapping)))  # Initialize all-zero matrix with shape num_question_types x num candidate answers (3 x 3129)
    for i in range(len(question_type_mapping)):
        binary_mapping[question_type_mapping[i]][i] = 1

    args = parse_args()
    result_dirs = args.res_dirs
    out_file = args.out
    question_ids = None
    soft_max_result = None
    ans_dic = None
    cnt = 0
    for res_dir in result_dirs:
        for file in glob.glob(res_dir + "/**/*.pkl", recursive=True):
            with open(file, 'rb') as f:
                print(file)
                cnt += 1
                sm = pickle.load(f)
                if soft_max_result is None:
                    soft_max_result = sm
                    question_ids = pickle.load(f)
                    ans_dic = pickle.load(f)
                else:
                    soft_max_result += sm

    print("ensemble total %d models" % cnt)

    #predicted_answers = np.argmax(soft_max_result, axis=1)

    pkl_file = out_file + ".pkl"

    print_result(question_ids, soft_max_result, ans_dic, out_file, False, pkl_file)

    print("Done")
