"""
MILQT - "Multiple interaction learning with question-type prior knowledge for constraining answer search space in visual question answering."
Do, Tuong, Binh X. Nguyen, Huy Tran, Erman Tjiputra, Quang D. Tran, and Thanh-Toan Do.
Our arxiv link: https://arxiv.org/abs/2009.11118

This code is written by Huy Tran.
"""

import json
import _pickle as pickle
import numpy as np

class answer_json:
    def __init__(self):
        self.answers = []

    def add(self, ques_id, ans):
        res = {
            "question_id": int(ques_id),
            "answer": ans
        }
        self.answers.append(res)

# Print and save ensemble results in json file
def print_result(question_ids,
                 soft_max_result,
                 ans_dic,
                 out_file,
                 json_only=True,
                 pkl_res_file=None):
    predicted_answers = np.argmax(soft_max_result, axis=1)

    if not json_only:
        with open(pkl_res_file, 'wb') as writeFile:
            pickle.dump(soft_max_result, writeFile,protocol = 4)
            pickle.dump(question_ids, writeFile, protocol = 4)
            pickle.dump(ans_dic, writeFile, protocol = 4)

    ans_json_out = answer_json()
    for idx, pred_idx in enumerate(predicted_answers):
        question_id = question_ids[idx]
        try:
            pred_ans = ans_dic[pred_idx]
        except:
            pred_ans = ans_dic.word_list[pred_idx]
        ans_json_out.add(question_id, pred_ans)

    with open(out_file, "w") as f:
        json.dump(ans_json_out.answers, f)
