"""
This code is modified from Jin-Hwa Kim, Jaehyun Jun, Byoung-Tak Zhang's repository.
https://github.com/jnhwkim/ban-vqa
"""

import torch
import torch.nn as nn
from attention import BiAttention, StackedAttention
from language_model import WordEmbedding, QuestionEmbedding
from classifier import SimpleClassifier
from fc import FCNet
from bc import BCNet
from counting import Counter
from MILQT import MILQT
from utils import tfidf_loading

# Create BAN model
class BAN_Model(nn.Module):
    def __init__(self, dataset, w_emb, q_emb, v_att, b_net, q_prj, c_prj, classifier, counter, args):
        super(BAN_Model, self).__init__()
        self.dataset = dataset
        self.op = args.op
        self.glimpse = args.gamma
        self.w_emb = w_emb
        self.q_emb = q_emb
        self.v_att = v_att
        self.b_net = nn.ModuleList(b_net)
        self.q_prj = nn.ModuleList(q_prj)
        if counter is not None:  # if do not use counter
            self.c_prj = nn.ModuleList(c_prj)
        self.classifier = classifier
        self.counter = counter
        self.drop = nn.Dropout(.5)
        self.tanh = nn.Tanh()

    def forward(self, v, b, q):
        """Forward

        v: [batch, num_objs, obj_dim]
        b: [batch, num_objs, b_dim]
        q: [batch_size, seq_length]

        return: logits, not probs
        """
        w_emb = self.w_emb(q)
        q_emb = self.q_emb.forward_all(w_emb) # [batch, q_len, q_dim]
        if self.counter is not None:
            boxes = b[:,:,:4].transpose(1,2)

        b_emb = [0] * self.glimpse
        att, logits = self.v_att.forward_all(v, q_emb) # b x g x v x q

        for g in range(self.glimpse):
            b_emb[g] = self.b_net[g].forward_with_weights(v, q_emb, att[:,g,:,:]) # b x l x h
            
            atten, _ = logits[:,g,:,:].max(2)
            if self.counter is not None:
                embed = self.counter(boxes, atten)

            q_emb = self.q_prj[g](b_emb[g].unsqueeze(1)) + q_emb
            if self.counter is not None:
                q_emb = q_emb + self.c_prj[g](embed).unsqueeze(1)

        return q_emb.sum(1)

    def classify(self, input_feats):
        return self.classifier(input_feats)

# Create SAN model
class SAN_Model(nn.Module):
    def __init__(self, w_emb, q_emb, v_att, classifier):
        super(SAN_Model, self).__init__()
        self.w_emb = w_emb
        self.q_emb = q_emb
        self.v_att = v_att
        self.classifier = classifier

    def forward(self, v, b, q):
        """Forward

        v: [batch, num_objs, obj_dim]
        b: [batch, num_objs, b_dim]
        q: [batch_size, seq_length]

        return: logits, not probs
        """
        w_emb = self.w_emb(q)
        q_emb = self.q_emb(w_emb)  # [batch, q_dim], return final hidden state

        att = self.v_att(v, q_emb)

        return att

    def classify(self, input_feats):
        return self.classifier(input_feats)

# Create question-type classification model
class QuestionType_Model(nn.Module):

    def __init__(self, w_emb, q_emb, classifier):
        super(QuestionType_Model, self).__init__()
        self.w_emb = w_emb
        self.q_emb = q_emb
        self.classifier = classifier

    def forward(self, q):
        w_emb = self.w_emb(q)
        q_emb = self.q_emb.forward_all(w_emb)

        return q_emb[:, -1, :]

    def classify(self, input_feats):
        return self.classifier(input_feats)

# Build BAN model
def build_BAN(dataset, args, priotize_using_counter=False):
    w_emb = WordEmbedding(dataset.dictionary.ntoken, 300, .0, args.op)
    q_emb = QuestionEmbedding(300 if 'c' not in args.op else 600, args.num_hid, 1, False, .0)
    v_att = BiAttention(dataset.v_dim, args.num_hid, args.num_hid, args.gamma)
    b_net = []
    q_prj = []
    c_prj = []

    # Loading tfidf weighted embedding
    if hasattr(args, 'tfidf'):
        w_emb = tfidf_loading(args.tfidf, w_emb, args)

    use_counter = args.use_counter if priotize_using_counter is None else priotize_using_counter

    if use_counter or priotize_using_counter:
        objects = 10  # minimum number of boxes
    for i in range(args.gamma):
        b_net.append(BCNet(dataset.v_dim, args.num_hid, args.num_hid, None, k=1))
        q_prj.append(FCNet([args.num_hid, args.num_hid], '', .2))
        if use_counter or priotize_using_counter:
            c_prj.append(FCNet([objects + 1, args.num_hid], 'ReLU', .0))
    classifier = SimpleClassifier(
        args.num_hid, args.num_hid * 2, dataset.num_ans_candidates, args)
    if use_counter or priotize_using_counter:
        counter = Counter(objects, counter_activation=args.counter_act)
    else:
        counter = None
    return BAN_Model(dataset, w_emb, q_emb, v_att, b_net, q_prj, c_prj, classifier, counter, args)

# Build BAN model with Counter sub-module
def build_BAN_COUNTER(dataset, args):
    return build_BAN(dataset, args, priotize_using_counter=True)

# Build SAN model
def build_SAN(dataset, args):
    w_emb = WordEmbedding(dataset.dictionary.ntoken, 300, 0.0, args.op)
    q_emb = QuestionEmbedding(300 if 'c' not in args.op else 600, args.num_hid, 1, False, 0.0)
    v_att = StackedAttention(args.num_stacks, dataset.v_dim, args.num_hid, args.num_hid, dataset.num_ans_candidates,
                             args.dropout)

    # Loading tfidf weighted embedding
    if hasattr(args, 'tfidf'):
        w_emb = tfidf_loading(args.tfidf, w_emb, args)

    classifier = SimpleClassifier(
        args.num_hid, 2 * args.num_hid, dataset.num_ans_candidates, args)
    return SAN_Model(w_emb, q_emb, v_att, classifier)

# Build question-type classification model
def build_question_type(dataset, args):
    w_emb = WordEmbedding(dataset.dictionary.ntoken, 300, 0.0, args.op)
    q_emb = QuestionEmbedding(300 if 'c' not in args.op else 600, args.num_hid, 1, False, 0.0)

    # Loading tfidf weighted embedding
    if hasattr(args, 'tfidf'):
        w_emb = tfidf_loading(args.tfidf, w_emb, args)

    classifier = nn.Linear(args.num_hid, dataset.num_qts_candidates)

    return QuestionType_Model(w_emb, q_emb, classifier)

# Build multi-hypothesis interaction model
def build_comp_attns(dataset, args):
    list_comp_names = args.comp_attns.replace(" ", "").split(",")
    comp_models = []
    for name in list_comp_names:
        constructor = 'build_%s' % name
        comp_models.append(globals()[constructor](dataset, args))
    return comp_models

# Build MILQT
def build_MILQT(dataset, args):
    # Build models
    question_type_model = build_question_type(dataset, args)
    models = build_comp_attns(dataset, args)
    # Get question type mapping
    # Read from file, get path from argument
    with open(args.question_type_mapping, 'r') as f:
        question_type_mapping = f.readlines()

    for i in range(len(question_type_mapping)):
        question_type_mapping[i] = int(question_type_mapping[i].split()[1])

    binary_mapping = torch.zeros((max(question_type_mapping) + 1, len(question_type_mapping)))  # Initialize all-zero matrix with shape num_question_types x num candidate answers (3 x 3129)
    for i in range(len(question_type_mapping)):
        binary_mapping[question_type_mapping[i]][i] = 1

    # Return a MILQT model
    return MILQT(question_type_model, models, binary_mapping, args.combination_operator)
