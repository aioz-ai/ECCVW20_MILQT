"""
MILQT - "Multiple interaction learning with question-type prior knowledge for constraining answer search space in visual question answering."
Do, Tuong, Binh X. Nguyen, Huy Tran, Erman Tjiputra, Quang D. Tran, and Thanh-Toan Do.
Our arxiv link: https://arxiv.org/abs/2009.11118

This code is written by Vuong Pham and Tuong Do.
"""

import torch
import torch.nn as nn

class MILQT(nn.Module):
    def __init__(self, qt_model, models, question_type_mapping, combination_operator='add'):
        super(MILQT, self).__init__()
        self.question_type_model = qt_model

        # initialize the models
        self.models = nn.ModuleList(model for model in models)
        self.num_models = len(models)
        self.combination_operator = combination_operator
        self.pred_combining_layer = nn.Linear(self.num_models, 1, bias=False)
        self.question_type_mapping = question_type_mapping
        self.features = [0]*self.num_models
        self.features_combined = [0]*self.num_models
        self.preds = [0]*len(self.models)
        self.preds_combined = [0]*self.num_models

    def forward(self, visuals, boxes, questions):
        # Do forward pass of every model
        question_emb = self.question_type_model(questions)  # b x 1024

        for idx, model in enumerate(self.models):
            self.features[idx] = model(visuals, boxes, questions)

        question_type_preds = self.question_type_model.classify(question_emb)
        for idx, model in enumerate(self.models):
            self.preds[idx] = model.classify(self.features[idx])

        for idx, model in enumerate(self.models):
            if self.combination_operator == 'add':
                self.features_combined[idx] = self.features[idx] + question_emb
            if self.combination_operator == 'mul':
                self.features_combined[idx] = self.features[idx] * question_emb

        for idx, model in enumerate(self.models):
            self.preds_combined[idx] = model.classify(self.features_combined[idx])

        # Do weight predictions of models
        model_preds = torch.cat([self.preds[idx].unsqueeze(2) for idx in range(self.num_models)], 2)
        model_preds = self.pred_combining_layer(model_preds).squeeze(2)

        model_preds_combined = torch.cat([self.preds_combined[idx].unsqueeze(2) for idx in range(self.num_models)], 2)
        model_preds_combined = self.pred_combining_layer(model_preds_combined).squeeze(2)

        # Get question type of questions of the current batch
        _, question_types = question_type_preds.max(1)

        # Weighting awareness
        mask = torch.zeros(model_preds.shape, device=model_preds.device)
        for i in range(len(question_types)):
            question_type = question_types[i]
            mask[i] = self.question_type_mapping[question_type]

        return self.preds_combined, model_preds, model_preds_combined, question_type_preds, mask
