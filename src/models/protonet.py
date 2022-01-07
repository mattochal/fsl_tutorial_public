from models.model_template import ModelTemplate

import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import os
import sys

class ProtoNet(ModelTemplate):
    
    def __init__(self, backbone, args, device):
        super().__init__(backbone, args, device)
        
    def setup_model(self):
        """
            Runs once, imidiately after object initialization
            This method sets up the loss function and any additional backbone settings
        """
        super().setup_model()
        self.loss_fn = nn.CrossEntropyLoss().to(self.device)
        self.prototypes = dict()
        
    def net_reset(self):
        """
            Resets any parameters between tasks
        """
        self.prototypes = dict()
        
    def net_train(self, support_set):
        """
            Inner-loop training.
            This is where the task adaptation / fine-tuning of the support set happens.
            In the case of ProtoNet, there is no strict 'inner-loop' optimization process.
            Instead, the learning process happens by calculating the prototypes.
        """
        supports_x, supports_y = support_set
        supports_h = self.backbone(supports_x)
        new_proto_h, new_proto_y = self.calc_prototypes(supports_h, supports_y)
        self.update_prototypes(new_proto_h, new_proto_y)

    def net_eval(self, target_set, ptracker):
        """
            Inner-loop evaluation.
            This is where evaluation on the query/target set happens
        """
        if len(target_set[0]) == 0: return torch.tensor(0.).to(self.device)
        targets_x, targets_y = target_set
        targets_h = self.backbone(targets_x)
        proto_h, proto_y = self.get_prototypes()
        dist = euclidean_dist(targets_h, proto_h)
        scores = -dist
        targets_y = targets_y
        loss = self.loss_fn(scores, targets_y)
        
        # Get the prediction label
        _, pred_y = torch.max(scores, 1)
        
        # Store performances for performance tracking
        ptracker.add_task_performance(
            pred_y.detach().cpu().numpy(),
            targets_y.detach().cpu().numpy(),
            loss.detach().cpu().numpy())
        return loss
        
    def calc_prototypes(self, h, y):
        """ 
            Compute prototypes
        
        Inputs:
            h : feature vectors of the support set
            y : the corresponding labels 
        """
        unique_labels = torch.unique(y)
        proto_h = []
        for label in unique_labels:
            """
                TODO: Calculate prototypes
            """
            ##### a possible solution #####
            prototype = h[y==label].mean(0)
            proto_h.append(prototype)
            ##############################
        
        return proto_h, unique_labels
    
    def update_prototypes(self, proto_h, proto_y):
        """
        Update memory for prototypes
        """
        labels = proto_y.detach().cpu().numpy()
        for i, label in enumerate(labels):
            self.prototypes[int(label)] = proto_h[i]
    
    def get_prototypes(self, y=None):
        """
        Returns prototypes for the corresponding labels
        """
        if y is None:
            y = np.arange(len(self.prototypes.keys()))
        else:
            y = y.detach().cpu().numpy()
        
        proto_h = []
        for l in y:
            h = self.prototypes[int(l)]
            proto_h.append(h)
            
        proto_h = torch.stack(proto_h, 0)
        return proto_h, y
    
    
def euclidean_dist(x, y):
    """
    Distance calculation between two sets of vectors x (n x d) and y (m x d)
    """
    n = x.size(0)
    m = y.size(0)
    d = x.size(1)
    assert d == y.size(1)

    x = x.unsqueeze(1).expand(n, m, d)
    y = y.unsqueeze(0).expand(n, m, d)
    
    return torch.pow(x - y, 2).sum(2)
