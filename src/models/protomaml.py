import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import torch.nn.functional as F
from models.maml import Maml
import copy
import argparse


class ProtoMaml(Maml):
    
    def __init__(self, backbone, args, device):
        super().__init__(backbone, args, device)
        
    def init_classifier(self, supports_x, supports_y):
        """
            Initializes the fast weights of FC layer with prototype embeddings
        """
        supports_h = self.backbone(supports_x)
        proto_h, proto_y = self.calc_prototypes(supports_h, supports_y)
        proto_h = torch.stack(proto_h, 0)[proto_y]
        proto_h = F.normalize(proto_h, p = 2, dim = 1)
        self.classifier.weight.data = 2 * nn.Parameter(proto_h, requires_grad=True).to(self.device)
        bias = torch.square(proto_h.norm(p=2,dim=1))
        self.classifier.bias.data = - nn.Parameter(bias, requires_grad=True).to(self.device)
    
    def calc_prototypes(self, h, y):
        """
            TODO: Compute prototypes
        """
        #### a possible solution ####
        unique_labels = torch.unique(y)
        proto_h = []
        for label in unique_labels:
            proto_h.append(h[y==label].mean(0))
        #############################
        return proto_h, unique_labels
    
    def net_train(self, support_set):
        """
            Inner-loop optimization of ProtoMAML
        """
        self.zero_grad()
        
        (support_x, support_y) = support_set
        
        # Initilize the final FC layer with prototypes
        self.init_classifier(support_x, support_y) 
        
        # Perform inner-loop optimization
        for _ in range(self.num_steps):
            """
                TODO: Finish the inner-loop optimization of ProtoMAML
            """
            #### a possible solution ####
            support_h = self.backbone.forward(support_x)
            scores  = self.classifier.forward(support_h)
            set_loss = self.loss_fn(scores, support_y)
            
            # build full graph support gradient of gradient
            grad = torch.autograd.grad(
                set_loss, 
                self.fast_parameters, 
                create_graph=True)
            
            if self.approx:
                grad = [ g.detach() for g in grad ] #do not calculate gradient of gradient if using first order approximation
            
            self.fast_parameters = []
            for k, weight in enumerate(self.parameters()):
                if weight.fast is None:
                    weight.fast = weight - self.inner_loop_lr * grad[k] # create weight.fast 
                else:
                    weight.fast = weight.fast - self.inner_loop_lr * grad[k] # update weight.fast
                
                # gradients are based on newest weights, but the graph will retain the link to old weight.fast
                self.fast_parameters.append(weight.fast)
            #############################