import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import torch.nn.functional as F
from models.model_template import ModelTemplate
from backbones.layers import Linear_fw
import copy
import argparse


class Maml(ModelTemplate):
    
    @staticmethod
    def get_parser(parser=None):
        """
            Returns a argparse.ArgumentParser() for MAML 
        """
        if parser is None: parser = argparse.ArgumentParser()
        parser = ModelTemplate.get_parser(parser)
        parser.add_argument('--num_inner_loop_steps', type=int, default=5)
        parser.add_argument('--inner_loop_lr', type=float, default=0.01)
        parser.add_argument('--approx', type=bool, default=False)
        parser.add_argument('--batch_size', type=int, default=4,
                           help='number of tasks before the outerloop update, eg. update meta learner every 4th task')
        parser.add_argument('--output_dim', type=dict, default={"train":-1, "val":-1, "test":-1},
                           help='output dimention for the classifer, if -1 set in code')
        return parser
    
    def __init__(self, backbone, args, device):
        super().__init__(backbone, args, device)
        self.approx = args.approx
        self.inner_loop_lr = args.inner_loop_lr
        self.num_steps = args.num_inner_loop_steps
        self.output_dim = args.output_dim
        self.batch_size = args.batch_size
        self.batch_count = 0
        self.batch_losses = []
        self.fast_parameters = []
        assert self.output_dim.train == self.output_dim.test, 'maml training output dim must mimic the testing scenario'
        
    def setup_model(self):
        """
            Set up linear classifier, outer-loop optimizer and lr scheduler
        """
        self.loss_fn = nn.CrossEntropyLoss()
        self.classifier = self.setup_classifier(self.output_dim.train)
        all_params = list(self.backbone.parameters()) + list(self.classifier.parameters())
        self.optimizer = torch.optim.Adam(all_params, lr=self.args.lr)
        self.lr_scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, 
                step_size=self.args.lr_decay_step, gamma=self.args.lr_decay)
        self.optimizer.zero_grad()
        self.optimizer.step()
        
    def setup_classifier(self, output_dim):
        """
            Setup last fully connected layer
        """
        classifier = Linear_fw(self.backbone.final_feat_dim, output_dim).to(self.device)
        classifier.bias.data.fill_(0)
        return classifier
        
    def meta_train(self, task, ptracker):
        """
            Single iteration of meta training (outer) loop 
        """
        self.mode='train'
        self.train()
        self.net_reset()
        self.batch_count += 1
        
        total_losses = []
        for support_set, target_set in task:
            self.net_train(support_set)
            loss = self.net_eval(target_set, ptracker)
            total_losses.append(loss)
        
        loss = torch.stack(total_losses).sum(0)
        self.batch_losses.append(loss)
        
        # Optimize algorithms on batches of tasks to stabilize learning
        if self.batch_count % self.batch_size == 0:
            self.optimizer.zero_grad()
            loss = torch.stack(self.batch_losses).sum(0)
            loss.backward()
            self.optimizer.step()
            self.batch_losses = []
        
    def meta_eval(self, task, ptracker):
        """
            Single iteration of the outer-loop evaluation 
        """ 
        self.net_reset()
        for support_set, target_set in task:
            self.net_train(support_set)
            self.net_eval(target_set, ptracker)
     
    def net_reset(self):
        """
            Inner-loop reset.
            Clears the fast weights of the model.

            Fast-weights are those weights used for task adaptation.
            The fast-weights should not overwrite meta-learned weights. 
        """
        self.fast_parameters = self.get_inner_loop_params()
        for weight in self.parameters():  # reset fast parameters
            weight.fast = None
    
    def net_train(self, support_set):
        """
            Inner-loop optimization.
        """ 
        self.zero_grad()

        support_x, support_y = support_set
        for _ in range(self.num_steps):

            # get an updated features, prediction scores, and loss
            support_h  = self.backbone.forward(support_x)
            scores  = self.classifier.forward(support_h)
            support_set_loss = self.loss_fn(scores, support_y)
            
            # build full graph support gradient of gradient
            grad = torch.autograd.grad(
                support_set_loss, 
                self.fast_parameters, 
                create_graph=True)
            
            # Do not calculate gradient of gradient if using first-order approximation
            if self.approx:
                grad = [ g.detach() for g in grad ] 
            
            self.fast_parameters = []
            for k, weight in enumerate(self.parameters()):
                """
                    TODO: Implement a simple SGD update step
                    HINT: Use predifined inner loop lr and the calculated gradients
                """
                ##### a possible solution #####
                step = - self.inner_loop_lr * grad[k] 
                ###############################

                if weight.fast is None:
                    # create weight.fast 
                    weight.fast = weight + step
                else:
                    # update weight.fast
                    weight.fast = weight.fast + step
                
                # Gradients are based on newest weights, but the graph (built by torch.autograd.grad)
                #   will retain a link to the original weights and perform second-order optimization 
                #   on meta-weights if the self.approx flag is enabled
                self.fast_parameters.append(weight.fast) 
                
    def net_eval(self, target_set, ptracker):
        """
            Inner-loop evalution on the query/target set after task adaptation.
        """
        if len(target_set[0]) == 0: return torch.tensor(0.).to(self.device)
        
        targets_x, targets_y = target_set
        """
            TODO: Implement forward propagation through the model to get final probability scores
        """
        #### a possible solution ####
        targets_h  = self.backbone.forward(targets_x)
        scores = self.classifier.forward(targets_h)
        #############################
        
        loss = self.loss_fn(scores, targets_y)
        
        _, pred_y = torch.max(scores, axis=1)
        
        ptracker.add_task_performance(
            pred_y.detach().cpu().numpy(),
            targets_y.detach().cpu().numpy(),
            loss.detach().cpu().numpy())
        
        return loss
    
    def get_inner_loop_params(self):
        """
            Get the inner loop parameters of the model
        """
        return list(self.backbone.parameters()) + list(self.classifier.parameters())
            