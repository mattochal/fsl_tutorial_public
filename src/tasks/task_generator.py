import numpy as np
import random
import time
import argparse


class TaskGenerator():
    
    def __init__(self,
                 dataset,
                 task,
                 num_tasks,
                 seed,
                 epoch,
                 deterministic,
                 fix_classes,
                 mode,
                 task_args):
        """
        Task Generator creates independent tasks for the outerloop of the algorithms.
        :param dataset: Dataset object to generate tasks from
        :param num_tasks: number of tasks
        :param seed: Start seed for the generator
        :param args: Arguments object
        """
        self.task = task
        self.dataset = dataset
        self.num_tasks = num_tasks
        self.epoch_seed = seed
        self.task_args = task_args
        self.epoch = epoch
        self.mode = mode
        self.fix_classes = fix_classes
        self.deterministic = deterministic
        self.task_rng = np.random.RandomState(self.epoch_seed)

    def __len__(self):
        return self.num_tasks
        
    def get_task_sampler(self):
        """
        return appropiate_task_sampler
        """
        
        if not self.deterministic:
            class_seed= None
            sample_seed= None
        else:
            class_seed= self.epoch_seed if self.fix_classes else self.task_seed
            sample_seed= self.task_seed

        return self.task(self.dataset,
                         self.task_args,
                         class_seed= class_seed,
                         sample_seed= sample_seed)
    
    def __iter__(self):
        """
        :returns: a sampler class that samples images from the dataset for the specific task
        """
        for i in range(self.num_tasks):
            self.task_seed = self.task_rng.randint(999999999)
            yield self.get_task_sampler()
