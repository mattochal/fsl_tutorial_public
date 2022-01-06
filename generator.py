import sys
sys.path.insert(0,'./src/')

from utils.utils import *
from utils.parser_utils import *

import itertools
import os
import copy
import json
import pdb
import argparse
from functools import partial
from pprint import pprint
from collections import defaultdict
import string


def flatten_dict(d, prefix=None, seperator='.', value_map=lambda x: x):
    new_d = {}
    for key, value in list(d.items()):
        new_key = key if prefix is None else prefix + seperator + key
        if type(value) == dict:
            flattened = flatten_dict(value, new_key, seperator)
            new_d.update(flattened)
        else:
            new_d[new_key] = value
    return new_d

    
def substitute_hyperparams(hyperparams, config=None, suppress_warning=True):
    hyperparams = from_syntactic_sugar(copy.deepcopy(hyperparams))
    
    if config is None:
        combined_args, excluded_args, _ = get_args([], hyperparams)
    else:
        # Convert from three phase syntactic sugar format
        for three_phase_args in ['dataset_args', 'ptracker_args', 'task_args']:
            if three_phase_args in hyperparams:
                hyperparams[three_phase_args] = expand_three_phase(hyperparams[three_phase_args])
        
        combined_args, excluded_args  = update_dict_exclusive(config, hyperparams)
    
    if not suppress_warning and len(excluded_args) > 0:
        print("""excluded""")
        pprint(excluded_args)
        
    return combined_args


def unpack(comb):
    new_comb = {}
    for key, value in comb.items():
        if type(key) is tuple and type(value) is tuple:
            unpacked_dict = {a[0]:a[1] for a in zip(key, value)}
            new_comb.update(unpacked_dict)
        else:
            new_comb[key] = value
    return new_comb
    
    
def hyperparameter_combinations(variables):
    '''
    Generates all possible combinations of variables
    :param variables: dictionary mapping variable names to a list of variable values that vary between experiments
    :returns experiments: a list of dictionaries mapping variable names to singular variable values
    '''
    # https://codereview.stackexchange.com/a/171189
    keys, values = zip(*variables.items())
    
    assert all([isinstance(v, list) for v in values]), "All variable values should be contained in a list!" +\
                                                        " Put square parentheses, ie. '[' and ']', around the lonely value. " 
    
    combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]
    
    unpacked_combinations = [unpack(comb) for comb in combinations]
    
    return unpacked_combinations


def get_python_script(exp_kwargs={}):
    script = 'python src/main.py '
    for arg in exp_kwargs:
        script += '--{} {} '.format(arg, exp_kwargs[arg])
    return script


def get_bash_script(exp_kwargs={}):
    script = 'export CUDA_VISIBLE_DEVICES=$1; \n'
    exp_kwargs['gpu'] = 0
    script += get_python_script(exp_kwargs) + ' $2'
    return script
    
    
def value_map(x):
    return int(x) if type(x) == bool else x
    
def generate_experiments(name_template, 
                         variables,
                         g_args,
                         default_config=None,
                         config_name='config', 
                         script_name='script', 
                         log_name='log', 
                         save=True):
    
    global GPU_COUNTER  # used to evenly distribute jobs among the gpus (optional)
    ngpus = len(g_args.gpu)
    
    # If dry run
    if g_args.dummy_run:
        variables.update({
                    'num_epochs': [3],
                    'num_tasks_per_epoch': [3],
                    'num_tasks_per_validation': [3],
                    'num_tasks_per_testing': [3]
        })
        if ('model' in variables and variables['model'] == 'simpleshot') or \
           (default_config is not None and 'model' in default_config and default_config['model'] == 'simpleshot'):
            variables['model_args.approx_train_mean'] = [True]
    
    scripts = []
    configs = []
    script_paths = []
    config_paths = []
    
    # Iterate over variable combinations
    combinations = hyperparameter_combinations(variables)
    
    for i_comb, hyperparams in enumerate(combinations):
        
        # Generate full config
        full_config = substitute_hyperparams(hyperparams, default_config)
        
        # Generate a compressed version of config
        compressed_config = compress_args(full_config)
        
        # Flattened config for template name
        sperator = '_'
        flat_config = {
            **flatten_dict(full_config, seperator=sperator, value_map=value_map),
            **flatten_dict(compressed_config, seperator=sperator, value_map=value_map)
        }
        
        # Assign experiment_name
        # pprint(flat_config)
        experiment_name = name_template.replace('.', '_').format(**flat_config)
        full_config['experiment_name'] = compressed_config['experiment_name'] = experiment_name
        
        # Setup paths
        experiment_path = os.path.join(g_args.results_folder, experiment_name)
        config_path = os.path.join(experiment_path, 'configs', '{}.json'.format(config_name))
        script_path = os.path.join(experiment_path, 'scripts', '{}.sh'.format(script_name))
        output_path = os.path.join(experiment_path, 'logs', '{}.txt'.format(log_name))
        
        # Select gpu
        gpu = g_args.gpu[GPU_COUNTER % ngpus]
        GPU_COUNTER+=1
        
        # Run from .sh script file or directly using python
        if g_args.bash:
            config_path = os.path.abspath(config_path)
            script_path = os.path.abspath(script_path)
            output_path = os.path.abspath(output_path)
            script_content = get_bash_script(exp_kwargs={'args_file': config_path, 'gpu':gpu}) + '\n'
            script_command = 'bash {} {} '.format(script_path, gpu)
        else:
            script_content = get_python_script(exp_kwargs={'args_file': config_path, 'gpu':gpu})
            script_command = script_content
        
        # Save script content
        if save:
            if not g_args.no_log: script_command += ' &> ' + output_path
            print(script_command)
            
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            os.makedirs(os.path.dirname(config_path), exist_ok=True)
            os.makedirs(os.path.dirname(script_path), exist_ok=True)
            
            with open(config_path,'w') as f:
                json.dump(compressed_config, f, indent=2)
            
            with open(script_path,'w') as f:
                f.write(script_content)
        
        # Save scripts for reference
        scripts.append(script_command)
        script_paths.append(script_path)
        configs.append(compressed_config)
        config_paths.append(config_path)
    
    return zip(scripts, script_paths, configs, config_paths)        
        
        
def make_names(settings, shot_half=True, query_half=True):
    names = []
    for setting in settings:
        s, q, w  = setting
        names.append(f'{s}-shot_{w}-way_{q}-query')
    return names

    

def generate_fsl_tasks(g_args, models=[], seeds=[], train_tasks=[], test_tasks=[], var_update={}, save=True, 
                   expfolder='', pretrained_backbone=None, slow_learning=False, dataset = 'mini', backbone='Conv4',
                   template_prefix='{dataset}/'):
    
    n_way = 5
    
    if slow_learning:
        train_setup = {
            'num_epochs': [200],
             'model_args.lr'              : [0.001],
             'model_args.lr_decay'        : [1.0],
             'model_args.lr_decay_step'   : [200],
             'num_tasks_per_epoch'        : [2500],
            }
    else:
        train_setup = {
            'num_epochs': [20],
             'model_args.lr'              : [0.001], 
             'model_args.lr_decay'        : [0.3],
             'model_args.lr_decay_step'   : [7],
             'num_tasks_per_epoch'        : [500],
            }
    
    is_baseline = lambda x: x in ['baseline', 'baselinepp', 'knn']
    
    experiement_files = []
    for seed in seeds:
        for model in models:
            for train_task in train_tasks:
                train_name = make_names([train_task])[0]
                
                variables = {
                    'results_folder'             : [os.path.abspath(g_args.results_folder)],
                    'seed'                       : [seed],
                    'backbone'                   : [backbone],
                    'num_tasks_per_validation'   : [200],
                    'num_tasks_per_testing'      : [600],
                    'model'                      : [model],
                    'task'                       : ['fsl'],
                    'task_args.batch_size'       : [1],
                    'task_args.num_classes'      : [n_way],
                    'dataset'                    : [dataset],
                    'dataset_args.data_path'     : [g_args.data_path],
                    'dataset_args.train.aug'     : [True],
                    'ptracker_args.test.metrics' : [['accuracy', 'loss', 'per_cls_stats']],
                    'tqdm'                       : [True]
                }
                variables.update(train_setup)
                
                variables[('task_args.num_supports', 
                            'task_args.num_targets',
                            'task_args.num_classes')] = [train_task]

                # experiment path
                template = os.path.join(expfolder, template_prefix, '{backbone}/{model}/')

                if is_baseline(model):
                    variables.update({
                        'no_val_loop'                       :[False],
                        'conventional_split'                :[True],
                        'conventional_split_from_train_only':[False],
                    })
                    template += "{task_args.train.batch_size}batch_train/"
                else:
                    template += "{task_args.train.num_supports}s_{task_args.train.num_targets}q_{task_args.train.num_classes}w_train/"
                
                
                if len(test_tasks) > 0:   # else if no test task is given, assume train task is the same as evaluation task 
                    variables[('task_args.eval.num_supports', 
                                'task_args.eval.num_targets', 
                                'task_args.eval.num_classes')] = test_tasks

                if slow_learning:
                    template += '{num_tasks_per_epoch}x{num_epochs}ep_{model_args.lr}lr_' +\
                                '{model_args.lr_decay_step}step/'
                
                if model in ['protonet']:
                    variables[(
                        'task_args.train.num_classes',
                        'task_args.train.num_targets')] = [ (20, 5), (5, 15) ]
                    # template += '{task_args.train.num_classes}trainway/'

                elif model in ['baseline']:
                    variables['task_args.trval.batch_size'] = [128]  # train and validation batch

                elif model in ['maml', 'protomaml']:
                    variables['task_args.train.num_targets'] = [5]
                    variables['model_args.batch_size'] = [1] #if model == 'maml' else [1]
                    variables['model_args.inner_loop_lr'] = [0.1] if model == 'maml' else [0.005]
                    variables['model_args.num_inner_loop_steps'] = [5] #if model == 'maml' else [5]
                    
                template += '{seed}/'
                variables.update(var_update)
                
                expfiles = generate_experiments(template, variables, g_args, save=save)
                experiement_files.extend(expfiles)
    return experiement_files


def generate_task_test(g_args, expfiles):

    test_settings = [
        (1, 15, 5),  # shots, queries, way
        (2, 15, 5),  # shots, queries, way
        (3, 15, 5),  # shots, queries, way
        (4, 15, 5),  # shots, queries, way
        (5, 15, 5),  # shots, queries, way
    ]
    
    test_names = make_names(test_settings)
    
    for experiment in expfiles:
        
        script, script_path, config, config_path = experiment
        
        # expanded args also useful for backward compatibility. 
        config = substitute_hyperparams(config)
        
        assert config['task'] == 'fsl'
        
        for t, test_setting in enumerate(test_settings):
            test_name = test_names[t]
            
            variables = {
                'continue_from' :                        ['best'],
                'evaluate_on_test_set_only':             [True],
                'test_performance_tag':                  [test_name],
            }
            
            variables[(
                'task_args.test.num_supports',
                'task_args.test.num_targets', 
                'task_args.test.num_classes',
            )] = [test_setting]
            
            generate_experiments(
                config['experiment_name'], 
                variables,
                g_args,
                default_config = config,
                save=True,
                config_name='config_test_on_{}'.format(test_name),
                script_name='script_test_on_{}'.format(test_name),
                log_name='log_test_on_{}'.format(test_name)
            )

            
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', default=['{gpu}'], type=str, nargs="+", help='GPU ID')
    parser.add_argument('--dummy_run', type=str2bool, nargs='?', const=True, default=False,
                        help='Produces scripts as a "dry run" with a reduced number of tasks, '
                             'and no mobel saving (useful for debugging)')
    parser.add_argument('--data_path', type=str, default='./data/',
                        help='Folder with data')
    parser.add_argument('--models', '--model', type=str, nargs="*", default=[],
                        help='Run selected models')
    parser.add_argument('--shot', type=int, default=5,
                        help='Run with selected shot number')
    parser.add_argument('--way', type=int, default=5,
                        help='Run with selected way number')
    parser.add_argument('--query', type=int, default=15,
                        help='Run with selected query number')
    parser.add_argument('--backbone', type=str, default='Conv4',
                        help='See ./src/utils/utils.py file for a list of valid backbones.')
    parser.add_argument('--seeds', '--seed', type=int, nargs="*", default=[],
                        help='Generate experiments using selected seed numbers')
    parser.add_argument('--results_folder', type=str, default='./experiments/',
                        help='Folder for saving the experiment config/scripts/logs into')
    parser.add_argument('--dataset', type=str, default='fish')
    parser.add_argument('--slow_learning', type=str, nargs='?', const=True, default=False,
                        help='If true, runs slower learning rate.')
    parser.add_argument('--test', type=str2bool, nargs='?', const=True, default=False,
                        help='Generate test tasks for models')
    parser.add_argument('--no_log', type=str2bool, nargs='?', const=True, default=True,
                        help='Output won''t get redirected to logs')
    parser.add_argument('--bash', type=str2bool, nargs='?', const=True, default=False,
                        help='Prints bash scripts instead of python')
    g_args = parser.parse_args()
    
    g_args.results_folder = os.path.abspath(g_args.results_folder)
    
    GPU_COUNTER = 0    

    if g_args.models is None or len(g_args.models) == 0:
        models = [
            'baseline',
            'protonet',
            'relationnet',
            'maml',
            'protomaml',
        ]
    else:
        models = g_args.models
    
    if g_args.seeds is None or len(g_args.seeds) == 0:
        seeds = [0]
    else:
        seeds = g_args.seeds
    
    tasks = [
        (g_args.shot, g_args.query, g_args.way), 
    ]

    assert g_args.way <= 5 or g_args.dataset != 'fish', 'Fish dataset supports only up to 5 classes'
    
    # Standard meta-training
    standard_expfiles = generate_fsl_tasks(g_args, models=models, seeds=seeds, train_tasks=tasks,
                                       save=(not g_args.test), expfolder='tutorial',  slow_learning=g_args.slow_learning,  
                                       backbone=g_args.backbone, dataset=g_args.dataset)
    if g_args.test: 
        generate_task_test(g_args, standard_expfiles)