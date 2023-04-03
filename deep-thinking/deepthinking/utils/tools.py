""" tools.py
    Utility functions that are common to all tasks

    Collaboratively developed
    by Avi Schwarzschild, Eitan Borgnia,
    Arpit Bansal, and Zeyad Emam.

    Developed for DeepThinking project
    October 2021
"""
import os
import logging
import random
import web_pdb as pdb
from datetime import datetime

import torch
from icecream import ic
from torch.optim import SGD, Adam, AdamW
from torch.optim.lr_scheduler import MultiStepLR, CosineAnnealingLR
from .lion_opt import Lion, AdamOnLion 

import deepthinking.models as models
from .mazes_data import prepare_maze_loader
from .prefix_sums_data import prepare_prefix_loader
from .chess_data import prepare_chess_loader #ADDED NEW
from .addition_data import prepare_addition_loader
from .. import adjectives, names

from .warmup import ExponentialWarmup, LinearWarmup

# Ignore statements for pylint:
#     Too many branches (R0912), Too many statements (R0915), No member (E1101),
#     Not callable (E1102), Invalid name (C0103), No exception (W0702),
#     Too many local variables (R0914), Missing docstring (C0116, C0115).
# pylint: disable=R0912, R0915, E1101, E1102, C0103, W0702, R0914, C0116, C0115


def generate_run_id():
    hashstr = f"{adjectives[random.randint(0, len(adjectives))]}-{names[random.randint(0, len(names))]}"
    return hashstr


def get_dataloaders(problem_args):
    if problem_args.name == "prefix_sums":
        return prepare_prefix_loader(train_batch_size=problem_args.hyp.train_batch_size,
                                     test_batch_size=problem_args.hyp.test_batch_size,
                                     train_data=problem_args.train_data,
                                     test_data=problem_args.test_data)
    elif problem_args.name == "mazes":
        return prepare_maze_loader(train_batch_size=problem_args.hyp.train_batch_size,
                                   test_batch_size=problem_args.hyp.test_batch_size,
                                   train_data=problem_args.train_data,
                                   test_data=problem_args.test_data)
    elif problem_args.name == "chess":
        return prepare_chess_loader(train_batch_size=problem_args.hyp.train_batch_size,
                                    test_batch_size=problem_args.hyp.test_batch_size,
                                    train_data=problem_args.train_data,
                                    test_data=problem_args.test_data)
    elif problem_args.name == "addition":
        return prepare_addition_loader(train_batch_size=problem_args.hyp.train_batch_size,
                                    test_batch_size=problem_args.hyp.test_batch_size,
                                    train_data=problem_args.train_data,
                                    test_data=problem_args.test_data)
    else:
        raise ValueError(f"Invalid problem spec. {problem_args.name}")


def get_model(model, width, max_iters, in_channels=3):
    model = model.lower()
    net = getattr(models, model)(width=width, in_channels=in_channels, max_iters=max_iters)
    print(net,'\n\n')
    return net


def get_optimizer(optim_args, model_args, net, state_dict):
    optimizer_name = optim_args.optimizer.lower()
    epochs = optim_args.epochs
    lr = optim_args.lr
    lr_decay = optim_args.lr_decay
    lr_schedule = optim_args.lr_schedule
    lr_factor = optim_args.lr_factor
    warmup_period = optim_args.warmup_period

    if optim_args.lr_throttle:
        # Reducing the lr here for the recurrent layers helps with stability,
        # To date (July 21, 2021), we may only need this for maze models.
        base_params = [p for n, p in net.named_parameters() if "recur" not in n]
        recur_params = [p for n, p in net.named_parameters() if "recur" in n]
        iters = model_args.max_iters
        all_params = [{"params": base_params}, {"params": recur_params, "lr": lr / iters}]
    else:
        base_params = [p for n, p in net.named_parameters()]
        recur_params = []
        iters = 1
        all_params = [{"params": base_params}]

    if optimizer_name == "sgd":
        optimizer = SGD(all_params, lr=lr, weight_decay=2e-3, momentum=0.9)
    elif optimizer_name == "adam":
        optimizer = Adam(all_params, lr=lr, weight_decay=2e-3)
    elif optimizer_name == "adamw":
        optimizer = AdamW(all_params, lr=lr, weight_decay=2e-3)
    elif optimizer_name == "lion":
        optimizer = Lion(all_params, lr=lr, weight_decay=2e-3, betas=(0.9, 0.99))
    elif optimizer_name == "adam_on_lion":
        optimizer = AdamOnLion(all_params, lr=lr, weight_decay=2e-3, betas=(0.9, 0.99))
    else:
        raise ValueError(f"{ic.format()}: Optimizer choise of {optimizer_name} not yet implmented.")

    if state_dict is not None:
        optimizer.load_state_dict(state_dict)
        optimizer.param_groups[0]["capturable"] = True # make optimizer capturable=True
        warmup_scheduler = ExponentialWarmup(optimizer, warmup_period=0)
        # warmup_scheduler = LinearWarmup(optimizer, warmup_period=0)
    else:
        warmup_scheduler = ExponentialWarmup(optimizer, warmup_period=warmup_period)
        # warmup_scheduler = LinearWarmup(optimizer, warmup_period=warmup_period)

    if lr_decay.lower() == "step":
        lr_scheduler = MultiStepLR(optimizer, milestones=lr_schedule,
                                   gamma=lr_factor, last_epoch=-1)
    elif lr_decay.lower() == "cosine":
        lr_scheduler = CosineAnnealingLR(optimizer, epochs, eta_min=0, last_epoch=-1, verbose=False)
    else:
        raise ValueError(f"{ic.format()}: Learning rate decay style {lr_decay} not yet implemented.")

    return optimizer, warmup_scheduler, lr_scheduler


def load_model_from_checkpoint(problem, model_args, device):
    model = model_args.model
    model_path = model_args.model_path
    width = model_args.width
    max_iters = model_args.max_iters
    epoch = 0
    optimizer = None
    new_state_dict = {}

    in_channels = 3
    if problem == "chess":
        in_channels = 12
    elif problem == 'addition':
        in_channels = 1

    net = get_model(model, width, in_channels=in_channels, max_iters=max_iters)
    net = net.to(device)
    if device == "cuda":
        net = net
    
    if model_path is not None and os.path.exists(model_path):
        logging.info(f"\n{'$'*50}\nLoading model from checkpoint {model_path}...\n{'$'*50}")
        state_dict = torch.load(model_path, map_location=device)

        # check if keys are prefixed with "module."
        new_state_dict = state_dict.copy()

        for key in list(new_state_dict["net"].keys()):
            new_key = key.replace('_orig_mod.', '') # remove _orig_mod. prefix
            new_state_dict["net"][new_key] = state_dict['net'][key]
            # remove old key
            del new_state_dict["net"][key]
        
        # Now load fixed state_dict
        net.load_state_dict(new_state_dict["net"])
        epoch = new_state_dict["epoch"] + 1
        optimizer = new_state_dict["optimizer"]

    # load AMP scaler
    scaler = new_state_dict["scaler"] if "scaler" in new_state_dict.keys() else None
    return net, epoch, optimizer, scaler


def now():
    return datetime.now().strftime("%Y%m%d %H:%M:%S")
