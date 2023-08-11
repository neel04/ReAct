""" training.py
    Utilities for training models

    Collaboratively developed
    by Avi Schwarzschild, Eitan Borgnia,
    Arpit Bansal, and Zeyad Emam.

    Developed for DeepThinking project
    October 2021
"""

from dataclasses import dataclass
from random import randrange, choices
from typing import Tuple, Any

import torch
from icecream import ic
from tqdm.auto import tqdm

from deepthinking.utils.testing import get_predicted

@dataclass
class TrainingSetup:
    """Attributes to describe the training precedure"""
    optimizer: "Any"
    scheduler: "Any"
    warmup: "Any"
    clip: "Any"
    alpha: "Any"
    max_iters: "Any"
    problem: "Any"

def get_skewed_n_and_k(max_iters: int) -> Tuple[int, int]:
    n = randrange(0, max_iters-1)
    k_range = range(1, max_iters - n)

    weights = lambda length, mean, stddev: [int(abs(i - mean) <= stddev) + 0.5 for i in range(length)] if len(k_range) > 1 else None
    weight = weights(len(k_range), len(k_range) // 0.2, 1)

    k = choices(k_range, weights=weight, k=1)[0] # weighted sampling
    return n, k

def get_output_for_prog_loss(inputs, max_iters, net):
    # get features from n iterations to use as input
    # do k iterations using intermediate features as input
    n, k = get_skewed_n_and_k(max_iters)

    if n > 0:
        _, interim_thought = net(inputs, iters_to_do=n)
        interim_thought = interim_thought.detach()
    else:
        interim_thought = None

    outputs, _ = net(inputs, iters_elapsed=n, iters_to_do=k, interim_thought=interim_thought)
    return outputs, k

def train(net, loaders, mode, train_setup, device, acc_obj=None):
    if mode == "progressive":
        loss, acc, train_mae, train_elem_acc, train_seq_acc, accelerator = train_progressive(net, loaders, train_setup, device, acc_obj)
    else:
        raise ValueError(f"{ic.format()}: train_{mode}() not implemented.")
    return loss, acc, train_mae, train_elem_acc, train_seq_acc, accelerator

def train_progressive(net, loaders, train_setup, device, accelerator=None):
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.enable_flash_sdp(False)
    
    trainloader = loaders["train"]
    net.train()
    optimizer = train_setup.optimizer
    lr_scheduler = train_setup.scheduler
    warmup_scheduler = train_setup.warmup
    alpha = train_setup.alpha
    max_iters = train_setup.max_iters
    problem = train_setup.problem
    clip = train_setup.clip

    weight = torch.ones(3).to(device)
    weight[2] = 0.2
    criterion = torch.nn.CrossEntropyLoss(reduction='none', weight=weight)

    train_loss = 0
    correct = 0
    total = 0
    train_metric, train_elem_acc, train_seq_acc = [], [], []
    
    for batch_idx, (inputs, targets) in enumerate(tqdm(trainloader, leave=False)):
        with accelerator.accumulate(net):
            # check inputs and targets if they're NaN
            if torch.isnan(inputs).any() or torch.isnan(targets).any():
                raise ValueError(f"\n{'!'*20}\n{ic.format()}: NaN detected in inputs or targets. inputs: {inputs},\ntargets: {targets}\n{'!'*20}\n")

            inputs, targets = inputs.int(), targets.long()
            
            targets = targets.view(targets.size(0), -1)
            if problem == "mazes":
                mask = inputs.view(inputs.size(0), inputs.size(1), -1).max(dim=1)[0] > 0
            
            # get fully unrolled loss if alpha is not 1 (if it is 1, this loss term is not used
            # so we save time by settign it equal to 0).
            with torch.backends.cuda.sdp_kernel(enable_flash=False) as disable:
                outputs_max_iters, _ = net(inputs, iters_to_do=max_iters)

            if alpha != 1:
                outputs_max_iters = outputs_max_iters.view(outputs_max_iters.size(0),
                                                        outputs_max_iters.size(1), -1).transpose(1, 2)
                with accelerator.autocast():
                    loss_max_iters = criterion(outputs_max_iters, targets)
            else:
                loss_max_iters = torch.zeros_like(targets).float()
            # get progressive loss if alpha is not 0 (if it is 0, this loss term is not used
            # so we save time by setting it equal to 0).
            if alpha != 0:
                outputs, k = get_output_for_prog_loss(inputs, max_iters, net)
                outputs = outputs.view(outputs.size(0), outputs.size(1), -1).transpose(1, 2)

                with accelerator.autocast():
                    # k is in [1, max_iters - 1] so we need to rescale it to [1, 2] to use as a weight
                    # for the progressive loss term
                    k = (k - 1) / (max_iters - 1)
                    loss_progressive = criterion(outputs, targets)
                    loss_progressive = loss_progressive * k
            else:
                loss_progressive = torch.zeros_like(targets).float()
            if problem == "mazes":
                loss_max_iters = (loss_max_iters * mask)
                loss_max_iters = loss_max_iters[mask > 0]
                loss_progressive = (loss_progressive * mask)
                loss_progressive = loss_progressive[mask > 0]

            loss_max_iters_mean = loss_max_iters.mean()
            loss_progressive_mean = loss_progressive.mean()
            loss = (1 - alpha) * loss_max_iters_mean + alpha * loss_progressive_mean
            #loss = loss / accum_iters # accumulate gradients
        
        accelerator.backward(loss)

        if clip is not None:
            accelerator.clip_grad_norm_(net.parameters(), clip)
            
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

        train_loss += loss.item()
        predicted = get_predicted(inputs, outputs_max_iters, problem, dim = 2 if alpha == 1 else 1)
        #predicted, targets = predicted.squeeze(), targets.squeeze()

        # Compute MAE b/w preds and targets
        train_metric.append(abs(predicted.float() - targets.float()).detach().mean()) #L1 metric, unrounded

        # compute elementwise accuracy, i.e compare each element of the prediction to the target
        train_elem_acc.append(torch.eq(targets.detach(), predicted.detach()).sum().item() / targets.numel())

        # Compute sequence accuracy, i.e compare the entire sequence to the target
        train_seq_acc.append(torch.eq(targets.detach(), predicted.detach()).all().item())

        correct += torch.eq(targets, predicted).all().item()
        total += targets.size(0)
    
    print(f'\nSample input: {trainloader.dataset.decode(inputs[0])} | Sample pred: {trainloader.dataset.decode(predicted[0])} | Sample answer: {trainloader.dataset.decode(targets[0])}')
    print(f"\n\nTrain metric (MAE): {(sum(train_metric)/len(train_metric)).item()}\n")
    print(f"\nTrain elementwise accuracy: {(sum(train_elem_acc)/len(train_elem_acc)) * 100}%\n")
    print(f"\nTrain sequence/batch accuracy: {(sum(train_seq_acc)/len(train_seq_acc)) * 100}%\n")

    train_loss = train_loss / (batch_idx + 1)
    acc = 100.0 * correct / total
    lr_scheduler.step()
    warmup_scheduler.dampen()

    return train_loss, acc, sum(train_metric)/len(train_metric), sum(train_elem_acc)/len(train_elem_acc), sum(train_seq_acc)/len(train_seq_acc), accelerator
    # train loss, accuracy, train MAE, train elementwise accuracy, train sequence accuracy, accelerator