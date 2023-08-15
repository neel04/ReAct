""" testing.py
    Utilities for testing models

    Collaboratively developed
    by Avi Schwarzschild, Eitan Borgnia,
    Arpit Bansal, and Zeyad Emam.

    Developed for DeepThinking project
    October 2021
"""

import einops
import torch
from icecream import ic
from tqdm import tqdm

# Ignore statements for pylint:
#     Too many branches (R0912), Too many statements (R0915), No member (E1101),
#     Not callable (E1102), Invalid name (C0103), No exception (W0702),
#     Too many local variables (R0914), Missing docstring (C0116, C0115, C0114).
# pylint: disable=R0912, R0915, E1101, E1102, C0103, W0702, R0914, C0116, C0115, C0114


def test(net, loaders, mode, iters, problem, device, extra_metrics=False):
    accs = []

    if extra_metrics:
        print(f'Entering with extra_metrics: {extra_metrics}')
        return test_default(net, loaders[0], iters, problem, device, extra_metrics)

    for idx, loader in enumerate(loaders):
        if mode == "default":
            if idx == 0 and len(loaders) > 1:  # ugly hack to not trigger on solely validation phase
                accuracy, elem_acc = test_default(net, loader, iters, problem, device, extra_metrics)
                accs.append(elem_acc)
            else:
                accuracy, _ = test_default(net, loader, iters, problem, device, extra_metrics)

        elif mode == "max_conf":
            accuracy = test_max_conf(net, loader, iters, problem, device)

        else:
            raise ValueError(f"{ic.format()}: test_{mode}() not implemented.")
        accs.append(accuracy)
    return accs


def get_predicted(inputs, outputs, problem, dim=1):
    predicted = outputs.argmax(dim)
    predicted = predicted.view(predicted.size(0), -1)

    if problem == "mazes":
        predicted = predicted * (inputs.max(1)[0].view(inputs.size(0), -1))
    elif problem == "chess":
        outputs = outputs.view(outputs.size(0), outputs.size(1), -1)
        top_2 = torch.topk(outputs[:, 1], 2, dim=1)[0].min(dim=1)[0]
        top_2 = einops.repeat(top_2, "n -> n k", k=8)
        top_2 = einops.repeat(top_2, "n m -> n m k", k=8).view(-1, 64)
        outputs[:, 1][outputs[:, 1] < top_2] = -float("Inf")
        outputs[:, 0] = -float("Inf")
        predicted = outputs.argmax(1)

    return predicted


def test_default(net, testloader, iters, problem, device, extra_metrics):
    max_iters = max(iters)
    net.eval()
    corrects = torch.zeros(max_iters)
    elemwise_corrects = torch.zeros(max_iters)
    total = 0
    incorrect_input, incorrect_output, incorrect_target = None, None, None

    with torch.no_grad():
        for inputs, targets in tqdm(testloader, leave=False):
            inputs, targets = inputs.to(device).int(), targets.to(device).long()

            all_outputs = net(inputs, iters_to_do=max_iters) # shape: (batch_size, max_iters, SEQ_LEN, tgt_vocab_size)

            for i in range(all_outputs.size(1)):
                outputs = all_outputs[:, i].transpose(1, 2) # shape: (batch_size, tgt_vocab_size, SEQ_LEN)
                old_predicted = get_predicted(inputs, outputs, problem)

                targets = targets.view(targets.size(0), -1) # shape: (batch_size, SEQ_LEN)
                predicted = old_predicted.view(targets.size(0), -1) # shape: (batch_size, SEQ_LEN)

                corrects[i] += torch.amin(predicted == targets, dim=[1]).sum().item()
                elemwise_corrects[i] += (predicted == targets).sum().item()

                # get a sample incorrect prediction to debug
                if (old_predicted != targets).any():
                        # find which one is incorrect
                        incorrect_idx = (predicted != targets).nonzero()[0][0]
                        incorrect_input, incorrect_output, incorrect_target = inputs[incorrect_idx], old_predicted[incorrect_idx].detach().round().int(), targets[incorrect_idx]

            total += targets.size(0)

    accuracy = 100.0 * corrects / total
    elemwise_accuracy = 100.0 * elemwise_corrects / targets.numel()

    ret_acc = {}
    for ite in iters:
        ret_acc[ite] = accuracy[ite-1].item()

    # ret_acc is a dictionary of accuracies for each iteration. with the key being the iteration and the value being the accuracy
    # best_val_acc is the best accuracy achieved. best_val_iteration is the iteration at which the best accuracy was achieved
    best_val_acc, best_val_iteration = max(ret_acc.values()), max(ret_acc, key=ret_acc.get)

    if extra_metrics:
        print(f'DEBUG: RETURNING best_val_acc: {best_val_acc} | best_val_iteration: {best_val_iteration} | ret_acc: {ret_acc}')
        print(f'\nDEBUG: INCORRECT VAL/TEST PREDICTION: input: {incorrect_input} | output: {incorrect_output} | target: {incorrect_target}')
        return ret_acc, best_val_acc, best_val_iteration # for validation set
    else:
        return ret_acc, elemwise_accuracy # for test set


def test_max_conf(net, testloader, iters, problem, device):
    max_iters = max(iters)
    net.eval()
    corrects = torch.zeros(max_iters).to(device)
    total = 0
    softmax = torch.nn.functional.softmax

    with torch.no_grad():
        for inputs, targets in tqdm(testloader, leave=False):
            inputs, targets = inputs.to(device), targets.to(device)
            targets = targets.view(targets.size(0), -1)
            total += targets.size(0)


            all_outputs = net(inputs, iters_to_do=max_iters)

            confidence_array = torch.zeros(max_iters, inputs.size(0)).to(device)
            corrects_array = torch.zeros(max_iters, inputs.size(0)).to(device)
            for i in range(all_outputs.size(1)):
                outputs = all_outputs[:, i]
                conf = softmax(outputs.detach(), dim=1).max(1)[0]
                conf = conf.view(conf.size(0), -1)
                if problem == "mazes":
                    conf = conf * inputs.max(1)[0].view(conf.size(0), -1)
                confidence_array[i] = conf.sum([1])
                predicted = get_predicted(inputs, outputs, problem)
                corrects_array[i] = torch.amin(predicted == targets, dim=[1])

            correct_this_iter = corrects_array[torch.cummax(confidence_array, dim=0)[1],
                                               torch.arange(corrects_array.size(1))]
            corrects += correct_this_iter.sum(dim=1)

    accuracy = 100 * corrects.long().cpu() / total
    ret_acc = {}
    for ite in iters:
        ret_acc[ite] = accuracy[ite-1].item()
    return ret_acc, accuracy