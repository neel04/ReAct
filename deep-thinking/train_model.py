""" train_model.py
    Train, test, and save models

    Collaboratively developed
    by Avi Schwarzschild, Eitan Borgnia,
    Arpit Bansal, and Zeyad Emam.

    Developed for DeepThinking project
    October 2021
"""
import wandb

import json
import logging
import os
import sys
from collections import OrderedDict

import hydra
import numpy as np
import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from icecream import ic
from omegaconf import DictConfig, OmegaConf
from accelerate import Accelerator
from accelerate.utils import DistributedDataParallelKwargs
from accelerate.utils import set_seed

import deepthinking as dt
import deepthinking.utils.logging_utils as lg
from deepthinking.utils.debug_utils import DebugUnderflowOverflow

# Ignore statements for pylint:
#     Too many branches (R0912), Too many statements (R0915), No member (E1101),
#     Not callable (E1102), Invalid name (C0103), No exception (W0702),
#     Too many local variables (R0914), Missing docstring (C0116, C0115).
# pylint: disable=R0912, R0915, E1101, E1102, C0103, W0702, R0914, C0116, C0115
# seed everything with torch

def cleanup():
    torch.distributed.destroy_process_group()

class DummyWandb:
    def __init__(self):
        pass
    def log(self, *args, **kwargs):
        pass
    def finish(self, *args, **kwargs):
        pass
    def save(self, *args, **kwargs):
        pass

@hydra.main(config_path="config", config_name="train_model_config")
def main(cfg: DictConfig):
    global wandb
    wandb.login(key="06af347f2f75679eaa6527711464b33792135f54")
    torch.backends.cudnn.benchmark = True

    log = logging.getLogger()
    log.info("\n_________________________________________________\n")
    log.info("train_model.py main() running.")
    log.info(OmegaConf.to_yaml(cfg))
    dic_cfg = OmegaConf.to_container(cfg, resolve=True)
    
    # ---- Distributed Data Parallel ----
    ddp_scaler = DistributedDataParallelKwargs(find_unused_parameters=False)
    accelerator = Accelerator(gradient_accumulation_steps=1, kwargs_handlers=[ddp_scaler])
    device = accelerator.device

    if accelerator.is_main_process:
        wandb.init(project="ReAct", config=dict(dic_cfg), 
                magic=True, sync_tensorboard=False, group='bADD_32')
        
        wandb.run.log_code("/fsx/awesome/DPT/", include_fn=lambda path: path.endswith(".py") or path.endswith(".ipynb") or path.endswith(".sh"))

        # write config YAML file to /tmp
        with open("/tmp/my_config.yaml", "w") as f:
            f.write(OmegaConf.to_yaml(cfg))
        
        wandb.save("/tmp/my_config.yaml") # save config file to wandb, so we can download it later
    else:
        # create a dummy wandb object so that we can call wandb.log() without checking if rank == 0
        wandb = DummyWandb()

    #cfg.problem.model.test_iterations = list(range(cfg.problem.model.test_iterations["low"],
    #                                               cfg.problem.model.test_iterations["high"] + 1))
    cfg.problem.model.test_iterations = list(range(cfg.problem.model.test_iterations["low"],
                                                    3 * cfg.problem.model.test_iterations["high"] + 1,
                                                    5))
    assert 0 <= cfg.problem.hyp.alpha <= 1, "Weighting for loss (alpha) not in [0, 1], exiting."

    ####################################################
    #               Dataset and Network and Optimizer
    loaders = dt.utils.get_dataloaders(cfg.problem)

    net, start_epoch, optimizer_state_dict, accelerator = dt.utils.load_model_from_checkpoint(cfg.problem.name,
                                                                                 cfg.problem.model,
                                                                                 device,
                                                                                 accelerator)
    pytorch_total_params = sum(p.numel() for p in net.parameters())
    log.info(f"This {cfg.problem.model.model} has {pytorch_total_params/1e6:0.3f} million parameters.")
    log.info(f"Training will start at epoch {start_epoch}.")
    optimizer, warmup_scheduler, lr_scheduler = dt.utils.get_optimizer(cfg.problem.hyp,
                                                                       cfg.problem.model,
                                                                       net,
                                                                       optimizer_state_dict,
                                                                       accelerator)
    # preparing for acceleration
    net, optimizer, loaders["train"], loaders["val"], lr_scheduler = accelerator.prepare(net, optimizer, loaders["train"], loaders["val"], lr_scheduler)

    train_setup = dt.TrainingSetup(optimizer=optimizer,
                                   scheduler=lr_scheduler,
                                   warmup=warmup_scheduler,
                                   clip=cfg.problem.hyp.clip,
                                   alpha=cfg.problem.hyp.alpha,
                                   max_iters=cfg.problem.model.max_iters,
                                   problem=cfg.problem.name)
    ####################################################
    # Register the LR scheduler for checkpointing
    accelerator.register_for_checkpointing(lr_scheduler)
    accelerator.register_for_checkpointing(optimizer)
    accelerator.register_for_checkpointing(net)

    ####################################################
    #        Train
    log.info(f"==> Starting training for {max(cfg.problem.hyp.epochs - start_epoch, 0)} epochs...")
    highest_val_acc_so_far = -1
    best_so_far = False
    train_elem_acc = 0

    # Curriculum learning
    trainloader = loaders["train"]
    tgt_upper_b = trainloader.dataset.upper_b # target upper bound, i.e. the upper bound we want to reach eventually
    trainloader.dataset.upper_b = trainloader.dataset.lower_b + 1 # initialize upper bound to 3 more than lower bound

    # Setting network weights initialization
    debug_overflow = DebugUnderflowOverflow(net) 
    
    for epoch in range(start_epoch, cfg.problem.hyp.epochs):
        # update upper bound for curriculum learning
        if train_elem_acc > (0.98 + epoch * (0.01 / cfg.problem.hyp.epochs)) and trainloader.dataset.upper_b < tgt_upper_b:
            trainloader.dataset.upper_b += 1
            loaders["train"] = trainloader # ensure to overwrite the dataloader
            print(f'{"~"*55}\n\t\tUpper bound is now {trainloader.dataset.upper_b}\n{"~"*55}')

            i,o = next(iter(trainloader)) # get a random sample for sanity check
            print(f'\nBound Sample: {i[0]} | {o[0]}')

        loss, acc, train_mae, train_elem_acc, train_seq_acc, accelerator = dt.train(net, loaders, cfg.problem.hyp.train_mode, train_setup, device, accelerator)
        val_acc, best_val_acc, best_val_it = dt.test(net, [loaders["val"]], cfg.problem.hyp.test_mode, [cfg.problem.model.max_iters],
                          cfg.problem.name, device, extra_metrics=True)[0] #TODO: [0][cfg.problem.model.max_iters]

        if best_val_acc > highest_val_acc_so_far:
            best_so_far = True
            highest_val_acc_so_far = best_val_acc

        log.info(f"Training loss at epoch {epoch}: {loss}")
        log.info(f"Training accuracy at epoch {epoch}: {acc}")
        log.info(f"Val accuracy at epoch {epoch}: {best_val_acc} | @ {best_val_it} iters")

        # if the loss is nan, then stop the training
        if np.isnan(float(loss)):
            raise ValueError(f"{ic.format()} Loss is nan, exiting...")

        wandb.log({"Loss/loss": loss, "Accuracy/acc": acc, "Accuracy/val_acc": best_val_acc,
                   "MAE/train_mae": train_mae, "Accuracy/train_elem_acc": train_elem_acc,
                   "Accuracy/train_seq_acc": train_seq_acc}, step=epoch)

        for i in range(len(optimizer.param_groups)):
            wandb.log({"Learning_rate/group{i}": optimizer.param_groups[i]["lr"]}, step=epoch)

        # evaluate the model periodically and at the final epoch
        if (epoch + 1) % cfg.problem.hyp.val_period == 0 or epoch + 1 == cfg.problem.hyp.epochs:
            test_elem, test_acc, val_acc, train_acc = dt.test(net,
                                                                [loaders["test"],
                                                                loaders["val"],
                                                                loaders["train"]],
                                                                cfg.problem.hyp.test_mode,
                                                                cfg.problem.model.test_iterations,
                                                                cfg.problem.name,
                                                                device)
            log.info(f"Training accuracy: {train_acc}")
            log.info(f"Val accuracy: {val_acc}")
            log.info(f"Test accuracy (hard data): {test_acc}")
            log.info(f"Test elementwise accuracy (hard data): {test_elem}")

            tb_last = cfg.problem.model.test_iterations[-1]
            wandb.log({
                "Accuracy/final_train_acc": train_acc[tb_last],
                "Accuracy/final_val_acc": val_acc[tb_last],
                "Accuracy/test_acc (hard_data)": test_acc[tb_last],
                "Accuracy/test_elem_acc (hard_data)": test_elem
                }, step=epoch)

        # check to see if we should save
        save_now = (epoch + 1) % cfg.problem.hyp.save_period == 0 or \
                   (epoch + 1) == cfg.problem.hyp.epochs or best_so_far
        if save_now:
            accelerator.wait_for_everyone()
            accelerator.save(epoch, "/fsx/awesome/DPT/outputs/epoch.pkl")

            out_str = f"model_{'best' if best_so_far else ''}.pth"
            best_so_far = False
            log.info(f"Saving model to: {out_str}")
            accelerator.save_state(output_dir=f"/fsx/awesome/DPT/outputs/{out_str}")
            wandb.save(out_str)

    # save some accuracy stats (can be used without testing to discern which models trained)
    stats = OrderedDict([("max_iters", cfg.problem.model.max_iters),
                         ("run_id", cfg.run_id),
                         ("test_acc", test_acc),
                         ("test_data", cfg.problem.test_data),
                         ("test_iters", list(cfg.problem.model.test_iterations)),
                         ("test_mode", cfg.problem.hyp.test_mode),
                         ("train_data", cfg.problem.train_data),
                         ("train_acc", train_acc),
                         ("test_elem_acc", test_elem),
                         ("val_acc", val_acc)])
    with open(os.path.join("stats.json"), "w") as fp:
        json.dump(stats, fp)
    
    log.info(stats)
    wandb.save("stats.json")
    ####################################################


if __name__ == "__main__":
    SEED = 420
    print(f'Using seed: {SEED}')
    set_seed(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    run_id = dt.utils.generate_run_id()
    sys.argv.append(f"+run_id={run_id}")
    main()
    wandb.finish() # this is needed to make sure the wandb process is closed