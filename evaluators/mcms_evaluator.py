
import torch
import numpy as np
from tqdm import tqdm

import torch.optim
import pickle

from utils.training_utils import load_model_data
from utils.MaskConfig import VertexInferenceConfig
import argparse
import os
from utils.task_datasets import IOIConfig
from utils.task_datasets import get_task_ds
from utils.training_utils import load_model_data, update_means_variances_mixed
from pruners.VertexPruner import VertexPruner
from mask_samplers.AblationMaskSampler import MultiComponentMaskSampler


def run_MCMS(ablation_type = "mean_agnostic",
              dataset = "ioi",
              n_samples = 1,
              batch_size = 100,
              model_name = "gpt2-small",
              owt_batch_size = 10,
              k = 1,
              max_batches = 10000,
              folder = "results/mcms"):

  if not os.path.exists(folder):
    print("Creating Folder", folder)
    os.makedirs(folder)

  # init model and tokenizer
  device, model, tokenizer, owt_iter = load_model_data(model_name, owt_batch_size)
  model.eval()
  n_layers = model.cfg.n_layers
  n_heads = model.cfg.n_heads

  # init pruning configs
  pruning_cfg = VertexInferenceConfig(model.cfg, device, folder, init_param=1)
  pruning_cfg.batch_size = batch_size
  pruning_cfg.n_samples = n_samples
  pruning_cfg.k = k
  print("---------------------------")
  print("Pruning Config")
  print("---------------------------")
  for k,v in pruning_cfg.__dict__.items():
    if k != "constant_prune_mask" and k!= "init_params":
      print(k,":",v)

  # init ds configs
  task_ds = get_task_ds(dataset, pruning_cfg.batch_size, device, ablation_type)

  for param in model.parameters():
      param.requires_grad = False
  print("---------------------------")
  print("Dataset Config")
  print("---------------------------")
  [print(k, ":", v) for k,v in task_ds.__dict__.items()]
  print("---------------------------")

  pruner_args = task_ds.get_pruner_args({"zero", "mean", "resample", "cf_mean", "cf", "oa", "oa_specific","mean_agnostic"})

  # init mask_sampler
  mask_sampler = MultiComponentMaskSampler(pruning_cfg)
  mask_sampler()
  print("Attn mask shape per layer", mask_sampler.sampled_mask["attn"][0].shape)
  print("MLP mask shape per layer", mask_sampler.sampled_mask["mlp"][0].shape)

  print("---------------------------")
  print("Starting Evaluation")
  print("---------------------------")

  # init vertex pruner
  vertex_pruner = VertexPruner(model, pruning_cfg, mask_sampler, **pruner_args)
  vertex_pruner.add_patching_hooks()
  vertex_pruner.modal_attention.requires_grad = False
  vertex_pruner.modal_mlp.requires_grad = False

  # init results variables
  sampling_optimizer = torch.optim.AdamW(mask_sampler.parameters(), lr=1, weight_decay=0)
  head_losses = torch.zeros((n_layers * n_heads,1)).to(device)
  head_vars = torch.zeros((n_layers * n_heads,1)).to(device)
  n_batches_by_head = torch.zeros_like(head_losses).to(device)
  n_samples_by_head = torch.zeros_like(head_losses).to(device)

  max_batches = int(max_batches / (batch_size * n_samples))


  for no_batches in tqdm(range(vertex_pruner.log.t, max_batches)):
      batch, last_token_pos,cf = task_ds.retrieve_batch_cf(tokenizer)
      last_token_pos = last_token_pos.int()

      sampling_optimizer.zero_grad()

      loss = vertex_pruner(batch, last_token_pos,timing = False, print_loss = False)
      loss.backward()

      atp_losses = torch.cat([ts.grad for ts in mask_sampler.mask_perturb['attn']], dim=0).unsqueeze(-1)

      batch_n_samples = []
      for ts in mask_sampler.sampled_mask['attn']:
          batch_n_samples.append((ts < 1-1e-3).sum(dim=0))
      batch_n_samples = torch.cat(batch_n_samples, dim=0).unsqueeze(-1)


      atp_losses = torch.where(
          batch_n_samples > 0,
          atp_losses / batch_n_samples * n_samples * batch_size,
          0
      )


      head_losses, head_vars, n_batches_by_head, n_samples_by_head = update_means_variances_mixed(head_losses, head_vars, atp_losses, n_batches_by_head, n_samples_by_head, batch_n_samples)
  print("---------------------------")
  print("Finished Evaluation")
  return head_losses, head_vars, n_batches_by_head, n_samples_by_head