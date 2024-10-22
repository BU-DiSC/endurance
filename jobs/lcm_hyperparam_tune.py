# #!/usr/bin/env python
# from typing import Any
# import toml
# import logging
# import os
# import tempfile
# from axe.util.lr_scheduler import LRSchedulerBuilder
#
# import ray
# import ray.train as RayTrain
# import ray.tune as RayTune
# from ray.tune.schedulers import ASHAScheduler
# import torch
# from torch.utils.data import DataLoader
#
# from axe.lcm.data.dataset import LCMDataSet
# from axe.lcm.model.builder import LearnedCostModelBuilder
# from axe.lsm.types import STR_POLICY_DICT, Policy
# from axe.util.losses import LossBuilder
# from axe.util.optimizer import OptimizerBuilder
#
#
# def build_train(cfg, lsm_design: Policy) -> LCMDataSet:
#     train_dir: str = os.path.join(
#         cfg["io"]["data_dir"],
#         cfg["job"]["LCMTrain"]["train"]["dir"],
#     )
#     train = LCMDataSet(
#         folder=train_dir,
#         lsm_design=lsm_design,
#         min_size_ratio=cfg["lsm"]["size_ratio"]["min"],
#         max_size_ratio=cfg["lsm"]["size_ratio"]["max"],
#         max_levels=cfg["lsm"]["max_levels"],
#         test=False,
#         shuffle=cfg["job"]["LCMTrain"]["train"]["shuffle"],
#     )
#
#     return train
#
#
# def build_validate(cfg, lsm_design: Policy) -> LCMDataSet:
#     validate_dir = os.path.join(
#         cfg["io"]["data_dir"],
#         cfg["job"]["LCMTrain"]["test"]["dir"],
#     )
#     validate = LCMDataSet(
#         folder=validate_dir,
#         lsm_design=lsm_design,
#         min_size_ratio=cfg["lsm"]["size_ratio"]["min"],
#         max_size_ratio=cfg["lsm"]["size_ratio"]["max"],
#         max_levels=cfg["lsm"]["max_levels"],
#         test=True,
#         shuffle=cfg["job"]["LCMTrain"]["test"]["shuffle"],
#     )
#
#     return validate
#
#
# def train_lcm(cfg: dict[str, Any]):
#     lsm_choice = STR_POLICY_DICT.get(cfg["lsm"]["design"], Policy.KHybrid)
#     size_ratio_min = cfg["lsm"]["size_ratio"]["min"]
#     size_ratio_max = cfg["lsm"]["size_ratio"]["max"]
#     net_builder = LearnedCostModelBuilder(
#         size_ratio_range=(size_ratio_min, size_ratio_max),
#         max_levels=cfg["lsm"]["max_levels"],
#         **cfg["lcm"]["model"],
#     )
#     net = net_builder.build_model(lsm_choice)
#
#     device = "cpu"
#     if torch.cuda.is_available():
#         device = "cuda:0"
#     net.to(device)
#
#     criterion = LossBuilder(cfg).build(cfg["job"]["LCMTrain"]["loss_fn"])
#     if criterion is None:
#         raise TypeError(f"Loss choice invalid: {cfg['job']['LCMTrain']['loss_fn']}")
#     assert criterion is not None
#     optimizer = OptimizerBuilder(cfg).build_optimizer(
#         cfg["job"]["LCMTrain"]["optimizer"], net
#     )
#     scheduler = LRSchedulerBuilder(cfg).build_scheduler(
#         optimizer, cfg["job"]["LCMTrain"]["lr_scheduler"]
#     )
#
#     if RayTrain.get_checkpoint():
#         loaded_checkpoint = RayTrain.get_checkpoint()
#         with loaded_checkpoint.as_directory() as loaded_checkpoint_dir:
#             model_state, optimizer_state = torch.load(
#                 os.path.join(loaded_checkpoint_dir, "checkpoint.pt")
#             )
#             net.load_state_dict(model_state)
#             optimizer.load_state_dict(optimizer_state)
#
#     train_data = build_train(cfg, lsm_choice)
#     train_set = DataLoader(
#         train_data,
#         batch_size=cfg["job"]["LCMTrain"]["train"]["batch_size"],
#         drop_last=cfg["job"]["LCMTrain"]["train"]["drop_last"],
#         num_workers=cfg["job"]["LCMTrain"]["train"]["num_workers"],
#         pin_memory=True,
#     )
#     validate_data = build_validate(cfg, lsm_choice)
#     validate_set = DataLoader(
#         validate_data,
#         batch_size=cfg["job"]["LCMTrain"]["test"]["batch_size"],
#         drop_last=cfg["job"]["LCMTrain"]["test"]["drop_last"],
#         num_workers=cfg["job"]["LCMTrain"]["test"]["num_workers"],
#         pin_memory=True,
#     )
#
#     for epoch in range(20):  # loop over the dataset multiple times
#         running_loss = 0.0
#         epoch_steps = 0
#         net.train()
#         for i, data in enumerate(train_set):
#             # get the inputs; data is a list of [inputs, labels]
#             labels, feats = data
#             labels, feats = labels.to(device), feats.to(device)
#
#             # zero the parameter gradients
#             optimizer.zero_grad()
#
#             # forward + backward + optimize
#             pred = net(feats)
#             loss = criterion(pred, labels)
#             loss.backward()
#             optimizer.step()
#
#             # print statistics
#             running_loss += loss.item()
#             epoch_steps += 1
#             if i % 2000 == 1999:  # print every 2000 mini-batches
#                 print(
#                     "[%d, %5d] loss: %.3f"
#                     % (epoch + 1, i + 1, running_loss / epoch_steps)
#                 )
#                 running_loss = 0.0
#             if scheduler is not None:
#                 scheduler.step()
#
#         # Validation loss
#         val_loss = 0.0
#         val_steps = 0
#         net.eval()
#         for i, data in enumerate(validate_set, 0):
#             with torch.no_grad():
#                 labels, feats = data
#                 labels, feats = labels.to(device), feats.to(device)
#
#                 pred = net(feats)
#                 loss = criterion(pred, labels)
#                 val_loss += loss.cpu().numpy()
#                 val_steps += 1
#
#         with tempfile.TemporaryDirectory() as temp_checkpoint_dir:
#             path = os.path.join(temp_checkpoint_dir, "checkpoint.pt")
#             torch.save((net.state_dict(), optimizer.state_dict()), path)
#             checkpoint = RayTrain.Checkpoint.from_directory(temp_checkpoint_dir)
#             RayTrain.report(
#                 {"loss": (val_loss / val_steps)},
#                 checkpoint=checkpoint,
#             )
#     print("Finished Training")
#
#
# def main():
#     from axe.data.io import Reader
#
#     config = Reader.read_config("axe.toml")
#
#     logging.basicConfig(
#         format=config["log"]["format"], datefmt=config["log"]["datefmt"]
#     )
#
#     log = logging.getLogger(config["log"]["name"])
#     log.setLevel(config["log"]["level"])
#
#     config["lcm"]["model"]["embedding_size"] = RayTune.choice([4, 8])
#     config["lcm"]["model"]["hidden_length"] = RayTune.choice([2, 3, 4])
#     config["lcm"]["model"]["hidden_width"] = RayTune.choice([32, 64, 128])
#     config["train"]["optimizer"]["Adam"]["lr"] = RayTune.loguniform(1e-4, 1e-1)
#     config["job"]["LCMTrain"]["lr_scheduler"] = RayTune.choice(
#         ["CosineAnnealing", "Constant"]
#     )
#     config["job"]["LCMTrain"]["train"]["batch_size"] = RayTune.choice(
#         [1024, 2048, 4096, 8192, 16384]
#     )
#     scheduler = ASHAScheduler(grace_period=3, max_t=20, reduction_factor=2)
#
#     ray.init(num_gpus=1)
#
#     tuner = RayTune.Tuner(
#         RayTune.with_resources(
#             RayTune.with_parameters(train_lcm),
#             resources={"cpu": 4, "gpu": 0},
#         ),
#         tune_config=RayTune.TuneConfig(
#             metric="loss",
#             mode="min",
#             scheduler=scheduler,
#             num_samples=2,
#         ),
#         param_space=config,
#     )
#     results = tuner.fit()
#     best_result = results.get_best_result("loss", "min")
#     assert best_result.config is not None
#     assert best_result.metrics is not None
#
#     print("Best trial config: {}".format(best_result.config))
#     with open('best.toml', "w") as fid:
#         toml.dump(best_result.config, fid)
#     print("Best trial final validation loss: {}".format(best_result.metrics["loss"]))
#
#
# if __name__ == "__main__":
#     main()
