#!/usr/bin/env python
import logging
import os
import tempfile

import ray.train as RayTrain
import ray.tune as RayTune
from ray.tune.schedulers import ASHAScheduler

import torch
from torch.utils.data import DataLoader

from endure.lcm.data.dataset import LCMDataSet
from endure.lcm.model.builder import LearnedCostModelBuilder
from endure.lsm.types import STR_POLICY_DICT, Policy
from endure.util.losses import LossBuilder
from endure.util.optimizer import OptimizerBuilder


def build_train(cfg, lsm_design: Policy) -> DataLoader:
    train_dir = os.path.join(
        cfg["io"]["data_dir"],
        cfg["job"]["LCMTrain"]["train"]["dir"],
    )
    train_data = LCMDataSet(
        folder=train_dir,
        lsm_design=lsm_design,
        min_size_ratio=cfg["lsm"]["size_ratio"]["min"],
        max_size_ratio=cfg["lsm"]["size_ratio"]["max"],
        max_levels=cfg["lsm"]["max_levels"],
        test=False,
        shuffle=cfg["job"]["LCMTrain"]["train"]["shuffle"],
    )
    train = DataLoader(
        train_data,
        batch_size=cfg["job"]["LCMTrain"]["train"]["batch_size"],
        drop_last=cfg["job"]["LCMTrain"]["train"]["drop_last"],
        num_workers=cfg["job"]["LCMTrain"]["train"]["num_workers"],
        pin_memory=True,
    )

    return train


def build_validate(cfg, lsm_design: Policy) -> DataLoader:
    validate_dir = os.path.join(
        cfg["io"]["data_dir"],
        cfg["job"]["LCMTrain"]["test"]["dir"],
    )
    validate_data = LCMDataSet(
        folder=validate_dir,
        lsm_design=lsm_design,
        min_size_ratio=cfg["lsm"]["size_ratio"]["min"],
        max_size_ratio=cfg["lsm"]["size_ratio"]["max"],
        max_levels=cfg["lsm"]["max_levels"],
        test=True,
        shuffle=cfg["job"]["LCMTrain"]["test"]["shuffle"],
    )
    validate = DataLoader(
        validate_data,
        batch_size=cfg["job"]["LCMTrain"]["test"]["batch_size"],
        drop_last=cfg["job"]["LCMTrain"]["test"]["drop_last"],
        num_workers=cfg["job"]["LCMTrain"]["test"]["num_workers"],
        pin_memory=True,
    )

    return validate


def train_lcm(cfg):
    lsm_choice = STR_POLICY_DICT.get(cfg["lsm"]["design"], Policy.KHybrid)
    net = LearnedCostModelBuilder(cfg).build_model(cfg["lsm"]["design"])

    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda:0"
    net.to(device)

    criterion = LossBuilder(cfg).build(cfg["job"]["LCMTrain"]["loss_fn"])
    if criterion is None:
        raise TypeError(f"Loss choice invalid: {cfg['job']['LCMTrain']['loss_fn']}")
    assert criterion is not None
    optimizer = OptimizerBuilder(cfg).build_optimizer(
        cfg["job"]["LCMTrain"]["optimizer"], net
    )

    if RayTrain.get_checkpoint():
        loaded_checkpoint = RayTrain.get_checkpoint()
        with loaded_checkpoint.as_directory() as loaded_checkpoint_dir:
            model_state, optimizer_state = torch.load(
                os.path.join(loaded_checkpoint_dir, "checkpoint.pt")
            )
            net.load_state_dict(model_state)
            optimizer.load_state_dict(optimizer_state)

    train_set = build_train(cfg, lsm_choice)
    validate_set = build_validate(cfg, lsm_choice)

    for epoch in range(10):  # loop over the dataset multiple times
        running_loss = 0.0
        epoch_steps = 0
        net.train()
        for i, data in enumerate(train_set):
            # get the inputs; data is a list of [inputs, labels]
            labels, feats = data
            labels, feats = labels.to(device), feats.to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            pred = net(feats)
            loss = criterion(pred, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            epoch_steps += 1
            if i % 2000 == 1999:  # print every 2000 mini-batches
                print(
                    "[%d, %5d] loss: %.3f"
                    % (epoch + 1, i + 1, running_loss / epoch_steps)
                )
                running_loss = 0.0

        # Validation loss
        val_loss = 0.0
        val_steps = 0
        net.eval()
        for i, data in enumerate(validate_set, 0):
            with torch.no_grad():
                labels, feats = data
                labels, feats = labels.to(device), feats.to(device)

                pred = net(feats)
                loss = criterion(pred, labels)
                val_loss += loss.cpu().numpy()
                val_steps += 1

        with tempfile.TemporaryDirectory() as temp_checkpoint_dir:
            path = os.path.join(temp_checkpoint_dir, "checkpoint.pt")
            torch.save((net.state_dict(), optimizer.state_dict()), path)
            checkpoint = RayTrain.Checkpoint.from_directory(temp_checkpoint_dir)
            RayTrain.report(
                {"loss": (val_loss / val_steps)},
                checkpoint=checkpoint,
            )
    print("Finished Training")


def main():
    from endure.data.io import Reader

    config = Reader.read_config("endure.toml")

    logging.basicConfig(
        format=config["log"]["format"], datefmt=config["log"]["datefmt"]
    )

    log = logging.getLogger(config["log"]["name"])
    log.setLevel(config["log"]["level"])

    config["lcm"]["model"]["embedding_size"] = RayTune.grid_search([4, 8])
    scheduler = ASHAScheduler(max_t=10, grace_period=1, reduction_factor=2)

    tuner = RayTune.Tuner(
        RayTune.with_resources(
            RayTune.with_parameters(train_lcm),
            resources={"cpu": 2, "gpu": 0},
        ),
        tune_config=RayTune.TuneConfig(
            metric="loss",
            mode="min",
            scheduler=scheduler,
            num_samples=2,
        ),
        param_space=config,
    )
    results = tuner.fit()
    best_result = results.get_best_result("loss", "min")
    assert best_result.metrics is not None

    print("Best trial config: {}".format(best_result.config))
    print("Best trial final validation loss: {}".format(best_result.metrics["loss"]))


if __name__ == "__main__":
    main()
