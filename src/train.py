import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import pytorch_pfn_extras as ppe
import pandas as pd
import matplotlib.pyplot as plt
import pytorch_pfn_extras.training.extensions as extensions
import numpy as np
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Subset
from KineNet import KineNet, KNData


def get_data_loaders(args):
    """
    Builds splits validation and training data from KNData loader. Ensures loaders are shuffled and randomly
    sampled.
    :param args: Useful arguments declared in main body
    :type args: argparse structure
    :return: loaders for both train and validation data
    :rtype: tuple of torch tensors - dims = ([(batch) x (joints) x 3], [(batch) x (joints - T)])
    """
    dataset = KNData(args.data_type, args.robot_path, args.joint_coord, args.joint_states)
    train_size = int(len(dataset) * args.train_val_ratio)
    train_dataset = Subset(dataset, list(range(0, train_size)))
    val_dataset = Subset(dataset, list(range(train_size, len(dataset))))
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=True)
    return train_loader, val_loader


def train(manager, args, model, device, train_loader, results):
    """
    Model train step.
    """
    while not manager.stop_trigger:
        model.train()
        loss_fn = nn.MSELoss()
        for data, target in train_loader:
            with manager.run_iteration(step_optimizers=["main"]):
                data, target = data.to(device), target.to(device)
                output = model(data.float())
                loss = loss_fn(output, target)
                ppe.reporting.report({"train/loss": loss})
                loss.backward()


def validate(args, model, device, data, target, results):
    """
    Model validation step.
    """
    model.eval()
    loss_fn = nn.MSELoss()
    data, target = data.to(device), target.to(device)
    output = model(data)
    loss = loss_fn(output, target)
    ppe.reporting.report({"val/loss": loss})
    for i in range(output.shape[1]):
        if (i == 0):
            results.append([loss_fn(output[:, i], target[:, i]).tolist()])
        else:
            results[-1].append(loss_fn(output[:, i], target[:, i]).tolist())


def plot_loss(df):
    """
    Plots training results by joint per epoch.
    :param df: validation data results
    :type df: pandas dataframe
    """
    figure, axes = plt.subplots(nrows=2, ncols=3, sharex=True, sharey=True, figsize=(12, 5))
    for i in range(len(df.columns)):
        axes[int(np.floor(i / 3)), i % 3].plot(df[df.columns[i]])
        axes[int(np.floor(i / 3)), i % 3].set_title(f'{df.columns[i]}-Joint Angle')
    figure.text(0.5, 0.04, 'Epochs', ha='center', fontsize=16)
    figure.text(0.08, 0.5, 'MSE Loss', va='center', rotation='vertical', fontsize=16)
    figure.savefig('result/train_loss.png')


def main():
    # Defaults arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-type", type=str, default="synth")
    parser.add_argument("--joint-coord", type=str, default="../data/train/synth/")
    parser.add_argument("--joint-states", type=str, default="../data/train/synth/")
    parser.add_argument("--robot-path", type=str, default="../data/urdf/mh5l.urdf")
    parser.add_argument("--robot", type=str, default="mh5l")
    parser.add_argument("--train-val-ratio", type=float, default=0.8)
    parser.add_argument("--batch-size", type=int, default=1000)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--save-model", action="store_true", default=True)
    args, unknown = parser.parse_known_args()

    # Define torch device based upon GPU availability
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = KineNet()
    # Assign device to model
    model.to(device)
    train_loader, val_loader = get_data_loaders(args)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    trigger = ppe.training.triggers.EarlyStoppingTrigger(
        check_trigger=(3, "epoch"), monitor="val/loss"
    )

    # Configure extensions
    results = []
    my_extensions = [
        extensions.LogReport(),
        extensions.ProgressBar(),
        extensions.observe_lr(optimizer=optimizer),
        extensions.ParameterStatistics(model, prefix="model"),
        extensions.VariableStatisticsPlot(model),
        extensions.Evaluator(
            val_loader,
            model,
            eval_func=lambda data, target: validate(args, model, device, data, target, results),
            progress_bar=True
        ),
        extensions.PlotReport(["train/loss", "val/loss"], "epoch", filename="loss.png"),
        extensions.PrintReport(
            [
                "epoch",
                "iteration",
                "train/loss",
                "lr",
                "val/loss",
            ]
        ),
    ]

    # Setup pfn extensions manager
    manager = ppe.training.ExtensionsManager(
        model,
        optimizer,
        args.epochs,
        extensions=my_extensions,
        iters_per_epoch=len(train_loader),
        stop_trigger=trigger,
    )

    train(manager, args, model, device, train_loader, results)
    plot_loss(pd.DataFrame(results, columns=['S', 'L', 'U', 'B', 'R']))

    if args.save_model:
        torch.save(model.state_dict(), "../model/mh5l_kinenet.pt")


if __name__ == "__main__":
    main()
