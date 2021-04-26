import sys
import argparse
import torch
import numpy as np
import torch.nn as nn
from torch.utils.data import DataLoader
from KineNet import KineNet, KNData


def main():
    # Defaults arguments
    dtype = sys.argv[1] if len(sys.argv) > 1 and sys.argv[1] is not None else "real"
    parser = argparse.ArgumentParser()
    if dtype == "synth":
        parser.add_argument("--data-type", type=str, default="synth")
        parser.add_argument("--joint-coord", type=str, default="../data/test/synth/")
        parser.add_argument("--joint-states", type=str, default="../data/test/synth/")
    else:
        parser.add_argument("--data-type", type=str, default="real")
        parser.add_argument("--joint-coord", type=str, default="../data/test/set6_output.npy")
        parser.add_argument("--joint-states", type=str, default="../data/test/set6_slu/")
    parser.add_argument("--robot-path", type=str, default="../data/urdf/mh5l.urdf")
    parser.add_argument("--robot", type=str, default="mh5l")
    parser.add_argument("--batch-size", type=int, default=100)
    args, unknown = parser.parse_known_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = KineNet()
    model.load_state_dict(torch.load("../model/mh5l_kinenet.pt"))
    model.to(device)
    model.eval()

    dataset = KNData(args.data_type, args.robot_path, args.joint_coord, args.joint_states)
    test_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

    total_loss = 0.0
    loss_fn = nn.MSELoss()
    result = None
    for data, target in test_loader:
        data, target = data.to(device), target.to(device)
        output = model(data)
        if result is not None:
            result = torch.cat((result, output-target))
        else:
            result = output-target
        total_loss += loss_fn(output, target)
    print(f"Total loss = {total_loss}")
    print(f'Mean difference measured in radians in order of joint (S,L,U,R,B):')
    print(torch.mean(result, 0).detach().cpu().numpy())


if __name__ == "__main__":
    main()
