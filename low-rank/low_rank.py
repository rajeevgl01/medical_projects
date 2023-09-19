import torch
import torch.nn as nn
import argparse
import numpy as np

import torch.backends.cudnn as cudnn
import logging
from timm.utils import setup_default_logging
import time

_logger = logging.getLogger('train')


def parse_option():
    parser = argparse.ArgumentParser('Low rank computation', add_help=False)
    parser.add_argument('--rank_factor', type=float, default=0.1)
    parser.add_argument('--is_relu', action='store_true')
    parser.add_argument('--mapping_path', type=str, default=None)
    parser.add_argument('--model', type=str, default=None)
    parser.add_argument('--out_dir', type=str, default=None)

    args, unparsed = parser.parse_known_args()

    return args


class LowRankLinear(nn.Module):
    def __init__(self, dim, rank):
        super().__init__()
        self.lowRankLinear = nn.Linear(dim, rank)

    def forward(self, x):
        return self.lowRankLinear(x)


def main(args):
    input_tensor = torch.load(
        args.mapping_path, map_location=torch.device('cpu'))
    start = time.time()
    n, dim = input_tensor.shape
    _logger.info(f"dim: {n}")

    kernel_matrix = input_tensor @ input_tensor.transpose(-2, -1)
    kernel_matrix = kernel_matrix.numpy()
    U, S, V = np.linalg.svd(kernel_matrix)

    end = time.time()

    U, S = torch.tensor(U), torch.tensor(S)
    _logger.info(end-start)
    torch.save(U, f'{args.out_dir}/U_{args.model}.pt')
    torch.save(S, f'{args.out_dir}/S_{args.model}.pt')

    # r = int(args.rank_factor * n)
    # _logger.info(f"rank: {r}")
    # U_r = U[:, :r]
    # S_r = S[:r]
    # target_tensor = U_r @ np.sqrt(np.diag(S_r))
    # target_tensor = torch.from_numpy(target_tensor)

    # device = "cuda:0"
    # torch.cuda.set_device(0)
    # model = LowRankLinear(dim, r).to(device)
    # model.cuda()

    # criterion = nn.MSELoss().cuda()
    # optimizer = torch.optim.Adam(model.parameters(), lr=5e-4)

    # input_tensor = input_tensor.cuda(non_blocking=True)
    # target_tensor = target_tensor.cuda(non_blocking=True)

    # num_epochs = 100000
    # best_loss = 1000000000
    # for epoch in range(num_epochs):
    # 	pred = model(input_tensor)
    # 	loss = criterion(pred, target_tensor)

    # 	optimizer.zero_grad()
    # 	loss.backward()
    # 	optimizer.step()

    # 	if epoch % 5000 == 0:
    # 		_logger.info(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item(): .8f}')

    # 	best_loss = min(best_loss, loss)

    # 	torch.save(model.state_dict(), f'{args.out_dir}/checkpoint_{args.rank_factor}_{r}.pt')
    # 	if best_loss <= loss:
    # 		torch.save(model.state_dict(), f'{args.out_dir}/best_checkpoint_{args.rank_factor}_{r}.pt')

    # 	if best_loss <= 0.000001:
    # 		_logger.info("Early Stopping")
    # 		break
    # return r


if __name__ == "__main__":
    args = parse_option()
    setup_default_logging()

    main(args)
    # _logger.info(f"end for rank {r}")
