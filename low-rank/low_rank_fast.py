import os

import torch
import torch.nn as nn
import argparse
import numpy as np
import scipy

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
	input_tensor = torch.load(args.mapping_path, map_location=torch.device('cpu'))
	start = time.time()
	n, dim = input_tensor.shape
	_logger.info(f"dim: {n}")

	kernel_matrix = input_tensor #@ input_tensor.transpose(-2, -1)
	kernel_matrix = kernel_matrix.numpy()
	# U, S, V = np.linalg.svd(kernel_matrix)
	# kernel_matrix = scipy.sparse.csr_matrix(kernel_matrix)
	# U, S, V = scipy.sparse.linalg.svds(kernel_matrix, dim, solver='propack')
	U, S, V = scipy.linalg.svd(kernel_matrix, full_matrices=False)


	end = time.time()
	
	# S = np.power(S, 2)
	U, S, V = torch.tensor(U), torch.tensor(S), torch.tensor(V)
	_logger.info(end-start)
	torch.save(U, f'{args.out_dir}/U_{args.model}_{dim}_fast.pt')
	torch.save(S, f'{args.out_dir}/S_{args.model}_{dim}_fast.pt')
	torch.save(V, f'{args.out_dir}/V_{args.model}_{dim}_fast.pt')

if __name__ == "__main__":
	args = parse_option()
	setup_default_logging()

	main(args)
	# _logger.info(f"end for rank {r}")