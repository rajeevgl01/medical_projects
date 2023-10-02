import torch
import torch.nn as nn
import argparse
import numpy as np
import io

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
	parser.add_argument('--S_path', type=str, default=None)
	parser.add_argument('--U_path', type=str, default=None)
	parser.add_argument('--V_path', type=str, default=None)
	parser.add_argument('--resume', type=str, default=None)

	args, unparsed = parser.parse_known_args()

	return args

class LowRankLinear(nn.Module):
	def __init__(self, dim, rank):
		super().__init__()
		self.lowRankLinear = nn.Linear(dim, rank)

	def forward(self, x):
		return self.lowRankLinear(x) 


def main(args):
	try:
		input_tensor = torch.load(args.mapping_path)
	except:
		file = args.mapping_path
		with open(file, 'rb') as f:
			buffer = io.BytesIO(f.read())

		input_tensor = torch.load(buffer)
	# start = time.time()
	n, dim = input_tensor.shape
	_logger.info(f"dims: {n} X {dim}")

	# kernel_matrix = input_tensor @ input_tensor.transpose(-2, -1)
	# kernel_matrix = kernel_matrix.numpy()
	# U, S, V = np.linalg.svd(kernel_matrix)

	# end = time.time()

	# U, S, V = torch.tensor(U), torch.tensor(S), torch.tensor(V)
	# _logger.info(end-start)
	# torch.save(U, f'{args.out_dir}/U_{args.model}.pt')
	# torch.save(S, f'{args.out_dir}/S_{args.model}.pt')
	# torch.save(V, f'{args.out_dir}/V_{args.model}.pt')
	
	assert args.model in args.U_path, "Model and tensor paths do not match"
	assert args.model in args.S_path, "Model and tensor paths do not match"

	U, S, V = torch.load(args.U_path), torch.load(args.S_path), torch.load(args.V_path)
	
	r = int(args.rank_factor * dim)
	_logger.info(f"rank: {r}")
	U_r = U[:, :r]
	S_r = S[:r]
	V_r = V[:r, :r]

	target_tensor = U_r @ (torch.diag(S_r) @ V_r)

	# target_tensor = torch.from_numpy(target_tensor)

	device = "cuda:0"
	torch.cuda.set_device(0)
	model = LowRankLinear(dim, r).to(device)
	model.cuda()	
	
	criterion = nn.MSELoss().cuda()
	optimizer = torch.optim.Adam(model.parameters(), lr=5e-4)

	input_tensor = input_tensor.cuda(non_blocking=True)
	target_tensor = target_tensor.cuda(non_blocking=True)
	start_epoch = 0
	t = 0
	num_epochs = 50000
	best_loss = 10000000
	total_time = 0
	total_epochs = 0
	
	if args.resume:
		checkpoint = torch.load(args.resume)
		model.load_state_dict(checkpoint['model'])
		optimizer.load_state_dict(checkpoint['optimizer'])
		start_epoch = int(checkpoint['epoch']) + 1
		best_loss = float(checkpoint['best_loss'])

	for epoch in range(start_epoch, num_epochs):
		start = time.time()
		pred = model(input_tensor)
		loss = criterion(pred, target_tensor)
		
		optimizer.zero_grad()
		loss.backward()
		optimizer.step()
		end = time.time()

		total_time += (end - start)
		total_epochs += 1

		if epoch % 500 == 0 or epoch == start_epoch:
			average_time = total_time/total_epochs
			_logger.info(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item(): .8f}, time: {average_time: .4f}s')
		
		best_loss = min(best_loss, loss)

		to_save = {
			'model': model.state_dict(),
			'optimizer': optimizer.state_dict(),
			'epoch': epoch,
			'loss': loss,
			'best_loss': best_loss
            }
		
		torch.save(to_save, f'{args.out_dir}/checkpoint_{args.model}_rank_{r}_dim_{dim}.pt')
		if best_loss <= loss:
			torch.save(to_save, f'{args.out_dir}/best_checkpoint_{args.model}_rank_{r}_dim_{dim}.pt')

		if best_loss <= 1e-4:
			_logger.info("Early Stopping")
			break
	return r

if __name__ == "__main__":
	args = parse_option()
	setup_default_logging()

	r = main(args)
	_logger.info(f"end for rank {r}")