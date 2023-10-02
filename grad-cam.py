import torch
import torchvision.transforms as transforms
from torchvision import models
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import argparse
import models_vit
import models_resnet
import models_densenet

import os
from torchvision import datasets, transforms

from timm.data import create_transform
from timm.utils import accuracy
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from .custom_transforms import GaussianBlur
import torch
from .augment import new_data_aug_generator

def get_args_parser():
	parser = argparse.ArgumentParser('grad-cam', add_help=False)

	# Model parameters
	parser.add_argument('--model', default='vit_large_patch16', type=str, metavar='MODEL',
						help='Name of model to train')
	parser.add_argument('--softRank', type=int, default=76, help='low rank')
	parser.add_argument('--retrain', type=str, default=None,
						help='retraining the model')

	parser.add_argument('--input_size', default=224, type=int,
						help='images input size')

	parser.add_argument('--drop_path', type=float, default=0.1, metavar='PCT',
						help='Drop path rate (default: 0.1)')

	parser.add_argument('--global_pool', action='store_true')
	parser.set_defaults(global_pool=True)
	
	parser.add_argument('--nb_classes', default=1000, type=int,
						help='number of the classification types')

	parser.add_argument('--output_dir', default='./output_dir',
						help='path where to save, empty for no saving')
	parser.add_argument('--data_path', help='path to data directory')
	
	parser.add_argument('--vit_dropout_rate', type=float, default=0,
						help='Dropout rate for ViT blocks (default: 0.0)')
	parser.add_argument("--grad_list", type=str, required=True)
	parser.add_argument("--dataset", type=str, default='chestxray')

	return parser

def build_transform(args):
	try:
		if args.dataset == 'chestxray' or args.dataset == 'covidx' or args.dataset == 'chexpert':
			mean = (0.5056, 0.5056, 0.5056)
			std = (0.252, 0.252, 0.252)
	except:
		mean = IMAGENET_DEFAULT_MEAN
		std = IMAGENET_DEFAULT_STD

    # eval transform
    t = []
    if args.input_size <= 224:
        crop_pct = 224 / 256
    else:
        crop_pct = 1.0
    size = int(args.input_size / crop_pct)
    t.append(
        transforms.Resize(size, interpolation=PIL.Image.BICUBIC),  # to maintain same ratio w.r.t. 224 images
    )
    t.append(transforms.CenterCrop(args.input_size))

    t.append(transforms.ToTensor())
    t.append(transforms.Normalize(mean, std))
    return transforms.Compose(t)


def main(args, input_image):

dataset_grad = build_dataset_chest_xray(split='grad', args=args)
dataset_train = build_dataset_chest_xray(split='train', args=args)dataset_train = build_dataset_chest_xray(split='train', args=args)
# Load a pre-trained model (e.g., ResNet50)
	if 'vit' in args.model:
		model = models_vit.__dict__[args.model](
			img_size=args.input_size,
			num_classes=args.nb_classes,
			drop_rate=args.vit_dropout_rate,
			drop_path_rate=args.drop_path,
			global_pool=args.global_pool,
		)

	elif 'densenet' in args.model:
		model = models_densenet.__dict__[args.model](num_classes=args.nb_classes)
	elif 'resnet' in args.model:
		model = models_resnet.__dict__[args.model](num_classes=args.nb_classes)
	else:
		raise NotImplementedError

	model.eval()

	# Load and preprocess an image
	image_path = input_image
	image = Image.open(image_path)
	preprocess = build_transform(args)
	input_tensor = preprocess(image)
	input_batch = input_tensor.unsqueeze(0)

	# Forward pass
	output = model(input_batch)

	# Get the class index (e.g., for a specific class)TODO
	class_idx = 123  # Replace with the class index you're interested in

	# Compute the gradient-weighted activation map (Grad-CAM)
	grad_cam = compute_grad_cam(model, input_batch, class_idx)

	# Overlay Grad-CAM on the original image
	heatmap = cv2.applyColorMap(np.uint8(255 * grad_cam), cv2.COLORMAP_JET)
	output_image = cv2.addWeighted(cv2.cvtColor(image, cv2.COLOR_RGB2BGR), 0.5, heatmap, 0.5, 0)

	# Display the result
	plt.imshow(cv2.cvtColor(output_image, cv2.COLOR_BGR2RGB))
	plt.axis('off')
	if args.softRank:
		plt.savefig(os.path.join(args.output_dir, f"{image_path}_softrank.png"))
	else:
		plt.savefig(os.path.join(args.output_dir, f"{image_path}_normal.png"))


if __name__ == "__main__":
	args = get_args_parser()
	args = args.parse_args()

	main(args)