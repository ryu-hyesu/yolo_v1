import cv2
import torch
import torchvision
from torch.utils import data

from utils.utils import Timer
from .yolo import nms
from IPython import display
from utils.visualize import *
from utils.metrics import *


def test_mAP_IoU(net: torch.nn.Module, test_iter_raw: data.DataLoader, device: torch.device):
	"""Calculate VOCmAP and COCOmAP on test dataset, and draw VOC-AP for each category

	Args:
		net (torch.nn.Module): network
		test_iter_raw (data.DataLoader): test dataloader (raw)
		device (torch.device): device
	"""
	with torch.no_grad():
		net.eval()
		net.to(device)

		# metrics calculation
		calc = ObjectDetectionMetricsCalculator(20, 0.1)

		for i, (X, YRaw) in enumerate(test_iter_raw):
			print("Batch %d / %d" % (i, len(test_iter_raw)))
			display.clear_output(wait=True)
			X = X.to(device)
			YHat = net(X)
			for yhat, yraw in zip(YHat, YRaw):
				yhat = nms(yhat)
				calc.add_image_data(yhat.cpu(), yraw)
			for cat in calc.data:
				for i in cat['data']:
					i.IoU

		print("Test VOC mAP:", calc.calculate_VOCmAP())
		print("Test IoU:", calc.calculate_VOCIoU())

