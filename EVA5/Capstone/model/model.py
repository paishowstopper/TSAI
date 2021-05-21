import torch
import torch.nn as nn

from encoder.encoder import _make_resnet_encoder

from decoder_midas.midas_net import MidasNet
from decoder_planercnn.model import MaskRCNN
from decoder_yolo.models import *
from decoder_yolo.utils import torch_utils

class Model(nn.Module):

	def __init__(self, yolo_cfg, midas_cfg, planercnn_cfg, path = None):
		
		super(Model, self).__init__()
		
		self.yolo_params = yolo_cfg
		self.midas_params = midas_cfg
		self.planercnn_params = planercnn_cfg
		self.path = path

		self.encoder = _make_resnet_encoder()

		self.yolo_decoder = Darknet(self.yolo_params)
		self.midas_decoder = MidasNet(path)
		self.planercnn_decoder = MaskRCNN(self.planercnn_params)

		self.conv1 = nn.Conv2d(in_channels=2048, out_channels=1024, kernel_size=(1, 1), padding=0, bias=False)
		self.conv2 = nn.Conv2d(in_channels=1024, out_channels=512, kernel_size=(1, 1), padding=0, bias=False)
		self.conv3 = nn.Conv2d(in_channels=512, out_channels=256, kernel_size=(1, 1), padding=0, bias=False)
		
		self.info(False)
		
	def forward(self, yolo_ip, midas_ip, planercnn_ip):

		x = yolo_ip

		# Encoding
		layer_1 = self.encoder.layer1(x)
		layer_2 = self.encoder.layer2(layer_1)
		layer_3 = self.encoder.layer3(layer_2)
		layer_4 = self.encoder.layer4(layer_3)

		yolo_75 = self.conv1(layer_4)
		yolo_61 = self.conv2(layer_3)
		yolo_36 = self.conv3(layer_2)

		# Decoding
		# Mask RCNN
		planercnn_op = None
		if planercnn_ip is not None:
			planercnn_ip['input'][0] = yolo_ip
			planercnn_op = self.planercnn_decoder.forward(planercnn_ip, [layer_1, layer_2, layer_3, layer_4])

		# MiDaS
		midas_op = None
		if midas_ip is not None:
			midas_op = self.midas_decoder([layer_1, layer_2, layer_3, layer_4])

		# Yolo
		if not self.training:
			inference_op, training_op = self.yolo_decoder(yolo_75, yolo_61, yolo_36)
			yolo_op = [inference_op, training_op]
		else:
			yolo_op = self.yolo_decoder(yolo_75, yolo_61, yolo_36)

		return  yolo_op, midas_op, planercnn_op

	def info(self, verbose=False):
		torch_utils.model_info(self, verbose)