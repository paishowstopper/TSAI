import torch
import torch.nn as nn

from Encoder.encoder import _make_resnet_encoder
from Decoder.MiDaS.midas_net import MidasNet #TO BE CHANGED
from Decoder.PlaneRCNN.model import MaskRCNN
from Decoder.YoloV3.models import *
from Decoder.YoloV3.utils import torch_utils

class VisionNet(nn.Module):
	'''
		Network for detecting objects, generate depth map and identify plane surfaces
	'''
	def __init__(self,yolo_cfg,midas_cfg,planercnn_cfg,path=None):
		super(VisionNet, self).__init__()
		"""
			Get required configuration for all the 3 models
		
		"""
		self.yolo_params = yolo_cfg
		self.midas_params = midas_cfg
		self.planercnn_params = planercnn_cfg
		self.path = path

		use_pretrained = False if path is None else True

		print('use_pretrained',use_pretrained)
		print('path',path)
		
		self.encoder = _make_resnet_encoder(use_pretrained)

		self.plane_decoder = MaskRCNN(self.planercnn_params,self.encoder) #options, config, modelType='final'

		self.depth_decoder = MidasNet(path) # TO BE CHANGED

		self.bbox_decoder =  Darknet(self.yolo_params)

		self.conv1 = nn.Conv2d(in_channels=2048, out_channels=1024, kernel_size=(1, 1), padding=0, bias=False)
		self.conv2 = nn.Conv2d(in_channels=1024, out_channels=512, kernel_size=(1, 1), padding=0, bias=False)
		self.conv3 = nn.Conv2d(in_channels=512, out_channels=256, kernel_size=(1, 1), padding=0, bias=False)
		
		self.info(False)

	def forward(self,yolo_ip,midas_ip,plane_ip):

		x = yolo_ip

		# Encoder blocks
		layer_1 = self.encoder.layer1(x)
		layer_2 = self.encoder.layer2(layer_1)
		layer_3 = self.encoder.layer3(layer_2)
		layer_4 = self.encoder.layer4(layer_3)

		Yolo_75 = self.conv1(layer_4)
		Yolo_61 = self.conv2(layer_3)
		Yolo_36 = self.conv3(layer_2)

		if plane_ip is not None:
			plane_ip['input'][0] = yolo_ip
			# PlaneRCNN decoder
			plane_out = self.plane_decoder.forward(plane_ip,[layer_1, layer_2, layer_3, layer_4])
		else:
			plane_out = None

		if midas_ip is not None:
			# MiDaS depth decoder
			depth_out = self.depth_decoder([layer_1, layer_2, layer_3, layer_4])
		else:
			depth_out = None

		#YOLOv3 bbox decoder
		if not self.training:
			inf_out, train_out = self.bbox_decoder(Yolo_75,Yolo_61,Yolo_36)
			bbox_out=[inf_out, train_out]
		else:
			bbox_out = self.bbox_decoder(Yolo_75,Yolo_61,Yolo_36)

		return  plane_out, bbox_out, depth_out

	def info(self, verbose=False):
		torch_utils.model_info(self, verbose)