""" File that serves as a pytorch transform to remove background from images """

import torch
import os

from .u2net import U2NETP # Lightweight U2net, path is attached in github

U2NETP_PATH = os.path.join(os.path.dirname(os.path.expanduser(__file__)), 'u2netp.pth')

class BGMask(torch.nn.Module):
	""" Like a torchvision.transforms object.
		Should operate on tensors (cuda or o/w)
	"""

	def __init__(self, background='r'):
		super().__init__()
		assert background in ['r', 'g'] # English sucks, grey==gray
		self.background = background

		masker = U2NETP(3, 1)
		masker.load_state_dict(torch.load(U2NETP_PATH, map_location='cpu'))
		self.masker = masker
		self.sample_param = self.masker.outconv.weight # for checking devices

	def forward(self, x, new_bg=None):
		""" X is an image(s) to remove background from
			if new_bg is not None, is a tensor of same shape as x
		"""
		# Step 0: Move masking network to wherever the data lives
		if self.sample_param.device != x.device:
			self.masker.to(x.device)
		# Step 1: identify background
		# this is kinda voodoo, don't worry about it
		singleton = False
		with torch.no_grad():
			if x.dim() == 3:
				singleton = True
				x = x.unsqueeze(0)
			mask = (sum(self.masker(x)) / 7.0).squeeze() > 0.66
			if mask.dim() == 2:
				mask = mask.unsqueeze(0)
			mask = mask.unsqueeze(1).expand_as(x)

		# Step 2: make new image
		if new_bg is not None:
			new_bg = new_bg.clone()
		else:
			if self.background == 'r':
				new_bg = torch.rand_like(x)
			elif self.background == 'g':
				new_bg = torch.ones_like(x) * 0.5
		new_bg[mask] = x[mask]

		if singleton: # some stupid hacky nonsense to make this work as a transform
			return new_bg.squeeze(0)
		return new_bg
