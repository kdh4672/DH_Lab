import torch
import torch.nn as nn
import torch.nn.functional as F

class FocalLoss(nn.Module):
	def __init__(self, gamma=2, alpha=0.25, size_average=True):
		super(FocalLoss, self).__init__()
		self.gamma = gamma
		self.alpha = alpha
		if isinstance(alpha, (float, int)): self.alpha = torch.Tensor([alpha, 1 - alpha])
		if isinstance(alpha, list): self.alpha = torch.Tensor(alpha)
		self.size_average = size_average

	def forward(self, input, target):
		if input.dim() > 2:
			input = input.view(input.size(0), input.size(1), -1)  # N,C,H,W => N,C,H*W
			input = input.transpose(1, 2)  # N,C,H*W => N,H*W,C
			input = input.contiguous().view(-1, input.size(2))  # N,H*W,C => N*H*W,C

		target = target.view(-1, 1)

		logpt = F.log_softmax(input)
		if self.alpha is not None:
			if self.alpha.type() != input.data.type():
				self.alpha = self.alpha.type_as(input.data)
		TF = (torch.argmax(logpt, dim=1) == target.squeeze()) * 1
		at = TF * self.alpha[0] + (1 - TF) * self.alpha[1] * 0.75

		logpt = logpt.gather(1, target)
		logpt = logpt.view(-1)
		pt = logpt.data.exp()
		logpt = logpt * at

		loss = -1 * (1 - pt) ** self.gamma * logpt
		if self.size_average:
			return loss.mean()
		else:
			return loss.sum()