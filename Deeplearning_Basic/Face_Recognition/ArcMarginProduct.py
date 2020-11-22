import math
import torch
from torch import nn
from torch.nn import Parameter
import torch.nn.functional as F

class ArcMarginProduct(nn.Module):
    def __init__(self, in_feature=1000, out_feature=100, s=64.0, m=0.50, easy_margin=False):
        super(ArcMarginProduct, self).__init__()
        self.in_feature = in_feature
        self.out_feature = out_feature
        self.s = s
        self.m = m
        self.weight = Parameter(torch.Tensor(out_feature, in_feature))
        nn.init.xavier_uniform_(self.weight)

        self.easy_margin = easy_margin
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)

        # make the function cos(theta+m) monotonic decreasing while theta in [0°,180°]
        self.th = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m

    def forward(self, x, label):
        # cos(theta)
        cosine = F.linear(F.normalize(x), F.normalize(self.weight))
        # cos(theta + m)
        sine = torch.sqrt(1.0 - torch.pow(cosine, 2))
        phi = cosine * self.cos_m - sine * self.sin_m

        if self.easy_margin:
            phi = torch.where(cosine > 0, phi, cosine)
        else:
            phi = torch.where((cosine - self.th) > 0, phi, cosine - self.mm)

        #one_hot = torch.zeros(cosine.size(), device='cuda' if torch.cuda.is_available() else 'cpu')
        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, label.view(-1, 1), 1)
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        cos = cosine*one_hot
        degree = math.acos(cos[0][label[0]])*180/math.pi
        degree2 = math.acos(cos[0][label[0]-1])*180/math.pi
        print("{:.2f}도".format(degree))
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)

        output = output * self.s
        return output,degree,degree2


class ArcMarginForTest(nn.Module):
    def __init__(self, in_feature=1000, out_feature=100, s=64.0, m=0.50, easy_margin=False):
        super(ArcMarginForTest, self).__init__()
        self.in_feature = in_feature
        self.out_feature = out_feature
        self.s = s
        self.m = m
        self.weight = Parameter(torch.Tensor(out_feature, in_feature))
        nn.init.xavier_uniform_(self.weight)

        self.easy_margin = easy_margin


    def forward(self, x): #label 제거

        cosine = F.linear(F.normalize(x), F.normalize(self.weight))
        sine = torch.sqrt(1.0 - torch.pow(cosine, 2))
        one_hot = torch.ones_like(cosine)
        output = (one_hot * cosine)
        cos = torch.max(output)
        degree = math.acos(cos)*180/(math.pi)
        output = output * self.s

        return output

if __name__ == '__main__':
    pass