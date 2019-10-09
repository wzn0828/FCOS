import torch
import torch.nn as nn
import torch.nn.functional as F
import math

Linear_Class = nn.Linear
Con2d_Class = nn.Conv2d
Con2d_2 = nn.Conv2d
BN_Class = nn.BatchNorm2d
_detach = None
_normlinear = None
_normconv2d = None
_coeff = True


def set_gl_variable(linear=nn.Linear, conv1=nn.Conv2d, conv2=nn.Conv2d, bn=nn.BatchNorm2d, detach=None, normlinear=None, normconv2d=None, coeff=True):

    global Linear_Class
    Linear_Class = linear

    global Con2d_Class
    Con2d_Class = conv1

    global Con2d_2
    Con2d_2 = conv2

    global BN_Class
    BN_Class = bn

    global _detach
    if detach is not None:
        _detach = detach

    global _normlinear
    if normlinear is not None:
        _normlinear = normlinear

    global _normconv2d
    if normconv2d is not None:
        _normconv2d = normconv2d

    global _coeff
    if coeff is not None:
        _coeff = coeff


class Linear(nn.Linear):

    def __init__(self, in_features, out_features, bias=True, eps=1e-8):
        super(Linear, self).__init__(in_features, out_features, bias)
        # self.register_buffer('eps', torch.tensor(eps))
        self.eps = eps

    def forward(self, x):
        w_len = torch.sqrt(torch.t(self.weight.pow(2).sum(dim=1, keepdim=True)).clamp_(min=self.eps))  # 1*num_classes
        x_len = torch.sqrt(x.pow(2).sum(dim=1, keepdim=True).clamp_(min=self.eps))   # batch*1

        wx_len = torch.matmul(x_len, w_len)  # batch*num_classes
        cos_theta = (torch.matmul(x, torch.t(self.weight)) / (wx_len.clamp_(min=self.eps))).clamp_(-1.0, 1.0)    # batch*num_classes
        del wx_len

        if _detach is not None:
            if _detach == 'w':
                w_len = w_len.detach()
            elif _detach == 'x':
                x_len = x_len.detach()
            elif _detach == 'wx':
                w_len = w_len.detach()
                x_len = x_len.detach()
            elif _detach == 'theta':
                cos_theta = cos_theta.detach()
            else:
                raise AssertionError('_detach is not valid!')

        out = (torch.matmul(x_len, w_len).clamp_(min=self.eps)) * cos_theta
        del x_len, w_len, cos_theta

        if self.bias is not None:
            out = out + self.bias

        return out


class Conv2d(nn.Conv2d):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True, eps=1e-8):
        super(Conv2d, self).__init__(
            in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)

        self.eps = eps
        # self.register_buffer('eps', torch.tensor(eps))
        # self.ones_weight = torch.ones((1, 1, self.weight.size(2), self.weight.size(3))).cuda()
        self.register_buffer('ones_weight', torch.ones((1, 1, self.weight.size(2), self.weight.size(3))))

        # self.register_buffer('a1_min', torch.tensor(0.))
        # self.register_buffer('a2_max', torch.tensor(0.))

    def forward(self, input):
        w_len = torch.sqrt(self.weight.view(self.weight.size(0), -1).pow(2).sum(dim=1, keepdim=True).t().clamp_(min=self.eps))  # 1*out_channels
        x_len = input.pow(2).sum(dim=1, keepdim=True)                                          # batch*1*H_in*W_in
        x_len = torch.sqrt(F.conv2d(x_len, self.ones_weight, None,
                              self.stride,
                              self.padding, self.dilation, self.groups).clamp_(min=self.eps))                             # batch*1*H_out*W_out

        wx_len = x_len * (w_len.unsqueeze(-1).unsqueeze(-1))                        # batch*out_channels*H_out*W_out

        cos_theta = (F.conv2d(input, self.weight, None, self.stride,
                       self.padding, self.dilation, self.groups) / wx_len.clamp_(min=self.eps)).clamp_(-1.0, 1.0)                                   # batch*out_channels*H_out*W_out
        del wx_len

        if _detach is not None:
            if _detach == 'w':
                w_len = w_len.detach()
            elif _detach == 'x':
                x_len = x_len.detach()
            elif _detach == 'wx':
                w_len = w_len.detach()
                x_len = x_len.detach()
            elif _detach == 'theta':
                cos_theta = cos_theta.detach()
            else:
                raise AssertionError('_detach is not valid!')

        out = (x_len * (w_len.unsqueeze(-1).unsqueeze(-1))).clamp_(min=self.eps) * cos_theta
        del x_len, w_len, cos_theta

        if self.bias is not None:
            out = out + self.bias.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)

        return out


class LinearPR(nn.Linear):

    def __init__(self, in_features, out_features, bias=True):
        super(LinearPR, self).__init__(in_features, out_features, bias)
        self.register_buffer('one', torch.tensor(1.0))

    def forward(self, x):
        w_len_pow2 = torch.t(self.weight.pow(2).sum(dim=1, keepdim=True))  # 1*num_classes
        x_len_pow2 = x.pow(2).sum(dim=1, keepdim=True)  # batch*1

        wx_len_pow_2 = torch.matmul(x_len_pow2, w_len_pow2)  # batch*num_classes
        del w_len_pow2, x_len_pow2

        pro = torch.matmul(x, torch.t(self.weight))  # batch*num_classes
        dis_ = torch.sqrt(F.relu(wx_len_pow_2 - pro.pow(2), inplace=True))  # batch*num_classes
        wx_len = torch.sqrt(F.relu(wx_len_pow_2, inplace=True))
        del wx_len_pow_2
        dis = wx_len - dis_  # batch*num_classes

        wx_len_detach = wx_len.detach()
        del wx_len
        if _coeff:
            a_1 = dis_.detach() / (wx_len_detach + 1e-15)
        else:
            a_1 = self.one
        del dis_
        if _coeff:
            a_2 = pro.detach() / (wx_len_detach + 1e-15)
        else:
            a_2 = torch.sign(pro)
        del wx_len_detach

        out = a_1 * pro + a_2 * dis

        del dis, pro

        del a_1, a_2

        if self.bias is not None:
            out = out + self.bias

        return out


class Conv2dPR(nn.Conv2d):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True):
        super(Conv2dPR, self).__init__(
            in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)

        # self.ones_weight = torch.ones((1, 1, self.weight.size(2), self.weight.size(3))).cuda()
        self.register_buffer('ones_weight', torch.ones((1, 1, self.weight.size(2), self.weight.size(3))))
        self.register_buffer('one', torch.tensor(1.0))

        # self.register_buffer('a1_min', torch.tensor(0.))
        # self.register_buffer('a2_max', torch.tensor(0.))

    def forward(self, input):
        w_len_pow2 = self.weight.view(self.weight.size(0), -1).pow(2).sum(dim=1, keepdim=True).t()  # 1*out_channels
        x_len_pow2 = input.pow(2).sum(dim=1, keepdim=True)                                          # batch*1*H_in*W_in
        x_len_pow2 = F.conv2d(x_len_pow2, self.ones_weight, None,
                              self.stride,
                              self.padding, self.dilation, self.groups)                             # batch*1*H_out*W_out

        wx_len_pow_2 = x_len_pow2 * (w_len_pow2.unsqueeze(-1).unsqueeze(-1))                        # batch*out_channels*H_out*W_out
        del w_len_pow2, x_len_pow2

        pro = F.conv2d(input, self.weight, None, self.stride,
                       self.padding, self.dilation, self.groups)                                    # batch*out_channels*H_out*W_out

        dis_ = torch.sqrt(
            F.relu(wx_len_pow_2 - pro.pow(2), inplace=True))                        # batch*out_channels*H_out*W_out
        wx_len = torch.sqrt(F.relu(wx_len_pow_2, inplace=True))                     # batch*out_channels*H_out*W_out
        del wx_len_pow_2
        dis = wx_len - dis_  # batch*out_channels*H_out*W_out

        wx_len_detach = wx_len.detach()
        del wx_len
        if _coeff:
            a_1 = dis_.detach() / (wx_len_detach + 1e-15)
        else:
            a_1 = self.one
        del dis_
        if _coeff:
            a_2 = pro.detach() / (wx_len_detach + 1e-15)
        else:
            a_2 = torch.sign(pro)
        del wx_len_detach

        out = a_1 * pro + a_2 * dis

        del dis, pro
        del a_1, a_2

        if self.bias is not None:
            out = out + self.bias.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)

        return out


class LinearPR_Detach(nn.Linear):

    def __init__(self, in_features, out_features, bias=True, eps=1e-8):
        super(LinearPR_Detach, self).__init__(in_features, out_features, bias)

        self.eps = eps
        # self.register_buffer('eps', torch.tensor(eps))
        # self.register_buffer('a1_min', torch.tensor(0.))
        # self.register_buffer('a2_max', torch.tensor(0.))

    def forward(self, x):
        w_len = torch.sqrt((torch.t(self.weight.pow(2).sum(dim=1, keepdim=True))).clamp_(min=self.eps))  # 1*num_classes
        x_len = torch.sqrt((x.pow(2).sum(dim=1, keepdim=True)).clamp_(min=self.eps))  # batch*1

        cos_theta = (torch.matmul(x, torch.t(self.weight)) / torch.matmul(x_len, w_len).clamp_(min=self.eps)).clamp_(-1.0, 1.0)    # batch*num_classes
        abs_sin_theta = torch.sqrt(1.0 - cos_theta**2)    # batch*num_classes

        if _detach is not None:
            if _detach == 'w':
                w_len = w_len.detach()
            elif _detach == 'x':
                x_len = x_len.detach()
            elif _detach == 'wx':
                w_len = w_len.detach()
                x_len = x_len.detach()
            elif _detach == 'theta':
                cos_theta = cos_theta.detach()
                abs_sin_theta = abs_sin_theta.detach()
            else:
                raise AssertionError('_detach is not valid!')

        out = torch.matmul(x_len, w_len).clamp_(min=self.eps) * (abs_sin_theta.detach()*cos_theta + cos_theta.detach()*(1.0-abs_sin_theta))
        del w_len, x_len, cos_theta, abs_sin_theta

        if self.bias is not None:
            out = out + self.bias

        return out


class Conv2dPR_Detach(nn.Conv2d):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True, eps=1e-8):
        super(Conv2dPR_Detach, self).__init__(
            in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)

        # self.ones_weight = torch.ones((1, 1, self.weight.size(2), self.weight.size(3))).cuda()
        # self.register_buffer('eps', torch.tensor(eps))
        self.eps = eps
        self.register_buffer('ones_weight', torch.ones((1, 1, self.weight.size(2), self.weight.size(3))))

        # self.register_buffer('a1_min', torch.tensor(0.))
        # self.register_buffer('a2_max', torch.tensor(0.))

    def forward(self, input):
        w_len = torch.sqrt((self.weight.view(self.weight.size(0), -1).pow(2).sum(dim=1, keepdim=True).t()).clamp_(min=self.eps))  # 1*out_channels
        x_len = input.pow(2).sum(dim=1, keepdim=True)                                          # batch*1*H_in*W_in
        x_len = torch.sqrt((F.conv2d(x_len, self.ones_weight, None,
                              self.stride,
                              self.padding, self.dilation, self.groups)).clamp_(min=self.eps))                             # batch*1*H_out*W_out

        wx_len = x_len * (w_len.unsqueeze(-1).unsqueeze(-1))                        # batch*out_channels*H_out*W_out

        cos_theta = (F.conv2d(input, self.weight, None, self.stride,
                       self.padding, self.dilation, self.groups) / wx_len.clamp_(min=self.eps)).clamp_(-1.0, 1.0)                                     # batch*out_channels*H_out*W_out
        del wx_len
        abs_sin_theta = torch.sqrt(1.0 - cos_theta ** 2)

        if _detach is not None:
            if _detach == 'w':
                w_len = w_len.detach()
            elif _detach == 'x':
                x_len = x_len.detach()
            elif _detach == 'wx':
                w_len = w_len.detach()
                x_len = x_len.detach()
            elif _detach == 'theta':
                cos_theta = cos_theta.detach()
                abs_sin_theta = abs_sin_theta.detach()
            else:
                raise AssertionError('_detach is not valid!')

        out = (x_len * (w_len.unsqueeze(-1).unsqueeze(-1))).clamp_(min=self.eps) * (
                            abs_sin_theta.detach() * cos_theta + cos_theta.detach() * (1.0 - abs_sin_theta))
        del w_len, x_len, cos_theta, abs_sin_theta

        # self.a1_min = a_1.view(a_1.size(0), -1).min(dim=1)[0].mean()
        # self.a2_max = torch.abs(a_2).view(a_2.size(0), -1).max(dim=1)[0].mean()
        # del a_1, a_2

        if self.bias is not None:
            out = out + self.bias.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)

        return out


class LinearNorm(nn.Linear):

    def __init__(self, in_features, out_features, bias=True, eps=1e-8):
        super(LinearNorm, self).__init__(in_features, out_features, bias)
        # self.register_buffer('eps', torch.tensor(eps))

        self.eps = eps
        self.lens = nn.Parameter(torch.ones(out_features, 1))

        if _normlinear == '3-1' or _normlinear is None:
            self.g = nn.Parameter(torch.ones(out_features, 1))
        if _normlinear == '3-5':
            self.g = nn.Parameter(torch.ones(out_features, 1))
            self.weight.register_hook(lambda grad: self.lens/self.g*grad)
        elif _normlinear == '3-2':
            self.g = nn.Parameter(torch.ones(1, 1))
        elif _normlinear == '3-3':
            self.g = nn.Parameter(torch.ones(1, 1))
            self.g.register_hook(lambda grad: grad/out_features)
        elif _normlinear == '3-4':
            self.g = nn.Parameter(torch.ones(1, 1))
            self.g.register_hook(lambda grad: grad/math.sqrt(out_features))
        elif _normlinear == '4' or _normlinear == '7':
            self.v = nn.Parameter(torch.ones(1, 1))
        elif _normlinear == '5-1':
            self.g = nn.Parameter(torch.ones(out_features, 1))
            self.v = nn.Parameter(torch.ones(1, 1))
        elif _normlinear == '5-2':
            self.g = nn.Parameter(torch.ones(1, 1))
            self.v = nn.Parameter(torch.ones(1, 1))
        elif _normlinear == '8':
            self.g = nn.Parameter(torch.ones(1, 1))

    def forward(self, x):

        lens = torch.sqrt((self.weight.pow(2).sum(dim=1, keepdim=True)).clamp(min=self.eps))   # out_feature*1
        self.lens.data = lens.data

        if _normlinear == '1':
            weight = self.weight / lens  # out_feature*1

        elif _normlinear == '2':
            x = x / torch.sqrt(x.pow(2).sum(dim=1, keepdim=True).clamp_(min=self.eps))  # batch*1
            weight = self.weight

        elif _normlinear == '3-1' or _normlinear == '3-2' or _normlinear == '3-3' or _normlinear == '3-4' or _normlinear == '3-5':
            weight = torch.abs(self.g) * self.weight / lens  # out_feature*in_features

        elif _normlinear == '4':
            x = torch.abs(self.v) * x / torch.sqrt(x.pow(2).sum(dim=1, keepdim=True).clamp_(min=self.eps))  # batch*1
            weight = self.weight

        elif _normlinear == '5-1' or _normlinear == '5-2':
            weight = torch.abs(self.g) * self.weight / torch.sqrt(
                (self.weight.pow(2).sum(dim=1, keepdim=True)).clamp_(min=self.eps))  # out_feature*in_features
            x = torch.abs(self.v) * x / torch.sqrt(x.pow(2).sum(dim=1, keepdim=True).clamp_(min=self.eps))  # batch*1

        elif _normlinear == '6':
            x = x / torch.sqrt(x.pow(2).sum(dim=1, keepdim=True).clamp_(min=self.eps))  # batch*1
            weight = self.weight / lens  # out_feature*1

        elif _normlinear == '7':
            x = torch.abs(self.v) * x / torch.sqrt(x.pow(2).sum(dim=1, keepdim=True).clamp_(min=self.eps))  # batch*1
            weight = self.weight / torch.sqrt(
                (self.weight.pow(2).sum(dim=1, keepdim=True)).clamp_(min=self.eps))  # out_feature*1

        elif _normlinear == '8':
            weight = lens.mean(dim=0, keepdim=True) * self.weight / lens
            self.g.data = lens.mean(dim=0, keepdim=True)

        elif _normlinear is None:
            weight = self.weight
            self.g.data = lens.data

        else:
            raise AssertionError('_norm is not valid!')

        return F.linear(x, weight, self.bias)


class Conv2dNorm(nn.Conv2d):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True, eps=1e-8):
        super(Conv2dNorm, self).__init__(
            in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)

        self.eps = eps
        self.lens = nn.Parameter(torch.ones(out_channels, 1, 1, 1))

        self.register_buffer('ones_weight', torch.ones((1, 1, self.weight.size(2), self.weight.size(3))))
        if _normconv2d == '3-1' or _normconv2d is None:
            self.g = nn.Parameter(torch.ones(out_channels, 1, 1, 1))
        if _normconv2d == '3-5':
            self.g = nn.Parameter(torch.ones(out_channels, 1, 1, 1))
            self.weight.register_hook(lambda grad: self.lens/self.g*grad)
        elif _normconv2d == '3-2':
            self.g = nn.Parameter(torch.ones(1, 1, 1, 1))
        elif _normconv2d == '3-3':
            self.g = nn.Parameter(torch.ones(1, 1, 1, 1))
            self.g.register_hook(lambda grad: grad/out_channels)
        elif _normconv2d == '3-4':
            self.g = nn.Parameter(torch.ones(1, 1, 1, 1))
            self.g.register_hook(lambda grad: grad/math.sqrt(out_channels))
        elif _normconv2d == '4' or _normconv2d == '7':
            self.v = nn.Parameter(torch.ones(1, 1, 1, 1))
        elif _normconv2d == '5-1':
            self.g = nn.Parameter(torch.ones(out_channels, 1, 1, 1))
            self.v = nn.Parameter(torch.ones(1, 1, 1, 1))
        elif _normconv2d == '5-2':
            self.g = nn.Parameter(torch.ones(1, 1, 1, 1))
            self.v = nn.Parameter(torch.ones(1, 1, 1, 1))
        elif _normconv2d == '8':
            self.g = nn.Parameter(torch.ones(1, 1, 1, 1))

    def forward(self, x):

        lens = torch.sqrt(
                self.weight.view(self.weight.size(0), -1).pow(2).sum(dim=1, keepdim=True).clamp(
                    min=self.eps)).unsqueeze(-1).unsqueeze(-1)       # out*1*1*1
        self.lens.data = lens.data

        if _normconv2d == '1':
            weight = self.weight / lens  # out*in*H*W

        elif _normconv2d == '2':
            x_len = x.pow(2).sum(dim=1, keepdim=True)  # batch*1*H_in*W_in
            x_len = torch.sqrt(F.conv2d(x_len, self.ones_weight, None,
                                        self.stride,
                                        self.padding, self.dilation, self.groups).clamp_(
                min=self.eps))  # batch*1*H_out*W_out

            out = F.conv2d(x, self.weight, None, self.stride, self.padding, self.dilation, self.groups) / x_len
            del x_len

            if self.bias is not None:
                out = out + self.bias.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)

            return out

        elif _normconv2d == '3-1' or _normconv2d == '3-2' or _normconv2d == '3-3' or _normconv2d == '3-4' or _normconv2d == '3-5':
            weight = torch.abs(self.g) * self.weight / lens  # out*in*H*W

        elif _normconv2d == '4':
            x_len = x.pow(2).sum(dim=1, keepdim=True)  # batch*1*H_in*W_in
            x_len = torch.sqrt(F.conv2d(x_len, self.ones_weight, None,
                                        self.stride,
                                        self.padding, self.dilation, self.groups).clamp_(
                min=self.eps))  # batch*1*H_out*W_out

            out = torch.abs(self.v) * F.conv2d(x, self.weight, None, self.stride, self.padding, self.dilation, self.groups) / x_len
            del x_len

            if self.bias is not None:
                out = out + self.bias.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)

            return out

        elif _normconv2d == '5-1' or _normconv2d == '5-2':
            weight = torch.abs(self.g) * self.weight / torch.sqrt(
                self.weight.view(self.weight.size(0), -1).pow(2).sum(dim=1, keepdim=True).clamp_(
                    min=self.eps)).unsqueeze(-1).unsqueeze(-1)  # out*in*H*W

            x_len = x.pow(2).sum(dim=1, keepdim=True)  # batch*1*H_in*W_in
            x_len = torch.sqrt(F.conv2d(x_len, self.ones_weight, None,
                                        self.stride,
                                        self.padding, self.dilation, self.groups).clamp_(
                min=self.eps))  # batch*1*H_out*W_out

            out = torch.abs(self.v) * F.conv2d(x, weight, None, self.stride, self.padding, self.dilation,
                                               self.groups) / x_len
            del x_len

            if self.bias is not None:
                out = out + self.bias.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)

            return out

        elif _normconv2d == '6':
            weight = self.weight / torch.sqrt(
                self.weight.view(self.weight.size(0), -1).pow(2).sum(dim=1, keepdim=True).clamp_(
                    min=self.eps)).unsqueeze(-1).unsqueeze(-1)  # out*in*H*W

            x_len = x.pow(2).sum(dim=1, keepdim=True)  # batch*1*H_in*W_in
            x_len = torch.sqrt(F.conv2d(x_len, self.ones_weight, None,
                                        self.stride,
                                        self.padding, self.dilation, self.groups).clamp_(
                min=self.eps))  # batch*1*H_out*W_out

            out = F.conv2d(x, weight, None, self.stride, self.padding, self.dilation, self.groups) / x_len
            del x_len

            if self.bias is not None:
                out = out + self.bias.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)

            return out

        elif _normconv2d == '7':
            weight = self.weight / torch.sqrt(
                self.weight.view(self.weight.size(0), -1).pow(2).sum(dim=1, keepdim=True).clamp_(
                    min=self.eps)).unsqueeze(-1).unsqueeze(-1)  # out*in*H*W

            x_len = x.pow(2).sum(dim=1, keepdim=True)  # batch*1*H_in*W_in
            x_len = torch.sqrt(F.conv2d(x_len, self.ones_weight, None,
                                        self.stride,
                                        self.padding, self.dilation, self.groups).clamp_(
                min=self.eps))  # batch*1*H_out*W_out

            out = torch.abs(self.v) * F.conv2d(x, weight, None, self.stride, self.padding, self.dilation, self.groups) / x_len
            del x_len

            if self.bias is not None:
                out = out + self.bias.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)

            return out

        elif _normconv2d == '8':
            weight = lens.mean(dim=0, keepdim=True) * self.weight / lens
            self.g.data = lens.mean(dim=0, keepdim=True)

        elif _normconv2d is None:
            weight = self.weight
            self.g.data = torch.sqrt(
                self.weight.view(self.weight.size(0), -1).pow(2).sum(dim=1, keepdim=True).clamp_(
                    min=self.eps)).unsqueeze(-1).unsqueeze(-1)

        else:
            raise AssertionError('_norm is not valid!')

        return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)


def init_g_conv(l):
    if not isinstance(l, (Con2d_Class, Con2d_2)):
        return

    if 'g' in l.__dict__['_parameters']:
        if _normconv2d is None:
            return
        if _normconv2d == '3-1' or _normconv2d == '3-5':
            l.g.data = torch.sqrt(
                l.weight.view(l.weight.size(0), -1).pow(2).sum(dim=1, keepdim=True).clamp_(
                    min=l.eps)).unsqueeze(-1).unsqueeze(-1)
        elif _normconv2d == '3-2' or _normconv2d == '3-3' or _normconv2d == '3-4':
            l.g.data = torch.sqrt(
                l.weight.view(l.weight.size(0), -1).pow(2).sum(dim=1, keepdim=True).clamp_(
                    min=l.eps)).unsqueeze(-1).unsqueeze(-1).mean(dim=0, keepdim=True)

# # test code
# torch.manual_seed(123)
# ori_linear = nn.Linear(512, 1024)
# torch.manual_seed(123)
# cus_linear = LinearProDis(512, 1024)
# input_linear = torch.randn(20, 512)
# ori_linear_out = ori_linear(input_linear)
# cus_linear_out = cus_linear(input_linear)
# linear_diff = ori_linear_out - cus_linear_out
# print(linear_diff.min(), linear_diff.max())
# print(ori_linear_out.min(), ori_linear_out.max())
# print(cus_linear_out.min(), cus_linear_out.max())
#
# torch.manual_seed(123)
# ori_conv = nn.Conv2d(16, 33, 3, 2)
# torch.manual_seed(123)
# cus_conv = Conv2dProDis(16, 33, 3, 2)
# input_conv = torch.randn(20, 16, 50, 100)
# ori_conv_out = ori_conv(input_conv)
# cus_conv_out = cus_conv(input_conv)
# conv_diff = ori_conv_out - cus_conv_out
# print(conv_diff.min(), conv_diff.max())
# print(ori_conv_out.min(), ori_conv_out.max())
# print(cus_conv_out.min(), cus_conv_out.max())



