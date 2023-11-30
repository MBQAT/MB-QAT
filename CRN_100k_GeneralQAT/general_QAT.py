import copy
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function


class Round(Function):
    @staticmethod
    def forward(self, input):
        sign = torch.sign(input)
        output = sign * torch.floor(torch.abs(input) + 0.5)
        return output

    @staticmethod
    def backward(self, grad_output):
        grad_input = grad_output.clone()
        return grad_input


class AQAT(Function):
    @staticmethod
    def forward(ctx, weight, alpha, g, Qn, Qp, zero_point):
        ctx.save_for_backward(weight, alpha, zero_point)
        ctx.other = g, Qn, Qp

        x_max = weight.max()
        x_min = weight.min()
        alpha = float(x_max - x_min) / (Qp - Qn)
        zero_point = (Qp - (x_max / alpha)).round()

        w_q = Round.apply((torch.div(weight, alpha) + zero_point).clamp(Qn, Qp))
        w_q = (w_q - zero_point) * alpha

        return w_q

    @staticmethod
    def backward(ctx, grad_weight):
        weight, alpha, zero_point = ctx.saved_tensors
        g, Qn, Qp = ctx.other

        alpha = alpha.cuda()
        zero_point = zero_point.cuda()
        q_w = torch.div(weight, alpha) + zero_point

        smaller = (q_w < Qn).float()
        bigger = (q_w > Qp).float()
        between = 1.0 - smaller - bigger

        grad_weight = between * grad_weight
        return grad_weight, None, None, None, None, None


class WQAT(Function):
    @staticmethod
    def forward(ctx, weight, alpha, g, Qn, Qp, per_channel):
        ctx.save_for_backward(weight, alpha)
        ctx.other = g, Qn, Qp, per_channel

        x_max = weight.max()
        x_min = weight.min()
        x_absmax = max(abs(x_min), x_max)
        x_min, x_max = -x_absmax, x_absmax
        alpha = float(x_max - x_min) / (Qp - Qn)

        w_q = Round.apply(torch.div(weight, alpha).clamp(Qn, Qp))
        w_q = w_q * alpha

        return w_q

    @staticmethod
    def backward(ctx, grad_weight):
        weight, alpha = ctx.saved_tensors
        g, Qn, Qp, per_channel = ctx.other

        alpha = alpha.cuda()
        q_w = torch.div(weight, alpha)
        smaller = (q_w < Qn).float()
        bigger = (q_w > Qp).float()
        between = 1.0 - smaller - bigger
        grad_weight = between * grad_weight
        return grad_weight, None, None, None, None, None


class QATActivationQuantizer(nn.Module):
    def __init__(self, a_bits, all_positive=False, batch_init=20):
        super(QATActivationQuantizer, self).__init__()
        self.a_bits = a_bits
        self.all_positive = all_positive
        self.batch_init = batch_init
        if self.all_positive:
            self.Qn = 0
            self.Qp = 2 ** self.a_bits - 1
        else:
            self.Qn = - 2 ** (self.a_bits - 1)
            self.Qp = 2 ** (self.a_bits - 1) - 1
        self.s = torch.ones(1)
        self.zero_point = torch.tensor([float(-1e-9)])
        self.init_state = 1
        self.g = 0

    def forward(self, activation):
        if self.a_bits == 32:
            q_a = activation
        elif self.a_bits == 1:
            print('！Binary quantization is not supported ！')
            q_a = activation
            assert self.a_bits != 1
        else:
            if self.init_state == 0:
                self.g = 1.0 / math.sqrt(activation.numel() * self.Qp)
                self.init_state += 1
            q_a = AQAT.apply(activation, self.s, self.g, self.Qn, self.Qp, self.zero_point)
        return q_a


class QATWeightQuantizer(nn.Module):
    def __init__(self, w_bits, all_positive=False, per_channel=False, batch_init=20):
        super(QATWeightQuantizer, self).__init__()
        self.w_bits = w_bits
        self.all_positive = all_positive
        self.batch_init = batch_init
        if self.all_positive:
            self.Qn = 0
            self.Qp = 2 ** w_bits - 1
        else:
            self.Qn = - 2 ** (w_bits - 1)
            self.Qp = 2 ** (w_bits - 1) - 1
        self.per_channel = per_channel
        self.init_state = 1
        self.s = torch.ones(1)
        self.g = 0

    def forward(self, weight):
        if self.init_state == 0:
            self.g = 1.0 / math.sqrt(weight.numel() * self.Qp)
            self.div = 2 ** self.w_bits - 1
            if self.per_channel:
                weight_tmp = weight.detach().contiguous().view(weight.size()[0], -1)
                mean = torch.mean(weight_tmp, dim=1)
                std = torch.std(weight_tmp, dim=1)
                self.s.data[0], _ = torch.max(torch.stack([torch.abs(mean - 3 * std), torch.abs(mean + 3 * std)]),
                                              dim=0)
                self.s.data[0] = self.s.data[0] / self.div
            else:
                mean = torch.mean(weight.detach())
                std = torch.std(weight.detach())
                self.s.data[0] = max([torch.abs(mean - 3 * std), torch.abs(mean + 3 * std)]) / self.div
            self.init_state += 1
        elif self.init_state < self.batch_init:
            self.div = 2 ** self.w_bits - 1
            if self.per_channel:
                weight_tmp = weight.detach().contiguous().view(weight.size()[0], -1)
                mean = torch.mean(weight_tmp, dim=1)
                std = torch.std(weight_tmp, dim=1)
                self.s.data[0], _ = torch.max(torch.stack([torch.abs(mean - 3 * std), torch.abs(mean + 3 * std)]),
                                              dim=0)
                self.s.data[0] = self.s.data[0] * 0.9 + 0.1 * self.s.data[0] / self.div
            else:
                mean = torch.mean(weight.detach())
                std = torch.std(weight.detach())
                self.s.data[0] = self.s.data[0] * 0.9 + 0.1 * max(
                    [torch.abs(mean - 3 * std), torch.abs(mean + 3 * std)]) / self.div
            self.init_state += 1
        elif self.init_state == self.batch_init:
            self.init_state += 1

        if self.w_bits == 32:
            output = weight
        elif self.w_bits == 1:
            print('！Binary quantization is not supported ！')
            assert self.w_bits != 1
        else:
            w_q = WQAT.apply(weight, self.s, self.g, self.Qn, self.Qp, self.per_channel)

        return w_q


class QuantConv2d(nn.Conv2d):
    def __init__(self,
                 in_channels, out_channels, kernel_size, stride=1, padding=0,
                 dilation=1, groups=1, bias=True, padding_mode='zeros', a_bits=8,
                 w_bits=8, quant_inference=False, all_positive=False, per_channel=False,
                 batch_init=20):
        super(QuantConv2d, self).__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, groups,
                                          bias, padding_mode)
        self.quant_inference = quant_inference
        self.activation_quantizer = QATActivationQuantizer(a_bits=a_bits, all_positive=all_positive,
                                                           batch_init=batch_init)
        self.weight_quantizer = QATWeightQuantizer(w_bits=w_bits, all_positive=all_positive,
                                                   per_channel=per_channel, batch_init=batch_init)

    def forward(self, input):
        self.input = input
        self.quant_input = self.activation_quantizer(self.input)
        if not self.quant_inference:
            self.quant_weight = self.weight_quantizer(self.weight)
        else:
            self.quant_weight = self.weight

        output = F.conv2d(self.quant_input, self.quant_weight, self.bias, self.stride,
                          self.padding, self.dilation, self.groups)
        return output


class QuantConvTranspose2d(nn.ConvTranspose2d):
    def __init__(self,
                 in_channels, out_channels, kernel_size, stride=1, padding=0,
                 output_padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros',
                 a_bits=8, w_bits=8, quant_inference=False, all_positive=False, per_channel=False,
                 batch_init=20):
        super(QuantConvTranspose2d, self).__init__(in_channels, out_channels, kernel_size, stride, padding,
                                                   output_padding,
                                                   groups, bias, dilation, padding_mode)
        self.quant_inference = quant_inference
        self.activation_quantizer = QATActivationQuantizer(a_bits=a_bits, all_positive=all_positive,
                                                           batch_init=batch_init)
        self.weight_quantizer = QATWeightQuantizer(w_bits=w_bits, all_positive=all_positive,
                                                   per_channel=per_channel, batch_init=batch_init)

    def forward(self, input):
        self.input = input
        self.quant_input = self.activation_quantizer(self.input)
        if not self.quant_inference:
            self.quant_weight = self.weight_quantizer(self.weight)
        else:
            self.quant_weight = self.weight
        output = F.conv_transpose2d(self.quant_input, self.quant_weight, self.bias, self.stride,
                                    self.padding, self.output_padding, self.groups, self.dilation)
        return output


class QuantLinear(nn.Linear):
    def __init__(self,
                 in_features, out_features, bias=True, a_bits=8, w_bits=8,
                 quant_inference=False, all_positive=False, per_channel=False,
                 batch_init=20):
        super(QuantLinear, self).__init__(in_features, out_features, bias)
        self.quant_inference = quant_inference
        self.activation_quantizer = QATActivationQuantizer(a_bits=a_bits, all_positive=all_positive,
                                                           batch_init=batch_init)
        self.weight_quantizer = QATWeightQuantizer(w_bits=w_bits, all_positive=all_positive,
                                                   per_channel=per_channel, batch_init=batch_init)

    def forward(self, input):
        self.input = input
        self.quant_input = self.activation_quantizer(self.input)
        if not self.quant_inference:
            self.quant_weight = self.weight_quantizer(self.weight)
        else:
            self.quant_weight = self.weight
        output = F.linear(self.quant_input, self.quant_weight, self.bias)
        return output


def add_quant_op(module, layer_counter, a_bits=8, w_bits=8, quant_inference=False, all_positive=False,
                 per_channel=False, batch_init=20):
    for name, child in module.named_children():
        if isinstance(child, nn.Conv2d):
            layer_counter[0] += 1
            if layer_counter[0] >= 1:
                if child.bias is not None:
                    quant_conv = QuantConv2d(child.in_channels, child.out_channels,
                                             child.kernel_size, stride=child.stride,
                                             padding=child.padding, dilation=child.dilation,
                                             groups=child.groups, bias=True, padding_mode=child.padding_mode,
                                             a_bits=a_bits, w_bits=w_bits, quant_inference=quant_inference,
                                             all_positive=all_positive, per_channel=per_channel, batch_init=batch_init)
                    quant_conv.bias.data = child.bias
                else:
                    quant_conv = QuantConv2d(child.in_channels, child.out_channels,
                                             child.kernel_size, stride=child.stride,
                                             padding=child.padding, dilation=child.dilation,
                                             groups=child.groups, bias=False, padding_mode=child.padding_mode,
                                             a_bits=a_bits, w_bits=w_bits, quant_inference=quant_inference,
                                             all_positive=all_positive, per_channel=per_channel, batch_init=batch_init)
                quant_conv.weight.data = child.weight
                module._modules[name] = quant_conv
        elif isinstance(child, nn.ConvTranspose2d):
            layer_counter[0] += 1
            if layer_counter[0] >= 1:
                if child.bias is not None:
                    quant_conv_transpose = QuantConvTranspose2d(child.in_channels, child.out_channels,
                                                                child.kernel_size, stride=child.stride,
                                                                padding=child.padding,
                                                                output_padding=child.output_padding,
                                                                dilation=child.dilation, groups=child.groups,
                                                                bias=True, padding_mode=child.padding_mode,
                                                                a_bits=a_bits, w_bits=w_bits,
                                                                quant_inference=quant_inference,
                                                                all_positive=all_positive, per_channel=per_channel,
                                                                batch_init=batch_init)
                    quant_conv_transpose.bias.data = child.bias
                else:
                    quant_conv_transpose = QuantConvTranspose2d(child.in_channels, child.out_channels,
                                                                child.kernel_size, stride=child.stride,
                                                                padding=child.padding,
                                                                output_padding=child.output_padding,
                                                                dilation=child.dilation, groups=child.groups,
                                                                bias=False, padding_mode=child.padding_mode,
                                                                a_bits=a_bits, w_bits=w_bits,
                                                                quant_inference=quant_inference,
                                                                all_positive=all_positive, per_channel=per_channel,
                                                                batch_init=batch_init)
                quant_conv_transpose.weight.data = child.weight
                module._modules[name] = quant_conv_transpose
        elif isinstance(child, nn.Linear):
            layer_counter[0] += 1
            if layer_counter[0] >= 1:
                if child.bias is not None:
                    quant_linear = QuantLinear(child.in_features, child.out_features,
                                               bias=True, a_bits=a_bits, w_bits=w_bits,
                                               quant_inference=quant_inference,
                                               all_positive=all_positive, per_channel=per_channel,
                                               batch_init=batch_init)
                    quant_linear.bias.data = child.bias
                else:
                    quant_linear = QuantLinear(child.in_features, child.out_features,
                                               bias=False, a_bits=a_bits, w_bits=w_bits,
                                               quant_inference=quant_inference,
                                               all_positive=all_positive, per_channel=per_channel,
                                               batch_init=batch_init)
                quant_linear.weight.data = child.weight
                module._modules[name] = quant_linear
        else:
            add_quant_op(child, layer_counter, a_bits=a_bits, w_bits=w_bits,
                         quant_inference=quant_inference, all_positive=all_positive,
                         per_channel=per_channel, batch_init=batch_init)


def prepare(model, inplace=False, a_bits=8, w_bits=8, quant_inference=False,
            all_positive=False, per_channel=False, batch_init=20):
    if not inplace:
        model = copy.deepcopy(model)
    layer_counter = [0]
    add_quant_op(model, layer_counter, a_bits=a_bits, w_bits=w_bits,
                 quant_inference=quant_inference,
                 all_positive=all_positive, per_channel=per_channel, batch_init=batch_init)
    return model
