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


class ALearnScale(Function):
    @staticmethod
    def forward(ctx, weight, alpha, g, Qn, Qp, beta):
        ctx.save_for_backward(weight, alpha, beta)
        ctx.other = g, Qn, Qp

        w_q = Round.apply(torch.div((weight - beta), alpha).clamp(Qn, Qp))
        w_q = w_q * alpha + beta
        return w_q

    @staticmethod
    def backward(ctx, grad_weight):
        weight, alpha, beta = ctx.saved_tensors
        g, Qn, Qp = ctx.other

        q_w = (weight - beta) / alpha

        smaller = (q_w < Qn).float()
        bigger = (q_w > Qp).float()
        between = 1.0 - smaller - bigger

        grad_alpha = ((smaller * Qn + bigger * Qp +
                       between * Round.apply(q_w) - between * q_w) * grad_weight * g).sum().unsqueeze(dim=0)
        grad_beta = ((smaller + bigger) * grad_weight * g).sum().unsqueeze(dim=0)
        grad_weight = between * grad_weight
        return grad_weight, grad_alpha, None, None, None, grad_beta


class WLearnScale(Function):
    @staticmethod
    def forward(ctx, weight, alpha, g, Qn, Qp, per_channel):
        ctx.save_for_backward(weight, alpha)
        ctx.other = g, Qn, Qp, per_channel
        if per_channel:
            sizes = weight.size()
            weight = weight.contiguous().view(weight.size()[0], -1)
            weight = torch.transpose(weight, 0, 1)
            alpha = torch.broadcast_to(alpha, weight.size())
            w_q = Round.apply(torch.div(weight, alpha).clamp(Qn, Qp))
            w_q = w_q * alpha
            w_q = torch.transpose(w_q, 0, 1)
            w_q = w_q.contiguous().view(sizes)
        else:
            w_q = Round.apply(torch.div(weight, alpha).clamp(Qn, Qp))
            w_q = w_q * alpha
        return w_q

    @staticmethod
    def backward(ctx, grad_weight):
        weight, alpha = ctx.saved_tensors
        g, Qn, Qp, per_channel = ctx.other
        if per_channel:
            sizes = weight.size()
            weight = weight.contiguous().view(weight.size()[0], -1)
            weight = torch.transpose(weight, 0, 1)
            alpha = torch.broadcast_to(alpha, weight.size())
            q_w = weight / alpha
            q_w = torch.transpose(q_w, 0, 1)
            q_w = q_w.contiguous().view(sizes)
        else:
            q_w = weight / alpha
        smaller = (q_w < Qn).float()
        bigger = (q_w > Qp).float()
        between = 1.0 - smaller - bigger
        if per_channel:
            grad_alpha = ((smaller * Qn + bigger * Qp +
                           between * Round.apply(q_w) - between * q_w) * grad_weight * g)
            grad_alpha = grad_alpha.contiguous().view(grad_alpha.size()[0], -1).sum(dim=1)
        else:
            grad_alpha = ((smaller * Qn + bigger * Qp +
                           between * Round.apply(q_w) - between * q_w) * grad_weight * g).sum().unsqueeze(dim=0)
        grad_weight = between * grad_weight
        return grad_weight, grad_alpha, None, None, None, None


class LearnScaleActivationQuantizer(nn.Module):
    def __init__(self, a_bits, all_positive=False, batch_init=20):
        super(LearnScaleActivationQuantizer, self).__init__()
        self.a_bits = a_bits
        self.all_positive = all_positive
        self.batch_init = batch_init
        if self.all_positive:
            self.Qn = 0
            self.Qp = 2 ** self.a_bits - 1
        else:
            self.Qn = - 2 ** (self.a_bits - 1)
            self.Qp = 2 ** (self.a_bits - 1) - 1
        self.s = torch.nn.Parameter(torch.ones(1), requires_grad=True)
        self.beta = torch.nn.Parameter(torch.tensor([float(-1e-9)]), requires_grad=True)
        self.init_state = 1
        self.g = 0

    def forward(self, activation):
        if self.a_bits == 32:
            q_a = activation
        elif self.a_bits == 1:
            print('！Binary quantization is not supported ！')
            assert self.a_bits != 1
        else:
            if self.init_state == 0:
                self.g = 1.0 / math.sqrt(activation.numel() * self.Qp)
                self.init_state += 1
            q_a = ALearnScale.apply(activation, self.s, self.g, self.Qn, self.Qp, self.beta)
        return q_a


class LearnScaleWeightQuantizer(nn.Module):
    def __init__(self, w_bits, all_positive=False, per_channel=False, batch_init=20):
        super(LearnScaleWeightQuantizer, self).__init__()
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
        self.s = torch.nn.Parameter(torch.ones(1), requires_grad=True)
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
            w_q = WLearnScale.apply(weight, self.s, self.g, self.Qn, self.Qp, self.per_channel)

        return w_q


def update_LearnScale_activation_Scalebeta(model):
    for name, child in model.named_children():
        if isinstance(child, (QuantConv2d, QuantConvTranspose2d, QuantLinear)):
            s = child.activation_quantizer.s.data
            beta = child.activation_quantizer.beta.data
            Qn = child.activation_quantizer.Qn
            Qp = child.activation_quantizer.Qp
            g = child.activation_quantizer.g
            q_input = (child.input - beta) / s
            smaller = (q_input < Qn).float()
            bigger = (q_input > Qp).float()
            between = 1.0 - smaller - bigger
            grad_alpha = ((smaller * Qn + bigger * Qp +
                           between * Round.apply(q_input) - between * q_input) * g).sum().unsqueeze(dim=0)
            grad_beta = ((smaller + bigger) * g).sum().unsqueeze(dim=0)
            child.activation_quantizer.s.grad.data.add_(
                g * (2 * (child.quant_input - child.input) * grad_alpha).sum().unsqueeze(dim=0))
            child.activation_quantizer.beta.grad.data.add_(
                g * (2 * (child.quant_input - child.input) * grad_beta).sum().unsqueeze(dim=0))

            model._modules[name] = child
        else:
            child = update_LearnScale_activation_Scalebeta(child)
            model._modules[name] = child
    return model


class QuantConv2d(nn.Conv2d):
    def __init__(self,
                 in_channels, out_channels, kernel_size, stride=1, padding=0,
                 dilation=1, groups=1, bias=True, padding_mode='zeros', a_bits=8,
                 w_bits=8, quant_inference=False, all_positive=False, per_channel=False,
                 batch_init=20):
        super(QuantConv2d, self).__init__(in_channels, out_channels, kernel_size, stride,
                                          padding, dilation, groups, bias, padding_mode)
        self.quant_inference = quant_inference
        self.activation_quantizer = LearnScaleActivationQuantizer(a_bits=a_bits, all_positive=all_positive,
                                                                  batch_init=batch_init)
        self.weight_quantizer = LearnScaleWeightQuantizer(w_bits=w_bits, all_positive=all_positive,
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


class QuantConv1d(nn.Conv1d):
    def __init__(self,
                 in_channels, out_channels, kernel_size, stride=1, padding=0,
                 dilation=1, groups=1, bias=True, padding_mode='zeros',
                 a_bits=8, w_bits=8, quant_inference=False, all_positive=False,
                 per_channel=False, batch_init=20):
        super(QuantConv1d, self).__init__(in_channels, out_channels, kernel_size, stride,
                                          padding, dilation, groups, bias, padding_mode)
        self.quant_inference = quant_inference
        self.activation_quantizer = LearnScaleActivationQuantizer(a_bits=a_bits, all_positive=all_positive,
                                                                  batch_init=batch_init)
        self.weight_quantizer = LearnScaleWeightQuantizer(w_bits=w_bits, all_positive=all_positive,
                                                          per_channel=per_channel, batch_init=batch_init)

    def forward(self, input):
        self.input = input
        self.quant_input = self.activation_quantizer(self.input)
        if not self.quant_inference:
            self.quant_weight = self.weight_quantizer(self.weight)
        else:
            self.quant_weight = self.weight

        output = F.conv1d(self.quant_input, self.quant_weight, self.bias, self.stride,
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
        self.activation_quantizer = LearnScaleActivationQuantizer(a_bits=a_bits, all_positive=all_positive,
                                                                  batch_init=batch_init)
        self.weight_quantizer = LearnScaleWeightQuantizer(w_bits=w_bits, all_positive=all_positive,
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
        self.activation_quantizer = LearnScaleActivationQuantizer(a_bits=a_bits, all_positive=all_positive,
                                                                  batch_init=batch_init)
        self.weight_quantizer = LearnScaleWeightQuantizer(w_bits=w_bits, all_positive=all_positive,
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


def add_quant_op(module, layer_counter, a_bits=8, w_bits=8,
                 quant_inference=False, all_positive=False, per_channel=False, batch_init=20):
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
        elif isinstance(child, nn.Conv1d):
            layer_counter[0] += 1
            if layer_counter[0] >= 1:
                if child.bias is not None:
                    quant_conv = QuantConv1d(child.in_channels, child.out_channels,
                                             child.kernel_size, stride=child.stride,
                                             padding=child.padding, dilation=child.dilation,
                                             groups=child.groups, bias=True, padding_mode=child.padding_mode,
                                             a_bits=a_bits, w_bits=w_bits, quant_inference=quant_inference,
                                             all_positive=all_positive, per_channel=per_channel,
                                             batch_init=batch_init)
                    quant_conv.bias.data = child.bias
                else:
                    quant_conv = QuantConv1d(child.in_channels, child.out_channels,
                                             child.kernel_size, stride=child.stride,
                                             padding=child.padding, dilation=child.dilation,
                                             groups=child.groups, bias=False, padding_mode=child.padding_mode,
                                             a_bits=a_bits, w_bits=w_bits, quant_inference=quant_inference,
                                             all_positive=all_positive, per_channel=per_channel,
                                             batch_init=batch_init)
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
