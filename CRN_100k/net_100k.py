import sys
from pathlib import Path
import torch
import torch.nn as nn
import torch.autograd.variable
import torch.nn.functional as F
import os

sys.path.append(str(Path(os.path.abspath(__file__)).parent.parent))


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=2, kernel_size=(2, 3), stride=(1, 2), padding=(1, 0))
        self.conv2 = nn.Conv2d(in_channels=2, out_channels=4, kernel_size=(2, 3), stride=(1, 2), padding=(1, 0))
        self.conv3 = nn.Conv2d(in_channels=4, out_channels=8, kernel_size=(2, 3), stride=(1, 2), padding=(1, 0))
        self.conv4 = nn.Conv2d(in_channels=8, out_channels=11, kernel_size=(2, 3), stride=(1, 2), padding=(1, 0))
        self.conv5 = nn.Conv2d(in_channels=11, out_channels=19, kernel_size=(2, 3), stride=(1, 2), padding=(1, 0))

        self.lstm = nn.LSTM(19 * 4, 19 * 4, 2, batch_first=True)

        self.conv5_t = nn.ConvTranspose2d(in_channels=38, out_channels=11, kernel_size=(2, 3), stride=(1, 2),
                                          padding=(1, 0))
        self.conv4_t = nn.ConvTranspose2d(in_channels=22, out_channels=8, kernel_size=(2, 3), stride=(1, 2),
                                          padding=(1, 0))
        self.conv3_t = nn.ConvTranspose2d(in_channels=16, out_channels=4, kernel_size=(2, 3), stride=(1, 2),
                                          padding=(1, 0))
        self.conv2_t = nn.ConvTranspose2d(in_channels=8, out_channels=2, kernel_size=(2, 3), stride=(1, 2),
                                          padding=(1, 0), output_padding=(0, 1))
        self.conv1_t = nn.ConvTranspose2d(in_channels=4, out_channels=1, kernel_size=(2, 3), stride=(1, 2),
                                          padding=(1, 0))

        self.bn1 = nn.BatchNorm2d(2)
        self.bn2 = nn.BatchNorm2d(4)
        self.bn3 = nn.BatchNorm2d(8)
        self.bn4 = nn.BatchNorm2d(11)
        self.bn5 = nn.BatchNorm2d(19)

        self.bn5_t = nn.BatchNorm2d(11)
        self.bn4_t = nn.BatchNorm2d(8)
        self.bn3_t = nn.BatchNorm2d(4)
        self.bn2_t = nn.BatchNorm2d(2)
        self.bn1_t = nn.BatchNorm2d(1)

        self.elu = nn.ELU(inplace=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out = x.unsqueeze(dim=1)

        e1 = self.elu(self.bn1(self.conv1(out)[:, :, :-1, :].contiguous()))
        e2 = self.elu(self.bn2(self.conv2(e1)[:, :, :-1, :].contiguous()))
        e3 = self.elu(self.bn3(self.conv3(e2)[:, :, :-1, :].contiguous()))
        e4 = self.elu(self.bn4(self.conv4(e3)[:, :, :-1, :].contiguous()))
        e5 = self.elu(self.bn5(self.conv5(e4)[:, :, :-1, :].contiguous()))

        out = e5.contiguous().transpose(1, 2)
        q1 = out.size(2)
        q2 = out.size(3)
        out = out.contiguous().view(out.size(0), out.size(1), -1)
        self.lstm.flatten_parameters()
        out, _ = self.lstm(out)
        out = out.contiguous().view(out.size(0), out.size(1), q1, q2)
        out = out.contiguous().transpose(1, 2)

        out = torch.cat([out, e5], dim=1)

        d5 = self.elu(torch.cat([F.pad(self.bn5_t(self.conv5_t(out)).contiguous(), [0, 0, 1, 0]), e4], dim=1))
        d4 = self.elu(torch.cat([F.pad(self.bn4_t(self.conv4_t(d5)).contiguous(), [0, 0, 1, 0]), e3], dim=1))
        d3 = self.elu(torch.cat([F.pad(self.bn3_t(self.conv3_t(d4)).contiguous(), [0, 0, 1, 0]), e2], dim=1))
        d2 = self.elu(torch.cat([F.pad(self.bn2_t(self.conv2_t(d3)).contiguous(), [0, 0, 1, 0]), e1], dim=1))
        d1 = self.sigmoid(F.pad(self.bn1_t(self.conv1_t(d2)).contiguous(), [0, 0, 1, 0]))

        out = torch.squeeze(d1, dim=1)

        return out
