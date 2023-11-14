from http.client import UnimplementedFileMode
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.nn.conv import GatedGraphConv

torch.manual_seed(2020)


def get_conv_mp_out_size(in_size, last_layer, mps):
    size = in_size

    for mp in mps:
        size = round((size - mp["kernel_size"]) / mp["stride"] + 1)

    size = size + 1 if size % 2 != 0 else size

    return int(size * last_layer["out_channels"])


def init_weights(m):
    if type(m) == nn.Linear or type(m) == nn.Conv1d:
        torch.nn.init.xavier_uniform_(m.weight)

class Readout(nn.Module):
    def __init__(self, max_nodes, graph_out_chs, emb_size):
        super(Readout, self).__init__()
        self.max_nodes = max_nodes

        # 设置卷积层和池化层参数
        self.conv1_size = {
            "in_channels": self.max_nodes,
            "out_channels": 64,
            "kernel_size": 3,
            "padding": 1
        }
        self.conv2_size = {
            "in_channels": 64,
            "out_channels": 16,
            "kernel_size": 2,
            "padding": 1
        }
        self.maxp1_size = {
            "kernel_size": 3,
            "stride": 2
        }
        self.maxp2_size = {
            "kernel_size": 2,
            "stride": 2
        }

        self.feature1 = nn.Conv1d(**self.conv1_size)
        self.maxpool1 = nn.MaxPool1d(**self.maxp1_size)
        self.feature2 = nn.Conv1d(**self.conv2_size)
        self.maxpool2 = nn.MaxPool1d(**self.maxp2_size)

        # 根据conv和maxpool参数计算mlp尺寸
        self.mlp1_size = get_conv_mp_out_size(
            graph_out_chs + emb_size,
            self.conv2_size,
            [self.maxp1_size, self.maxp2_size]
        )
        self.mlp2_size = get_conv_mp_out_size(
            graph_out_chs,
            self.conv2_size,
            [self.maxp1_size, self.maxp2_size]
        )

        self.mlp1 = nn.Linear(1200, 1)
        self.mlp2 = nn.Linear(self.mlp2_size, 1)


    def forward(self, h, x):
        z_feature = torch.cat([h, x], 1)
        z_feature = z_feature.view(-1, self.max_nodes, h.shape[1] + x.shape[1])
        out_z = self.maxpool1(F.relu(self.feature1(z_feature)))
        out_z = self.maxpool2(F.relu(self.feature2(out_z)))
        out_z = out_z.view(-1, int(out_z.shape[1] * out_z.shape[-1]))
        out_z = self.mlp1(out_z)

        y_feature = h.view(-1, self.max_nodes, h.shape[1])
        out_y = self.maxpool1(F.relu(self.feature1(y_feature)))
        out_y = self.maxpool2(F.relu(self.feature2(out_y)))
        out_y = out_y.view(-1, int(out_y.shape[1] * out_y.shape[-1]))
        out_y = self.mlp2(out_y)

        out = out_z * out_y
        out = torch.sigmoid(torch.flatten(out))
        return out


class Net(nn.Module):
    def __init__(self, gated_graph_conv_args, emb_size, max_nodes, device):
        super(Net, self).__init__()
        self.ggc = GatedGraphConv(**gated_graph_conv_args).to(device)
        self.emb_size = emb_size
        self.readout = Readout(max_nodes, gated_graph_conv_args["out_channels"], emb_size)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.ggc(x, edge_index)
        x = self.readout(x, data.x) # data.x是原始节点特征
        return x

    def save(self, path):
        torch.save(self.state_dict(), path)

    def load(self, path):
        self.load_state_dict(torch.load(path))
