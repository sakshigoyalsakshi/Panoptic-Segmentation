"""
Author: Davy Neven
Licensed under the CC BY-NC 4.0 license (https://creativecommons.org/licenses/by-nc/4.0/)
"""
import torch
import torch.nn as nn
import models.erfnet as erfnet


class BranchedERFNet(nn.Module):
    def __init__(self, num_classes, encoder=None):
        super().__init__()

        print('Creating branched erfnet with {} classes'.format(num_classes))

        if encoder is None:
            self.encoder = erfnet.Encoder(sum(num_classes))
        else:
            self.encoder = encoder

        self.seed_layer = erfnet.seed_layer(8)

        self.decoders = nn.ModuleList()
        for n in num_classes:
            self.decoders.append(erfnet.Decoder(n))

    def init_output(self, n_sigma=1):
        with torch.no_grad():
            output_conv = self.decoders[1].output_conv
            print('initialize last layer with size: ',
                  output_conv.weight.size())

            output_conv.weight[:, 0:2, :, :].fill_(0)
            output_conv.bias[0:2].fill_(0)

            output_conv.weight[:, 2:2 + n_sigma, :, :].fill_(0)
            output_conv.bias[2:2 + n_sigma].fill_(1)

    def predict(self, output, type=" "):
        if type == "sigma":
            return self.decoders[1].forward(output)
        if type == "seed":
            return self.decoders[2].forward(output)
        if type == "sem":
            return self.decoders[0].forward(output)
        elif type == "ins":
            return torch.cat([self.decoders[1].forward(output), self.decoders[2].forward(output)], 1)

    def forward(self, input, only_encode=False):
        if only_encode:
            return self.encoder.forward(input, predict=True)
        else:
            output = self.encoder(input)
        features, output_layer = self.decoders[0].forward(output)
        seed_prediction = self.seed_layer.forward(features)
        _, offset_output = self.decoders[1].forward(output)

        return output_layer, torch.cat([offset_output, seed_prediction], 1)
