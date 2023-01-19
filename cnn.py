import numpy as np

class CNN:
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=True) -> None:
        kernel_size = (kernel_size, kernel_size) if isinstance(kernel_size, int) else kernel_size
        stride = (stride, stride) if isinstance(stride, int) else stride
        padding = (padding, padding) if isinstance(padding, int) else padding

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

        self.weight = np.random.randn(out_channels, in_channels, *kernel_size)
        self.bias = np.ones((out_channels,)) if bias else None
        # print('kernel_shape', self.weight.shape)
        # print('bias_shape', self.bias.shape)


    def forward(self, x):
        batch_size, channels, width, height = x.shape
        pad_w, pad_h = self.padding
        kernel_w, kernel_h = self.kernel_size
        stride_w, stride_h = self.stride

        if pad_w + pad_h > 0:
            x = np.pad(x, ((0,0), (0,0), (pad_w, pad_w), (pad_h, pad_h)), 'constant', constant_values=0)
        # print('padding_shape', x.shape)

        W = width - kernel_w + 2*pad_w
        H = height - kernel_h + 2*pad_h

        out_w = int(W / stride_w) + 1
        out_h = int(H / stride_h) + 1
        output = np.zeros((batch_size, self.out_channels, out_w, out_h))

        # x: 5차원 배열
        x = x[:, None, ...]  # x[: np.newaxis, :, :, :]와 동일
        for j, h in enumerate(range(0, H+1, stride_h)):
            for i, w in enumerate(range(0, W+1, stride_w)):
                convolution = x[..., w:w+kernel_w, h:h+kernel_h]
                convolution = self.weight * convolution
                convolution = convolution.sum(axis=(2, 3, 4))

                if self.bias is not None:
                    convolution += self.bias
                output[:, :, i, j] = convolution

        # # x: 4차원 배열
        # for n in range(batch_size):
        #     for j, h in enumerate(range(0, H+1, stride_h)):
        #         for i, w in enumerate(range(0, W+1, stride_w)):
        #             convolution = x[n, :, w:w+self.kernel_size[0], h:h+self.kernel_size[1]][None, ...]
        #             convolution = self.weight * convolution
        #             convolution = convolution.sum(axis=(1, 2, 3))

        #             if self.bias is not None:
        #                 convolution += self.bias
        #             output[n, :, i, j] = convolution

        return output

x = np.random.randn(2, 3, 16, 16)
numpy_cnn = CNN(3, 10, kernel_size=(5,5), stride=(1,2), padding=(1,1))
y = numpy_cnn.forward(x)
print('numpy', y.shape, y.sum().item())

# 파이토치 확인
import torch
import torch.nn as nn

x = torch.from_numpy(x).float()
torch_cnn = nn.Conv2d(3, 10, kernel_size=(5,5), stride=(1,2), padding=(1,1))
torch_cnn.weight.data = torch.from_numpy(numpy_cnn.weight).float()
torch_cnn.bias.data = torch.from_numpy(numpy_cnn.bias).float()

y2 = torch_cnn.forward(x)
print('torch', y2.shape, y2.sum().item())