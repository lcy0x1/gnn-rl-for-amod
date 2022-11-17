import math

import torch
from torch import Tensor
from torch.nn import Module, Parameter, init


class TrueBilinear(Module):
    r"""Applies a bilinear transformation to the incoming data:
    :math:`y = x_1 A x_2^T + b`

    Args:
        in1_features: size of last dimension of input 1
        in2_features: size of last dimension of input 2
        out1_features: size of last dimension of input 1
        out2_features: size of last dimension of input 2
        bias: If set to False, the layer will not learn an additive bias.
            Default: ``True``

    Shape:
        - Input1: :math:`(*, H_{out1}, H_{in1})` where :math:`H_{in1}=\text{in1\_features}`,
          :math:`H_{out1}=\text{out1\_features}`, and
          :math:`*` means any number of additional dimensions including none. All but the last dimension
          of the inputs should be the same.
        - Input2: :math:`(*, H_{out2}, H_{in2})` where :math:`H_{in2}=\text{in2\_features}`, and
          :math:`H_{out2}=\text{out2\_features}`.
        - Output: :math:`(*, H_{out1}, H_{out2})` all but the last 2 dimension are the same shape as the input.

    Attributes:
        weight: the learnable weights of the module of shape
            :math:`(\text{in1\_features}, \text{in2\_features})`.
            The values are initialized from :math:`\mathcal{U}(-\sqrt{k}, \sqrt{k})`, where
            :math:`k = \frac{1}{\text{in1\_features}}`
        bias:   the learnable bias of the module of shape :math:`(\text{out1\_features}, \text{out2\_features})`.
                If :attr:`bias` is ``True``, the values are initialized from
                :math:`\mathcal{U}(-\sqrt{k}, \sqrt{k})`, where
                :math:`k = \frac{1}{\text{in1\_features}}`

    Examples::

        >>> m = TrueBilinear(20, 30, 40, 50)
        >>> input1 = torch.randn(128, 40, 20)
        >>> input2 = torch.randn(128, 50, 30)
        >>> output = m(input1, input2)
        >>> print(output.size())
        torch.Size([128, 40, 50])
    """

    __constants__ = ['in1_features', 'in2_features']
    in1_features: int
    in2_features: int
    weight: Tensor

    def __init__(self, in1_features: int, in2_features: int,
                 out1_features: int, out2_features: int,
                 bias: bool = True, device=None, dtype=None) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super(TrueBilinear, self).__init__()
        self.in1_features = in1_features
        self.in2_features = in2_features
        self.weight = Parameter(torch.empty((in1_features, in2_features), **factory_kwargs))

        if bias:
            self.bias = Parameter(torch.empty(out1_features, out2_features, **factory_kwargs))
        else:
            self.bias = None
        self.reset_parameters()

    def reset_parameters(self) -> None:
        bound = 1 / math.sqrt(self.weight.size(1))
        if self.bias is not None:
            init.uniform_(self.bias, -bound, bound)
        init.uniform_(self.weight, -bound, bound)

    def forward(self, input1: Tensor, input2: Tensor) -> Tensor:
        n = len(input1.size())
        perm = [i for i in range(n - 2)] + [n - 1, n - 2]
        x = input2.matmul(input1.matmul(self.weight).permute(perm)).permute(perm)
        return x if self.bias is None else x + self.bias

    def extra_repr(self) -> str:
        return 'in1_features={}, in2_features={}, bias={}'.format(
            self.in1_features, self.in2_features, self.bias is not None
        )
