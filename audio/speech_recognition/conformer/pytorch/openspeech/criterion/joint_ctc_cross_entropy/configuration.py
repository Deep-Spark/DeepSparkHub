# MIT License
#
# Copyright (c) 2021 Soohwan Kim and Sangchun Ha and Soyoung Cho
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

from dataclasses import dataclass, field

from ...dataclass.configurations import OpenspeechDataclass


@dataclass
class JointCTCCrossEntropyLossConfigs(OpenspeechDataclass):
    """
    This is the configuration class to store the configuration of
    a :class:`~openspeech.criterion.JointCTCCrossEntropyLoss`.

    It is used to initiated an `CTCLoss` criterion.

    Configuration objects inherit from :class: `~openspeech.dataclass.configs.OpenspeechDataclass`.

    Configurations:
        criterion_name (str): name of criterion. (default: joint_ctc_cross_entropy)
        reduction (str): reduction method of criterion. (default: mean)
        ctc_weight (float): weight of ctc loss for training. (default: 0.3)
        cross_entropy_weight (float): weight of cross entropy loss for training. (default: 0.7)
        smoothing (float): ratio of smoothing loss (confidence = 1.0 - smoothing) (default: 0.0)
        zero_infibity (bool): whether to zero infinite losses and the associated gradients. (default: True)
    """
    criterion_name: str = field(
        default="joint_ctc_cross_entropy", metadata={"help": "Criterion name for training."}
    )
    reduction: str = field(
        default="mean", metadata={"help": "Reduction method of criterion"}
    )
    ctc_weight: float = field(
        default=0.3, metadata={"help": "Weight of ctc loss for training."}
    )
    cross_entropy_weight: float = field(
        default=0.7, metadata={"help": "Weight of cross entropy loss for training."}
    )
    smoothing: float = field(
        default=0.0, metadata={"help": "Ratio of smoothing loss (confidence = 1.0 - smoothing)"}
    )
    zero_infinity: bool = field(
        default=True, metadata={"help": "Whether to zero infinite losses and the associated gradients."}
    )
