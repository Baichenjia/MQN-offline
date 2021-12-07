import os
import gym
import numpy as np
import torch
from torch.distributions import uniform
from torch.distributions.normal import Normal
from torch.distributions import Bernoulli


class LinearSchedule(object):
    def __init__(self, schedule_timesteps, initial=1., final=0.):
        """Linear interpolation between initial_p and final_p over
        schedule_timesteps. After this many timesteps pass final_p is
        returned.

        Parameters
        ----------
        schedule_timesteps: int
            Number of timesteps for which to linearly anneal initial_p
            to final_p
        initial_p: float
            initial output value
        final_p: float
            final output value
        """
        self.schedule_timesteps = schedule_timesteps
        self.final = final
        self.initial = initial

    def __call__(self, t):
        """See Schedule.value"""
        fraction = min(float(t) / self.schedule_timesteps, 1.0)
        return self.initial + fraction * (self.final - self.initial)


class Wang(object):
    """Sample quantile levels for the Wang risk measure.
    Wang 2000

    Parameters
    ----------
    eta: float. Default: -0.75
        for eta < 0 produces risk-averse.
    """

    def __init__(self, eta=-0.75):
        self.eta = eta
        self.normal = Normal(loc=torch.Tensor([0]), scale=torch.Tensor([1]))

    def sample(self, num_samples):
        """
        Parameters
        ----------
        :param num_samples: tuple. (num_samples,)
        :param taus_uniform:

        """
        # 从0-1之间的均匀分布中采样 num_samples 个数字, 如 num_samples=(10,), 则采样10个数. 注意采样后的数并不是排序好的
        taus_uniform = uniform.Uniform(0., 1.).sample(num_samples)

        # icdf 函数输入的是分位数，输出分位数在 cdf 分布中对应的值
        # 将 icdf 得到的值 +eta, 随后由 cdf 函数得到累积分布.
        # 如果 eta > 0, 则分位数的分布由 uniform 向右倾斜, 变成 risk-seeking; 否则是 risk-aware.
        wang_tau = self.normal.cdf(
            value=self.normal.icdf(value=taus_uniform) + self.eta)

        return wang_tau


class CPW(object):
    """Sample quantile levels for the CPW risk measure.

    Parameters
    ----------
    eta: float.  这个分布可以调节关注 两侧 或者 中间.
    """

    def __init__(self, eta=0.71):
        self.eta = eta

    def sample(self, num_samples):
        """
        Parameters
        ----------
        :param num_samples:
        :param taus_uniform:
        """
        taus_uniform = uniform.Uniform(0., 1.).sample(num_samples)    # 采样 num_samples 个元素
        tau_eta = taus_uniform ** self.eta                            # 对 taus_uniform 中的元素, eta < 1 时会增加 taus_uniform 的值
        one_tau_eta = (1 - taus_uniform) ** self.eta                  # 对 (1-taus_uniform) 中的元素, eta < 1 会增加各元素的值
        # 原来 tau_eta 和 (1 - taus_uniform) 各个位置之和为1. 经过上述变换后, tau_eta + one_tau_eta 各位置之和均大于1
        # ((tau_eta + one_tau_eta) ** (1. / self.eta)) 在此将分布向右移动，每个位置均大于1
        # cpw_tau 最终相比于 taus_uniform 每个位置均向左移动
        cpw_tau = tau_eta / ((tau_eta + one_tau_eta) ** (1. / self.eta))
        return cpw_tau


class Power(object):
    """Sample quantile levels for the Power risk measure.
    Parameters
    ----------
    eta: float. if eta < 0 is risk averse, if eta > 0 is risk seeking.
    """

    def __init__(self, eta=-2):
        self.eta = eta
        self.exponent = 1 / (1 + np.abs(eta))

    def sample(self, num_samples):
        """
        Parameters
        ----------
        num_samples: tuple. (num_samples,)

        """
        taus_uniform = uniform.Uniform(0., 1.).sample(num_samples)

        if self.eta > 0:
            return taus_uniform ** self.exponent          # exponent 是 0-1 之间的数, 能够增大 taus_uniform 但不超过1
        else:
            return 1 - (1 - taus_uniform) ** self.exponent

