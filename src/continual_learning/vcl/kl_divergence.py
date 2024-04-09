import torch
import torch.nn as nn


def kl_divergence(
    p_mean: nn.Parameter,
    p_sigma: nn.Parameter,
    q_mean: nn.Parameter,
    q_sigma: nn.Parameter,
) -> torch.Tensor:
    kl = (
        0.5
        * (
            2 * torch.log(p_sigma / q_sigma)
            - 1
            + (q_sigma / p_sigma).pow(2)
            + ((p_mean - q_mean) / p_sigma).pow(2)
        ).sum()
    )
    return kl


def js_divergence(
    p_mean: nn.Parameter,
    p_sigma: nn.Parameter,
    q_mean: nn.Parameter,
    q_sigma: nn.Parameter,
) -> torch.Tensor:
    m_mean = 0.5 * (p_mean + q_mean)
    m_sigma = torch.sqrt(0.5 * (p_sigma.pow(2) + q_sigma.pow(2)))

    kl_pm = kl_divergence(p_mean, p_sigma, m_mean, m_sigma)
    kl_qm = kl_divergence(q_mean, q_sigma, m_mean, m_sigma)

    js = 0.5 * (kl_pm + kl_qm)
    return js
