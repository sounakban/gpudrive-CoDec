import torch

def average_displacement_error(pred, gt):
    """
    Calculate the average displacement error between predicted and ground truth points.

    Args:
        pred (torch.Tensor): Predicted points of shape (N, 2).
        gt (torch.Tensor): Ground truth points of shape (N, 2).

    Returns:
        float: Average displacement error.
    """
    return torch.mean(torch.sqrt(torch.sum((pred - gt) ** 2, dim=1))).item()



def log_likelihood(pred, gt): ####FIXME
    """
    Calculate the log likelihood of the predicted points given the ground truth points.

    Args:
        pred (torch.Tensor): Predicted points of shape (N, M).
        gt (torch.Tensor): Ground truth points of shape (N, M).

    Returns:
        float: Log likelihood.
    """
    return -0.5 * torch.sum((pred - gt) ** 2).item()