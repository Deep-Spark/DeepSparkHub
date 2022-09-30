# Copyright (c) OpenMMLab. All rights reserved.
import math

import cv2
import mmcv
import numpy as np
from scipy.ndimage import convolve
from scipy.special import gamma

from .metric_utils import gauss_gradient


def sad(alpha, trimap, pred_alpha):
    if alpha.ndim != 2 or trimap.ndim != 2 or pred_alpha.ndim != 2:
        raise ValueError(
            'input alpha, trimap and pred_alpha should has two dimensions, '
            f'alpha {alpha.shape}, please check their shape: '
            f'trimap {trimap.shape}, pred_alpha {pred_alpha.shape}')
    assert (pred_alpha[trimap == 0] == 0).all()
    assert (pred_alpha[trimap == 255] == 255).all()
    alpha = alpha.astype(np.float64) / 255
    pred_alpha = pred_alpha.astype(np.float64) / 255
    sad_result = np.abs(pred_alpha - alpha).sum() / 1000
    return sad_result


def mse(alpha, trimap, pred_alpha):
    if alpha.ndim != 2 or trimap.ndim != 2 or pred_alpha.ndim != 2:
        raise ValueError(
            'input alpha, trimap and pred_alpha should has two dimensions, '
            f'alpha {alpha.shape}, please check their shape: '
            f'trimap {trimap.shape}, pred_alpha {pred_alpha.shape}')
    assert (pred_alpha[trimap == 0] == 0).all()
    assert (pred_alpha[trimap == 255] == 255).all()
    alpha = alpha.astype(np.float64) / 255
    pred_alpha = pred_alpha.astype(np.float64) / 255
    weight_sum = (trimap == 128).sum()
    if weight_sum != 0:
        mse_result = ((pred_alpha - alpha)**2).sum() / weight_sum
    else:
        mse_result = 0
    return mse_result


def gradient_error(alpha, trimap, pred_alpha, sigma=1.4):
    """Gradient error for evaluating alpha matte prediction.

    Args:
        alpha (ndarray): Ground-truth alpha matte.
        trimap (ndarray): Input trimap with its value in {0, 128, 255}.
        pred_alpha (ndarray): Predicted alpha matte.
        sigma (float): Standard deviation of the gaussian kernel. Default: 1.4.
    """
    if alpha.ndim != 2 or trimap.ndim != 2 or pred_alpha.ndim != 2:
        raise ValueError(
            'input alpha, trimap and pred_alpha should has two dimensions, '
            f'alpha {alpha.shape}, please check their shape: '
            f'trimap {trimap.shape}, pred_alpha {pred_alpha.shape}')
    if not ((pred_alpha[trimap == 0] == 0).all() and
            (pred_alpha[trimap == 255] == 255).all()):
        raise ValueError(
            'pred_alpha should be masked by trimap before evaluation')
    alpha = alpha.astype(np.float64)
    pred_alpha = pred_alpha.astype(np.float64)
    alpha_normed = np.zeros_like(alpha)
    pred_alpha_normed = np.zeros_like(pred_alpha)
    cv2.normalize(alpha, alpha_normed, 1., 0., cv2.NORM_MINMAX)
    cv2.normalize(pred_alpha, pred_alpha_normed, 1., 0., cv2.NORM_MINMAX)

    alpha_grad = gauss_gradient(alpha_normed, sigma).astype(np.float32)
    pred_alpha_grad = gauss_gradient(pred_alpha_normed,
                                     sigma).astype(np.float32)

    grad_loss = ((alpha_grad - pred_alpha_grad)**2 * (trimap == 128)).sum()
    # same as SAD, divide by 1000 to reduce the magnitude of the result
    return grad_loss / 1000


def connectivity(alpha, trimap, pred_alpha, step=0.1):
    """Connectivity error for evaluating alpha matte prediction.

    Args:
        alpha (ndarray): Ground-truth alpha matte with shape (height, width).
            Value range of alpha is [0, 255].
        trimap (ndarray): Input trimap with shape (height, width). Elements
            in trimap are one of {0, 128, 255}.
        pred_alpha (ndarray): Predicted alpha matte with shape (height, width).
            Value range of pred_alpha is [0, 255].
        step (float): Step of threshold when computing intersection between
            `alpha` and `pred_alpha`.
    """
    if alpha.ndim != 2 or trimap.ndim != 2 or pred_alpha.ndim != 2:
        raise ValueError(
            'input alpha, trimap and pred_alpha should has two dimensions, '
            f'alpha {alpha.shape}, please check their shape: '
            f'trimap {trimap.shape}, pred_alpha {pred_alpha.shape}')
    if not ((pred_alpha[trimap == 0] == 0).all() and
            (pred_alpha[trimap == 255] == 255).all()):
        raise ValueError(
            'pred_alpha should be masked by trimap before evaluation')
    alpha = alpha.astype(np.float32) / 255
    pred_alpha = pred_alpha.astype(np.float32) / 255

    thresh_steps = np.arange(0, 1 + step, step)
    round_down_map = -np.ones_like(alpha)
    for i in range(1, len(thresh_steps)):
        alpha_thresh = alpha >= thresh_steps[i]
        pred_alpha_thresh = pred_alpha >= thresh_steps[i]
        intersection = (alpha_thresh & pred_alpha_thresh).astype(np.uint8)

        # connected components
        _, output, stats, _ = cv2.connectedComponentsWithStats(
            intersection, connectivity=4)
        # start from 1 in dim 0 to exclude background
        size = stats[1:, -1]

        # largest connected component of the intersection
        omega = np.zeros_like(alpha)
        if len(size) != 0:
            max_id = np.argmax(size)
            # plus one to include background
            omega[output == max_id + 1] = 1

        mask = (round_down_map == -1) & (omega == 0)
        round_down_map[mask] = thresh_steps[i - 1]
    round_down_map[round_down_map == -1] = 1

    alpha_diff = alpha - round_down_map
    pred_alpha_diff = pred_alpha - round_down_map
    # only calculate difference larger than or equal to 0.15
    alpha_phi = 1 - alpha_diff * (alpha_diff >= 0.15)
    pred_alpha_phi = 1 - pred_alpha_diff * (pred_alpha_diff >= 0.15)

    connectivity_error = np.sum(
        np.abs(alpha_phi - pred_alpha_phi) * (trimap == 128))
    # same as SAD, divide by 1000 to reduce the magnitude of the result
    return connectivity_error / 1000


def reorder_image(img, input_order='HWC'):
    """Reorder images to 'HWC' order.

    If the input_order is (h, w), return (h, w, 1);
    If the input_order is (c, h, w), return (h, w, c);
    If the input_order is (h, w, c), return as it is.

    Args:
        img (ndarray): Input image.
        input_order (str): Whether the input order is 'HWC' or 'CHW'.
            If the input image shape is (h, w), input_order will not have
            effects. Default: 'HWC'.

    Returns:
        ndarray: reordered image.
    """

    if input_order not in ['HWC', 'CHW']:
        raise ValueError(
            f'Wrong input_order {input_order}. Supported input_orders are '
            '"HWC" and "CHW"')
    if len(img.shape) == 2:
        img = img[..., None]
        return img
    if input_order == 'CHW':
        img = img.transpose(1, 2, 0)
    return img


def psnr(img1, img2, crop_border=0, input_order='HWC', convert_to=None):
    """Calculate PSNR (Peak Signal-to-Noise Ratio).

    Ref: https://en.wikipedia.org/wiki/Peak_signal-to-noise_ratio

    Args:
        img1 (ndarray): Images with range [0, 255].
        img2 (ndarray): Images with range [0, 255].
        crop_border (int): Cropped pixels in each edges of an image. These
            pixels are not involved in the PSNR calculation. Default: 0.
        input_order (str): Whether the input order is 'HWC' or 'CHW'.
            Default: 'HWC'.
        convert_to (str): Whether to convert the images to other color models.
            If None, the images are not altered. When computing for 'Y',
            the images are assumed to be in BGR order. Options are 'Y' and
            None. Default: None.

    Returns:
        float: psnr result.
    """

    assert img1.shape == img2.shape, (
        f'Image shapes are different: {img1.shape}, {img2.shape}.')
    if input_order not in ['HWC', 'CHW']:
        raise ValueError(
            f'Wrong input_order {input_order}. Supported input_orders are '
            '"HWC" and "CHW"')
    img1 = reorder_image(img1, input_order=input_order)
    img2 = reorder_image(img2, input_order=input_order)

    img1, img2 = img1.astype(np.float32), img2.astype(np.float32)
    if isinstance(convert_to, str) and convert_to.lower() == 'y':
        img1 = mmcv.bgr2ycbcr(img1 / 255., y_only=True) * 255.
        img2 = mmcv.bgr2ycbcr(img2 / 255., y_only=True) * 255.
    elif convert_to is not None:
        raise ValueError('Wrong color model. Supported values are '
                         '"Y" and None.')

    if crop_border != 0:
        img1 = img1[crop_border:-crop_border, crop_border:-crop_border, None]
        img2 = img2[crop_border:-crop_border, crop_border:-crop_border, None]

    mse_value = np.mean((img1 - img2)**2)
    if mse_value == 0:
        return float('inf')
    return 20. * np.log10(255. / np.sqrt(mse_value))


def mae(img1, img2, crop_border=0, input_order='HWC', convert_to=None):
    """Calculate mean average error for evaluation.

    Args:
        img1 (ndarray): Images with range [0, 255].
        img2 (ndarray): Images with range [0, 255].
        crop_border (int): Cropped pixels in each edges of an image. These
            pixels are not involved in the PSNR calculation. Default: 0.
        input_order (str): Whether the input order is 'HWC' or 'CHW'.
            Default: 'HWC'.
        convert_to (str): Whether to convert the images to other color models.
            If None, the images are not altered. Options are 'RGB2Y', 'BGR2Y'
            and None. Default: None.

    Returns:
        float: mae result.
    """

    assert img1.shape == img2.shape, (
        f'Image shapes are different: {img1.shape}, {img2.shape}.')
    if input_order not in ['HWC', 'CHW']:
        raise ValueError(
            f'Wrong input_order {input_order}. Supported input_orders are '
            '"HWC" and "CHW"')
    img1 = reorder_image(img1, input_order=input_order)
    img2 = reorder_image(img2, input_order=input_order)

    img1, img2 = img1.astype(np.float32), img2.astype(np.float32)
    img1, img2 = img1 / 255., img2 / 255.
    if isinstance(convert_to, str) and convert_to.lower() == 'rgb2y':
        img1 = mmcv.rgb2ycbcr(img1, y_only=True)
        img2 = mmcv.rgb2ycbcr(img2, y_only=True)
    elif isinstance(convert_to, str) and convert_to.lower() == 'bgr2y':
        img1 = mmcv.bgr2ycbcr(img1, y_only=True)
        img2 = mmcv.bgr2ycbcr(img2, y_only=True)
    elif convert_to is not None:
        raise ValueError('Wrong color model. Supported values are '
                         '"RGB2Y", "BGR2Y" and None.')

    if crop_border != 0:
        img1 = img1[crop_border:-crop_border, crop_border:-crop_border, None]
        img2 = img2[crop_border:-crop_border, crop_border:-crop_border, None]

    l1_value = np.mean(np.abs(img1 - img2))

    return l1_value


def _ssim(img1, img2):
    """Calculate SSIM (structural similarity) for one channel images.

    It is called by func:`calculate_ssim`.

    Args:
        img1, img2 (ndarray): Images with range [0, 255] with order 'HWC'.

    Returns:
        float: ssim result.
    """

    C1 = (0.01 * 255)**2
    C2 = (0.03 * 255)**2

    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    kernel = cv2.getGaussianKernel(11, 1.5)
    window = np.outer(kernel, kernel.transpose())

    mu1 = cv2.filter2D(img1, -1, window)[5:-5, 5:-5]
    mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]
    mu1_sq = mu1**2
    mu2_sq = mu2**2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = cv2.filter2D(img1**2, -1, window)[5:-5, 5:-5] - mu1_sq
    sigma2_sq = cv2.filter2D(img2**2, -1, window)[5:-5, 5:-5] - mu2_sq
    sigma12 = cv2.filter2D(img1 * img2, -1, window)[5:-5, 5:-5] - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) *
                (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) *
                                       (sigma1_sq + sigma2_sq + C2))
    return ssim_map.mean()


def ssim(img1, img2, crop_border=0, input_order='HWC', convert_to=None):
    """Calculate SSIM (structural similarity).

    Ref:
    Image quality assessment: From error visibility to structural similarity

    The results are the same as that of the official released MATLAB code in
    https://ece.uwaterloo.ca/~z70wang/research/ssim/.

    For three-channel images, SSIM is calculated for each channel and then
    averaged.

    Args:
        img1 (ndarray): Images with range [0, 255].
        img2 (ndarray): Images with range [0, 255].
        crop_border (int): Cropped pixels in each edges of an image. These
            pixels are not involved in the SSIM calculation. Default: 0.
        input_order (str): Whether the input order is 'HWC' or 'CHW'.
            Default: 'HWC'.
        convert_to (str): Whether to convert the images to other color models.
            If None, the images are not altered. When computing for 'Y',
            the images are assumed to be in BGR order. Options are 'Y' and
            None. Default: None.

    Returns:
        float: ssim result.
    """

    assert img1.shape == img2.shape, (
        f'Image shapes are different: {img1.shape}, {img2.shape}.')
    if input_order not in ['HWC', 'CHW']:
        raise ValueError(
            f'Wrong input_order {input_order}. Supported input_orders are '
            '"HWC" and "CHW"')
    img1 = reorder_image(img1, input_order=input_order)
    img2 = reorder_image(img2, input_order=input_order)

    if isinstance(convert_to, str) and convert_to.lower() == 'y':
        img1, img2 = img1.astype(np.float32), img2.astype(np.float32)
        img1 = mmcv.bgr2ycbcr(img1 / 255., y_only=True) * 255.
        img2 = mmcv.bgr2ycbcr(img2 / 255., y_only=True) * 255.
        img1 = np.expand_dims(img1, axis=2)
        img2 = np.expand_dims(img2, axis=2)
    elif convert_to is not None:
        raise ValueError('Wrong color model. Supported values are '
                         '"Y" and None')

    if crop_border != 0:
        img1 = img1[crop_border:-crop_border, crop_border:-crop_border, None]
        img2 = img2[crop_border:-crop_border, crop_border:-crop_border, None]

    ssims = []
    for i in range(img1.shape[2]):
        ssims.append(_ssim(img1[..., i], img2[..., i]))
    return np.array(ssims).mean()


class L1Evaluation:
    """L1 evaluation metric.

    Args:
        data_dict (dict): Must contain keys of 'gt_img' and 'fake_res'. If
            'mask' is given, the results will be computed with mask as weight.
    """

    def __call__(self, data_dict):
        gt = data_dict['gt_img']
        if 'fake_img' in data_dict:
            pred = data_dict.get('fake_img')
        else:
            pred = data_dict.get('fake_res')
        mask = data_dict.get('mask', None)

        from mmedit.models.losses.pixelwise_loss import l1_loss
        l1_error = l1_loss(pred, gt, weight=mask, reduction='mean')

        return l1_error


def estimate_aggd_param(block):
    """Estimate AGGD (Asymmetric Generalized Gaussian Distribution) parameters.

    Args:
        block (ndarray): 2D Image block.

    Returns:
        tuple: alpha (float), beta_l (float) and beta_r (float) for the AGGD
            distribution (Estimating the parames in Equation 7 in the paper).
    """
    block = block.flatten()
    gam = np.arange(0.2, 10.001, 0.001)  # len = 9801
    gam_reciprocal = np.reciprocal(gam)
    r_gam = np.square(gamma(gam_reciprocal * 2)) / (
        gamma(gam_reciprocal) * gamma(gam_reciprocal * 3))

    left_std = np.sqrt(np.mean(block[block < 0]**2))
    right_std = np.sqrt(np.mean(block[block > 0]**2))
    gammahat = left_std / right_std
    rhat = (np.mean(np.abs(block)))**2 / np.mean(block**2)
    rhatnorm = (rhat * (gammahat**3 + 1) *
                (gammahat + 1)) / ((gammahat**2 + 1)**2)
    array_position = np.argmin((r_gam - rhatnorm)**2)

    alpha = gam[array_position]
    beta_l = left_std * np.sqrt(gamma(1 / alpha) / gamma(3 / alpha))
    beta_r = right_std * np.sqrt(gamma(1 / alpha) / gamma(3 / alpha))
    return (alpha, beta_l, beta_r)


def compute_feature(block):
    """Compute features.

    Args:
        block (ndarray): 2D Image block.

    Returns:
        list: Features with length of 18.
    """
    feat = []
    alpha, beta_l, beta_r = estimate_aggd_param(block)
    feat.extend([alpha, (beta_l + beta_r) / 2])

    # distortions disturb the fairly regular structure of natural images.
    # This deviation can be captured by analyzing the sample distribution of
    # the products of pairs of adjacent coefficients computed along
    # horizontal, vertical and diagonal orientations.
    shifts = [[0, 1], [1, 0], [1, 1], [1, -1]]
    for shift in shifts:
        shifted_block = np.roll(block, shift, axis=(0, 1))
        alpha, beta_l, beta_r = estimate_aggd_param(block * shifted_block)
        mean = (beta_r - beta_l) * (gamma(2 / alpha) / gamma(1 / alpha))
        feat.extend([alpha, mean, beta_l, beta_r])
    return feat

