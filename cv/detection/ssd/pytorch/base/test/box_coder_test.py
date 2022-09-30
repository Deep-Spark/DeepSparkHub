import sys
sys.path.append("..")

from box_coder import DefaultBoxes, Encoder
import torch


def dboxes300_coco():
    figsize = 300
    feat_size = [38, 19, 10, 5, 3, 1]
    steps = [8, 16, 32, 64, 100, 300]
    # use the scales here: https://github.com/amdegroot/ssd.pytorch/blob/master/data/config.py
    scales = [21, 45, 99, 153, 207, 261, 315]
    aspect_ratios = [[2], [2, 3], [2, 3], [2, 3], [2], [2]]
    dboxes = DefaultBoxes(figsize, feat_size, steps, scales, aspect_ratios)
    return dboxes


if __name__ == "__main__":
    dboxes = dboxes300_coco()
    encoder = Encoder(dboxes, fast_nms=False)
    encoder_fast = Encoder(dboxes, fast_nms=True)

    saved_inputs = torch.load('inputs.pth')

    bboxes = saved_inputs['bbox'].float()
    scores = saved_inputs['scores'].float()
    criteria = float(saved_inputs['criteria'])
    max_num = int(saved_inputs['max_output'])

    print('bboxes: {}, scores: {}'.format(bboxes.shape, scores.shape))

    for i in range(bboxes.shape[0]):
        box1, label1, score1 = encoder.decode_batch(bboxes[i, :, :].unsqueeze(0), scores[i, :, :].unsqueeze(0), criteria, max_num)[0]

        box2, label2, score2 = \
        encoder_fast.decode_batch(bboxes[i, :, :].unsqueeze(0), scores[i, :, :].unsqueeze(0), criteria, max_num)[0]

        print('label: {}, fast label: {}'.format(label1, label2))
