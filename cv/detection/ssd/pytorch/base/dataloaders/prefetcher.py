import torch

def eval_prefetcher(load_iterator, device, pad_input=False, nhwc=False, fp16=False):
    prefetch_stream = torch.cuda.Stream()

    def _prefetch():
        try:
            # Note: eval has 5 outputs, only care about 3
            img, img_id, img_size, _, _ = next(load_iterator)
        except StopIteration:
            return None, None, None

        with torch.cuda.stream(prefetch_stream):
            img = img.to(device, non_blocking=True)
            if fp16:
                img = img.half()
            if pad_input:
                s = img.shape
                s = [s[0], 1, s[2], s[3]]
                img = torch.cat([img, torch.ones(s, device=img.device, dtype=img.dtype)], dim=1)
            if nhwc:
                img = img.permute(0, 2, 3, 1).contiguous()

        return img, img_id, img_size

    next_img, next_img_id, next_img_size = _prefetch()

    while next_img is not None:
        torch.cuda.current_stream().wait_stream(prefetch_stream)
        current_img, current_img_id, current_img_size = next_img, next_img_id, next_img_size
        next_img, next_img_id, next_img_size = _prefetch()
        yield current_img, current_img_id, current_img_size

