import torch
import numpy as np
import torch.nn.functional as F
import cv2
from typing import Optional

from utils import (
    global_max_seq_len,
    image_prepare_val,
    tok_to_ind,
    ind_to_tok,
    N_CAPTIONS
)

def generate(
    model,
    image,
    max_seq_len: Optional[int] = global_max_seq_len,
    top_p: Optional[float] = None,
    top_k: Optional[int] = None,
    greedy: bool = False,
    temperature = 1.0,
    device='cpu'
):
    assert top_p is None or top_k is None, "Don't use top_p and top_k at the same time"

    model = model.to(device)
    model.eval()

    image = image_prepare_val(image)

    img_batch = image[None, ...].to(device)

    result_tokens = ['<BOS>']

    with torch.no_grad():
        while result_tokens[-1] != '<EOS>' and len(result_tokens) < max_seq_len:
            result_indexes = [tok_to_ind[tok] for tok in result_tokens]
            captions_batch = torch.tensor([result_indexes], dtype=torch.int64)
            captions_batch = captions_batch.repeat((1, N_CAPTIONS, 1)).to(device)

            pred = model(img_batch, captions_batch)
            pred = pred[:, 0, -1, ...].reshape(-1)
            preds = torch.argsort(pred, dim=-1, descending=True)

            if top_k is not None:
                preds = preds[:top_k]

            elif top_p is not None:
                mask = np.cumsum(pred.cpu()) <= top_p
                mask[0] = 1
                preds = preds[mask]

            elif greedy:
                preds = preds[:1]

            logits_left = pred[preds]
            dist_next = F.softmax(logits_left / temperature, dim=-1).cpu().data.numpy()

            next_idx = np.random.choice(preds.cpu().data.numpy(), p=dist_next)

            result_tokens.append(ind_to_tok[next_idx])

        if result_tokens[-1] != '<EOS>':
            result_tokens[-1] = '<EOS>'

        result_text = ' '.join(result_tokens)

        return result_tokens, result_text


def run(model, image_path):
    image = cv2.imread(image_path)[:, :, ::-1]

    tokens, text = generate(
        model,
        image,
        greedy=True
    )
    return image, text
