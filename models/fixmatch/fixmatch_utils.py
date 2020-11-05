import torch
import torch.nn.functional as F
from train_utils import ce_loss


class Get_Scalar:
    def __init__(self, value):
        """
        Initialize the value

        Args:
            self: (todo): write your description
            value: (todo): write your description
        """
        self.value = value
        
    def get_value(self, iter):
        """
        Return the value of an iterable.

        Args:
            self: (todo): write your description
            iter: (todo): write your description
        """
        return self.value
    
    def __call__(self, iter):
        """
        Calls the given call.

        Args:
            self: (todo): write your description
            iter: (int): write your description
        """
        return self.value


def consistency_loss(logits_w, logits_s, name='ce', T=1.0, p_cutoff=0.0, use_hard_labels=True):
    """
    Consistency loss.

    Args:
        logits_w: (todo): write your description
        logits_s: (todo): write your description
        name: (str): write your description
        T: (array): write your description
        p_cutoff: (array): write your description
        use_hard_labels: (bool): write your description
    """
    assert name in ['ce', 'L2']
    logits_w = logits_w.detach()
    if name == 'L2':
        assert logits_w.size() == logits_s.size()
        return F.mse_loss(logits_s, logits_w, reduction='mean')
    
    elif name == 'L2_mask':
        pass

    elif name == 'ce':
        pseudo_label = torch.softmax(logits_w, dim=-1)
        max_probs, max_idx = torch.max(pseudo_label, dim=-1)
        mask = max_probs.ge(p_cutoff).float()
        
        if use_hard_labels:
            masked_loss = ce_loss(logits_s, max_idx, use_hard_labels, reduction='none') * mask
        else:
            pseudo_label = torch.softmax(logits_w/T, dim=-1)
            masked_loss = ce_loss(logits_s, pseudo_label, use_hard_labels) * mask
        return masked_loss.mean(), mask.mean()

    else:
        assert Exception('Not Implemented consistency_loss')
            