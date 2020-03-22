import torch
import torch.nn as nn


class CrossEntropyLoss(object):
    """
    Cross entropy with instance-wise weights. Leave `aggregate` to None to obtain a loss
    vector of shape (batch_size,).
    """
    def __init__(self, aggregate='mean'):
        super(CrossEntropyLoss, self).__init__()
        assert aggregate in ['sum', 'mean', None]
        self.aggregate = aggregate

    def build_loss(self, mode='ce'):
        """Choices: ['ce' or 'focal']"""
        if mode == 'ce':
            return self.cross_entropy_with_weights

        else:
            raise NotImplementedError


    def forward(self, input, target, weights=None):
        if self.aggregate == 'sum':

            return cross_entropy_with_weights(input, target, weights).sum()

        elif self.aggregate == 'mean':
            return cross_entropy_with_weights(input, target, weights).mean()

        elif self.aggregate is None:
            return cross_entropy_with_weights(input, target, weights)


    def log_sum_exp(x):
        b, _ = torch.max(x, 1)
        y = b + torch.log(torch.exp(x - b.expand_as(x)).sum(1))
        # y.size() = [N, 1]. Squeeze to [N] and return
        return y.squeeze(1)


    def class_select(logits, target):

        # in numpy, this would be logits[:, target].
        batch_size, num_classes = logits.size()
        if target.is_cuda:
            device = target.data.get_device()
            one_hot_mask = torch.autograd.Variable(torch.arange(0, num_classes)
                                                   .long()
                                                   .repeat(batch_size, 1)
                                                   .cuda(device)
                                                   .eq(target.data.repeat(num_classes, 1).t()))
        else:
            one_hot_mask = torch.autograd.Variable(torch.arange(0, num_classes)
                                                   .long()
                                                   .repeat(batch_size, 1)
                                                   .eq(target.data.repeat(num_classes, 1).t()))
        return logits.masked_select(one_hot_mask)


    def cross_entropy_with_weights(logits, target, weights=None):
        assert logits.dim() == 2
        assert not target.requires_grad
        target = target.squeeze(1) if target.dim() == 2 else target
        assert target.dim() == 1
        loss = log_sum_exp(logits) - class_select(logits, target)
        if weights is not None:

            # loss.size() = [N]. Assert weights has the same shape
            assert list(loss.size()) == list(weights.size())

            # Weight the loss
            loss = loss * weights

        return loss


if __name__ == "__main__":
    loss = CrossEntropyLoss(cuda=True)
    a = torch.rand(1, 3, 7, 7).cuda()
    b = torch.rand(1, 7, 7).cuda()
    print(loss.CrossEntropyLoss(a, b).item())
