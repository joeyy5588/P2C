import torch
import torch.nn.functional as F
# from torch.nn.functional import gumbel_softmax

def sample_gumbel(shape, eps=1e-20):
    U = torch.rand(shape)
    return -torch.log(-torch.log(U + eps) + eps)


def gumbel_softmax_sample(logits, temperature=0.5):
    y = logits + sample_gumbel(logits.size()).to(logits.device)
    return F.softmax(y / temperature, dim=-1)

def gumbel_softmax(logits, tau=0.5):
    """
    input: [*, n_class]
    return: [*, n_class] an one-hot vector
    """
    y = gumbel_softmax_sample(logits, tau)
    shape = y.size()
    _, ind = y.max(dim=-1)
    y_hard = torch.zeros_like(y).view(-1, shape[-1])
    y_hard.scatter_(1, ind.view(-1, 1), 1)
    y_hard = y_hard.view(*shape)
    gumbel_logits = (y_hard - y).detach() + y
    return gumbel_logits

# def gumbel_softmax(logits, temperature=0.5, hard=False):
#     """
#     ST-gumple-softmax
#     input: [*, n_class]
#     return: flatten --> [*, n_class] an one-hot vector
#     """
#     y = gumbel_softmax_sample(logits, temperature)
    
#     if not hard:
#         return y.view(-1, latent_dim * categorical_dim)

#     shape = y.size()
#     _, ind = y.max(dim=-1)
#     y_hard = torch.zeros_like(y).view(-1, shape[-1])
#     y_hard.scatter_(1, ind.view(-1, 1), 1)
#     y_hard = y_hard.view(*shape)
#     # Set gradients w.r.t. y_hard gradients w.r.t. y
#     y_hard = (y_hard - y).detach() + y
#     return y_hard.view(-1, latent_dim * categorical_dim)

# logits = torch.randn(5, 10)
# print(torch.nn.Softmax(dim=1)(logits))
# print(torch.argmax(logits, dim=1))
# print(gumbel_softmax(logits))
