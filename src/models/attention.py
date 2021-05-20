import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler

class Attention(nn.Module):
    """
    Additive attention layer - Custom layer to perform weighted average over the second axis (axis=1)
        Transform a tensor of size [N, W, H] to [N, 1, H]
        N: batch size
        W: number of tokens
        H: hidden state dimension or word embedding dimension

    Example:
        m = Attention(300)
        input = Variable(torch.randn(4, 128, 300))
        output = m(input)
        print(output.size())
    """

    def __init__(self, dim, attn_bias = False):
        super(Attention, self).__init__()
        self.dim = dim
        self.attn_bias = attn_bias
        self.attn_weights = None

        self.attn = nn.Linear(self.dim, self.dim, bias = self.attn_bias)
        self.attn_combine = nn.Linear(self.dim, 1, bias = self.attn_bias)

    def forward(self, input, attention_mask = None):
        wplus = self.attn(input)
        wplus = torch.tanh(wplus)

        att_w = self.attn_combine(wplus)
        att_w = att_w.view(-1, wplus.size()[1])

        # apply attention mask to remove weight on padding
        if attention_mask is not None:
            # invert attention mask [0, 1] -> [-10000, 0] and add to attention weights
            attention_mask = (1 - attention_mask) * -10000
            att_w = att_w + attention_mask
        
        att_w = torch.softmax(att_w, dim=1)

        # save attention weights for visualization
        self.attn_weights = att_w

        # multiply input by attention weights
        after_attention = torch.bmm(att_w.unsqueeze(1), input)
        after_attention = torch.squeeze(after_attention, dim=1)

        return after_attention