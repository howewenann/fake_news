import torch
from torch import nn, optim

# Transformers
from transformers import BertTokenizerFast, BertModel, AdamW, get_linear_schedule_with_warmup
from transformers import BertPreTrainedModel


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


class LinearBlock(nn.Module):

    def __init__(self, in_dim, out_dim):
        super(LinearBlock, self).__init__()

        self.in_dim = in_dim
        self.out_dim = out_dim
        self.fc = nn.Linear(self.in_dim, self.out_dim)
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.fc(x)
        out = self.relu(out)
        return out


class HIBERT(BertPreTrainedModel):

    def __init__(self, 
                    config, 
                    n_classes, 
                    add_linear=None, 
                    attn_bias=False, 
                    freeze_layer_count=-1):

        super(HIBERT, self).__init__(config)

        self.n_classes = n_classes
        self.add_linear = add_linear
        self.attn_bias = attn_bias
        self.freeze_layer_count = freeze_layer_count
        self.attn_weights = None

        # Define model objects
        self.bert = BertModel(config, add_pooling_layer=False)
        self.fc_in_size = self.bert.config.hidden_size

        # Control layer freezing
        if freeze_layer_count == -1:
            # freeze all bert layers
            for param in self.bert.parameters():
                param.requires_grad = False

        if freeze_layer_count == -2:
            # unfreeze all bert layers
            for param in self.bert.parameters():
                param.requires_grad = True

        if freeze_layer_count > 0:
            # freeze embedding layer
            for param in self.bert.embeddings.parameters():
                param.requires_grad = False

            # freeze the top `freeze_layer_count` encoder layers
            for layer in self.bert.encoder.layer[:freeze_layer_count]:
                for param in layer.parameters():
                    param.requires_grad = False

        # Attention pooling layer
        self.attention = Attention(dim=self.bert.config.hidden_size, attn_bias=self.attn_bias)

        # fully connected layers
        if self.add_linear is None:
            self.fc = nn.ModuleList([nn.Linear(self.fc_in_size, self.n_classes)])
        
        else: 
            self.fc_layers = [self.fc_in_size] + self.add_linear
            self.fc = nn.ModuleList([
                LinearBlock(self.fc_layers[i], self.fc_layers[i+1])
                for i in range(len(self.fc_layers) - 1)
                ])
            
            # no relu after last dense (cannot use LinearBlock)
            self.fc.append(nn.Linear(self.fc_layers[-1], self.n_classes))

    def forward(self, input_ids, attention_mask, n_chunks):

        # Bert transformer (take sequential output)
        output, _ = self.bert(
            input_ids = input_ids, 
            attention_mask = attention_mask, 
            return_dict=False
            )

        # group chunks together
        chunks = output.split_with_sizes(n_chunks.tolist())

        # loop through attention layer (need a loop as there are different sized chunks)
        # collect attention output and attention weights for each call of attention
        after_attn_list = []
        self.attn_weights = []

        for chunk in chunks:
            after_attn_list.append(self.attention(chunk.view(1, -1, self.bert.config.hidden_size)))
            self.attn_weights.append(self.attention.attn_weights)

        output = torch.cat(after_attn_list)

        # fully connected layers
        for fc in self.fc:
            output = fc(output)

        return output