import torch
from torch import nn, optim

from src.models.attention import Attention

# Transformers
from transformers import BertTokenizerFast, BertModel, AdamW, get_linear_schedule_with_warmup

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


class HIBERT(nn.Module):

    def __init__(self, 
                    PRE_TRAINED_MODEL_NAME, 
                    n_classes, 
                    add_linear=None, 
                    attn_bias=False, 
                    freeze_layer_count=-1):

        super(HIBERT, self).__init__()

        self.n_classes = n_classes
        self.add_linear = add_linear
        self.attn_bias = attn_bias
        self.freeze_layer_count = freeze_layer_count

        # Define model objects
        self.bert = BertModel.from_pretrained(PRE_TRAINED_MODEL_NAME, return_dict=False, add_pooling_layer=False)
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
            attention_mask = attention_mask
            )

        # group chunks together
        output = output.split_with_sizes(n_chunks)

        # loop through attention layer (need a loop as there are different sized chunks)
        output = torch.stack([
            self.attention(chunk.view(1, -1, self.bert.config.hidden_size)) 
            for chunk in output
            ])

        # fully connected layers
        for fc in self.fc:
            output = fc(output)

        return output