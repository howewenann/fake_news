from torch.utils.data import Dataset, DataLoader
import torch

# Create a pytorch dataset object

class CreateDataset(Dataset):

    '''
    A custom Dataset class must implement three functions: __init__, __len__, and __getitem__.
        The __init__ function is run once when instantiating the Dataset object.
        The __len__ function returns the number of samples in our dataset.
        The __getitem__ function loads and returns a sample from the dataset at the given index idx. Returns a python dict
    '''

    def __init__(self, text, targets, tokenizer, max_len):
        self.text = text
        self.targets = targets
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.text)

    def __getitem__(self, idx):
        # get a single item based on index and return a dict
        text = str(self.text[idx])
        target = self.targets[idx]

        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens = True,
            max_length = self.max_len,
            return_token_type_ids = False,
            padding = 'max_length',
            truncation = True,
            return_attention_mask = True,
            return_tensors = 'pt')

        out_dict = {
            'text': text,
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'target': torch.tensor(target, dtype=torch.long)
        }

        return out_dict


# Create dataloader
def create_data_loader(df, tokenizer, max_len, batch_size, sampler = None, shuffle = False, drop_last = False):

    # create dataset object
    ds = CreateDataset(
        text = df['content'].to_numpy(), 
        target = df['target'].to_numpy(), 
        tokenizer = tokenizer, 
        max_len = max_len
        )

    return DataLoader(dataset=ds, batch_size=batch_size, sampler=sampler, shuffle=shuffle, drop_last=drop_last)