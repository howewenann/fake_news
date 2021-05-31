from torch.utils.data import Dataset, DataLoader
import torch
import numpy as np

# Create a pytorch dataset object
class CreateDataset(Dataset):

    '''
    A custom Dataset class must implement three functions: __init__, __len__, and __getitem__.
        The __init__ function is run once when instantiating the Dataset object.
        The __len__ function returns the number of samples in our dataset.
        The __getitem__ function loads and returns a sample from the dataset at the given index idx. Returns a python dict
    '''

    def __init__(self, text, targets, tokenizer, max_len, chunksize, is_unseen):
        self.text = text
        self.targets = targets
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.chunksize = chunksize
        self.is_unseen = is_unseen

    def __len__(self):
        return len(self.text)

    def __getitem__(self, idx):
        # get a single item based on index and return a dict
        text = str(self.text[idx])
        target = self.targets[idx]

        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens = False,
            max_length = self.max_len,
            return_token_type_ids = False,
            padding = False,
            truncation = True,
            return_attention_mask = True,
            return_tensors = 'pt')

        input_id_chunks = list(encoding['input_ids'][0].split(self.chunksize - 2))
        mask_chunks = list(encoding['attention_mask'][0].split(self.chunksize - 2))

        # loop through each chunk
        for i in range(len(input_id_chunks)):
            # add CLS and SEP tokens to input IDs
            input_id_chunks[i] = torch.cat([
                torch.tensor([101]), input_id_chunks[i], torch.tensor([102])
            ])
            # add attention tokens to attention mask
            mask_chunks[i] = torch.cat([
                torch.tensor([1]), mask_chunks[i], torch.tensor([1])
            ])
            # get required padding length
            pad_len = self.chunksize - input_id_chunks[i].shape[0]
            # check if tensor length satisfies required chunk size
            if pad_len > 0:
                # if padding length is more than 0, we must add padding
                input_id_chunks[i] = torch.cat([
                    input_id_chunks[i], torch.Tensor([0] * pad_len)
                ])
                mask_chunks[i] = torch.cat([
                    mask_chunks[i], torch.Tensor([0] * pad_len)
                ])
        
        # Stack output
        input_ids = torch.stack(input_id_chunks)
        attention_mask = torch.stack(mask_chunks)

        # in case targets dont exist when predicting on unseen data
        if self.is_unseen:
            target_ = torch.tensor([target])
        else:
            target_ = torch.tensor([target], dtype=torch.long)

        out_dict = {
            'text': text,
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'n_chunks': torch.tensor([len(input_id_chunks)], dtype=torch.long),
            'target': target_
        }

        return out_dict


def custom_collate_fn(batch):
    # return batches
    batched = {}

    for item in batch:
        for key, value in item.items():
            if key not in batched:
                batched[key] = []
            batched[key].append(value)

    # concat list of tensors (if all elements in the list are tensors)
    for key in batched.keys():
        if all(isinstance(item, torch.Tensor) for item in batched[key]):
            batched[key] = torch.cat(batched[key])

    return batched


# Create dataloader
def create_data_loader(df, tokenizer, max_len, batch_size, chunksize = 512, is_unseen=False, sampler = None, shuffle = False, drop_last = False):

    # create dataset object
    ds = CreateDataset(
        text = df['content'].to_numpy(), 
        targets = df['target'].to_numpy(), 
        tokenizer = tokenizer, 
        max_len = max_len,
        chunksize = chunksize,
        is_unseen = is_unseen
        )

    return DataLoader(
        dataset=ds, 
        batch_size=batch_size, 
        sampler=sampler, 
        shuffle=shuffle, 
        drop_last=drop_last, 
        collate_fn=custom_collate_fn)