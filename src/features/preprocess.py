# Contains preprocessor class
# useful if there is some parameters learned during training

from transformers import BertTokenizerFast

from tqdm import tqdm
tqdm.pandas()

from sklearn.model_selection import train_test_split

class DataCleaner():

    def __init__(self, model_dir, random_state=None):
        self.model_dir = model_dir
        self.random_state = random_state
        self.tokenizer = BertTokenizerFast.from_pretrained(self.model_dir)

    def clean_data(self, df, min_tokens=None, max_tokens=None, subsample=None):
        
        df = df.copy()

        # rename columns
        df = df.rename(columns = {'review': 'content'})

        # create target variable
        df['target'] = (df['sentiment'] == 'positive').astype(int)

        # count number of tokens        
        df['tokens'] = df['content'].progress_apply(self.tokenizer.tokenize)
        df['n_tokens'] = df['tokens'].apply(len)

        if min_tokens is not None:
            df = df.loc[df['n_tokens'] >= min_tokens, :]

        if max_tokens is not None:
            df = df.loc[df['n_tokens'] <= max_tokens, :]

        # drop columns which are note needed
        df = df.drop(['tokens', 'n_tokens'], axis=1)

        # subsample if needed
        if subsample is not None:
            df, _ = train_test_split(
                df, 
                train_size=subsample, 
                random_state=self.random_state, 
                shuffle=True, 
                stratify=df['target'])

        return df


class Preprocessor():
    
    def __init__(self):
        pass