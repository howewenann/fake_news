import pandas as pd
import numpy as np
import matplotlib.cm as cm
import matplotlib
from matplotlib.colors import LinearSegmentedColormap

from sklearn.preprocessing import StandardScaler
from transformers import BertTokenizerFast

from tqdm import tqdm
tqdm.pandas()

import ast

import logging
log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
logging.basicConfig(level=logging.INFO, format=log_fmt)
logger = logging.getLogger("AttentionVisualizer")

class AttentionVisualizer():

    def __init__(self, model_dir):
        self.model_dir = model_dir

        # tokenizer
        self.tokenizer = BertTokenizerFast.from_pretrained(self.model_dir)

    # combine word tokens
    def combine_word_tokens(self, tokens_list, attn_list, agg_fn):

        df_html = pd.DataFrame({'tokens':tokens_list, 'attention':attn_list})

        # filter to remove special tokens
        special_tokens = ['[CLS]', '[PAD]', '[SEP]']
        df_html = df_html.loc[~df_html['tokens'].isin(special_tokens), :].reset_index(drop=True)
        df_html['word_grp'] = 0

        group = 0
        for i in range(len(df_html['tokens'])):
            word = df_html['tokens'].tolist()[i]
            
            # if word is not a word piece, count as word
            if '##' not in word:
                group = group + 1
            
            df_html.loc[i, 'word_grp'] = group

        combined_tokens = df_html.groupby('word_grp')['tokens'].apply(lambda x: ' '.join(x))
        combined_attn = df_html.groupby('word_grp')['attention'].agg(agg_fn)

        # rescale attention
        combined_attn = combined_attn / combined_attn.sum()

        # clean up tokens
        combined_tokens = combined_tokens.str.replace(' ##', '')

        # output lists
        combined_tokens = combined_tokens.tolist()
        combined_attn = combined_attn.tolist()

        return combined_tokens, combined_attn


    # Create html visualization of attention weights
    def create_html(self, tokens_list, attn_list, clip_neg=True):

        # create custom colour map
        cmap = LinearSegmentedColormap.from_list('rg', ['r', 'w', 'g'], N=256)

        df_html = pd.DataFrame({'tokens':tokens_list, 'attention':attn_list})

        # filter to remove special tokens
        special_tokens = ['[CLS]', '[PAD]', '[SEP]']
        df_html = df_html.loc[~df_html['tokens'].isin(special_tokens), :].reset_index(drop=True)

        # Rescale attention weights
        df_html['attention'] = df_html['attention'] / df_html['attention'].sum()

        # create colour map
        norm = matplotlib.colors.TwoSlopeNorm(vmin=-3, vcenter=0, vmax=3)
        m = cm.ScalarMappable(norm=norm, cmap=cmap)

        # standardize attention weights
        norm_attn = pd.Series(
            StandardScaler().fit_transform(df_html['attention'].values.reshape(-1,1)).flatten()
        )

        # clip outliers
        norm_attn[norm_attn > 3] = 3
        norm_attn[norm_attn < -3] = -3

        # clip norm weights <= 0
        if clip_neg:
            norm_attn[norm_attn <= 0] = 0

        # get colours
        df_html['colour'] = norm_attn.apply(lambda x: matplotlib.colors.to_hex(m.to_rgba(x)))

        html_text = '<span style="background-color:' + df_html['colour'] + ';">' \
                    + df_html['tokens'] + '</span>'

        html_text = ' '.join(html_text)

        return html_text


    def visualize_attention(self, df, agg_fn='max'):
        df = df.copy()

        # convert strings to back to lists
        if isinstance(df['input_ids'].values[0], str):
            logger.info('Convert input_ids strings back to lists...')
            df['input_ids'] = df['input_ids'].progress_apply(ast.literal_eval)

        if isinstance(df['attn_wts'].values[0], str):
            logger.info('Convert attn_wts strings back to lists...')
            df['attn_wts'] = df['attn_wts'].progress_apply(ast.literal_eval)
        
        # get tokens from ids
        logger.info('Convert ids to tokens...')
        df['tokens'] = df['input_ids'].progress_apply(self.tokenizer.convert_ids_to_tokens)

        # combine tokens and attention weights from word pieces
        logger.info('Combine word pieces and attention weights...')
        new_tokens_attn = \
        df.progress_apply(lambda x: self.combine_word_tokens(x['tokens'], x['attn_wts'], agg_fn=agg_fn), axis=1)

        # convert to structured form
        new_tokens_attn = new_tokens_attn.apply(pd.Series)

        # concat with original dataframe
        df['new_tokens'] = new_tokens_attn.iloc[:,0]
        df['new_attn_wts'] = new_tokens_attn.iloc[:,1]

        # create html strings
        logger.info('Generate HTML text field...')
        df['html'] = \
        df.progress_apply(lambda x: self.create_html(x['new_tokens'], x['new_attn_wts']), axis=1)

        return df