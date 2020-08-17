import os
import json
import time
import torch
import pandas as pd
from transformers import AutoTokenizer, AutoModelWithLMHead


'''
    timing decorator
'''
def timing(f):
    def wrap(*args, **kwargs):
        time1 = time.time()
        ret = f(*args, **kwargs)
        time2 = time.time()
        print('{:s} function took {:.3f} ms'.format(f.__name__, (time2-time1)*1000.0))
        return ret
    return wrap


class Translator():
    def __init__(
            self,
            source_lang="heb",
            target_lang="arb",
            max_cache_entries=1024
    ):
        '''
        :param source_lang: source langauge to be translated from ["heb","arb","eng"]
        :param target_lang: target langauge to be translated to ["heb","arb", "eng"]
        :param max_cache_entries: maximum entrie
        '''
        with open("translator_config.json", "r") as fp:
            languages_dict = json.load(fp)
        try:
            pretrained_model = languages_dict[source_lang][target_lang]["model_name"]
            special_token = languages_dict[source_lang][target_lang]["special_tok"]
        except:
            raise NotImplementedError
        self.source_lang = source_lang
        self.target_lang = target_lang

        trained_model_path = os.path.join("trained_models", pretrained_model)
        self.special_tok = special_token

        # loading models
        self.tokenizer = AutoTokenizer.from_pretrained(trained_model_path)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(self.device)
        self.model : AutoModelWithLMHead = AutoModelWithLMHead.from_pretrained(trained_model_path)
        self.model.to(self.device)
        self.max_sequence_len = 512
        self.translation_df = pd.DataFrame(columns=["src_text", "tgt_text"])
        self.max_cache_entries = max_cache_entries

    def _get_special_token(self):
        return f">>{self.special_tok}<< ";

    def preprocess(self, x : str):
        '''
        this method can be overriden to define specific preprocess behavior
        :param x: string to be preprocessed for translation
        :return: processed string
        '''
        cur_x = self._get_special_token() + x
        return cur_x

    def to(self, device):
        '''
        mode model to device (cuda/cpu)
        :param device:
        :return:
        '''
        self.device = device
        self.model.to(device)

    def _check_cache(self, dataframe: pd.DataFrame):
        cached_indexes = self.translation_df["src_text"].isin(dataframe["src_text"])
        if cached_indexes.any():
            cached_dataframe = self.translation_df.loc[cached_indexes]
            indexes_to_check = ~dataframe["src_text"].isin(self.translation_df["src_text"])
            return indexes_to_check, cached_dataframe
        return None, None

    def _update_cache(self, dataframe: pd.DataFrame):
        dataframe = pd.concat([self.translation_df, dataframe]).drop_duplicates(keep=False)
        if len(dataframe) > 0:
            self.translation_df = self.translation_df.append(dataframe, ignore_index=True)[-self.max_cache_entries:]
            self.translation_df.drop_duplicates(inplace=True)

    @timing
    def translate(self, x):
        '''
        translation function
        :param x : [str, [str], pd.Series(str)] - data to be translated:
        :return: returns df : pd.DataFrame with original and translated text
        '''
        cur_translation_df = pd.DataFrame(columns=["src_text", "tgt_text"])
        if type(x) == list:
            cur_translation_df["src_text"] = x
        elif type(x) == str:
            cur_translation_df["src_text"] = [x]
        indexes_to_check, cached_dataframe = self._check_cache(cur_translation_df)
        if indexes_to_check is not None:
            cur_translation_df = cur_translation_df.loc[indexes_to_check]
        if len(cur_translation_df) > 0:
            processed_text_series = cur_translation_df["src_text"].apply(lambda x: self.preprocess(x))
            input_dict = self.tokenizer.prepare_translation_batch(processed_text_series.tolist())
            input_dict.to(self.device)
            translated = self.model.generate(**input_dict)
            tgt_text = [self.tokenizer.decode(t, skip_special_tokens=True) for t in translated]
            cur_translation_df["tgt_text"] = tgt_text
            self._update_cache(cur_translation_df)
            return cur_translation_df.append(cached_dataframe)
        return cached_dataframe




