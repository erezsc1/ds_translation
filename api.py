import json
import torch
import pandas as pd
from typing import List
from translator import Translator
from fastapi import FastAPI, Query


app = FastAPI()

running_models = {}


@app.get("/")
def get_available_models():
    with open("translator_config.json") as fp:
        model_list = []
        config = json.load(fp)
        for src_lang in config.keys():
            for dest_lang in config[src_lang].keys():
                model_list.append(config[src_lang][dest_lang]["model_name"])
        return {"models": model_list}


@app.get("/translate")
def translate(source_lang, target_lang, data):
    '''
    :param source_lang: [arb,eng,heb]
    :param target_lange: [arb,eng,heb]
    :param data:
    :return: str
    '''
    model_index = f"{source_lang}_{target_lang}"
    if model_index not in running_models.keys():
        # need to init model
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        running_models[model_index] = Translator(
            source_lang=source_lang,
            target_lang=target_lang,
        )
        running_models[model_index].to(device)
    print(data)
    translation_df = running_models[model_index].translate(data)
    print(translation_df)
    if len(translation_df) == 1:
        return translation_df["tgt_text"].loc[0]
    return translation_df.to_dict()

@app.get("/translate_list")
def translate_list(source_lang, target_lang, data_list : List[str] = Query(None)):
    '''
    :param source_lang: source language [arb, eng, heb]
    :param target_lang: target language [arb, eng, heb]
    :param data_list: list[str]
    :return: list[str]
    '''
    model_index = f"{source_lang}_{target_lang}"
    if model_index not in running_models.keys():
        # need to init model
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        running_models[model_index] = Translator(
            source_lang=source_lang,
            target_lang=target_lang,
        )
        running_models[model_index].to(device)
    translation_df = running_models[model_index].translate(data_list)
    print(translation_df)
    return translation_df["tgt_text"].tolist()




@app.post("/free_model")
def free_model(source_lang, target_lang):
    model_index = f"{source_lang}_{target_lang}"
    print(model_index)
    print(running_models.keys())
    if model_index in running_models.keys():
        running_models[model_index].to("cpu")
        del running_models[model_index]
        torch.cuda.empty_cache()

@app.post("/empty_cache")
def empty_cache():
    with open("translator_config.json") as fp:
        model_list = []
        config = json.load(fp)
        for src_lang in config.keys():
            for dest_lang in config[src_lang].keys():
                model_index = f"{src_lang}_{dest_lang}"
                if model_index in running_models.keys():
                    running_models[model_index].translation_df = pd.DataFrame(columns=["src_text", "tgt_text"])
    return True