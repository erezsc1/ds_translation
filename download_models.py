import os
import json
from transformers import AutoTokenizer, AutoModelWithLMHead

MODELS_BASE_PATH = "trained_models"
TRANSLATOR_CONFIG = "translator_config.json"


SPECIAL_CASE_TGT_MAP = {
    "he": "heb",
    "ara": "arb",
    "ar": "arb",
    "en": "eng",
    "fr" : "fra"
}

MODELS = [
    "Helsinki-NLP/opus-mt-ar-fr",
    "Helsinki-NLP/opus-mt-ara-fra",
    "Helsinki-NLP/opus-mt-ar-de",
    "Helsinki-NLP/opus-mt-ar-en",
    "Helsinki-NLP/opus-mt-en-ar",

    # heb -> *
    "Helsinki-NLP/opus-mt-heb-ara",
    "Helsinki-NLP/opus-mt-heb-epo",
    "Helsinki-NLP/opus-mt-heb-ukr",
    "Helsinki-NLP/opus-mt-he-de",
    "Helsinki-NLP/opus-mt-he-fi",
    "Helsinki-NLP/opus-mt-he-sv",
    
    # * -> heb
    "Helsinki-NLP/opus-mt-en-he",
    "Helsinki-NLP/opus-mt-de-he",
    "Helsinki-NLP/opus-mt-es-he",
    "Helsinki-NLP/opus-mt-fr-he",
    "Helsinki-NLP/opus-mt-fi-he",
    "Helsinki-NLP/opus-mt-sv-he",
    "Helsinki-NLP/opus-mt-epo-heb",
    "Helsinki-NLP/opus-mt-jpn-heb",
    "Helsinki-NLP/opus-mt-ukr-heb",

    # * -> ara
    "Helsinki-NLP/opus-mt-fra-ara",
    "Helsinki-NLP/opus-mt-ita-ara",
    "Helsinki-NLP/opus-mt-jpn-ara",
    "Helsinki-NLP/opus-mt-tur-ara",
    "Helsinki-NLP/opus-mt-spa-ara",
    "Helsinki-NLP/opus-mt-ell-ara",
    # ara -> *
    "Helsinki-NLP/opus-mt-rus-ara",
    "Helsinki-NLP/opus-mt-ara-rus",
    "Helsinki-NLP/opus-mt-ara-spa",
    "Helsinki-NLP/opus-mt-ara-ell",
    "Helsinki-NLP/opus-mt-ara-epo",
    "Helsinki-NLP/opus-mt-ara-ita",
    "Helsinki-NLP/opus-mt-ara-tur",
    "Helsinki-NLP/opus-mt-ara-pol",
    "Helsinki-NLP/opus-mt-ara-heb",
]

if __name__ == '__main__':
    models_count = 0

    UNIQUE_MODELS = list(set(MODELS))

    with open(TRANSLATOR_CONFIG, "r") as fp:
        config = json.load(fp)

        for model_name in UNIQUE_MODELS:
            hf_model_name = model_name
            model_name = hf_model_name.replace("/","_").replace("-","_")
            print(hf_model_name)
            if os.path.exists(MODELS_BASE_PATH):
                MODEL_PATH = os.path.join(MODELS_BASE_PATH, model_name)
                if not os.path.exists(MODEL_PATH):
                    try:
                        tokenizer = AutoTokenizer.from_pretrained(hf_model_name)
                        model = AutoModelWithLMHead.from_pretrained(hf_model_name)
                        print(model_name.split("_"))
                        src_tag = model_name.split("_")[-2]
                        tgt_tag = model_name.split("_")[-1]
                        # saving models
                        tokenizer.save_pretrained(MODEL_PATH)
                        model.save_pretrained(MODEL_PATH)
                        models_count += 1

                        # TODO update config json
                        temp_dict = {
                            "model_name": model_name,
                            "special_tok": tgt_tag
                        }
                        try:
                            mod_src_tag = SPECIAL_CASE_TGT_MAP.get(src_tag, src_tag)
                            mod_tgt_tag = SPECIAL_CASE_TGT_MAP.get(tgt_tag, tgt_tag)
                            config[mod_src_tag][mod_tgt_tag] = temp_dict
                        except Exception as ex:
                            config[mod_src_tag] = {}
                            config[mod_src_tag][mod_tgt_tag] = temp_dict
                    except Exception as ex:
                        print(f"model {hf_model_name} does not exist in huggingface.co/models")
                else:
                    print(f"model {model_name} already exists.")
        with open(TRANSLATOR_CONFIG, "w") as fp:
            json.dump(config, fp, indent=4, sort_keys=True)
        assert models_count == len(MODELS)



