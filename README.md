
# MarianMT Service
Machine Translation RESTful API service. Based on Helsinki-NLP repository, and supports the following languages:
- Arabic -> English
- Arabic -> Hebrew
- English -> Hebrew
- English -> Arabic
- Hebrew -> Arabic

## Examples
### library calls
```python
from translator import Translator


if __name__ == '__main__':
    # heb -> arb
    translator = Translator("heb","arb")
    seq = ["שלום לכולם", "ארבעים ושתיים", "ארבעים וחמש"]

    result = translator.translate(seq)
```
### API calls
```python
import requests

URL = r"http://127.0.0.1:8000/translate_list/"

if __name__ == '__main__':
    query = [
        "שלום, אחמד",
        "מה שלומך היום?"
    ]
    src_lang = "heb"
    tgt_lang = "arb"
    request = {
        "source_lang": src_lang,
        "target_lang": tgt_lang,
        "data_list": query,
        "content-type":"application/json"
    }
    response = requests.get(URL, params=request).json()
```


## Docker
### build: 
```bash
sudo docker build . -t translation_image
```
### run:
with gpu:
```bash
sudo nvidia-docker run -p 80:80 translation_image
```
without gpu:
```bash
sudo docker run -p 80:80 translation_image
```
access via ```localhost:80/docs```

