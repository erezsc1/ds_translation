import requests

URL = r"http://127.0.0.1:8000/translate_list/"

if __name__ == '__main__':
    query_1 = [
        "שלום, אחמד",
        "מה שלומך היום?"
    ]

    query_2 = [
        "שלום, אחמד",
        "שלום, סלים"
    ]

    src_lang = "heb"
    tgt_lang = "arb"


    request_1 = {
        "source_lang": src_lang,
        "target_lang": tgt_lang,
        "data_list": query_1,
        "content-type":"application/json"
    }

    request_2 = {
        "source_lang": src_lang,
        "target_lang": tgt_lang,
        "data_list": query_2,
        "content-type": "application/json"
    }

    response = requests.get(
        URL,
        params=request_1
    ).json()

    print(response)

    response = requests.get(
        URL,
        params=request_2
    ).json()

    print(response)
