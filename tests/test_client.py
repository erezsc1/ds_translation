import requests

URL = r"http://127.0.0.1:8000/translate_list/"

if __name__ == '__main__':
    query = [
        "שלום, אחמד",
        "מה שלומך היום?"
    ]

    src_lang = "heb"
    tgt_lang = "arb"

    print(query)

    request = {
        "source_lang": src_lang,
        "target_lang": tgt_lang,
        "data_list": query,
        "content-type": "application/json"
    }
    response = requests.get(
        URL,
        params=request
    ).json()

    print(response)
