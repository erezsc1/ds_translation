import os
from translator import Translator

if __name__ == '__main__':
    os.chdir("..")
    query_1 = [
        "بالأمس ذهبت إلى محل بقالة",
        "أخبر ديفيد عمه أنه اشترى تذاكر بسعر مخفض للفيلم الجديد",
        "مرحبًا بكل أصدقاء باسل"
    ]

    query_2 = query_1.copy()
    query_2.reverse()


    src_lang = "arb"
    tgt_lang = "heb"

    translator = Translator(src_lang, tgt_lang)
    response_1 = translator.translate(query_1)

    response_2 = translator.translate(query_2)

    print(response_1 == response_2)