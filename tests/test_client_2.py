import os
from translator import Translator

if __name__ == '__main__':
    os.chdir("..")
    # query = [
    #     "بالأمس ذهبت إلى محل بقالة",
    #     "أخبر ديفيد عمه أنه اشترى تذاكر بسعر مخفض للفيلم الجديد",
    #     "مرحبًا بكل أصدقاء باسل"
    # ]

    query = "Paz y buenas noches a todos los habitantes de Israel."

    src_lang = "es"
    tgt_lang = "heb"

    translator = Translator(src_lang, tgt_lang)
    response = translator.translate(query)

    print(response)