import pprint
from translator import Translator


class AugmenText:
    def __init__(self, src_lang, target_langs, special_tokens=None):
        self.src_lang = src_lang
        self.target_langs = target_langs
        self.special_tokens = special_tokens

    def augment(self, query):
        augmentations = {}
        for lang in self.target_langs:
            try:
                translator_front = Translator(source_lang=self.src_lang, target_lang=lang, special_tokens=self.special_tokens)
            except NotImplementedError:
                print(f"translation {self.src_lang} --> {lang} does not exist.")
                continue
            try:
                translator_back = Translator(source_lang=lang, target_lang=self.src_lang, special_tokens=self.special_tokens)
            except NotImplementedError:
                print(f"translation {lang} --> {self.src_lang} does not exist.")
                continue
            trans_text = translator_front.translate(query)
            aug_text = translator_back.translate(trans_text["tgt_text"][0])
            augmentations[lang] = aug_text["tgt_text"][0]
            del translator_front
            del translator_back
        return augmentations


if __name__ == '__main__':
    #query = "בוקר טוב לכולם, היום נתחיל את [TIME_E] התרגול הראשון [TIME_S] במבוא לאלגברה לינארית"
    #query = "בוקר טוב לכולם. היום נתחיל את התרגול הראשון במבוא לאלגברה לינארית"
    #query = "אתמול בבוקר יוסי ומירב הלכו לבית הספר. שם הם פגשו את חבריהם לאחר שלא ראו אותם הרבה זמן עקב הקורונה"
    query = "חנות המכולת של סלים"

    special_tokens = ["[TIME_S]", "[TIME_E]"]

    target_langs = [
        "fi",
        "de",
        "arb",
        "eng",
        "es",
        "sv"
    ]

    ta = AugmenText("heb", target_langs, special_tokens=None)

    augmentations_heb = ta.augment(query)
    #----------------------------------------------------#
    query = "ذهب يوسي وميراف صباح أمس إلى المدرسة. هناك التقوا بأصدقائهم بعد أن لم يروهم لفترة طويلة بسبب كورونا"
    src_lang = "arb"

    target_langs = [
        "heb",
        "eng",
        "rus",
        "spa",
        "tur",
    ]

    ta = AugmenText("arb", target_langs)
    augmentations_arb = ta.augment(query)

    print("Augmentations")
    print("#-------- HEB --------#")
    pprint.pprint(augmentations_heb)
    print("#-------- ARB --------#")
    pprint.pprint(augmentations_arb)
