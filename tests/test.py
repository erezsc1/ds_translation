from translator import Translator


if __name__ == '__main__':
    # heb -> arb
    translator = Translator("heb","arb")
    seq_1 = ["שלום לכולם", "ארבעים ושתיים", "ארבעים וחמש"]
    seq_2 = ["ארבעים ושתיים"]

    print(translator.translate(seq_2))
    translator.to("cpu")
    print(translator.translate(seq_2))
    translator.to("cuda")
    print(translator.translate(seq_1))
    print(translator.translate(seq_2))
    print(translator.translate(seq_2))

    # eng -> heb
    translator = Translator("eng", "arb")
    translator.to("cpu")
    print(translator.translate(["hello, world"]))
    #print(translator.translate("hello my name is yossi"))
    print(translator.translate(["hello, my name is Yossi", "hello, my name is Avi"]))

    translator = Translator("eng", "heb")
    translator.to("cuda")
    print(translator.translate(["hello, my name is Yossi", "hello, my name is Avi"]))
    print(translator.translate(["hello, my name is Yossi"]))

    # arb -> eng
    translator = Translator(source_lang="arb", target_lang="eng")
    print(translator.translate(["مرحبا احمد"]))

    # arb -> heb
    translator = Translator(source_lang="arb", target_lang="heb")
    #print(translator.translate(["مرحبا احمد"]))
    print(translator.translate("بالأمس ذهبت إلى محل بقالة"))

    # eng -> arb
    translator = Translator(source_lang="eng", target_lang="arb")
    print(translator.translate(["hello, my name is yossi"]))