from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
from flores200_codes import flores_codes

model_dict = {}


def load_models(model_name: str):
    # build model and tokenizer
    model_name_dict = {
        "nllb-1.3B": "facebook/nllb-200-1.3B",
        "nllb-distilled-1.3B": "facebook/nllb-200-distilled-1.3B",
        "nllb-3.3B": "facebook/nllb-200-3.3B",
    }[model_name]

    print("\tLoading model: %s" % model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name_dict)
    tokenizer = AutoTokenizer.from_pretrained(model_name_dict)
    model_dict[model_name + "_model"] = model
    model_dict[model_name + "_tokenizer"] = tokenizer

    return model_dict


def translation(model_name: str, source, target, text: str):

    model_dict = load_models(model_name)

    source = flores_codes[source]
    target = flores_codes[target]

    model = model_dict[model_name + "_model"]
    tokenizer = model_dict[model_name + "_tokenizer"]

    translator = pipeline(
        "translation",
        model=model,
        tokenizer=tokenizer,
        src_lang=source,
        tgt_lang=target,
    )
    output = translator(text, max_length=400)

    output = output[0]["translation_text"]
    result = {
        "source": source,
        "target": target,
        "result": output,
    }

    return result


NLLB_EXAMPLES = [
    ["nllb-distilled-1.3B", "English", "Shan", "Hello, how are you today?"],
    ["nllb-distilled-1.3B", "Shan", "English", "မႂ်ႇသုင်ၶႃႈ ယူႇလီယူႇၶႃႈၼေႃႈ"],
    [
        "nllb-distilled-1.3B",
        "English",
        "Shan",
        "Forming Myanmar’s New Political System Will Remain an Ideal but Never in Practicality",
    ],
]
