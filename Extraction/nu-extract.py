#### Extraction with open source models...



import json
import torch

from transformers import AutoModelForCausalLM, AutoTokenizer


def predict_NuExtract(model, tokenizer, texts, template, 
                      batch_size=1, max_length=10000, max_new_tokens=4000):

    template = json.dumps(json.loads(template), indent=4)

    prompts = [f"""<|input|>\n### Template:\n{template}\n### Text:\n{text}\n\n<|output|>""" for text in texts]

    outputs = []

    with torch.no_grad():

        for i in range(0, len(prompts), batch_size):
            batch_prompts = prompts[i:i+batch_size]
            batch_encodings = tokenizer(batch_prompts, return_tensors="pt", truncation=True,
                                        padding=True, max_length=max_length).to(model.device)
            

            pred_ids = model.generate(**batch_encodings, max_new_tokens=max_new_tokens)
            outputs += tokenizer.batch_decode(pred_ids, skip_special_tokens=True)

    return [output.split("<|output|>")[1] for output in outputs]

model_name = "numind/NuExtract-v1.5"
device = "cpu"

model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16, trust_remote_code=True).to(device).eval()
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)



text = """We introduce Mistral 7B, a 7–billion-parameter language model engineered for
superior performance and efficiency. Mistral 7B outperforms the best open 13B
model (Llama 2) across all evaluated benchmarks, and the best released 34B
model (Llama 1) in reasoning, mathematics, and code generation. Our model
leverages grouped-query attention (GQA) for faster inference, coupled with sliding
window attention (SWA) to effectively handle sequences of arbitrary length with a
reduced inference cost. We also provide a model fine-tuned to follow instructions,
Mistral 7B – Instruct, that surpasses Llama 2 13B – chat model both on human and
automated benchmarks. Our models are released under the Apache 2.0 license.
Code: <https://github.com/mistralai/mistral-src>
Webpage: <https://mistral.ai/news/announcing-mistral-7b/>"""

template = """{
    "Model": {
        "Name": "",
        "Number of parameters": "",
        "Number of max token": "",
        "Architecture": []
    },
    "Usage": {
        "Use case": [],
        "Licence": ""
    }
}"""

prediction = predict_NuExtract(model, tokenizer, [text], template)[0]

print(dir(prediction))

print(prediction["Model"])
