import os
import pandas as pd
from pathlib import Path
from PyPDF2 import PdfReader
import openai

import prompts
from dotenv import load_dotenv

load_dotenv()

openai.api_key = os.getenv("OPENAI_API_KEY")

role = (
        "You are a helpful hiring assistant to analyze the text in science articles."
        " Your job is compare the content of the abstract with that of the results."
)


def getdocs(directory: str, one_page):
    files = Path(directory).glob('*')
    file_arr = []
    for file in files:
        reader = PdfReader(file)
        if not one_page:
            text = ""
            for page in reader.pages:
                text += page.extract_text()
            file_arr.append(text)
        else:
            file_arr.append(reader.pages[0].extract_text())

    return file_arr


def getRes(messages):
    res = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=messages
    )
    return res

def chatGPTResponse (prompt1):
    messages = [{"role": "system", "content": role},
                {"role": "user", "content": prompt1}]
    res = getRes(messages)
    return res["choices"][0]["message"]["content"]


first_page_arr = getdocs("data/papers1/", True)
abstract_arr = []
for txt in first_page_arr:
    abstract_arr.append(chatGPTResponse(prompts.abstract_prompt + txt))

document_arr = getdocs("data/papers1/", False)
results_arr = []
for txt in document_arr:
    results_arr.append(chatGPTResponse(prompts.results_prompt + txt))

dep_var_ab = []
for x in abstract_arr:
    dep_var_ab.append(chatGPTResponse(prompts.compare_prompt + x))

dep_var_res = []
for x in results_arr:
    dep_var_res.append(chatGPTResponse(prompts.compare_prompt + x))

data = {'Papers': first_page_arr, 'DV from Abstract': dep_var_ab, 'DV from Results': dep_var_res}
df = pd.DataFrame(data)
df.to_csv("output4.csv")

# abstract = res["choices"][0]["message"]["content"]
#     # print(abstract)
#     # import pdb; pdb.set_trace()
#     messages.append({"role": "assistant", "content": abstract})
#     messages.append({"role": "user", "content": prompt2})
#     res = getRes(messages)
