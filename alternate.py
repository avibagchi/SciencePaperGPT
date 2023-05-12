import os
import pandas as pd
from llama_index import GPTListIndex, SimpleDirectoryReader
import uuid
import prompts
from dotenv import load_dotenv
import openai
import re
import prompts2

# Delete what is in data/index_dir before running

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

directory_path = 'data/papers2'
papers = []
for filename in os.listdir(directory_path):
    papers.append(filename)


def index_name(f: str):
    f = f.split("/")[-1].split(".")[0]
    return {"file": f}


def index(doc_dir: str, index_dir: str):
    documents = SimpleDirectoryReader(doc_dir, file_metadata=index_name).load_data()
    for doc in documents:
        index = GPTListIndex(documents=[doc])
        # papers.append(doc.extra_info_str)
        # f = doc.extra_info["file"]
        f = str(uuid.uuid4())
        index.save_to_disk(f"{index_dir}/{f}")

def query(index: str, q: str):
    index = GPTListIndex.load_from_disk(index)
    result = index.query(q)
    return result


if __name__ == "__main__":
    doc_dir, index_dir = "data/papers2", "data/index_dir"
    index(doc_dir, index_dir)

    full_ab_arr = []
    dep_var_ab_arr = []
    dep_var_res_arr = []
    ind_var_ab_arr = []
    ind_var_res_arr = []
    sample_ab_arr = []
    sample_res_arr = []
    population_ab_arr = []
    population_res_arr = []
    exp_ab_arr = []
    exp_res_arr = []
    mech_ab_arr = []
    mech_res_arr = []
    temp_ab_arr = []
    temp_res_arr = []
    rec_ab_arr = []
    rec_res_arr = []

    for f in os.listdir(index_dir):
        out_ab = str(query(f"{index_dir}/{f}", prompts2.full_abstract_prompt))
        out_res = str(query(f"{index_dir}/{f}", prompts2.full_results_prompt))


        def getregex(s):
            return s + ". ([^" + str(int(s) + 1) + "]+)"

        m1 = re.search(getregex("1"), out_ab)
        m2a = re.search(getregex("2"), out_ab)
        m2b = re.search(getregex("2"), out_res)
        m3a = re.search(getregex("3"), out_ab)
        m3b = re.search(getregex("3"), out_res)
        m4a = re.search(getregex("4"), out_ab)
        m4b = re.search(getregex("4"), out_res)
        m5a = re.search(getregex("5"), out_ab)
        m5b = re.search(getregex("5"), out_res)
        m6a = re.search(getregex("6"), out_ab)
        m6b = re.search(getregex("6"), out_res)
        m7a = re.search(getregex("7"), out_ab)
        m7b = re.search(getregex("7"), out_res)
        m8a = re.search(getregex("8"), out_ab)
        m8b = re.search(getregex("8"), out_res)
        m9a = re.search(getregex("9"), out_ab)
        m9b = re.search(getregex("9"), out_res)

        full_ab_arr.append(m1.group(1)) if m1 is not None else None

        dep_var_ab_arr.append(m2a.group(1)) if m2a is not None else None
        dep_var_res_arr.append(m2b.group(1)) if m2b is not None else None

        ind_var_ab_arr.append(m3a.group(1)) if m3a is not None else None
        ind_var_res_arr.append(m3b.group(1)) if m3b is not None else None

        sample_ab_arr.append(m4a.group(1)) if m4a is not None else None
        sample_res_arr.append(m4b.group(1)) if m4b is not None else None

        population_ab_arr.append(m5a.group(1)) if m5a is not None else None
        population_res_arr.append(m5b.group(1)) if m5b is not None else None

        exp_ab_arr.append(m6a.group(1)) if m6a is not None else None
        exp_res_arr.append(m6b.group(1)) if m6b is not None else None

        mech_ab_arr.append(m7a.group(1)) if m7a is not None else None
        mech_res_arr.append(m7b.group(1)) if m7b is not None else None

        temp_ab_arr.append(m8a.group(1)) if m8a is not None else None
        temp_res_arr.append(m8b.group(1)) if m8b is not None else None

        rec_ab_arr.append(m9a.group(1)) if m9a is not None else None
        rec_res_arr.append(m9b.group(1)) if m9b is not None else None

    data = {'papers': papers, 'full_ab_arr': full_ab_arr,
            'dep_var_ab': dep_var_ab_arr, 'dep_var_res': dep_var_res_arr,
            'ind_var_ab': ind_var_ab_arr, 'ind_var_res': ind_var_res_arr,
            'sample_var_ab': sample_ab_arr, 'sample_var_res': sample_res_arr,
            'population_ab': population_ab_arr, 'population_res': population_res_arr,
            'exp_ab': exp_ab_arr, 'exp_res': exp_res_arr,
            'mech_ab': mech_ab_arr, 'mech_res': mech_res_arr,
            'temp_ab': temp_ab_arr, 'temp_res': temp_res_arr,
            'rec_ab': rec_ab_arr, 'rec_res': rec_res_arr}
    df = pd.DataFrame(data)
    df.to_csv("output21.csv")
