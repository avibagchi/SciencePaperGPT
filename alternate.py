import os
import pandas as pd
from llama_index import GPTListIndex, SimpleDirectoryReader
import uuid
import prompts
from dotenv import load_dotenv
import openai
import re
import prompts2

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
            return r"^" + s + "\.\s+(.*)"

        dep_var_ab_arr.append(re.search(getregex("2"), out_ab).group(1))
        dep_var_res_arr.append(re.search(getregex("2"), out_res).group(1))

        ind_var_ab_arr.append(re.search(getregex("3"), out_ab).group(1))
        ind_var_res_arr.append(re.search(getregex("3"), out_res).group(1))

        sample_ab_arr.append(re.search(getregex("4"), out_ab).group(1))
        sample_res_arr.append(re.search(getregex("4"), out_res).group(1))

        population_ab_arr.append(re.search(getregex("5"), out_ab).group(1))
        population_res_arr.append(re.search(getregex("5"), out_res).group(1))

        exp_ab_arr.append(re.search(getregex("6"), out_ab).group(1))
        exp_res_arr.append(re.search(getregex("6"), out_res).group(1))

        mech_ab_arr.append(re.search(getregex("7"), out_ab).group(1))
        mech_res_arr.append(re.search(getregex("7"), out_res).group(1))

        temp_ab_arr.append(re.search(getregex("8"), out_ab).group(1))
        temp_res_arr.append(re.search(getregex("8"), out_res).group(1))

        rec_ab_arr.append(re.search(getregex("9"), out_ab).group(1))
        rec_res_arr.append(re.search(getregex("9"), out_res).group(1))
        break


    data = {'papers': papers,
            'dep_var_ab': dep_var_ab_arr, 'dep_var_res': dep_var_res_arr,
            'ind_var_ab': ind_var_ab_arr, 'ind_var_res': ind_var_res_arr,
            'sample_var_ab': sample_ab_arr, 'sample_var_res': sample_res_arr,
            'population_ab': population_ab_arr, 'population_res': population_res_arr,
            'exp_ab': exp_ab_arr, 'exp_res': exp_res_arr,
            'mech_ab': mech_ab_arr, 'mech_res': mech_res_arr,
            'temp_ab': temp_ab_arr, 'temp_res': temp_res_arr,
            'rec_ab': rec_ab_arr, 'rec_res': rec_res_arr}
    df = pd.DataFrame(data)
    df.to_csv("output18.csv")
