from langchain.chains import RetrievalQA, ConversationalRetrievalChain,LLMChain
from langchain.llms import HuggingFaceHub
from langchain.prompts import PromptTemplate
from create_db import generate_data_store

vectordb = generate_data_store()
retriever = vectordb.as_retriever(search_type = "similarity", search_kwargs={"k": 5})

def main(query):
    llm = HuggingFaceHub(
        repo_id = "mistralai/Mistral-7B-Instruct-v0.1",
        model_kwargs={"temperature":0.2, "max_length":512},
        huggingfacehub_api_token = "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"
    )

    print('In Search Mode')
    rqa_prompt_template = "Who was the top scorer for Chennai Super Kings in the first match of the IPL 2024 season against Royal Challengers Bengaluru?"
                    
    RQA_PROMPT = PromptTemplate(
        template = rqa_prompt_template, input_variables = ["context","question"]
    )
    rqa_chain_type_kwargs = {"prompt": RQA_PROMPT}

    qa = RetrievalQA.from_chain_type(llm,
                                     chain_type="stuff",
                                     retriever = retriever,
                                     chain_type_kwargs=rqa_chain_type_kwargs,
                                     return_source_documents = True,
                                     verbose = False)
    result = qa({"query": query})
    return result

if __name__ == "__main__":
    query = input("Ask a question: ")
    print("looking for results")
    result = main(query)
    print(result['result'])
