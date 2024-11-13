# RAG-Question-Answering-Pipeline

A Retrieval Augmented Generation Pipeline to generate responses for questions related to Indian Premier League (IPL) matches.
 
## This involves :
Download the HTML files from the URL's from espncricinfo.com
Processing the HTML files with BeautifulSoup library
Generating questions using a Language Model, 
Ingesting the data into ChromaDB, implementing RAG retrieval processes using LangChain, 
And evaluating the performance of naive and advanced RAG techniques.

## To run:

- Install requirements using pip install -r requirements.txt

- "data" folder is where we have the data of ipl matches

- Run the code using python query_data.py

To run this pipeline, you need to provide huggingface api token. Follow below site to get your api token.
https://huggingface.co/docs/hub/security-tokens

Here Mistral-7B-Instruct-v0.1 Large Language Model (LLM) is used for Generation part, follow the below link to know more about it. 

