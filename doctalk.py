import pickle
import os
import glob
import getopt
import sys
from langchain.text_splitter import MarkdownHeaderTextSplitter
from langchain import LlamaCpp
from langchain.chains import LLMChain
from langchain.embeddings import LlamaCppEmbeddings
from sklearn.metrics.pairwise import cosine_similarity
from langchain import PromptTemplate
import re
import rerank


def display_help():
    """ """

    help = """ In Progress """
    print(help)


def load_data(data_folder, vector_file, model_path):
    """Parse md file, chunk content and perform content embeddings"""

    # init splitter
    headers_to_split_on = [
        ("#", "Header 1"),
        ("##", "Header 2"),
        ("###", "Header 3"),
    ]
    markdown_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=headers_to_split_on
    )

    # loop over folder and load .md files as doc
    text_list = []
    doc_list = []
    for doc in glob.glob(f"{data_folder}/*.md"):

        log_data = open("bot.log", "a")
        log_data.write(f"{doc} detected in {data_folder}\n")
        log_data.close()

        # get MD content
        with open(doc, "r") as file:
            md_text = file.read()

        # MD splits
        markdown_splitter = MarkdownHeaderTextSplitter(
            headers_to_split_on=headers_to_split_on
        )
        md_header_splits = markdown_splitter.split_text(md_text)

        # Char-level splits
        from langchain.text_splitter import RecursiveCharacterTextSplitter

        chunk_size = 250
        chunk_overlap = 30
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size, chunk_overlap=chunk_overlap
        )

        # Split
        texts = text_splitter.split_documents(md_header_splits)

        # convert langchain doc to str
        for i in range(len(texts)):

            # clean text before adding to vectorisation list
            t = re.sub(r"\{\{.+\}\}", "", texts[i].page_content)
            t = re.sub(r" \n", "", t)

            # add to vectorisation list
            text_list.append(t)
            doc_list.append(texts[i])

    # embed list of texts
    text_to_vector = {}
    embeddings = LlamaCppEmbeddings(model_path=model_path)
    embedded_texts = embeddings.embed_documents(text_list)
    for i in range(len(text_list)):
        text_to_vector[text_list[i]] = embedded_texts[i]

    log_data = open("bot.log", "a")
    log_data.write(f"{len(text_to_vector)} element vectorized\n")
    log_data.write(f"{len(doc_list)} document found\n")
    log_data.close()

    # save results
    with open(vector_file, "wb") as file:
        pickle.dump(text_to_vector, file)

    # return text to vector
    return text_to_vector


def pick_context(text_to_vectors, query, model_path, nb_elt):
    """Compute proximity betwen query and vectorised documents to define a context for question answering"""

    # test if text_to_vectors is a file
    if os.path.isfile(text_to_vectors):
        with open(text_to_vectors, "rb") as file:
            text_to_vectors = pickle.load(file)

    # vectorize query
    embeddings = LlamaCppEmbeddings(model_path=model_path)
    embedded_query = embeddings.embed_query(query)

    # compute similarity
    text_to_proximity = {}
    for text in text_to_vectors:
        vector = text_to_vectors[text]
        proximity = cosine_similarity([embedded_query], [vector])[0][0]
        text_to_proximity[text] = proximity

    # sort similarity
    n = nb_elt
    text_to_proximity = dict(
        sorted(text_to_proximity.items(), key=lambda item: item[1])
    )
    last_n_elements = list(text_to_proximity.items())[-n:]
    context_list = list(dict(last_n_elements).keys())
    context_list.reverse()

    # save context
    context = ". ".join(context_list)
    log_data = open("bot.log", "a")
    log_data.write(f"pick context : {context}\n")
    log_data.close()

    # return context
    return context_list


def rerank_context(context_list, query, nb_to_keep):
    """ """

    # rerank contexts
    context_list = rerank.bert_rerank(context_list, query, nb_to_keep)

    # craft context for prompt
    context = ". ".join(context_list)

    # save log
    log_data = open("bot.log", "a")
    log_data.write(f"save context : {context}\n")
    log_data.close()

    # return context
    return context


def get_answer(question, context, model_path):
    """ask question to model given a specific context"""

    # import LLM
    llm = LlamaCpp(model_path=model_path)

    # Define a template
    template = """Instruction: Utilise uniquement les éléments de contexte suivant pour répondre à la question avec une phrase courte.
    Si tu ne connais pas la réponse, dis que tu ne sais pas, n'esssaye surtout pas d'inventer une réponse.
    {context}
    Question: {question}
    Réponse:"""

    # Create prompt from template
    prompt = PromptTemplate.from_template(template)

    # Define chain
    llm_chain = LLMChain(prompt=prompt, llm=llm)

    # Run the Chain
    answer = llm_chain.run({"context": context, "question": question})

    # return answer
    return answer


def run(vector_file, doc_repo, question, model_path):
    """main function"""

    # load data
    if not os.path.isfile(vector_file):
        load_data(doc_repo, vector_file, model_path)
        log_data = open("bot.log", "a")
        log_data.write("data vectorized\n")
        log_data.close()

    # pick context
    nb_context_to_scan = 10
    context_list = pick_context(vector_file, question, model_path, nb_context_to_scan)

    # rerank context
    nb_context_to_keep = 2
    context = rerank_context(context_list, question, nb_context_to_keep)

    # answer question
    answer = get_answer(question, context, model_path)

    print("#" * 42)
    print(f"[QUERY] => {question}")
    print("=" * 42)
    print("[CONTEXT]" + "-" * 35)
    print(context)
    print("=" * 42)
    print(f"[ANSWER] => {answer}")

    # return answer
    return (answer, context)


def run_demo():
    """Run sloubinator example"""

    # parameters
    model_path = "/home/bran/Workspace/misc/llama/models/llama-2-13b-chat.Q5_K_M.gguf"
    vector_file = "vectors/llama-2-13b-chat.Q5_doc.pkl"
    question = "Le sloubinator est il dangereux pour les ordinateurs quantiques ?"

    # init log file
    log_data = open("bot.log", "w")
    log_data.close()

    # run
    run(vector_file, doc_repo, question, model_path)


def test_all_models(query):
    """ """

    # parameters
    model_list = [
        "/home/bran/Workspace/misc/llama/models/mistral-7b-v0.1.Q5_K_M.gguf",
        "/home/bran/Workspace/misc/llama/models/llama-2-7b-chat.Q6_K.gguf",
        "/home/bran/Workspace/misc/llama/models/llama-2-13b-chat.Q5_K_M.gguf",
        "/home/bran/Workspace/misc/llama/models/llama-2-70b-chat.Q5_K_M.gguf",
    ]

    # loop over model
    model_to_answer = {}
    model_to_context = {}
    for llm_model in model_list:

        # get vector folder
        vector_folder = llm_model.split("/")[-1].split("b-")[0] + "b"
        vector_folder = f"vectors/{vector_folder}"

        # run llm
        ctx = context.pick_context(vector_folder, query, llm_model)
        answer = get_answer(query, ctx, llm_model)
        model_to_answer[llm_model] = answer
        model_to_context[llm_model] = ctx

    # display results
    print(f"[*][USER INPUT] {query}")
    print("##############")
    for model in model_list:
        print(f"[+][MODEL] {model}")
        print(f"\t -> CONTEXT :\n{model_to_context[model]}\n")
        print(f"\t -> ANSWER :\n{model_to_answer[model]}\n")
        print("-" * 45)


if __name__ == "__main__":

    # parameters
    doc_repo = "docs"
    model_path = "/home/bran/Workspace/misc/llama/models/llama-2-13b-chat.Q5_K_M.gguf"
    vector_file = "vectors/llama-2-13b-chat.Q5_doc.pkl"
    question = "Un sloubinator est il dangereux pour les ordinateur quantique ?"
    reload = False

    # init log file
    log_data = open("bot.log", "w")
    log_data.close()

    # catch arguments
    argv = sys.argv[1:]
    if argv[0] == "demo":
        run_demo()
        sys.exit()
    try:
        opts, args = getopt.getopt(argv, "hm:q:r:", ["model=", "query=", "reload="])
    except getopt.GetoptError:
        display_help()
        sys.exit(2)
    for opt, arg in opts:
        if opt in ("-h", "--help"):
            display_help()
            sys.exit()
        elif opt in ("-m", "--model"):
            model_path = arg
        elif opt in ("-q", "--query"):
            question = arg
        elif opt in ("-r", "--reload"):
            reload = arg

    # load vector if reload is set to True or vector file does not exist
    vector_file = f"vectors/{model_path.split('/')[-1].replace('.gguf', '.pkl')}"
    if reload or not os.path.isfile(vector_file):
        load_data(doc_repo, vector_file, model_path)

    # run
    a = run(vector_file, doc_repo, question, model_path)
