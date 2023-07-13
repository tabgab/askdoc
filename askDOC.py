#!pip install llama-index
#!pip install openai
#!pip install pypdf
#!pip install langchain
#!pip install faiss-cpu
#!pip install tiktoken (MAC-re kellett terminalban panaszkodott miatta)
#!pip install colorama


import argparse
import os
import sys
import subprocess
import pkg_resources
from colorama import Fore



# Provide the path to the folder containing the PDF documents
folder_path = "./"


#Store what OS we are running on. This will be important when composing links to PDF pages.
os_name = sys.platform

# Conditional branch based on the operating system
if os_name.startswith('linux'):
    # Linux-specific code
    print("Running on Linux")
    # Add your Linux-specific code here

elif os_name.startswith('darwin'):
    # macOS-specific code
    print("Running on macOS")
    # Add your macOS-specific code here

elif os_name.startswith('win'):
    # Windows-specific code
    print("Running on Windows")
    # Add your Windows-specific code here

else:
    # Code for other operating systems
    print("Running on an unrecognized operating system")
    # Add code for other operating systems here


# Create an ArgumentParser object
parser = argparse.ArgumentParser(prog='askDOC',
    description="""askDOC is an assistant style helper for an OMNEST/OMNeT++ user to find answers
    to their product-related questions from the documentation.

    This is not meant to replace the documentation itself, but it may help you to quickly find hints
    or answers to questions you might have about the product. 

    This application has limited knowledge, and may occasionally give incorrect or misleading answers.
    Use common sense, verify the answers and check the documentation for a better understanding. 

    You use this tool entirely at your own discression and responsibility, no guarantees, implied or
    otherwise are provided.

    askDoc works from a local indexed and embedded database made from the documentation.
    It uses OPENAI to give nicely worded answers to your questions and understand the fully formed 
    questions you send in.
\n
    ---:» You have to specify an OpenAI API key for this tool to work. «:---\n
\n
    By default, askDOC will assume that the first command line argument is going to be your question,
    and assume that it will find the documents at the $DOCUMENTSPATH location you specified in the
    configuration.

    For example:
    > askDOC "How can I colorize an icon?"

    You can also use the -q argument to prefix your question, and the -pdfp argument to tell
    askDOC to look for the documentation in a different folder.

For example:
    > python askDOC -q "How can I colorize an icon?" -pdfp "../omnetpp-6.0.1/doc/"

    Note, the askDOC will not regenerate the enbedding database for the application, unless explicitly
    instructed to do so, to ensure resources are not wasted, so you may get an error if such a database
    cannot be found at the specified folder.

    In this case, you must use the genDCO tool to generate this database before queries can be made.

    """,
    epilog=''
)

# Add an argument for the question
parser.add_argument('question', type=str, help='The question to process')

# Add an optional argument for the path
parser.add_argument('-pdfp', default='./', help = "The path of the directory containing your PDF documents to search", required=False)

# Parse the command-line arguments
args = parser.parse_args()

# Access the question argument
question = args.question

# Access the optional path argument
pdf_path = args.pdfp


#Check if there is an OpenAI key available. (There should be an openai_key.txt file in the same directory as searchPDF.py)
if os.path.exists("openai_key.txt"):
    print("OpenAI key found, proceeding.")
else:
    print(Fore.RED+"This program cannot function without an OpenAI key. There must be an"+Fore.CYAN+" openai_key.txt"+Fore.RED+
    " file in the same directory as searchPDF, and it must contain your own unique OpenAI API key. Refer to the internet on how to get one."
    +Fore.WHITE)
    sys.exit()


    #List of required packages
packages = [
    "llama-index",
    "openai",
    "pypdf",
    "langchain",
    "faiss-cpu",
    "tiktoken",
    "colorama",
    "texttable",
    "Chromadb"
]
#Check for required packages and install them if needed.
def check_package(package_name):
    try:
        dist = pkg_resources.get_distribution(package_name)
        #print(f"{package_name} {dist.version} is already installed.")
        return True
    except pkg_resources.DistributionNotFound:
        print(f"{package_name} is not installed.")
        return False

def install_package(package_name):
    subprocess.check_call(["pip", "install", package_name])
    print(f"{package_name} has been installed.")

#Install missing packages
for package in packages:
    if not check_package(package):
        install_package(package)

def printhelp():
    print ("""   askDOC is an assistant style helper for an OMNEST/OMNeT++ user to find answers
    to their product-related questions from the documentation.

    This is not meant to replace the documentation itself, but it may help you to quickly find hints
    or answers to questions you might have about the product. 

    This application has limited knowledge, and may occasionally give incorrect or misleading answers.
    Use common sense, verify the answers and check the documentation for a better understanding. 

    You use this tool entirely at your own discression and responsibility, no guarantees, implied or
    otherwise are provided.

    askDoc works from a local indexed and embedded database made from the documentation.
    It uses OPENAI to give nicely worded answers to your questions and understand the fully formed 
    questions you send in.
\n
    ---:» You have to specify an OpenAI API key for this tool to work. «:---\n
\n
    By default, askDOC will assume that the first command line argument is going to be your question,
    and assume that it will find the documents at the $DOCUMENTSPATH location you specified in the
    configuration.

    For example:
    > askDOC "How can I colorize an icon?"

    You can also use the -q argument to prefix your question, and the -pdfp argument to tell
    askDOC to look for the documentation in a different folder.

    For example:
    > python askDOC -q "How can I colorize an icon?" -pdfp "../omnetpp-6.0.1/doc/"

    Note, the askDOC will not regenerate the enbedding database for the application, unless explicitly
    instructed to do so, to ensure resources are not wasted, so you may get an error if such a database
    cannot be found at the specified folder.

    In this case, you must use the genDCO tool to generate this database before queries can be made.

    """)

#Check command line arguments, we need a question at least.
if len(sys.argv) < 2:

       printhelp()


#get OPENAI_API_KEY from config file
def load_openai_key():
    
    with open('openai_key.txt', 'r') as file:
        key = file.read().strip()
    return key

# Set key from config file
openai_key = load_openai_key()
os.environ['OPENAI_API_KEY'] = openai_key


from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.text_splitter import CharacterTextSplitter
from langchain.llms import OpenAI
from langchain.chains import RetrievalQA
from langchain.document_loaders import PyPDFDirectoryLoader

llm = OpenAI(temperature=0)
llm.model_name="text-davinci-003"
llm.top_p = 0.2
llm.max_tokens = 1500
llm.best_of = 5



from langchain.document_loaders import TextLoader

import joblib
from chromadb.api.types import Metadata
from chromadb.types import MetadataEmbeddingRecord

loader = PyPDFDirectoryLoader(args.pdfp)

embeddings = OpenAIEmbeddings()
#embeddings.__setattr__(model_name, "gpt-3.5-turbo")
#embeddings = OpenAIEmbeddings(model_name="gpt-3.5-turbo")

persist_directory = args.pdfp+'db'

# Check if the local persistent database exists
if os.path.exists(persist_directory):
    # Check if the folder is not empty
    if len(os.listdir(persist_directory)) > 0:
        databasealreadyexists = True
    else:
        databasealreadyexists = False
else:
    databasealreadyexists = False

text_splitter = CharacterTextSplitter(chunk_size=1500, chunk_overlap=100)


if databasealreadyexists == False:
    print("Generating persistent database. This will be used for future questions as well.")
    from chromadb.api.types import Metadata
    from chromadb.types import MetadataEmbeddingRecord
    docs = loader.load()
    omnetpp_texts = text_splitter.split_documents(docs)
    omnetpp_db = Chroma.from_documents(omnetpp_texts, embeddings, persist_directory=persist_directory, collection_name="omnetpp")
    omnetpp = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=omnetpp_db.as_retriever())
    #Added this to retain metadata, as it seems that this was removed with the metadata=None in the original omnetpp object that is created.
    omnetpp.metadata=omnetpp_db.search("metadata", "similarity")
    omnetpp_db.persist #call the persist function to ensure, the database is stored.

    
else:
    #skip generating embeddings, as they are already available loally /
    print("Found database, using persistent database.")
    omnetpp_db = Chroma(persist_directory=persist_directory, embedding_function=embeddings)
    omnetpp = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=omnetpp_db.as_retriever())
    omnetpp.metadata=omnetpp_db.search("metadata", "similarity")
    llm.metadata = omnetpp.metadata

# Import things that are needed generically
from langchain.agents import initialize_agent, Tool
from langchain.agents import AgentType
from langchain.tools import BaseTool
from langchain.llms import OpenAI
from langchain import LLMMathChain, SerpAPIWrapper

tools = [
    Tool(
        name="OMNeT++ and INET QA System",
        func=omnetpp.run,
        description="useful for when you need to answer questions about using OMNeT++"
    ),
]

# Construct the agent. We will use the default agent type here.
# See documentation for a full list of options.
agent = initialize_agent(
tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True
)

response = agent.run(question)


print (response)
print("............................................................")
#print(agent.acall)
#print("............................................................")

#print(agent.prep_inputs)

