import os
import streamlit as st

# 从系统环境变量中获取 API 密钥
os.environ["OPENAI_API_KEY"] = st.secrets["KEY"]

from llama_index import download_loader
from llama_index.node_parser import SimpleNodeParser
from llama_index import GPTSimpleVectorIndex
from llama_index import LLMPredictor, GPTSimpleVectorIndex, PromptHelper, ServiceContext
from langchain import OpenAI

#doc_path ='D://pythonsave//data'
doc_path = './data/'
index_file = 'index.json'

if 'response' not in st.session_state:
    st.session_state.response = ''

if not os.path.exists(doc_path):
    os.makedirs(doc_path)

def send_click():
    st.session_state.response  = index.query(st.session_state.prompt)

index = None
st.title("Yeyu's Doc Chat")

sidebar_placeholder = st.sidebar.container()
#uploaded_file = st.file_uploader("Choose a file")
# 设置密码
password = "123456"

# 在侧边栏添加一个文本框，用于输入密码
password_placeholder = st.sidebar.empty()
password_input = password_placeholder.text_input("Enter the password", type="password")

# 声明 uploaded_file 变量
uploaded_file = None

if password_input == password:
    # 密码正确，允许上传文件
    uploaded_file = st.file_uploader("Choose a file")
    # 其他操作...
else:
    # 密码错误，禁止上传文件
    st.warning("您好，我是丽达小丽，见到你很高兴！")










if uploaded_file is not None:

    doc_files = os.listdir(doc_path)
    for doc_file in doc_files:
        os.remove(doc_path  + doc_file)

    bytes_data = uploaded_file.read()
    with open(f"{doc_path}{uploaded_file.name}", 'wb') as f: 
        f.write(bytes_data)

    SimpleDirectoryReader = download_loader("SimpleDirectoryReader")

    loader = SimpleDirectoryReader(doc_path, recursive=True, exclude_hidden=True)
    documents = loader.load_data()
    sidebar_placeholder.header('Current Processing Document:')
    sidebar_placeholder.subheader(uploaded_file.name)
    sidebar_placeholder.write(documents[0].get_text()[:10000]+'...')

    llm_predictor = LLMPredictor(llm=OpenAI(temperature=0, model_name="text-davinci-003"))

    max_input_size = 4096
    num_output = 256
    max_chunk_overlap = 20
    prompt_helper = PromptHelper(max_input_size, num_output, max_chunk_overlap)

    service_context = ServiceContext.from_defaults(llm_predictor=llm_predictor, prompt_helper=prompt_helper)

    index = GPTSimpleVectorIndex.from_documents(
        documents, service_context=service_context
    )

    index.save_to_disk(index_file)

elif os.path.exists(index_file):
    index = GPTSimpleVectorIndex.load_from_disk(index_file)

    SimpleDirectoryReader = download_loader("SimpleDirectoryReader")
    loader = SimpleDirectoryReader(doc_path, recursive=True, exclude_hidden=True)
    documents = loader.load_data()
    doc_filename = os.listdir(doc_path)[0]
    sidebar_placeholder.header('Current Processing Document:')
    sidebar_placeholder.subheader(doc_filename)
    sidebar_placeholder.write(documents[0].get_text()[:10000]+'...')

if index != None:
    st.text_input("Ask something: ", key='prompt', value='请输入问题：')
    st.button("Send", on_click=send_click)
    if st.session_state.response:
        st.subheader("Response: ")
        st.success(st.session_state.response, icon= "🤖")