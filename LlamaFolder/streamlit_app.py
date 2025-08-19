import streamlit as st
import os, json, base64, re
from io import BytesIO
from datetime import datetime

import difflib

from time import time

from langchain.prompts import PromptTemplate
from langchain.retrievers import MergerRetriever

from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain

from langchain_chroma import Chroma

from llm_handler import OllamaLLM
from document_parser import DocumentParser

# prevent pytorch from exploding (usually)
os.environ["TORCH_DISABLE_SOURCE_FINDER"] = "1"

# define exact file paths
IMAGE_PATH = r"C:\Users\maxwell.boutte\Techneaux Interns\imageByte.json"
PDF_PATH = r"C:\Users\maxwell.boutte\Techneaux Interns\PDFs"
VECTOR_PATH = r"C:\Users\maxwell.boutte\Techneaux Interns\sql_chroma_db"
MEMORY_PATH = r"C:\Users\maxwell.boutte\Techneaux Interns\memory_chroma_db"
start_time = 0


class StreamlitApp:
    # Instantiate other classes when StreamLit runs
    st.session_state.initialized = False
    # loading initial time
    start_time = time()
    # check if this run has intential LLM calls
    if "AI_mode" not in st.session_state:
        st.session_state.AI_mode = False
    if "AI_mode_next" not in st.session_state:
        st.session_state.AI_mode_next = False

    def __init__(self):
        
        # define path and filetype for docs
        self.parser = DocumentParser(
            pdf_path=PDF_PATH,
            vector_path=VECTOR_PATH,
            memory_vector_path=MEMORY_PATH
        )
        

        # define LLM model 
        self.llm = OllamaLLM(model="llama3:8b")

        # define prompt template for context and input insertion
        self.prompt_template = PromptTemplate.from_template("""
        You are a helpful assistant trained on structured step-by-step instructions extracted from a SharePoint page.

        Your job is to help users understand or complete tasks using the full body of available instructional content.
        - Always base your answers on the most relevant and reliable information from the entire corpus.
        - When specific content areas or documents are mentioned in the user's question, give those priority in your response.
        - If no specific context is given, use the broader knowledge base to provide the best possible answer.  

        You should:

        - Answer like a friendly expert or trainer.
        - Refer to the steps only when needed.
        - Cite the source used (source URL)
        - Use the glossary to clarify terminology or concepts.
        - Do NOT repeat full step lists or attempt to number steps.
        - Any instructions taken from the CONTEXT should be stripped of their step number.
        - Give practical advice in natural language.
        - Use the CONTEXT as your background knowledge to inform your response.
        - Do NOT reference or rely on any previous responses. Treat each question as independent unless the user explicitly refers to an earlier answer.
                                                                    
        CONTEXT (glossary or other documents, selected URL):
        {context}

        QUESTION:
        {input}

        ---
        ANSWER (helpful):
        
        """)
 

        # get ready freddy
        if not st.session_state.initialized:
            st.session_state.initialized = True
            self.initialize_session()
            self.build_rag_chain()
            self.show_intro()



    # initialize vector stores and short-term memory file
    def initialize_session(self):
        
        st.session_state.shown_alt_idx = 0
        st.session_state.use_saved_response = False
        st.session_state.AI_selected = False
        st.session_state.prompt = ""
        st.session_state.AI_response_generated = False
        st.session_state.feedback_saved = False

        if "i" not in st.session_state:
            st.session_state.i = 0

        # dictionary for streamlit logic
        for key, val in {
            "intro_shown": False,
            "i": 0,
            "conversationBufferMemory": [],
            "messages": [],#
            "exit_flag": False
        }.items():
            # default mapping for reference 
            st.session_state.setdefault(key, val)


        # load dictionary of URL:Image
        if "image_dict" not in st.session_state:
            with open(IMAGE_PATH, "r") as f:
                st.session_state.image_dict = json.load(f)

        # create .json and vector store for q/a memory 
        if "memory_vector_store" not in st.session_state:
            memory = self.parser.load_memory_json()
            store = self.parser.initialize_memory_store(memory)
            st.session_state.memory_vector_store = store
            if not store:
                st.info("No prior memory found. Starting fresh.")

        # create document vector store
        if "vector_store" not in st.session_state:
            embeddings = self.parser.embedding
            st.warning("Loading Database...")
            st.session_state.vector_store = Chroma(
                persist_directory=self.parser.vector_path,
                embedding_function=embeddings)
            st.success("Database loaded successfully.")
        

    # define search types, merge the RAG retrievers for both vector stores and link to session 
    def build_rag_chain(self):

        # retriever for content
        doc_retriever = st.session_state.vector_store.as_retriever(search_type="similarity_score_threshold", search_kwargs={"k": 3, "score_threshold": 0.05})
        # retriever for Q/A memory
        memory_retriever = st.session_state.memory_vector_store.as_retriever(search_type="similarity_score_threshold", search_kwargs={"k": 1, "score_threshold": 0.2})#search_type="similarity", search_kwargs={"k": 2}) if st.session_state.memory_vector_store else doc_retriever
        
        # RAG 
        doc_chain = create_stuff_documents_chain(self.llm, self.prompt_template)
        # expose to session 
        st.session_state.rag_chain = create_retrieval_chain(doc_retriever, doc_chain)

        # separate retriever for comparing inputs to previously saved good responses
        st.session_state.memory_retriever = memory_retriever

    def rerank_documents(self, prompt: str, docs: list):
        scored_docs = []
        for doc in docs:
            source = doc.metadata['source']
            page_url = source
            score = difflib.SequenceMatcher(None, prompt.lower(), page_url).ratio()
            scored_docs.append((score, doc))
        if not scored_docs:
            return [], 0.0  # Avoid UnboundLocalError
        
        # Sort by score in descending order
        sorted_docs = [doc for score, doc in sorted(scored_docs, key=lambda x: x[0], reverse=True)]
        return sorted_docs, score
    
    def rerank_memory(self, prompt: str, docs: list):
        scored_docs = []
        for doc in docs:
            previous_prompt = doc.metadata['user']
            score = difflib.SequenceMatcher(None, prompt.lower(), previous_prompt).ratio()
            scored_docs.append((score, doc))
        if not scored_docs:
            return [], 0.0  # Avoid UnboundLocalError
        
        # Sort by score in descending order
        sorted_docs = [doc for score, doc in sorted(scored_docs, key=lambda x: x[0], reverse=True)]
        return sorted_docs, score
    
    # streamlit data 
    def run_chat(self):
    # Handle user prompt and check if user exited
    # Method exists to prevent handle_prompt from executing when the program is terminated
        if not st.session_state.exit_flag:
            session_continue = self.handle_prompt()
            if session_continue:
                return


    # display intro
    def show_intro(self):
        # difference in time between pressing start and the program activating
        print("Load time (s): ", (time() - self.start_time))
        #st.image(image=r"C:\Users\maxwell.boutte\Techneaux Interns\Enterprise Services Logo - Back (RGB).png",width=500)
        
        st.title("🟧 🟧 Welcome ES Team! 🟧 🟧")

        st.write("Please input key words below to search through training documents. If your question isn't answered, please enable AI mode for enhanced search.")


    # initiate response chain
    def handle_prompt(self):
        
        # accept user input
        new_prompt = st.chat_input("Type here")

        if st.session_state.get("AI_response_generated"):
            if new_prompt:  # If new input has arrived
                st.session_state.AI_response_generated = False

        if new_prompt:
            st.session_state.prompt = new_prompt
            st.session_state.i += 1
            
### NOTE ###            #st.session_state.AI_response_generated = False  # Reset on new input
### NOTE VERY IMPORTANT?###

        if not st.session_state.AI_selected:
            # loop forever until input is received
            if not st.session_state.prompt:
                return False
            else:
                st.session_state.i += 1
            
            # break if exit flag is raised
            if st.session_state.prompt.lower().strip() == "exit":
                    st.session_state.exit_flag = True
                    st.info("🚪 The application has ended. Thanks for using it!")
                    return True 
        
        # insert next message and increment message counter
        if st.session_state.prompt:
            #st.session_state.messages.append({"role": "user", "content": st.session_state.prompt})
            
            # If AI was requested for this prompt in the previous run, activate it
            if st.session_state.get("AI_mode_next", False):
                st.session_state.AI_mode = True
                st.session_state.AI_mode_next = True  # reset the flag immediately


            user_input = st.session_state.prompt
            # empty list of relevant docs
            st.session_state.backup_responses = []
            
            # blank response to populate with text and pictures
            full_response = ""

            # Run rough similarity
            quicktriever = st.session_state.vector_store.as_retriever(search_type="similarity_score_threshold",search_kwargs={"k": 10, "score_threshold": 0.40})
            if not st.session_state.AI_selected:
                #retrieve and rerank relevant documents
                quick_docs = quicktriever.invoke(st.session_state.prompt)
                sorted_docs, scores = self.rerank_documents(st.session_state.prompt, quick_docs)
                quick_docs = sorted_docs
            else:
                quick_docs = None
            
            assistant_response = ""
            
            if quick_docs:
                #assistant_response = "**Select from relevant documents in the sidebar**"
                # find nearest documents to offer the user
                i=0
                while i < len(quick_docs) and i < 10:
                    #print(quick_docs[i].metadata["source"])
                    st.session_state.backup_responses.append(quick_docs[i])
                    i+=1
            else:
                assistant_response = "Please retry with a different spelling..."

            # If AI box is checked, prepare to handle the AI response and Previous response buttons
            if st.session_state.AI_mode:

                # find previous question matches and sort by relevance
                memory_matches = st.session_state.memory_retriever.get_relevant_documents(user_input)
                memory_matches, score = self.rerank_memory(docs=memory_matches, prompt=user_input)
                
                # load the relevance score and associated response
                if memory_matches:
                    top_match = memory_matches[0]
                    st.session_state.score = score
                    st.session_state.suggested_memory_response = {
                        "user": top_match.metadata["user"],
                        "response": top_match.page_content
                        #"source": top_match.metadata.get("source", "Unknown")
                    }

                else:
                    # need to add conditional message when no relevant responses appear
                    st.session_state.suggested_memory_response = None   

                # if the user decides to view the response, display it 
                if st.session_state.use_saved_response:
                    if st.session_state.get("suggested_memory_response"):
                        suggested = st.session_state.suggested_memory_response
                        st.session_state.messages.append({
                            "role": "assistant",
                            "content": suggested["response"]
                        })
                        st.session_state.suggested_memory_response = None
                        st.session_state.use_saved_response = False
                        return False  # 👈 EXIT before generating anything else
                
                # if the user decides to invoke the LLM
                if st.session_state.AI_selected:    
                    # Call LLM with populated prompt template
                    result = st.session_state.rag_chain.invoke({"input": st.session_state.prompt})
                    assistant_response = result.get("answer", "No answer found.")
                    # set flag 
                    st.session_state.AI_response = assistant_response
                    # print response
                    print("full unbridled response:\n", assistant_response)
                    
                    # append the response to the messages list 
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": assistant_response
                    })

                    # reset flags to break out of loop
                    #st.session_state.AI_mode = False
                    st.session_state.AI_selected = False
                    st.session_state.AI_response_generated = True  # ✅ Prevents duplicate calls
                    st.session_state.AI_mode_next = True

                    # prepare response for saving to memory
                    st.session_state.conversationBufferMemory.append({
                        "index": st.session_state.i,
                        "user": st.session_state.prompt,
                        "response": assistant_response,
                        "timestamp": datetime.now().isoformat(),
                        "feedback": None
                    })
                    return False

        # display agent icon then empty message for response to fill
        if not st.session_state.get("AI_response"):
            # user input is displayed with the UI
            with st.empty():
                
                assistant_response = ""

                if st.session_state.get("AI_response"):
                    assistant_response = st.session_state.get("AI_response")

                # find instances of URL pattern in the list of steps
                url_pattern = re.compile(r'(https?://\S+)')
                tokens = re.split(url_pattern, assistant_response)

                for token in tokens:
                    token_strip = token.strip()
                    
                    # if split value contains a URL extension in the image dict, pull image bytes from the dict
                    if token_strip in st.session_state.image_dict:
                        if full_response:
                            # with st.chat_message(name="assistant",avatar="🪨"):
                            #     st.markdown(full_response, unsafe_allow_html=True)
                            full_response = ""

                        # decode bytes into image and display as the response is typed
                        image_bytes = base64.b64decode(st.session_state.image_dict[token_strip])
                        st.image(BytesIO(image_bytes))
                    
                    else:
                        # if not a picture, append text to response
                        full_response += token

                # if the question counter doesnt match the response counter, add it to the list of messages
                if full_response and st.session_state.get("just_added_response") != st.session_state.i:
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": full_response
                    })
                    
                    # match counters
                    st.session_state.just_added_response = st.session_state.i
                    
                # insert q/a metadata into buffer (won't be saved)
                st.session_state.conversationBufferMemory.append({
                        "index": st.session_state.i,
                        "user": st.session_state.prompt,
                        "response": full_response,
                        "timestamp": datetime.now().isoformat(),
                        "feedback": None
                    })
                # ancillery functions are enabled after first run
                st.session_state.started = True
        
        # we aint done yet boy get back
        return False   
    

# round 1.... BEGIN 
if __name__ == "__main__":
    
    # initialize session on startup
    if "app" not in st.session_state:
        st.session_state.app = StreamlitApp()

    # chatting time
    st.session_state.app.run_chat()

    # begin rendering after first user input
    if "started" in st.session_state:

        # Reset flags if new prompt is entered
        if "AI_response_generated" in st.session_state and "prompt" in st.session_state:
            last_input = st.session_state.conversationBufferMemory[-1]["user"] if st.session_state.conversationBufferMemory else None
            if last_input != st.session_state.prompt:
                st.session_state.AI_response_generated = False

        # add latest prompt to list of messages
        st.session_state.messages.append({
            "role": "user",
            "content": st.session_state.prompt
        })

        # record messages currently on screen for refresh
        seen_messages = set()
        for msg in st.session_state.messages:
            role = msg["role"]
            msg_key = f"{msg['role']}::{msg['content']}"
            # display messages in list that are not on screen
            if msg_key not in seen_messages:
                seen_messages.add(msg_key)
                # square is user, diamond is AI
                avatar = "👨‍💻" if role == "user" else "🔮"
                with st.chat_message(name=role, avatar=avatar):
                    st.markdown(msg["content"], unsafe_allow_html=True)


        # Show feedback saved message
        if st.session_state.get("feedback_saved"):
            st.success("Thanks for the feedback!")
            st.session_state.feedback_saved = False


        # Sidebar: show relevant sources
        with st.sidebar:
            if "backup_responses" in st.session_state:
        
                source_options = {
                    st.session_state.app.parser.get_url_name(doc.metadata["source"]): doc
                    for doc in st.session_state.backup_responses
                }
                selected_source = st.selectbox(
                    "### 🔄 Relevant Sources",
                    options=list(source_options.keys()),
                    key="source_selectbox"
                )
                if st.button("✅ Confirm Selection", key="confirm_source_button"):
                    
                    st.session_state.shown_alt = source_options[selected_source]
                    st.session_state.shown_alt_idx = list(st.session_state.backup_responses).index(st.session_state.shown_alt)
                    content = st.session_state.shown_alt.page_content + "\n" + st.session_state.shown_alt.metadata["source"]
                    st.session_state.just_confirmed_source = content

                # Show selected doc if any
                if "shown_alt" in st.session_state:
                    with st.chat_message(name="assistant",avatar="🦢"):
                        content = st.session_state.shown_alt.page_content + "\n\nSource: " + st.session_state.shown_alt.metadata["source"]
                        st.markdown(content.replace('\n', '  \n'), unsafe_allow_html=True)
        
        # prompt user with suggested memory document before they are presented buttons
        if st.session_state.AI_mode and not st.session_state.AI_selected and not st.session_state.use_saved_response and not st.session_state.AI_response_generated:
            # 💡 Show suggestion before buttons
            if st.session_state.get("suggested_memory_response") and st.session_state.AI_mode_next:
                user_string = st.session_state.suggested_memory_response.get("user", "previous input")
                with st.chat_message(name="assistant",avatar="♻️"):
                    st.markdown(f"""
                    💡You may choose to use the AI or view the saved response for: 
                        `"{user_string}"` Score:`"{st.session_state.score:.2f}"`
                    Use the buttons below to proceed.
                    """)
            # buttons to use the AI mode functions
            col1, col2, col3 = st.columns(3)
            with col1:    
                if st.button("Generate AI Response"):
                    st.session_state.AI_selected = True
                    st.rerun()
            if st.session_state.get("suggested_memory_response"):        
                with col2:
                    if st.button("✅ Use Saved Answer"):
                        st.session_state.use_saved_response = True
                        st.rerun()
        
        # Good response button after AI response comes through
        if st.session_state.AI_response_generated and not st.session_state.feedback_saved:
            col1, col2 = st.columns(2)
            with col1:
                if st.button("Good Response 👍"):
                    # Tie user question to shown response and mark feedback good
                    st.session_state.conversationBufferMemory[-1]["user"] = st.session_state.prompt
                    st.session_state.conversationBufferMemory[-1]["response"] = st.session_state.AI_response
                    st.session_state.conversationBufferMemory[-1]["feedback"] = "good"
                    if "AI_response" in st.session_state:
                        del st.session_state.AI_response
                    st.session_state.feedback_saved = True
                    st.success("Thanks for the feedback!")
                    # Save feedback (your existing save function)
                    st.session_state.app.parser.save_memory_json(st.session_state.conversationBufferMemory)

    # Before first input, show AI mode toggle button only
    col1, col2, col3 = st.columns(3)
    with col3:
        ai_mode_enabled = st.checkbox("Enable AI mode", key="AI_mode_toggle")

    # set flags for next run
    if ai_mode_enabled:
        st.session_state.AI_mode_next = True
        st.success("AI mode is active! Please ask a question.")
    else:
        st.session_state.AI_mode_next = False

