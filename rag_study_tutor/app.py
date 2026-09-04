"""Streamlit web UI:  streamlit run app.py"""

import streamlit as st

from ragtutor.config import settings
from ragtutor.quiz import make_quiz
from ragtutor.retriever import build_index
from ragtutor.tutor import Tutor

st.set_page_config(page_title="RAG Study Tutor", page_icon="📚")


@st.cache_resource(show_spinner=False)
def get_tutor(_version: int = 0) -> Tutor:
    return Tutor(settings=settings)


st.title("📚 RAG Study Tutor")
st.caption("Answers come from the files in your `data/` folder — with sources.")

with st.sidebar:
    st.subheader("Index")
    st.write(f"Notes folder: `{settings.data_dir}`")
    if st.button("Rebuild index"):
        with st.spinner("Reading notes and building the index..."):
            build_index(settings, verbose=False)
        st.cache_resource.clear()
        st.success("Index rebuilt.")
    top_k = st.slider("Chunks to retrieve", 1, 10, settings.top_k)

try:
    tutor = get_tutor()
except FileNotFoundError:
    st.warning("No index yet. Put notes in `data/`, then click **Rebuild index**.")
    st.stop()

st.sidebar.write(f"Model: `{tutor.llm.name}`")
st.sidebar.write(f"Chunks indexed: {len(tutor.store)}")

tab_ask, tab_quiz = st.tabs(["Ask", "Quiz me"])

with tab_ask:
    question = st.text_input("Your question", placeholder="e.g. what is normalization?")
    if question:
        with st.spinner("Thinking..."):
            answer = tutor.ask(question, k=top_k)
        st.markdown(answer.text)
        if answer.sources:
            st.caption("Sources: " + ", ".join(answer.sources))
        with st.expander("Retrieved passages"):
            for hit in answer.hits:
                st.markdown(f"**{hit.source}** · match {hit.score:.2f}")
                st.text(hit.text)

with tab_quiz:
    topic = st.text_input("Topic", key="quiz_topic", placeholder="e.g. transactions")
    count = st.number_input("Questions", 1, 15, 5)
    if st.button("Generate quiz") and topic:
        with st.spinner("Writing questions..."):
            items = make_quiz(tutor, topic, n=int(count))
        if not items:
            st.info("Nothing found on that topic in your notes.")
        for i, item in enumerate(items, start=1):
            st.markdown(f"**Q{i}.** {item.question}")
            with st.expander("Show answer"):
                st.write(item.answer)
                if item.source:
                    st.caption(item.source)
