import streamlit as st
import asyncio
from agent import ResearchAssistantAgent
from agents import Runner

st.set_page_config(page_title="Research Agent", layout="centered")
st.title("🔬 AI Research Agent")
st.write("This agent can help you find and summarize research papers on a given topic. Just enter your research topic below, and it will fetch relevant papers and provide a summary.")

 

user_query = st.text_input("Enter your research topic:", placeholder="e.g., computer vision")

if st.button("Get Research Paper") and user_query:
    async def fetch_result():
        with st.spinner("🔎 Getting paper information and summarizing..."):
            result = await Runner.run(
                ResearchAssistantAgent,
                f"give me a research paper on {user_query}"
            )
            return result.final_output


    result_output = asyncio.run(fetch_result())

    # Try to split the result into 'info' and 'summary'
    parts = result_output.split("**Summary:**")

    if len(parts) == 2:
        paper_info, paper_summary = parts
        st.subheader("📄 Paper Information")
        st.markdown(paper_info)

        # Extract and link the PDF URL if present
        if "PDF URL:" in paper_info:
            try:
                pdf_line = [line for line in paper_info.splitlines() if "PDF URL:" in line][0]
                pdf_url = pdf_line.split("**PDF URL:**")[-1].strip()
                st.markdown(f"📎 [Click here to read the full paper]({pdf_url})")
            except IndexError:
                pass

        st.subheader("🧠 Summary")
        st.markdown(paper_summary)
    else:
        st.subheader("🔍 Result")
        st.markdown(result_output)

    st.success("Done!")
