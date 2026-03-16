import streamlit as st
from huggingface_hub import InferenceClient
from dotenv import load_dotenv
import os

# Load your Hugging Face token
load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN")

# Initialize client with a Hugging Face chat-capable model
client = InferenceClient(
    model="mistralai/Mistral-7B-Instruct-v0.2",
    token=HF_TOKEN
)

st.title("SmartResumeAI")
st.subheader("Compare your resume with a job description using AI")

resume = st.text_area("Paste your Resume", height=250)
job = st.text_area("Paste Job Description", height=200)


def analyze_resume(resume, job):
    prompt = f"""
You are an expert career advisor.

Compare this resume with the job description.

Resume:
{resume}

Job Description:
{job}

Tasks:
1. List missing skills
2. Suggest improvements
3. Suggest ways to better match the job
"""

    response = client.chat_completion(
        messages=[{"role": "user", "content": prompt}],
        max_tokens=500
    )

    return response.choices[0].message.content


if st.button("Analyze Resume"):
    if resume and job:
        with st.spinner("Analyzing your resume..."):
            try:
                result = analyze_resume(resume, job)
                st.success("Analysis Complete!")
                st.markdown(result)
            except Exception as e:
                st.error(f"Something went wrong: {e}")
    else:
        st.warning("Please paste both the resume and job description.")
