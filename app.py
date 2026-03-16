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

resume = st.text_area("Paste your Resume")
job = st.text_area("Paste Job Description")

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

    # ✅ Correct call for HF chat-capable model
    response = client.chat(
        messages=[{"role": "user", "content": prompt}],
        max_new_tokens=400
    )

    # The API returns text directly
    return response["choices"][0]["message"]["content"]

if st.button("Analyze Resume"):

    if resume and job:
        result = analyze_resume(resume, job)
        st.write(result)
    else:
        st.warning("Please paste both the resume and job description.")
