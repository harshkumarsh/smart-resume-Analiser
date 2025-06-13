import streamlit as st
import torch
from transformers import BertTokenizer
import joblib
import base64
from utils import ResumeMultiTaskModel, extract_text, predict_resume

# --- Load Tokenizer and Models ---
st.session_state['tokenizer'] = BertTokenizer.from_pretrained('bert-base-uncased')

# Load encoders
le_job = joblib.load("label_encoder_job.pkl")
mlb_skills = joblib.load("mlb_skills.pkl")

# Load model
model = ResumeMultiTaskModel(
    num_job_roles=len(le_job.classes_),
    num_skills=len(mlb_skills.classes_)
)
model.load_state_dict(torch.load("resume_mtl_model.pt", map_location=torch.device('cpu')))
model.eval()

# --- Streamlit UI ---
st.set_page_config(page_title="Resume Analyzer", layout="wide")
st.title("📄 Resume Analyzer - ATS, Role, Skills, Tips, and Recommendations")

uploaded_file = st.file_uploader("Upload your Resume (PDF or TXT)", type=["pdf", "txt"])

def show_pdf(file):
    base64_pdf = base64.b64encode(file.read()).decode("utf-8")
    pdf_display = f'<iframe src="data:application/pdf;base64,{base64_pdf}" width="700" height="1000" type="application/pdf"></iframe>'
    st.markdown(pdf_display, unsafe_allow_html=True)

if uploaded_file:
    st.subheader("📑 Full Resume Preview")
    show_pdf(uploaded_file)

    uploaded_file.seek(0)
    resume_text = extract_text(uploaded_file)

    if resume_text:
        st.subheader("🔍 Resume Analysis Results")
        ats, job, skills, missing, skill_suggestions, job_objective = predict_resume(
            resume_text, model, le_job, mlb_skills, st.session_state['tokenizer']
        )

        st.metric("ATS Score", f"{int(abs(ats) * 100 * 100)}%")
        st.write("**Predicted Job Role:**", job)
        st.write("**Extracted Skills:**", ", ".join(skills) if skills else "None")
        st.write("**Missing Sections:**", ", ".join(missing) if missing else "None")

        st.subheader("🎯 Career Recommendations")
        st.write("- Suggested Career Objective:")
        st.info(job_objective)

        st.subheader("🧠 Skill Recommendations")
        st.write(", ".join(skill_suggestions) if skill_suggestions else "No new suggestions")

        st.subheader("✍️ Resume Writing Tips")
        st.write("- Use action verbs to describe achievements.")
        st.write("- Tailor the resume for each job role.")
        st.write("- Include quantifiable results.")
        st.write("- Ensure consistent formatting and structure.")
    else:
        st.error("❌ Could not extract text from the uploaded resume.")
