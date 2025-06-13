import torch
import torch.nn as nn
from transformers import BertModel
import re

# Define known skills
known_skills = [
    "python", "java", "c++", "javascript", "html", "css", "sql", "react", "node.js",
    "machine learning", "deep learning", "data analysis", "pandas", "numpy", "tensorflow",
    "keras", "flask", "django", "git", "linux", "excel", "power bi", "tableau", "nlp",
    "opencv", "cloud computing", "aws", "azure", "gcp", "docker", "kubernetes"
]
known_skills_set = set(skill.lower() for skill in known_skills)

tip_bank = [
    "Highlight your accomplishments using bullet points.",
    "Use active verbs and quantify achievements when possible.",
    "Keep your resume concise, ideally one page.",
    "Ensure formatting is consistent throughout the document."
]

class ResumeMultiTaskModel(nn.Module):
    def __init__(self, num_job_roles, num_skills):
        super(ResumeMultiTaskModel, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.dropout = nn.Dropout(0.3)
        self.ats_head = nn.Linear(768, 1)
        self.job_head = nn.Linear(768, num_job_roles)
        self.skills_head = nn.Linear(768, num_skills)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = self.dropout(outputs.pooler_output)
        ats_score = self.ats_head(pooled_output)
        job_logits = self.job_head(pooled_output)
        skills_logits = self.skills_head(pooled_output)
        return ats_score, job_logits, skills_logits

def extract_text(file):
    if file.name.endswith(".txt"):
        return file.read().decode("utf-8")
    elif file.name.endswith(".pdf"):
        import PyPDF2
        reader = PyPDF2.PdfReader(file)
        return "\n".join([page.extract_text() for page in reader.pages if page.extract_text()])
    return ""

def extract_skills_from_text(text):
    text_lower = text.lower()
    extracted = [skill for skill in known_skills if re.search(r'\b' + re.escape(skill.lower()) + r'\b', text_lower)]
    return extracted

def predict_resume(text, model, le_job, mlb_skills, tokenizer):
    enc = tokenizer(text, truncation=True, padding='max_length', max_length=512, return_tensors='pt')
    with torch.no_grad():
        ats_score, job_logits, skills_logits = model(enc['input_ids'], enc['attention_mask'])

    job_pred = le_job.inverse_transform([torch.argmax(job_logits).item()])[0]

    model_skill_probs = torch.sigmoid(skills_logits).squeeze().numpy()
    model_top_skills = [mlb_skills.classes_[i] for i, prob in enumerate(model_skill_probs) if prob > 0.5]

    extracted_skills = extract_skills_from_text(text)

    skill_suggestions = [s for s in known_skills if s not in extracted_skills][:5]
    job_objective = f"Seeking a challenging position as a {job_pred} where I can utilize my skills and grow professionally."

    missing_sections = []
    if 'experience' not in text.lower(): missing_sections.append("Experience")
    if 'education' not in text.lower(): missing_sections.append("Education")
    if 'projects' not in text.lower(): missing_sections.append("Projects")

    return ats_score.item(), job_pred, extracted_skills, missing_sections, skill_suggestions, job_objective
