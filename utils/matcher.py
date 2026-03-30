import json
import re
from groq import Groq
from config import GROQ_API_KEY, LLM_MODEL, LLM_TEMPERATURE, MAX_TOKENS

# Groq client
client = Groq(api_key=GROQ_API_KEY)

SYSTEM_PROMPT = """You are an expert HR recruiter and resume analyst.
You always respond with valid JSON only — no markdown, no explanation, no extra text."""

USER_PROMPT = """Compare the RESUME below against the JOB DESCRIPTION and return a structured analysis.

JOB DESCRIPTION:
{job_description}

RESUME:
{resume_text}

Respond ONLY with a valid JSON object in this exact format:
{{
  "match_score": <integer 0-100>,
  "summary": "<2-3 sentence overall assessment>",
  "matching_skills": ["skill1", "skill2"],
  "missing_skills": ["skill1", "skill2"],
  "experience_match": "<Strong | Moderate | Weak>",
  "education_match": "<Strong | Moderate | Weak | Not Required>",
  "recommendation": "<Shortlist | Consider | Reject>",
  "reasons": ["reason1", "reason2"]
}}"""


def score_resume(resume_text: str, job_description: str) -> dict:
    """Score a single resume against a job description using Groq."""
    response = client.chat.completions.create(
        model=LLM_MODEL,
        temperature=LLM_TEMPERATURE,
        max_tokens=MAX_TOKENS,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": USER_PROMPT.format(
                resume_text=resume_text[:4000],
                job_description=job_description[:2000],
            )},
        ],
    )

    raw     = response.choices[0].message.content
    cleaned = re.sub(r"```json|```", "", raw).strip()

    try:
        result = json.loads(cleaned)
    except json.JSONDecodeError:
        result = {
            "match_score":      0,
            "summary":          "Could not parse response. Please retry.",
            "matching_skills":  [],
            "missing_skills":   [],
            "experience_match": "Unknown",
            "education_match":  "Unknown",
            "recommendation":   "Review manually",
            "reasons":          [raw[:300]],
        }

    return result


def batch_score_resumes(resumes: list, job_description: str) -> list:
    """Score multiple resumes, return sorted by match_score descending."""
    results = []
    for resume in resumes:
        result = score_resume(resume["text"], job_description)
        result["name"] = resume["name"]
        results.append(result)

    results.sort(key=lambda x: x.get("match_score", 0), reverse=True)
    return results