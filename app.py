import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px

from utils.resume_parser import extract_resume_text, chunk_text
from utils.embedder import get_or_create_index, upsert_resume, query_similar_chunks
from utils.matcher import batch_score_resumes

# PAGE CONFIG

st.set_page_config(
    page_title="AI Resume Screener",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded",
)

# CUSTOM CSS

st.markdown("""
<style>
    /* Main header */
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        background: linear-gradient(90deg, #4F46E5, #7C3AED);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        color: #6B7280;
        font-size: 1rem;
        margin-bottom: 2rem;
    }

    /* Score cards */
    .score-card {
        background: #F9FAFB;
        border: 1px solid #E5E7EB;
        border-radius: 12px;
        padding: 1.2rem 1.5rem;
        margin-bottom: 1rem;
    }
    .score-card-header {
        font-size: 1rem;
        font-weight: 600;
        color: #111827;
        margin-bottom: 0.3rem;
    }

    /* Match score badge */
    .badge-green  { background:#D1FAE5; color:#065F46; padding:2px 10px; border-radius:12px; font-weight:600; font-size:0.85rem; }
    .badge-yellow { background:#FEF3C7; color:#92400E; padding:2px 10px; border-radius:12px; font-weight:600; font-size:0.85rem; }
    .badge-red    { background:#FEE2E2; color:#991B1B; padding:2px 10px; border-radius:12px; font-weight:600; font-size:0.85rem; }

    /* Skill pills */
    .pill-green { background:#D1FAE5; color:#065F46; padding:3px 10px; border-radius:20px; font-size:0.8rem; margin:2px; display:inline-block; }
    .pill-red   { background:#FEE2E2; color:#991B1B; padding:3px 10px; border-radius:20px; font-size:0.8rem; margin:2px; display:inline-block; }

    /* Section divider */
    .section-title {
        font-size:1.1rem; font-weight:600; color:#374151;
        border-left: 4px solid #4F46E5;
        padding-left: 0.6rem;
        margin: 1.5rem 0 0.8rem;
    }

    /* Sidebar */
    section[data-testid="stSidebar"] { background: #F3F4F6; }
    .stButton > button {
        width: 100%;
        background: linear-gradient(90deg, #4F46E5, #7C3AED);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.6rem 1rem;
        font-weight: 600;
        font-size: 0.95rem;
        cursor: pointer;
    }
    .stButton > button:hover { opacity: 0.9; }
</style>
""", unsafe_allow_html=True)

# SESSION STATE

if "resumes"        not in st.session_state: st.session_state.resumes        = []
if "jd_text"        not in st.session_state: st.session_state.jd_text        = ""
if "results"        not in st.session_state: st.session_state.results        = []
if "index"          not in st.session_state: st.session_state.index          = None
if "screening_done" not in st.session_state: st.session_state.screening_done = False



# SIDEBAR — INPUT PANEL

with st.sidebar:
    st.markdown("## 🎯 Resume Screener")
    st.markdown("---")

    # ── Step 1: Job Description ───────────────────────────
    st.markdown("### Step 1: Job Description")
    jd_input_mode = st.radio("Input method", ["Paste text", "Upload file"], horizontal=True)

    if jd_input_mode == "Paste text":
        jd_text = st.text_area(
            "Paste job description here",
            height=200,
            placeholder="Senior Python Developer...\n\nRequirements:\n- 5+ years Python\n- FastAPI, Docker...",
        )
    else:
        jd_file = st.file_uploader("Upload JD (PDF / DOCX / TXT)", type=["pdf", "docx", "txt"], key="jd_file")
        jd_text = ""
        if jd_file:
            jd_text = extract_resume_text(jd_file)
            st.success(f"JD loaded: {len(jd_text)} characters")

    st.session_state.jd_text = jd_text

    st.markdown("---")

    # Step 2: Resume Upload
    st.markdown("### Step 2: Upload Resumes")
    uploaded_files = st.file_uploader(
        "Upload resumes (PDF / DOCX / TXT)",
        type=["pdf", "docx", "txt"],
        accept_multiple_files=True,
        key="resume_files",
    )

    if uploaded_files:
        st.success(f"{len(uploaded_files)} resume(s) uploaded")

    st.markdown("---")

    # Step 3: Options
    st.markdown("### Options")
    use_pinecone = st.checkbox("Store embeddings in Pinecone", value=True,
                               help="Uncheck to run matching without vector storage (direct LLM only)")
    show_raw     = st.checkbox("Show raw resume text in results", value=False)

    st.markdown("---")

    # Run button
    run_btn = st.button(" Run Screening", use_container_width=True)


# HEADER

st.markdown('<div class="main-header"> AI Resume Screener</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Powered by Groq · LangChain · Pinecone</div>', unsafe_allow_html=True)

# Quick stats bar
col_a, col_b, col_c, col_d = st.columns(4)
col_a.metric("Resumes uploaded", len(uploaded_files) if uploaded_files else 0)
col_b.metric("JD characters",    len(st.session_state.jd_text))
col_c.metric("Results ready",    len(st.session_state.results))
col_d.metric("Top match",
             f"{st.session_state.results[0]['match_score']}%" if st.session_state.results else "—")

st.markdown("---")


# RUN SCREENING

if run_btn:
    if not st.session_state.jd_text.strip():
        st.error("Please enter or upload a job description first.")
        st.stop()

    if not uploaded_files:
        st.error("Please upload at least one resume.")
        st.stop()

    with st.spinner("Parsing resumes and running AI analysis…"):

        # Parse all resumes
        parsed_resumes = []
        progress = st.progress(0, text="Parsing resumes…")

        for i, f in enumerate(uploaded_files):
            text = extract_resume_text(f)
            parsed_resumes.append({"name": f.name, "text": text})
            progress.progress((i + 1) / len(uploaded_files), text=f"Parsed: {f.name}")

        # Optionally embed into Pinecone
        if use_pinecone:
            progress.progress(0, text="Connecting to Pinecone…")
            try:
                index = get_or_create_index()
                st.session_state.index = index
                for i, resume in enumerate(parsed_resumes):
                    chunks = chunk_text(resume["text"], metadata={"resume_name": resume["name"]})
                    upsert_resume(resume["name"], chunks, index=index)
                    progress.progress((i + 1) / len(parsed_resumes), text=f"Embedded: {resume['name']}")
            except Exception as e:
                st.warning(f"Pinecone upsert skipped: {e}")

        # Score all resumes with GPT-4
        progress.progress(0, text="Scoring with LLaMa…")
        results = batch_score_resumes(parsed_resumes, st.session_state.jd_text)
        st.session_state.results = results
        st.session_state.screening_done = True
        progress.empty()

    st.success("Screening complete!")
    st.rerun()


# RESULTS DISPLAY

if st.session_state.screening_done and st.session_state.results:
    results = st.session_state.results

    # Tab layout
    tab1, tab2, tab3 = st.tabs(["Overview", "Candidate Detail", "Export"])


    # TAB 1: Overview
    with tab1:
        st.markdown('<div class="section-title">Candidate Rankings</div>', unsafe_allow_html=True)

        # ── Ranking table ─────────────────────────────────
        df = pd.DataFrame([{
            "Rank":           i + 1,
            "Resume":         r["name"],
            "Match Score":    r["match_score"],
            "Experience":     r.get("experience_match", "—"),
            "Education":      r.get("education_match", "—"),
            "Recommendation": r.get("recommendation", "—"),
        } for i, r in enumerate(results)])

        st.dataframe(
            df.style.background_gradient(subset=["Match Score"], cmap="RdYlGn"),
            use_container_width=True,
            hide_index=True,
        )

        st.markdown('<div class="section-title">Score Distribution</div>', unsafe_allow_html=True)

        col1, col2 = st.columns(2)

        # Bar chart
        with col1:
            fig_bar = px.bar(
                df, x="Resume", y="Match Score",
                color="Match Score",
                color_continuous_scale="RdYlGn",
                range_color=[0, 100],
                title="Match Score by Candidate",
                text="Match Score",
            )
            fig_bar.update_traces(texttemplate="%{text}%", textposition="outside")
            fig_bar.update_layout(
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)",
                coloraxis_showscale=False,
                height=350,
                margin=dict(t=40, b=20),
            )
            st.plotly_chart(fig_bar, use_container_width=True)

        # Radar chart for top candidate
        with col2:
            top = results[0]
            categories = ["Match Score", "Exp. Match", "Edu. Match", "Skills Found"]

            def strength_to_score(val):
                mapping = {"Strong": 90, "Moderate": 60, "Weak": 30,
                           "Not Required": 75, "Unknown": 0}
                if isinstance(val, int): return val
                return mapping.get(val, 50)

            values = [
                top.get("match_score", 0),
                strength_to_score(top.get("experience_match", 0)),
                strength_to_score(top.get("education_match", 0)),
                min(len(top.get("matching_skills", [])) * 10, 100),
            ]
            values += [values[0]]   # close the polygon
            cats   = categories + [categories[0]]

            fig_radar = go.Figure(go.Scatterpolar(
                r=values, theta=cats, fill="toself",
                fillcolor="rgba(79,70,229,0.2)",
                line=dict(color="#4F46E5", width=2),
            ))
            fig_radar.update_layout(
                polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
                showlegend=False,
                title=f"Top candidate: {top['name'][:30]}",
                height=350,
                margin=dict(t=50, b=20),
                paper_bgcolor="rgba(0,0,0,0)",
            )
            st.plotly_chart(fig_radar, use_container_width=True)


    # TAB 2: Candidate Detail
    with tab2:
        selected_name = st.selectbox(
            "Select a candidate to review",
            [r["name"] for r in results],
        )
        selected = next(r for r in results if r["name"] == selected_name)

        # Score badge
        score = selected.get("match_score", 0)
        badge_cls = "badge-green" if score >= 70 else ("badge-yellow" if score >= 45 else "badge-red")
        rec = selected.get("recommendation", "—")
        rec_cls = "badge-green" if rec == "Shortlist" else ("badge-yellow" if rec == "Consider" else "badge-red")

        col_s1, col_s2, col_s3, col_s4 = st.columns(4)
        col_s1.metric("Match score",     f"{score}%")
        col_s2.metric("Experience",      selected.get("experience_match", "—"))
        col_s3.metric("Education",       selected.get("education_match",  "—"))
        col_s4.metric("Recommendation",  rec)

        # Gauge chart
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=score,
            domain={"x": [0, 1], "y": [0, 1]},
            title={"text": "Match Score", "font": {"size": 16}},
            gauge={
                "axis":      {"range": [0, 100]},
                "bar":       {"color": "#4F46E5"},
                "steps": [
                    {"range": [0,  45], "color": "#FEE2E2"},
                    {"range": [45, 70], "color": "#FEF3C7"},
                    {"range": [70, 100],"color": "#D1FAE5"},
                ],
                "threshold": {"line": {"color": "#1F2937", "width": 3}, "value": 70},
            },
        ))
        fig_gauge.update_layout(height=250, margin=dict(t=30, b=10),
                                paper_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig_gauge, use_container_width=True)

        # Summary
        st.markdown('<div class="section-title">Summary</div>', unsafe_allow_html=True)
        st.info(selected.get("summary", "No summary available."))

        # Skills
        col_m, col_g = st.columns(2)
        with col_m:
            st.markdown('<div class="section-title"> Matching skills</div>', unsafe_allow_html=True)
            skills_html = " ".join(
                f'<span class="pill-green">{s}</span>'
                for s in selected.get("matching_skills", [])
            ) or "<i>None identified</i>"
            st.markdown(skills_html, unsafe_allow_html=True)

        with col_g:
            st.markdown('<div class="section-title"> Missing skills</div>', unsafe_allow_html=True)
            missing_html = " ".join(
                f'<span class="pill-red">{s}</span>'
                for s in selected.get("missing_skills", [])
            ) or "<i>None identified</i>"
            st.markdown(missing_html, unsafe_allow_html=True)

        # Reasons
        st.markdown('<div class="section-title">Key reasons</div>', unsafe_allow_html=True)
        for reason in selected.get("reasons", []):
            st.markdown(f"• {reason}")

        # Raw text toggle
        if show_raw:
            with st.expander("Raw resume text"):
                resume_obj = next((r for r in st.session_state.get("resumes", [])
                                   if r.get("name") == selected_name), None)
                if resume_obj:
                    st.text(resume_obj.get("text", "Not available"))


    # TAB 3: Export
    with tab3:
        st.markdown('<div class="section-title">Download results</div>', unsafe_allow_html=True)

        export_df = pd.DataFrame([{
            "Resume":             r["name"],
            "Match Score (%)":    r.get("match_score", 0),
            "Summary":            r.get("summary", ""),
            "Matching Skills":    ", ".join(r.get("matching_skills", [])),
            "Missing Skills":     ", ".join(r.get("missing_skills", [])),
            "Experience Match":   r.get("experience_match", ""),
            "Education Match":    r.get("education_match", ""),
            "Recommendation":     r.get("recommendation", ""),
            "Reasons":            " | ".join(r.get("reasons", [])),
        } for r in results])

        st.dataframe(export_df, use_container_width=True, hide_index=True)

        csv = export_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label=" Download CSV",
            data=csv,
            file_name="screening_results.csv",
            mime="text/csv",
            use_container_width=True,
        )

# Empty state
elif not st.session_state.screening_done:
    st.markdown("""
    <div style="text-align:center; padding: 3rem 2rem; color:#9CA3AF;">
        <div style="font-size:3rem;">📄</div>
        <div style="font-size:1.1rem; margin-top:0.5rem;">
            Upload resumes and a job description, then click <b>Run Screening</b>
        </div>
        <div style="font-size:0.9rem; margin-top:0.5rem;">
            Results will show match scores, skill gaps, and ranked candidates
        </div>
    </div>
    """, unsafe_allow_html=True)
