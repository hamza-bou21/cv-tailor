import os
import re
import glob
import sqlite3
import time
import streamlit as st
import pandas as pd
import yaml
from core import (
    load_master_cv,
    analyze_job,
    select_cv_content,
    generate_cover_letter,
    extract_text_from_pdf,
    parse_pdf_to_master_cv,
    save_master_cv
)
from models import MasterCV
from latex_builder import compile_to_single_page, compile_letter_pdf
from db import init_db, log_document, delete_document_record, clear_all_history


def render_download_button(pdf_bytes: bytes, filename: str, label: str):
    """Native Streamlit download button (robust across versions, no HTML/base64 bloat)."""
    st.download_button(
        label=label,
        data=pdf_bytes,
        file_name=filename,
        mime="application/pdf",
        use_container_width=True,
        type="primary",
    )


@st.cache_data(show_spinner=False)
def get_demo_posting() -> str:
    """Cached read of the demo job posting so we don't hit disk on every rerun."""
    if os.path.exists("demo/posting.txt"):
        with open("demo/posting.txt", "r", encoding="utf-8") as f:
            return f.read()
    return ""


@st.cache_data(show_spinner=False)
def load_history_df() -> pd.DataFrame:
    """Cached read of the audit history table."""
    conn = sqlite3.connect("history.db")
    try:
        return pd.read_sql_query(
            "SELECT id, timestamp, doc_type, job_title, company, fit_score, "
            "verdict, language, tex_hash FROM generated_documents ORDER BY id DESC",
            conn,
        )
    finally:
        conn.close()


def invalidate_history_cache():
    load_history_df.clear()


st.set_page_config(
    page_title="CV Tailor — ATS Engine",
    page_icon="📄",
    layout="wide"
)

init_db()
PROFILES_DIR = "profiles"
os.makedirs(PROFILES_DIR, exist_ok=True)


# Helper to discover all profile YAML files
def get_available_profiles():
    files = glob.glob(f"{PROFILES_DIR}/*.yaml") + glob.glob(f"{PROFILES_DIR}/*.yml")
    if os.path.exists("demo/master_cv.yaml") and "demo/master_cv.yaml" not in files:
        files.insert(0, "demo/master_cv.yaml")
    if os.path.exists("master_cv.yaml") and "master_cv.yaml" not in files:
        files.insert(0, "master_cv.yaml")
    return files


st.title("📄 CV Tailor & Audit Engine")
st.caption("Deterministic CV tailoring, PDF-to-YAML parsing, and tamper-evident provenance.")

# Sidebar Configuration
st.sidebar.header("🔑 API Configuration")
api_key = st.sidebar.text_input(
    "Groq API Key:",
    type="password",
    placeholder="gsk_...",
    help="Pasted keys are held only in this session's memory and used per request. They are never written to disk.",
)
if api_key and api_key.strip():
    st.sidebar.success("API key provided.")
else:
    st.sidebar.info("A Groq API key is required to analyze or generate documents.")

st.sidebar.divider()
st.sidebar.header("Active Profile")
available_profiles = get_available_profiles()

if not available_profiles:
    st.sidebar.warning("No Master CV YAML found. Upload a PDF in the 'Sources' tab.")
    selected_profile_path = None
    master_cv = None
else:
    selected_profile_path = st.sidebar.selectbox("Select CV Profile:", available_profiles)
    try:
        master_cv = load_master_cv(selected_profile_path)
        st.sidebar.success(f"Active: **{master_cv.personal_info.name}**")
        # Reset in-flight results when the active profile switches
        if st.session_state.get("_active_profile") != selected_profile_path:
            st.session_state["_active_profile"] = selected_profile_path
            st.session_state.generated_cv = None
            st.session_state.generated_letter = None
            st.session_state.last_analysis = None
    except Exception as e:
        st.sidebar.error(f"Error loading profile: {e}")
        master_cv = None

language = st.sidebar.selectbox(
    "Target Language:",
    options=["en", "fr"],
    format_func=lambda x: "English (EN)" if x == "en" else "Français (FR)"
)
generate_letter_flag = st.sidebar.checkbox("Generate Tailored Cover Letter", value=True)
force_generate = st.sidebar.checkbox(
    "Force Generation on 'SKIP'",
    value=False,
    help="Compile CV and cover letter even if fit score is below 50."
)

# Main Navigation Tabs
tab_tailor, tab_sources, tab_history = st.tabs([
    "🎯 Job Tailoring",
    "📁 Sources & Profile Manager",
    "📜 Provenance & Audit History"
])

# ==========================================
# TAB 1: TAILORING WORKFLOW
# ==========================================
with tab_tailor:
    if not master_cv:
        st.info("Select or import a Master CV in the **Sources & Profile Manager** tab to begin.")
    else:
        # Initialize session state for generated outputs
        if "generated_cv" not in st.session_state:
            st.session_state.generated_cv = None
        if "generated_letter" not in st.session_state:
            st.session_state.generated_letter = None
        if "last_analysis" not in st.session_state:
            st.session_state.last_analysis = None

        col_input, col_output = st.columns([1, 1], gap="medium")

        with col_input:
            st.subheader("1. Target Job Description")
            job_title_input = st.text_input("Role Title (optional):", value="Senior Backend Engineer")
            company_input = st.text_input("Company Name (optional):", value="CloudNext")

            default_job_text = get_demo_posting()

            job_text = st.text_area("Paste Job Posting Text:", value=default_job_text, height=350)
            run_button = st.button("🚀 Analyze & Generate Documents", type="primary", use_container_width=True)

            if run_button:
                if not api_key or not api_key.strip():
                    st.error("Please enter your Groq API key in the sidebar before analyzing.")
                elif not job_text.strip():
                    st.error("Please provide a job description.")
                else:
                    try:
                        with st.spinner("Analyzing job fit against Master CV..."):
                            analysis = analyze_job(job_text, master_cv, api_key=api_key)
                            st.session_state.last_analysis = analysis

                        # --- Check Skip Gatekeeper with Force Override ---
                        if analysis.verdict == "skip" and not force_generate:
                            st.warning("Analysis verdict is 'SKIP'. Halting generation. (Check 'Force Generation on SKIP' in the sidebar to override.)")
                        else:
                            if analysis.verdict == "skip" and force_generate:
                                st.info("⚠️ Fit verdict is 'SKIP', but proceeding due to manual override.")

                            # Only clear previous outputs once we're actually generating
                            st.session_state.generated_cv = None
                            st.session_state.generated_letter = None

                            with st.spinner("Selecting & ranking CV items deterministically..."):
                                selection = select_cv_content(job_text, master_cv, api_key=api_key, language=language)

                            with st.spinner("Compiling LaTeX to single-page PDF..."):
                                run_tag = time.time_ns()
                                cv_job_name = f"tailored_cv_{run_tag}"

                                cv_pdf = compile_to_single_page(selection, master_cv, output_dir="output", job_name=cv_job_name)
                                cv_tex = os.path.join("output", f"{cv_job_name}.tex")

                                cv_id = log_document(
                                    job_title=selection.target_role or job_title_input,
                                    company=company_input,
                                    fit_score=analysis.fit_score,
                                    verdict=analysis.verdict,
                                    pdf_path=cv_pdf,
                                    tex_path=cv_tex,
                                    language=language,
                                    doc_type="cv"
                                )

                                with open(cv_pdf, "rb") as f:
                                    st.session_state.generated_cv = {
                                        "bytes": f.read(),
                                        "filename": f"CV_{master_cv.personal_info.name.replace(' ', '_')}_{language.upper()}.pdf",
                                        "audit_id": cv_id
                                    }

                            if generate_letter_flag:
                                with st.spinner("Writing and verifying cited cover letter..."):
                                    letter = generate_cover_letter(job_text, master_cv, api_key=api_key, language=language)
                                    letter_job_name = f"cover_letter_{run_tag}"
                                    letter_pdf = compile_letter_pdf(letter, master_cv, output_dir="output", job_name=letter_job_name)
                                    letter_tex = os.path.join("output", f"{letter_job_name}.tex")

                                    warnings_str = ", ".join(letter.verification_warnings) if letter.verification_warnings else None
                                    letter_id = log_document(
                                        job_title=letter.target_role or job_title_input,
                                        company=letter.company or company_input,
                                        fit_score=analysis.fit_score,
                                        verdict=analysis.verdict,
                                        pdf_path=letter_pdf,
                                        tex_path=letter_tex,
                                        language=language,
                                        doc_type="cover_letter",
                                        warnings=warnings_str
                                    )

                                    with open(letter_pdf, "rb") as f:
                                        st.session_state.generated_letter = {
                                            "bytes": f.read(),
                                            "filename": f"CoverLetter_{master_cv.personal_info.name.replace(' ', '_')}_{language.upper()}.pdf",
                                            "audit_id": letter_id,
                                            "passed": letter.verification_passed,
                                            "warnings": letter.verification_warnings
                                        }
                    except Exception as e:
                        st.error(f"Generation failed: {e}")
                        st.exception(e)
                    finally:
                        invalidate_history_cache()

        with col_output:
            st.subheader("2. Results & Provenance")

            # Render analysis if present in session state
            if st.session_state.last_analysis:
                analysis = st.session_state.last_analysis
                verdict_colors = {"apply": "green", "stretch": "orange", "skip": "red"}
                color = verdict_colors.get(analysis.verdict, "blue")
                st.markdown(f"### Fit Score: :{color}[{analysis.fit_score}/100 — {analysis.verdict.upper()}]")

                col_str, col_gap = st.columns(2)
                with col_str:
                    st.markdown("**Identified Strengths:**")
                    for s in analysis.strengths:
                        st.markdown(f"- {s}")
                with col_gap:
                    st.markdown("**Identified Gaps:**")
                    for g in analysis.gaps:
                        st.markdown(f"- {g}")

            # Persistent CV download button
            if st.session_state.generated_cv:
                cv_data = st.session_state.generated_cv
                st.success(f"CV compiled successfully (Audit ID: {cv_data['audit_id']})")
                render_download_button(
                    pdf_bytes=cv_data["bytes"],
                    filename=cv_data["filename"],
                    label="⬇️ Download Tailored CV (PDF)"
                )

            # Persistent Cover Letter download button
            if st.session_state.generated_letter:
                let_data = st.session_state.generated_letter
                if let_data["passed"]:
                    st.info("Cover letter passed deterministic honesty verification.")
                elif let_data["warnings"]:
                    st.warning(f"Verification warnings: {', '.join(let_data['warnings'])}")

                st.success(f"Cover letter compiled successfully (Audit ID: {let_data['audit_id']})")
                render_download_button(
                    pdf_bytes=let_data["bytes"],
                    filename=let_data["filename"],
                    label="⬇️ Download Tailored Cover Letter (PDF)"
                )

# ==========================================
# TAB 2: SOURCES & PROFILE MANAGER
# ==========================================
with tab_sources:
    col_import, col_manage = st.columns([1, 1], gap="large")

    with col_import:
        st.subheader("Import CV from PDF")
        st.caption("Upload an existing PDF CV. The LLM extracts the structure, creates ID slugs, and produces bilingual YAML.")

        uploaded_pdf = st.file_uploader("Select PDF Resume:", type=["pdf"])
        new_profile_slug = st.text_input("Target Profile Name (e.g. john_doe):", value="")

        if st.button("⚙️ Extract & Generate YAML Profile", type="primary", use_container_width=True):
            if not api_key or not api_key.strip():
                st.error("Please enter your Groq API key in the sidebar before extracting a profile.")
            elif not uploaded_pdf:
                st.error("Please upload a PDF file first.")
            elif not new_profile_slug.strip():
                st.error("Please enter a profile name.")
            else:
                # Sanitize the slug to prevent path traversal and invalid filenames
                slug = re.sub(r"[^a-z0-9_-]", "", new_profile_slug.strip().lower().replace(" ", "_"))
                if not slug:
                    st.error("Profile name must contain letters or numbers.")
                else:
                    try:
                        with st.spinner("Extracting text and generating structured YAML schema..."):
                            pdf_bytes = uploaded_pdf.read()
                            extracted_raw_text = extract_text_from_pdf(pdf_bytes)

                            if len(extracted_raw_text) < 50:
                                st.error("Could not extract sufficient text from PDF. Ensure it is not a scanned image.")
                            else:
                                parsed_cv = parse_pdf_to_master_cv(extracted_raw_text, api_key=api_key)
                                clean_filename = f"{slug}.yaml"
                                save_path = os.path.join(PROFILES_DIR, clean_filename)
                                save_master_cv(parsed_cv, save_path)
                                st.success(f"Saved new Master CV to: `{save_path}`")
                                st.rerun()
                    except Exception as err:
                        st.error(f"Import failed: {err}")
                        st.exception(err)

    with col_manage:
        st.subheader("Manage Saved YAML Profiles")
        if not available_profiles:
            st.write("No profiles available to manage.")
        else:
            profile_to_manage = st.selectbox("Select Profile to View / Delete:", available_profiles, key="manage_select")

            with open(profile_to_manage, "r", encoding="utf-8") as f:
                yaml_content = f.read()

            with st.expander("Inspect / Edit Raw YAML", expanded=False):
                edited_yaml = st.text_area("YAML Definition:", value=yaml_content, height=300)
                if st.button("💾 Save YAML Edits"):
                    try:
                        # Validate before saving
                        yaml_dict = yaml.safe_load(edited_yaml)
                        _ = MasterCV(**yaml_dict)
                        tmp_path = f"{profile_to_manage}.tmp"
                        with open(tmp_path, "w", encoding="utf-8") as f:
                            f.write(edited_yaml)
                        os.replace(tmp_path, profile_to_manage)  # atomic swap
                        st.success("Profile saved successfully!")
                        st.cache_data.clear()
                        st.rerun()
                    except Exception as err:
                        st.error(f"Validation Error: {err}")

            st.write("---")
            st.markdown(f"**Delete Profile:** `{profile_to_manage}`")
            if profile_to_manage == "demo/master_cv.yaml":
                st.caption("⚠️ The demo profile cannot be deleted.")
            else:
                if st.button("🗑️ Delete Selected Profile", type="secondary"):
                    try:
                        os.remove(profile_to_manage)
                        st.success(f"Deleted `{profile_to_manage}`")
                        st.rerun()
                    except Exception as err:
                        st.error(f"Failed to delete file: {err}")

# ==========================================
# TAB 3: AUDIT TRAIL & HISTORY
# ==========================================
with tab_history:
    st.subheader("Audit Trail & Provenance History")
    st.caption("All generated artifacts tracked with SHA-256 hashes and timestamp provenance.")

    df = load_history_df()

    if df.empty:
        st.write("No document generation history recorded yet.")
    else:
        st.dataframe(
            df,
            use_container_width=True,
            column_config={
                "tex_hash": st.column_config.TextColumn("TeX Hash (SHA-256)", help="Tamper-evident verification hash"),
                "fit_score": st.column_config.ProgressColumn("Fit Score", min_value=0, max_value=100, format="%d"),
            }
        )

        col_del_single, col_del_all = st.columns([1, 1])
        with col_del_single:
            doc_id_to_del = st.selectbox("Select Record ID to Delete:", df["id"].tolist())
            if st.button("🗑️ Delete Selected Record & Files"):
                if delete_document_record(doc_id_to_del, remove_files=True):
                    invalidate_history_cache()
                    st.success(f"Deleted record ID {doc_id_to_del} and associated files.")
                    st.rerun()

        with col_del_all:
            st.write("&nbsp;")
            confirm_purge = st.checkbox("I understand this permanently deletes all history & files.")
            if st.button("⚠️ Purge All History & Generated Files", type="secondary", disabled=not confirm_purge):
                clear_all_history(remove_files=True)
                invalidate_history_cache()
                st.success("All history and output artifacts cleared.")
                st.rerun()