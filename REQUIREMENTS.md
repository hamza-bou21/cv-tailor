CV Tailor: Requirements (v1)
Status: draft for v1
Last updated: 2026-11-08
1. Overview
CV Tailor takes a master CV and a job posting, gives an honest assessment of how well the candidate fits, and on
request produces a tailored one-page CV and a cover letter, each in English and French, as PDFs built from LaTeX.
Every run is recorded in a local SQLite database.
2. Goals
1. Save time on tailoring applications without lowering the quality of each one.
2. Never invent experience: every claim in the output must trace back to the master CV.
3. Produce clean, one-page, clickable PDFs in two languages.
4. Keep a history of every job analyzed and every document generated.
5. Be a clear, well-structured portfolio project that can grow into a web or mobile app.
3. Non-goals for v1
1. Job scraping from URLs or job boards.
2. Automatic submission of applications.
3. Application status tracking (planned for v2).
4. Languages other than English and French.
5. User accounts or multi-user support.
4. User flow
1. The user pastes or uploads a job posting.
2. The tool shows a fit report: score, verdict, strengths, gaps, missing keywords.
3. If the score is below the threshold, the tool asks whether to continue anyway. The verdict never blocks the user.
4. The user reviews the proposed selection of CV content and edits it.
5. The user reviews the cover letter text and edits it.
6. The tool builds the PDFs, shows them, and offers downloads.
7. The run is saved to the history.
5. Functional requirements
Master CV
1. FR-1: The master CV is a structured file (YAML or JSON) containing every role, bullet, project, skill, education
entry and link. This is the only source of facts.
2. FR-2: Every text item has an English version and a French version. French versions are produced once by
translation, approved by the user, and stored in the master file so later runs reuse them unchanged.
3. FR-3: Every bullet and project has a unique id.
Job input
4. FR-4: The job posting can be pasted as text or loaded from a text file.
Fit analysis
5. FR-5: The tool returns a score from 0 to 100, a verdict (apply, stretch, skip), strengths, gaps, and covered and
missing keywords.
6. FR-6: Analysis output is structured data with a fixed schema, validated before use. Any id or skill that does not
exist in the master CV is discarded.
7. FR-7: The score threshold is configurable. Below it, the tool asks for confirmation before continuing. A flag skips
the prompt for scripted runs.
Honesty rules
8. FR-8: CV content is selected, ranked and trimmed from the master CV only. The model never writes new CV
bullets.
9. FR-9: The cover letter may not claim any item listed as a gap or any fact absent from the master CV.
10. FR-10: After the letter is written, a verification pass checks each claim against the master CV. Unsupported
claims trigger one rewrite and, if any remain, a visible warning.
Review step
11. FR-11: Before any PDF is built, the user sees the selected bullets, projects and skills in ranked order and can
add, remove or reorder them.
12. FR-12: The user sees and can edit the cover letter text before it is compiled.
Output
13. FR-13: The CV is compiled from LaTeX to PDF, in English and French, and is exactly one page. If it overflows, the
lowest-ranked items are dropped automatically until it fits, and the user is told what was dropped.
14. FR-14: The cover letter is compiled to a one-page PDF in English and French, naming the company and role
from the posting. The French letter is a faithful translation of the English one.
15. FR-15: Email, LinkedIn, GitHub and project links are clickable in all PDFs.
16. FR-16: The LaTeX source files are kept alongside the PDFs.
History
17. FR-17: Each run is stored in SQLite (see section 7).
18. FR-18: A history view lists past jobs with date, company, role, fit score and download links.
Interfaces
19. FR-19: Primary interface is a Streamlit app covering the full flow above.
20. FR-20: A command line interface offers the same core functions, mainly for testing.
6. Non-functional requirements
1. Privacy: the master CV, the database and generated files stay local and are excluded from Git. The repository
ships with a fake demo CV and a fake demo database.
2. Secrets: the API key comes from an environment variable or secrets file, never from code.
3. Reliability: model calls have error handling and retries. Streamlit state is kept in session state so edits are not
lost on rerun, and model calls are not repeated by accident.
4. Cost control: each analysis is stored so it is never paid for twice for the same job and master CV version.
5. Dependencies: a local TeX Live installation with the French language support is required. The README
documents this.
6. Testability: core logic can be tested without the interface, and with a mocked model.
7. Data model (SQLite)
Keep a schema version number in the database.
jobs: id, company, role_title, source_url, posting_text, date_added, status (default "draft", unused in v1), notes.
analyses: id, job_id, date, fit_score, verdict, strengths, gaps, model_used, threshold_overridden (yes or no),
master_cv_hash.
documents: id, job_id, type (cv or letter), language, pdf_path, tex_path, date_generated, verification_warnings.
Store file paths, not PDF contents.
8. Architecture principles
1. Core library holds the logic: analysis, selection, translation, rendering, verification and database access.
2. Streamlit and the command line are thin layers that only call the core.
3. The review step is a function that returns a proposed selection and accepts an approved one, so any interface
can drive it.
4. Storage locations are configurable.
5. Model prompts live in their own files, not scattered in code.
9. Acceptance criteria
1. A real posting and a fake master CV produce four PDFs (CV and letter, English and French), each one page, with
working links.
2. A posting that clearly mismatches the master CV triggers the continue prompt, and continuing never adds skills
that are not in the master CV.
3. A deliberately planted false claim in a letter is caught by the verification pass.
4. Each run appears in the history view with correct data.
5. The repository clones and runs with the demo data and no personal information.
10. Roadmap
v1: Streamlit app and command line, analysis, review step, bilingual CV and letter, SQLite history.
v2: Application status tracking with a status event log, and a richer history view.
v3: Small API and a web page on top of the core.
Later: mobile friendly interface or phone app, if the web version proves useful.