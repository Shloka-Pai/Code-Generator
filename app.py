import html
import streamlit as st
from grounding import get_available_topics
from prompt_builder import build_prompt
from validator import extract_json, normalize_generated_io, validate_structure

try:
    from streamlit_ace import st_ace
    HAS_ACE = True
except ImportError:
    HAS_ACE = False

st.set_page_config(page_title="DSA Lab", page_icon="⚡", layout="wide",
                   initial_sidebar_state="expanded")

# ── Styling ───────────────────────────────────────────────────────────────────
st.markdown("""
<style>
:root{--bg:#0b1020;--side:#080c18;--panel:#131a2e;--line:#242c45;--card:#1b2340;
      --muted:#94a3b8;--blue:#2563eb;--green:#1f6f3f;}
header[data-testid="stHeader"]{display:none;}
.block-container{padding:0.8rem 1rem 0 1rem;max-width:100%;}
section[data-testid="stSidebar"]{background:var(--side);border-right:1px solid var(--line);}
section[data-testid="stSidebar"] .block-container{padding:1rem 0.8rem;}

div[data-testid="stVerticalBlockBorderWrapper"]{background:var(--panel);
    border:1px solid var(--line)!important;border-radius:10px;}
.panel-title{font-weight:600;font-size:15px;color:#e5e7eb;margin:0 0 10px 0;
    padding-bottom:8px;border-bottom:1px solid var(--line);}

.task-title{font-size:26px;font-weight:700;margin:6px 0 2px 0;}
.badge{display:inline-block;font-size:12px;padding:1px 8px;border-radius:6px;
    background:#3a3320;color:#e0b24a;border:1px solid #6b5a22;margin-left:8px;vertical-align:middle;}
.meta{color:var(--muted);font-size:13px;margin:4px 0;}
.meta b{color:#e5e7eb;} .ok{color:#34d399!important;}
.code-chip{background:#262626;border-radius:6px;padding:12px;font-family:monospace;font-size:13px;}

.success{background:var(--green);border-radius:10px;padding:16px 18px;margin-bottom:14px;}
.success h3{margin:0 0 6px 0;font-size:22px;}
.success p{margin:0;font-size:12px;}
.section-h{font-weight:600;margin:14px 0 8px 0;}
details{background:var(--card)!important;border:1px solid var(--line)!important;
    border-radius:8px!important;margin-bottom:8px;}

.stButton>button{background:var(--card);border:1px solid var(--line);color:#e5e7eb;border-radius:8px;}
.stButton>button[kind="primary"]{background:var(--blue);border-color:var(--blue);}

.nav-h{color:#64748b;font-size:11px;margin:14px 0 6px 4px;}
</style>
""", unsafe_allow_html=True)

# ── Session state ─────────────────────────────────────────────────────────────
DEFAULT_CODE = """# Your solution here
def solve():
    pass
"""

st.session_state.setdefault("code", DEFAULT_CODE)
st.session_state.setdefault("results", None)
st.session_state.setdefault("problem", None)
st.session_state.setdefault("generation_stats", None)


def run_tests(problem: dict, code: str, language: str):
    """Run the user's code against test cases (Python only)."""
    if language != "Python":
        return None  # Not supported
    from validator import run_test_cases
    temp = dict(problem)
    temp["reference_solution"] = code
    return run_test_cases(temp, language)


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ⚡ DSA Lab")
    st.markdown("---")

    st.markdown('<div class="nav-h">QUESTION SETUP</div>', unsafe_allow_html=True)

    topic = st.selectbox("Topic", get_available_topics())
    difficulty = st.selectbox("Difficulty", ["Easy", "Medium", "Hard"])
    language = "Python"
    test_case_count = st.slider("Test Cases", min_value=1, max_value=5, value=3)
    include_edge = st.checkbox("Include Edge Cases", value=True)

    st.markdown("---")
    generate_btn = st.button("⚡ Generate Problem", type="primary", use_container_width=True)

    if generate_btn:
        with st.spinner("Generating..."):
            try:
                from model import generate_problem
                messages = build_prompt(
                    topic=topic, difficulty=difficulty, language=language,
                    num_test_cases=test_case_count, include_edge_cases=include_edge,
                )
                raw_output, elapsed, token_count = generate_problem(messages)
                prob = extract_json(raw_output)
                if prob is None:
                    st.error(
                        "The model returned invalid JSON. Please try generating the problem again."
                    )
                else:
                    prob = normalize_generated_io(prob)
                    test_cases = prob.get("test_cases")
                    if isinstance(test_cases, list) and len(test_cases) > test_case_count:
                        prob["test_cases"] = test_cases[:test_case_count]
                    # For Python, recompute expected outputs by running the
                    # reference solution — fixes LLM-hallucinated expected values
                    if language == "Python":
                        from validator import recompute_expected_outputs
                        prob, recompute_warns = recompute_expected_outputs(prob)
                    else:
                        recompute_warns = []
                    warnings = validate_structure(prob, language, test_case_count)
                    hard = [w for w in warnings if "Missing" in w or "must be" in w or "Unsupported" in w or "Unexpected" in w]
                    soft = [w for w in warnings if w not in hard] + recompute_warns
                    if hard:
                        st.error("The generated problem is incomplete or malformed: " + " ".join(hard))
                    else:
                        prob["language"] = language
                        st.session_state.problem = prob
                        st.session_state.code = prob.get("reference_solution", "")
                        st.session_state.results = None
                        st.session_state.generation_stats = (elapsed, token_count)
                        if soft:
                            st.warning("Note: " + " ".join(soft))
            except RuntimeError as e:
                st.error(f"Model error: {e}")
            except Exception as e:
                st.error(f"Error: {e}")

    # Model info at bottom
    st.markdown("---")
    try:
        from model import get_model_info
        info = get_model_info()
        st.caption(f"{'🟢' if info['loaded'] else '🔴'} {info['model_id'].split('/')[-1]}")
        st.caption(f"Device: {info['device']}")
    except Exception:
        pass

# ── Top bar ───────────────────────────────────────────────────────────────────
problem = st.session_state.problem
problem_title = problem.get("title", "No problem generated") if problem else "No problem generated"

c_bc, c_regen = st.columns([9, 1])
c_bc.caption(f"DSA Lab  ›  **{html.escape(str(problem_title))}**")
if c_regen.button("↺ Regenerate", use_container_width=True):
    st.session_state.problem = None
    st.session_state.results = None
    st.rerun()

# ── Three columns ─────────────────────────────────────────────────────────────
H = 720
col_info, col_edit, col_res = st.columns([1.15, 1.45, 0.9], gap="small")

# ── Left: Problem Description ─────────────────────────────────────────────────
with col_info:
    with st.container(border=True, height=H):
        st.markdown('<div class="panel-title">ⓘ Problem Description</div>', unsafe_allow_html=True)

        if not problem:
            st.info("Configure settings in the sidebar and click ⚡ Generate Problem.")
        else:
            diff = problem.get("difficulty", difficulty)
            st.markdown(
                f'<div class="task-title">{html.escape(str(problem.get("title","Untitled")))}'
                f'<span class="badge">{html.escape(diff)}</span></div>'
                f'<div class="meta">Topic: <b>{html.escape(str(problem.get("topic", topic)))}</b></div>'
                '<hr style="border-color:#242c45">',
                unsafe_allow_html=True,
            )

            st.write(problem.get("problem_statement", ""))

            st.markdown("**Input Format**")
            st.write(problem.get("input_format", ""))

            st.markdown("**Output Format**")
            st.write(problem.get("output_format", ""))

            constraints = problem.get("constraints", [])
            if constraints:
                st.markdown("**Constraints**")
                for c in (constraints if isinstance(constraints, list) else [constraints]):
                    st.write(f"- {c}")

            examples = problem.get("examples", [])
            if examples:
                st.markdown("**Example**")
                ex = examples[0]
                st.markdown(
                    f'<div class="code-chip">Input: {html.escape(str(ex.get("input","")))}'
                    f'<br>Output: {html.escape(str(ex.get("output","")))}',
                    unsafe_allow_html=True,
                )
                if ex.get("explanation"):
                    st.caption(ex["explanation"])

            st.markdown("**Complexity**")
            st.write(f"⏱ Time: {problem.get('time_complexity','N/A')}")
            st.write(f"💾 Space: {problem.get('space_complexity','N/A')}")

# ── Center: Code Editor ───────────────────────────────────────────────────────
with col_edit:
    with st.container(border=True, height=H):
        st.markdown('<div class="panel-title">&lt;/&gt; Editor</div>', unsafe_allow_html=True)

        if not problem:
            st.info("Generate a problem to load the reference solution.")
        else:
            editor_language = problem.get("language", language)
            st.caption(f"Solution · {editor_language}")
            ace_mode = {"Java": "java", "Python": "python", "C++": "c_cpp"}.get(editor_language, "text")

            if HAS_ACE:
                edited_code = st_ace(
                    value=st.session_state.code, language=ace_mode,
                    theme="tomorrow_night", font_size=14, height=480,
                    show_gutter=True, auto_update=True, key="ace",
                )
                if edited_code is not None:
                    st.session_state.code = edited_code
            else:
                st.session_state.code = st.text_area(
                    "code", st.session_state.code, height=480,
                    label_visibility="collapsed",
                )

            b1, b2, b3 = st.columns([1, 1, 1])
            if b1.button("▷ Run", use_container_width=True):
                if editor_language != "Python":
                    st.warning(f"Auto-run only supports Python. Got: {editor_language}")
                else:
                    with st.spinner("Running..."):
                        st.session_state.results = run_tests(problem, st.session_state.code, editor_language)

            if b2.button("↺ Reset", use_container_width=True):
                st.session_state.code = problem.get("reference_solution", "")
                st.rerun()

            if b3.button("✕ Clear", use_container_width=True):
                st.session_state.code = ""
                st.rerun()

            st.markdown("**Explanation**")
            st.write(problem.get("solution_explanation", ""))

# ── Right: Results ────────────────────────────────────────────────────────────
with col_res:
    with st.container(border=True, height=H):
        st.markdown('<div class="panel-title">☑ Results</div>', unsafe_allow_html=True)

        if not problem:
            st.info("Results appear here after running.")
        else:
            # Generation stats
            stats = st.session_state.generation_stats
            if stats:
                elapsed, tokens = stats
                s1, s2 = st.columns(2)
                s1.metric("Gen Time", f"{elapsed:.1f}s")
                s2.metric("Tokens", tokens)

            res = st.session_state.results
            if res is None:
                st.info("Click ▷ Run to execute your solution.")
            else:
                passed = sum(1 for r in res if r.get("passed", False))
                total = sum(1 for r in res if "passed" in r)
                if total > 0:
                    if passed == total:
                        st.markdown(
                            f'<div class="success"><h3>✓ {passed}/{total} Tests Passed</h3>'
                            f'<p>All test cases passed.</p></div>',
                            unsafe_allow_html=True,
                        )
                    else:
                        st.error(f"✗ {passed}/{total} tests passed.")

                for r in res:
                    if "note" in r:
                        st.info(r["note"])
                        continue
                    icon = "✅" if r["passed"] else "❌"
                    with st.expander(f"{icon} Test {r['test_case']}"):
                        st.code(
                            f"Input:    {r['input']}\n"
                            f"Expected: {r['expected']}\n"
                            f"Got:      {r['got']}",
                            language="text",
                        )
                        if r.get("error"):
                            st.error(r["error"])

            # Test cases preview
            st.markdown('<div class="section-h">Test Cases</div>', unsafe_allow_html=True)
            test_cases = problem.get("test_cases", [])
            for i, tc in enumerate(test_cases if isinstance(test_cases, list) else []):
                with st.expander(f"Case {i+1}"):
                    st.code(
                        f"Input:    {tc.get('input','')}\n"
                        f"Expected: {tc.get('expected_output','')}",
                        language="text",
                    )

            with st.expander("{ } Raw JSON"):
                st.json(problem)
