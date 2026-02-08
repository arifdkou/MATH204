# streamlit_app_math204_week2.py
# OSTİM Technical University • Faculty of Engineering
# MATH 204 – Probability & Statistics • Week 2
# Interactive lecture + visuals + 10 solved examples + Week 2 Homework Module

import time
import math
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle

from PIL import Image

logo = Image.open("ostim_logo.png")  # aynı klasördeyse
st.image(logo, width=220)



# -----------------------------
# Page Config
# -----------------------------
st.set_page_config(
    page_title="MATH 204 • Week 2 • Conditional Probability (Interactive)",
    layout="wide"
)

# -----------------------------
# Utilities
# -----------------------------
def animate_progress(label="Computing...", steps=30, sleep_s=0.03):
    ph = st.empty()
    bar = st.progress(0)
    for i in range(steps + 1):
        ph.write(f"{label} {int(100*i/steps)}%")
        bar.progress(i / steps)
        time.sleep(sleep_s)
    ph.empty()

def clamp01(x: float) -> float:
    return max(0.0, min(1.0, x))

def approx_equal(a, b, tol=1e-3):
    return abs(a - b) <= tol

def venn_plot():
    """Conceptual Venn diagram: Omega rectangle + two circles A and B."""
    fig, ax = plt.subplots(figsize=(6.2, 3.8))
    ax.add_patch(Rectangle((0, 0), 10, 6, fill=False, linewidth=1.5))
    ax.add_patch(Circle((4, 3), 2.2, fill=False, linewidth=2.0))
    ax.add_patch(Circle((6, 3), 2.2, fill=False, linewidth=2.0))
    ax.text(9.5, 5.6, r"$\Omega$", ha="right", va="top")
    ax.text(3.1, 5.0, r"$A$", ha="center")
    ax.text(6.9, 5.0, r"$B$", ha="center")
    ax.text(5.0, 3.0, r"$A\cap B$", ha="center")
    ax.set_xlim(-0.5, 10.5)
    ax.set_ylim(-0.5, 6.5)
    ax.set_aspect("equal", adjustable="box")
    ax.axis("off")
    return fig

def reliability_series(Rs):
    p = 1.0
    for r in Rs:
        p *= r
    return p

def reliability_parallel(Rs):
    q = 1.0
    for r in Rs:
        q *= (1.0 - r)
    return 1.0 - q

def styled_box(text, kind="info"):
    if kind == "info":
        st.info(text)
    elif kind == "warning":
        st.warning(text)
    elif kind == "success":
        st.success(text)
    else:
        st.write(text)

# -----------------------------
# Header (Top of Page)
# -----------------------------
st.markdown("""
### OSTİM Technical University  
### Faculty of Engineering  
### **MATH 204 – Probability and Statistics**  
### **Week 2 – Interactive Lecture on Conditional Probability**
---
This Streamlit application is an **interactive lecture presentation** for **Week 2**, prepared for engineering students.
It combines **concept explanations**, **LaTeX formulas**, **visualizations**, **animations**, **solved examples**, and a **homework module**.

""")

st.markdown("""
## How to Use This App
- Use the **sidebar** to navigate between sections (concepts, rules, tools, examples, homework).
- Change values in **sliders / input boxes** to see probabilities update instantly.
- Use **animations** to build intuition (chain rule, Monte Carlo convergence).
- Review **10 fully solved engineering problems** (software, electronics, AI, aerospace, mechanical).
- Complete the **Week 2 Homework Module** to practice and self-check your solutions.

""")

st.markdown("""
## Learning Outcomes – Week 2
After completing this interactive lecture, students will be able to:
1. Define conditional probability and interpret conditioning as a reduced sample space.
2. Apply the conditional probability definition: \\( P(A\\mid B)=\\frac{P(A\\cap B)}{P(B)} \\) for \\(P(B)>0\\).
3. Explain and use the **axioms of conditional probability**.
4. Apply the **product rule** and **chain rule** to compute joint probabilities in sequential systems.
5. Use key properties (complement, bounds, monotonicity) under conditioning.
6. Test **independence** using \\(P(A\\mid B)=P(A)\\) and \\(P(A\\cap B)=P(A)P(B)\\).
7. Solve engineering-oriented problems across multiple disciplines using Week 2 tools.

""")

st.markdown("""
## Engineering Coverage of the Examples
The examples and homework questions are designed to include contexts from:
- **Software Engineering** (incident analysis, CI/CD pipelines)  
- **Electronics Engineering** (noise conditioning, board reliability)  
- **AI Engineering** (data filtering, model cascades)  
- **Aerospace Engineering** (launch decision chains, redundant avionics)  
- **Mechanical Engineering** (series systems, load-dependent failure)

---
""")

# -----------------------------
# Sidebar Navigation
# -----------------------------
st.sidebar.title("Week 2 Navigation")
section = st.sidebar.radio(
    "Go to",
    [
        "1) Concept & Definition",
        "2) Axioms of Conditional Probability",
        "3) Product Rule & Chain Rule",
        "4) Key Properties",
        "5) Interactive Visuals & Calculators",
        "6) 10 Solved Engineering Examples",
        "7) Week 2 Homework Module",
        "8) Mini Quiz (Self-check)"
    ]
)

st.sidebar.divider()
anim_speed = st.sidebar.slider("Animation speed", 1, 10, 6)
sleep_s = 0.06 / anim_speed
tolerance = st.sidebar.select_slider("Answer tolerance", options=[1e-1, 5e-2, 1e-2, 5e-3, 1e-3], value=1e-3)

# -----------------------------
# SECTION 1: Concept & Definition
# -----------------------------
if section == "1) Concept & Definition":
    st.markdown("## 1) Conditional Probability — Concept & Definition")

    st.markdown("""
**Idea:** If we learn that event \\(B\\) happened, the “world” is restricted to outcomes in \\(B\\).  
Then we measure how much of \\(B\\) is also inside \\(A\\).
""")

    st.markdown("### Definition")
    st.latex(r"P(A\mid B)=\frac{P(A\cap B)}{P(B)} \quad \text{for } P(B)>0.")

    st.markdown("### Interpretation")
    st.markdown("""
- \\(B\\) is the **observed evidence** (e.g., a test passed, a sensor state, a fault indicator).
- \\(P(A\\mid B)\\) answers: **“Given what we observed, how likely is event A?”**
""")

    st.markdown("### Visual intuition (conceptual Venn diagram)")
    st.pyplot(venn_plot(), clear_figure=True)
    styled_box("Conditioning on B means: focus only on outcomes inside B; then compute the fraction that also belongs to A.", "info")

# -----------------------------
# SECTION 2: Axioms
# -----------------------------
elif section == "2) Axioms of Conditional Probability":
    st.markdown("## 2) Axioms of Conditional Probability (Proof Sketch)")

    st.markdown("""
Fix an event \\(B\\) with \\(P(B)>0\\). Define \\(Q(A)=P(A\\mid B)\\).  
We show \\(Q\\) satisfies the probability axioms.
""")

    st.markdown("### Axiom 1 — Non-negativity")
    st.latex(r"P(A\mid B)=\frac{P(A\cap B)}{P(B)}\ge 0 \quad (\text{since } P(A\cap B)\ge 0,\ P(B)>0).")

    st.markdown("### Axiom 2 — Normalization")
    st.latex(r"P(\Omega\mid B)=\frac{P(\Omega\cap B)}{P(B)}=\frac{P(B)}{P(B)}=1.")

    st.markdown("### Axiom 3 — Additivity for disjoint events")
    st.markdown("If \\(A_1\\cap A_2=\\varnothing\\), then \\((A_1\\cap B)\\cap(A_2\\cap B)=\\varnothing\\).")
    st.latex(r"""
P(A_1\cup A_2\mid B)
=\frac{P((A_1\cup A_2)\cap B)}{P(B)}
=\frac{P((A_1\cap B)\cup(A_2\cap B))}{P(B)}
=\frac{P(A_1\cap B)+P(A_2\cap B)}{P(B)}
=P(A_1\mid B)+P(A_2\mid B).
""")

    styled_box("Conclusion: Conditional probability behaves like a real probability measure on the reduced sample space B.", "success")

# -----------------------------
# SECTION 3: Product Rule & Chain Rule
# -----------------------------
elif section == "3) Product Rule & Chain Rule":
    st.markdown("## 3) Multiplicative (Product) Rule & Chain Rule")

    st.markdown("### Product rule (two events)")
    st.latex(r"P(A\cap B)=P(A\mid B)P(B)=P(B\mid A)P(A).")

    st.markdown("### Chain rule (three events)")
    st.latex(r"P(A\cap B\cap C)=P(A)\,P(B\mid A)\,P(C\mid A\cap B).")

    st.markdown("### Animation: cumulative probability in a multi-step pipeline")
    c1, c2, c3 = st.columns(3)
    with c1:
        p1 = st.number_input("P(A) — Step 1", 0.0, 1.0, 0.90, 0.01)
    with c2:
        p2 = st.number_input("P(B|A) — Step 2 given Step 1", 0.0, 1.0, 0.95, 0.01)
    with c3:
        p3 = st.number_input("P(C|A∩B) — Step 3 given Step 1&2", 0.0, 1.0, 0.98, 0.01)

    if st.button("Run chain-rule animation", type="primary"):
        animate_progress("Animating...", steps=25, sleep_s=sleep_s)
        vals = [p1, p1*p2, p1*p2*p3]
        labels = ["After Step 1", "After Step 2", "After Step 3"]
        ph = st.empty()
        for i in range(3):
            fig, ax = plt.subplots(figsize=(6.5, 3))
            ax.bar([labels[i]], [vals[i]])
            ax.set_ylim(0, 1)
            ax.set_ylabel("Probability")
            ax.set_title(f"Cumulative Probability = {vals[i]:.6f}")
            ph.pyplot(fig, clear_figure=True)
            time.sleep(max(0.05, sleep_s * 8))
        ph.empty()
        st.latex(r"P(A\cap B\cap C)=P(A)\,P(B\mid A)\,P(C\mid A\cap B).")

# -----------------------------
# SECTION 4: Properties
# -----------------------------
elif section == "4) Key Properties":
    st.markdown("## 4) Key Properties of Conditional Probability")

    st.markdown("### Complement")
    st.latex(r"P(A^c\mid B)=1-P(A\mid B).")

    st.markdown("### Bounds")
    st.latex(r"0\le P(A\mid B)\le 1.")

    st.markdown("### Monotonicity")
    st.latex(r"A\subseteq C\Rightarrow P(A\mid B)\le P(C\mid B).")

    st.markdown("### Independence test")
    st.latex(r"A\ \text{and}\ B\ \text{independent} \Longleftrightarrow P(A\mid B)=P(A) \ (P(B)>0).")
    st.latex(r"\text{Equivalently: } P(A\cap B)=P(A)P(B).")

    styled_box("Common pitfall: In general, P(A|B) ≠ P(B|A). Do not swap them.", "warning")

# -----------------------------
# SECTION 5: Interactive Visuals & Calculators
# -----------------------------
elif section == "5) Interactive Visuals & Calculators":
    st.markdown("## 5) Interactive Visuals & Calculators")

    left, right = st.columns(2)

    with left:
        st.markdown("### (A) Conditional Probability Calculator")
        pab = st.number_input("Enter P(A ∩ B)", 0.0, 1.0, 0.20, 0.01)
        pb = st.number_input("Enter P(B)  (must be > 0)", 1e-9, 1.0, 0.50, 0.01)
        st.latex(r"P(A\mid B)=\frac{P(A\cap B)}{P(B)}")
        st.write(f"**P(A|B) = {pab/pb:.6f}**")

        st.markdown("---")
        st.markdown("### (B) Independence Checker")
        pa = st.number_input("P(A)", 0.0, 1.0, 0.40, 0.01)
        pb2 = st.number_input("P(B)", 0.0, 1.0, 0.50, 0.01, key="pb2_ind")
        pab2 = st.number_input("P(A ∩ B)", 0.0, 1.0, 0.20, 0.01, key="pab2_ind")
        st.latex(r"\text{Independent if } P(A\cap B)=P(A)P(B).")
        st.write(f"Left: **{pab2:.6f}**  |  Right: **{(pa*pb2):.6f}**")
        st.write("**Independent?**", "YES ✅" if abs(pab2 - pa*pb2) < 1e-6 else "NO ❌")

    with right:
        st.markdown("### (C) Conceptual Venn Diagram")
        st.pyplot(venn_plot(), clear_figure=True)

        st.markdown("---")
        st.markdown("### (D) Reliability (Series vs Parallel) + Monte Carlo Animation")
        n = st.slider("Number of components", 2, 5, 2)
        Rs = []
        cols = st.columns(n)
        for i in range(n):
            with cols[i]:
                Rs.append(st.number_input(f"R{i+1}", 0.0, 1.0, 0.95, 0.01, key=f"rel_{i}"))

        rs = reliability_series(Rs)
        rp = reliability_parallel(Rs)
        st.write(f"**Series reliability:** {rs:.6f}")
        st.write(f"**Parallel reliability:** {rp:.6f}")

        if st.button("Animate Monte Carlo convergence"):
            trials = 2500
            rng = np.random.default_rng(11)
            series_ok = 0
            parallel_ok = 0
            series_hist = []
            parallel_hist = []
            ph = st.empty()

            for t in range(1, trials + 1):
                works = rng.random(n) < np.array(Rs)
                if works.all():
                    series_ok += 1
                if works.any():
                    parallel_ok += 1

                series_hist.append(series_ok / t)
                parallel_hist.append(parallel_ok / t)

                if t % 50 == 0:
                    fig, ax = plt.subplots(figsize=(6.5, 3.1))
                    ax.plot(series_hist, label="Series (simulated)")
                    ax.plot(parallel_hist, label="Parallel (simulated)")
                    ax.set_ylim(0, 1)
                    ax.set_xlabel("Trials")
                    ax.set_ylabel("Estimated Probability")
                    ax.set_title("Monte Carlo Convergence")
                    ax.legend()
                    ph.pyplot(fig, clear_figure=True)
                    time.sleep(max(0.01, sleep_s))

            ph.empty()
            styled_box(
                f"Done. Final estimates — Series: {series_hist[-1]:.4f}, Parallel: {parallel_hist[-1]:.4f}",
                "success"
            )

# -----------------------------
# SECTION 6: 10 Solved Engineering Examples (English + LaTeX)
# -----------------------------
elif section == "6) 10 Solved Engineering Examples":
    st.markdown("## 6) Ten Solved Engineering Examples (Week 2)")

    st.markdown("""
Each example demonstrates a Week 2 rule:
- Conditional probability definition
- Product rule / Chain rule
- Complement under conditioning
- Total probability (used as a natural extension; still consistent with Week 2 conditioning logic)
""")

    examples = [
        ("1) Software — Incident Analysis",
         r"""A microservices system shows:
- \(A\): database latency spike
- \(B\): API error rate exceeds 2%

Given \(P(A)=0.12\), \(P(B)=0.08\), \(P(A\cap B)=0.03\).  
Find \(P(A\mid B)\).""",
         r"""Use definition:
\[
P(A\mid B)=\frac{P(A\cap B)}{P(B)}=\frac{0.03}{0.08}=0.375.
\]
Interpretation: Given high API errors, DB latency spike probability is 37.5%."""),

        ("2) Software — CI/CD Pipeline (Chain Rule)",
         r"""A CI/CD pipeline has stages:
\(A\)=build succeeds, \(B\)=unit tests pass, \(C\)=integration tests pass.

Given \(P(A)=0.98\), \(P(B\mid A)=0.95\), \(P(C\mid A\cap B)=0.92\).
Find \(P(A\cap B\cap C)\).""",
         r"""Chain rule:
\[
P(A\cap B\cap C)=P(A)\,P(B\mid A)\,P(C\mid A\cap B)
=0.98\times0.95\times0.92=0.85652.
\]"""),

        ("3) Electronics — Series Reliability",
         r"""A board works only if both components work (independent):
ADC reliability 0.97, MCU reliability 0.95. Find board reliability.""",
         r"""Series:
\[
P(\text{board works})=0.97\times0.95=0.9215.
\]"""),

        ("4) Electronics — Conditioning on Noise",
         r"""Receiver events:
\(S\)=symbol decoded correctly, \(N\)=strong noise present.
Given \(P(N)=0.20\), \(P(S\mid N)=0.85\), \(P(S\mid N^c)=0.98\).
Find \(P(S)\) and \(P(S^c\mid N)\).""",
         r"""Total probability:
\[
P(S)=0.85(0.20)+0.98(0.80)=0.954.
\]
Complement:
\[
P(S^c\mid N)=1-0.85=0.15.
\]"""),

        ("5) AI — Bias Filtering",
         r"""Dataset events:
\(B\)=sample is biased, \(A\)=removed by bias filter.
Given \(P(B)=0.30\), \(P(A\mid B)=0.80\), \(P(A\mid B^c)=0.10\).
Find \(P(A)\) and \(P(A\cap B)\).""",
         r"""Total probability:
\[
P(A)=0.80(0.30)+0.10(0.70)=0.31.
\]
Product rule:
\[
P(A\cap B)=P(A\mid B)P(B)=0.80\times0.30=0.24.
\]"""),

        ("6) AI — Two-stage Model Cascade",
         r"""Two-stage AI pipeline:
\(D\)=object detected, \(C\)=correctly classified given detected.
Given \(P(D)=0.90\), \(P(C\mid D)=0.88\).
Find \(P(D\cap C)\).""",
         r"""Product rule:
\[
P(D\cap C)=P(D)P(C\mid D)=0.90\times0.88=0.792.
\]"""),

        ("7) Aerospace — Launch Decision Chain",
         r"""Launch requires:
\(A\)=weather go, \(B\)=range safety go given \(A\),
\(C\)=vehicle health go given \(A\cap B\).
Given \(P(A)=0.75\), \(P(B\mid A)=0.96\), \(P(C\mid A\cap B)=0.98\).
Find probability of launch.""",
         r"""Chain rule:
\[
P(A\cap B\cap C)=0.75\times0.96\times0.98=0.7056.
\]"""),

        ("8) Aerospace — Redundant Avionics (Parallel)",
         r"""Two independent flight computers:
\(P(F_1)=0.98\), \(P(F_2)=0.97\).
System works if at least one works. Find \(P(S)\).""",
         r"""Parallel:
\[
P(S)=1-P(F_1^c\cap F_2^c)=1-(0.02)(0.03)=0.9994.
\]"""),

        ("9) Mechanical — Transmission Path (Series)",
         r"""Independent components in series:
\(P(S)=0.99\), \(P(B)=0.97\), \(P(G)=0.96\).
Find \(P(S\cap B\cap G)\).""",
         r"""Series:
\[
P(S\cap B\cap G)=0.99\times0.97\times0.96=0.921888\approx0.9219.
\]"""),

        ("10) Mechanical — Failure Under High Load",
         r"""Events:
\(F\)=failure within 100 hours, \(L\)=high load.
Given \(P(L)=0.25\), \(P(F\mid L)=0.12\), \(P(F\mid L^c)=0.03\).
Find overall \(P(F)\).""",
         r"""Total probability:
\[
P(F)=0.12(0.25)+0.03(0.75)=0.03+0.0225=0.0525.
\]"""),
    ]

    for title, prob, sol in examples:
        with st.expander(title, expanded=False):
            st.markdown("### Problem")
            st.markdown(prob)
            st.markdown("### Solution (explained)")
            st.markdown(sol)

# -----------------------------
# SECTION 7: Homework Module
# -----------------------------
elif section == "7) Week 2 Homework Module":
    st.markdown("## 7) Week 2 Homework Module (Auto-Generated)")
    st.markdown("""
This module generates **new numeric values each time** (if you wish) and checks your answers.
- Enter your numeric answer(s).
- Click **Check** to get immediate feedback.
- You may reveal the full solution after trying.

**Tip:** Use at least 3 decimal places.
""")

    # Initialize RNG seed and generated homework in session state
    if "hw_seed" not in st.session_state:
        st.session_state.hw_seed = 20402
    if "hw_generated" not in st.session_state:
        st.session_state.hw_generated = False
    if "hw_data" not in st.session_state:
        st.session_state.hw_data = {}

    colA, colB = st.columns([1, 1])
    with colA:
        new_seed = st.number_input("Homework seed (change to generate a new version)", value=int(st.session_state.hw_seed), step=1)
    with colB:
        if st.button("Generate / Regenerate Homework", type="primary"):
            st.session_state.hw_seed = int(new_seed)
            st.session_state.hw_generated = False

    # Generate if needed
    if not st.session_state.hw_generated:
        rng = np.random.default_rng(int(st.session_state.hw_seed))

        # HW1: Conditional probability P(A|B)
        pb = float(rng.uniform(0.25, 0.80))
        pa = float(rng.uniform(0.15, min(0.90, 1.0 - (1.0 - pb)*0.1)))
        pab = float(rng.uniform(0.05, min(pa, pb, 0.45)))
        pab = min(pab, pa, pb)
        hw1_ans = pab / pb

        # HW2: Chain rule
        pA = float(rng.uniform(0.70, 0.99))
        pB_A = float(rng.uniform(0.70, 0.99))
        pC_AB = float(rng.uniform(0.70, 0.99))
        hw2_ans = pA * pB_A * pC_AB

        # HW3: Independence check (compute P(A∩B) if independent)
        pa3 = float(rng.uniform(0.10, 0.90))
        pb3 = float(rng.uniform(0.10, 0.90))
        hw3_ans = pa3 * pb3

        # HW4: Complement under conditioning
        pX_given_Y = float(rng.uniform(0.60, 0.99))
        hw4_ans = 1.0 - pX_given_Y

        # HW5: Reliability (series vs parallel) (3 components)
        R1 = float(rng.uniform(0.80, 0.99))
        R2 = float(rng.uniform(0.80, 0.99))
        R3 = float(rng.uniform(0.80, 0.99))
        hw5_series = R1 * R2 * R3
        hw5_parallel = 1.0 - (1.0 - R1) * (1.0 - R2) * (1.0 - R3)

        st.session_state.hw_data = {
            "hw1": {"pb": pb, "pab": pab, "ans": hw1_ans},
            "hw2": {"pA": pA, "pB_A": pB_A, "pC_AB": pC_AB, "ans": hw2_ans},
            "hw3": {"pa": pa3, "pb": pb3, "ans": hw3_ans},
            "hw4": {"p": pX_given_Y, "ans": hw4_ans},
            "hw5": {"R1": R1, "R2": R2, "R3": R3, "ans_series": hw5_series, "ans_parallel": hw5_parallel},
        }
        st.session_state.hw_generated = True

    hw = st.session_state.hw_data

    st.divider()
    st.markdown("### Homework Questions (Week 2)")

    # Score tracking
    if "hw_score" not in st.session_state:
        st.session_state.hw_score = 0
    if "hw_checked" not in st.session_state:
        st.session_state.hw_checked = {"hw1": False, "hw2": False, "hw3": False, "hw4": False, "hw5": False}

    # ---- HW1
    with st.expander("HW1 — Conditional Probability (Software Incident Context)", expanded=True):
        st.markdown(f"""
A software platform reports:
- \(A\): database latency spike  
- \(B\): API error rate exceeds a threshold  

Given:
\[
P(B)={hw["hw1"]["pb"]:.3f},\quad P(A\cap B)={hw["hw1"]["pab"]:.3f}
\]
Compute:
\[
P(A\mid B)=\frac{{P(A\cap B)}}{{P(B)}}.
\]
""")
        user = st.number_input("Your answer for P(A|B)", 0.0, 1.0, value=0.0, step=0.001, key="hw1_user")
        c1, c2, c3 = st.columns(3)
        with c1:
            if st.button("Check HW1", key="check_hw1"):
                st.session_state.hw_checked["hw1"] = True
                if approx_equal(user, hw["hw1"]["ans"], tol=tolerance):
                    st.success("Correct ✅")
                else:
                    st.error("Not correct yet ❌")
        with c2:
            st.write(f"**Tolerance:** ±{tolerance}")
        with c3:
            if st.button("Reveal HW1 Solution", key="reveal_hw1"):
                st.latex(r"P(A\mid B)=\frac{P(A\cap B)}{P(B)}")
                st.write(f"Answer: **{hw['hw1']['ans']:.6f}**")

    # ---- HW2
    with st.expander("HW2 — Chain Rule (CI/CD Pipeline Context)", expanded=False):
        st.markdown(f"""
A CI/CD pipeline requires three sequential stages:
- \(A\): build success
- \(B\): unit tests pass given build success
- \(C\): integration tests pass given build & unit tests

Given:
\[
P(A)={hw["hw2"]["pA"]:.3f},\quad P(B\mid A)={hw["hw2"]["pB_A"]:.3f},\quad P(C\mid A\cap B)={hw["hw2"]["pC_AB"]:.3f}
\]
Compute:
\[
P(A\cap B\cap C).
\]
""")
        user = st.number_input("Your answer for P(A∩B∩C)", 0.0, 1.0, value=0.0, step=0.001, key="hw2_user")
        if st.button("Check HW2", key="check_hw2"):
            if approx_equal(user, hw["hw2"]["ans"], tol=tolerance):
                st.success("Correct ✅")
            else:
                st.error("Not correct yet ❌")
        if st.button("Reveal HW2 Solution", key="reveal_hw2"):
            st.latex(r"P(A\cap B\cap C)=P(A)P(B\mid A)P(C\mid A\cap B)")
            st.write(f"Answer: **{hw['hw2']['ans']:.6f}**")

    # ---- HW3
    with st.expander("HW3 — Independence (Electronics Context)", expanded=False):
        st.markdown(f"""
In an electronics lab:
- \(A\): ADC works
- \(B\): MCU works  

Assume **independence** and given:
\[
P(A)={hw["hw3"]["pa"]:.3f},\quad P(B)={hw["hw3"]["pb"]:.3f}
\]
Compute:
\[
P(A\cap B).
\]
""")
        user = st.number_input("Your answer for P(A∩B)", 0.0, 1.0, value=0.0, step=0.001, key="hw3_user")
        if st.button("Check HW3", key="check_hw3"):
            if approx_equal(user, hw["hw3"]["ans"], tol=tolerance):
                st.success("Correct ✅")
            else:
                st.error("Not correct yet ❌")
        if st.button("Reveal HW3 Solution", key="reveal_hw3"):
            st.latex(r"P(A\cap B)=P(A)P(B)")
            st.write(f"Answer: **{hw['hw3']['ans']:.6f}**")

    # ---- HW4
    with st.expander("HW4 — Complement Under Conditioning (AI Monitoring Context)", expanded=False):
        st.markdown(f"""
An AI monitoring system triggers an alert when an anomaly exists.
Let:
- \(X\): alert triggers correctly
- \(Y\): anomaly exists

Given:
\[
P(X\mid Y)={hw["hw4"]["p"]:.3f}
\]
Compute:
\[
P(X^c\mid Y).
\]
""")
        user = st.number_input("Your answer for P(Xᶜ|Y)", 0.0, 1.0, value=0.0, step=0.001, key="hw4_user")
        if st.button("Check HW4", key="check_hw4"):
            if approx_equal(user, hw["hw4"]["ans"], tol=tolerance):
                st.success("Correct ✅")
            else:
                st.error("Not correct yet ❌")
        if st.button("Reveal HW4 Solution", key="reveal_hw4"):
            st.latex(r"P(X^c\mid Y)=1-P(X\mid Y)")
            st.write(f"Answer: **{hw['hw4']['ans']:.6f}**")

    # ---- HW5
    with st.expander("HW5 — Reliability (Aerospace/Mechanical Context: Series vs Parallel)", expanded=False):
        st.markdown(f"""
Three independent components have reliabilities:
\[
R_1={hw["hw5"]["R1"]:.3f},\quad R_2={hw["hw5"]["R2"]:.3f},\quad R_3={hw["hw5"]["R3"]:.3f}
\]

1) Compute **series reliability** \(R_s = R_1R_2R_3\).  
2) Compute **parallel reliability** \(R_p = 1-(1-R_1)(1-R_2)(1-R_3)\).
""")
        user_series = st.number_input("Your series reliability Rs", 0.0, 1.0, value=0.0, step=0.001, key="hw5_user_series")
        user_parallel = st.number_input("Your parallel reliability Rp", 0.0, 1.0, value=0.0, step=0.001, key="hw5_user_parallel")

        if st.button("Check HW5", key="check_hw5"):
            ok1 = approx_equal(user_series, hw["hw5"]["ans_series"], tol=tolerance)
            ok2 = approx_equal(user_parallel, hw["hw5"]["ans_parallel"], tol=tolerance)
            if ok1 and ok2:
                st.success("Correct ✅ (both series and parallel)")
            else:
                st.error("Not correct yet ❌")
                st.write(f"Series correct? {'YES' if ok1 else 'NO'} | Parallel correct? {'YES' if ok2 else 'NO'}")

        if st.button("Reveal HW5 Solution", key="reveal_hw5"):
            st.latex(r"R_s=R_1R_2R_3")
            st.latex(r"R_p=1-(1-R_1)(1-R_2)(1-R_3)")
            st.write(f"Series answer: **{hw['hw5']['ans_series']:.6f}**")
            st.write(f"Parallel answer: **{hw['hw5']['ans_parallel']:.6f}**")

        st.markdown("#### Optional: visualize series vs parallel")
        if st.button("Plot system structures (conceptual)", key="plot_hw5"):
            fig, ax = plt.subplots(figsize=(7.5, 3))
            ax.axis("off")
            ax.text(0.02, 0.75, "Series:", fontsize=12)
            ax.text(0.20, 0.75, "[R1]—[R2]—[R3]", fontsize=12)
            ax.text(0.02, 0.35, "Parallel:", fontsize=12)
            ax.text(0.20, 0.35, "┌[R1]┐\n├[R2]┤  (at least one works)\n└[R3]┘", fontsize=12, family="monospace")
            st.pyplot(fig, clear_figure=True)

    st.divider()
    st.markdown("### Homework Notes")
    st.markdown("""
- HW1 emphasizes the **definition** of conditional probability.  
- HW2 emphasizes the **chain rule** (multi-stage systems).  
- HW3 emphasizes **independence** and product rule.  
- HW4 emphasizes **complement under conditioning**.  
- HW5 emphasizes reliability (series vs parallel), common in aerospace & mechanical systems.

If you want, you can set a new seed to generate a new homework version for a new class section.
""")

# -----------------------------
# SECTION 8: Mini Quiz
# -----------------------------
else:
    st.markdown("## 8) Mini Quiz (Self-check)")

    st.markdown("### Q1")
    st.latex(r"P(A)=0.30,\quad P(B)=0.50,\quad P(A\cap B)=0.12.\ \ \text{Find }P(A\mid B).")
    if st.button("Reveal Q1", key="q1"):
        animate_progress("Revealing...", steps=20, sleep_s=sleep_s)
        st.latex(r"P(A\mid B)=\frac{0.12}{0.50}=0.24")

    st.markdown("### Q2")
    st.latex(r"P(A)=0.4,\quad P(B)=0.6.\ \ \text{If independent, find } P(A\cap B).")
    if st.button("Reveal Q2", key="q2"):
        animate_progress("Revealing...", steps=20, sleep_s=sleep_s)
        st.latex(r"P(A\cap B)=P(A)P(B)=0.4\times 0.6=0.24")

    st.markdown("### Q3")
    st.latex(r"P(A)=0.9,\ P(B\mid A)=0.8,\ P(C\mid A\cap B)=0.95.\ \ \text{Find }P(A\cap B\cap C).")
    if st.button("Reveal Q3", key="q3"):
        animate_progress("Revealing...", steps=20, sleep_s=sleep_s)
        st.latex(r"P=0.9\times 0.8\times 0.95=0.684")

    styled_box("Remember: In general, P(A|B) ≠ P(B|A). Do not swap conditional probabilities.", "warning")

# -----------------------------
# Footer (sidebar)
# -----------------------------
st.sidebar.divider()
st.sidebar.caption("Run locally:\n\n`streamlit run streamlit_app_math204_week2.py`")
