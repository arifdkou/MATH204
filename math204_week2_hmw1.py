# math204_weeks1_4_homework_app.py
# OSTİM Technical University • Faculty of Engineering
# MATH 204 – Probability and Statistics
# Weeks 1–4 Homework Companion (Excel-oriented) with plots and downloadable CSV templates
#
# Run:
#   pip install streamlit numpy matplotlib
#   streamlit run math204_weeks1_4_homework_app.py

import io
import math
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt

st.set_page_config(page_title="MATH 204 • Weeks 1–4 Homework (Interactive)", layout="wide")

# -----------------------------
# Small math helpers (no SciPy)
# -----------------------------
def normal_pdf(x, mu, sigma):
    return (1.0/(sigma*math.sqrt(2*math.pi))) * math.exp(-0.5*((x-mu)/sigma)**2)

def normal_cdf(x, mu, sigma):
    # CDF using erf
    z = (x - mu) / (sigma * math.sqrt(2))
    return 0.5 * (1 + math.erf(z))

def to_csv_bytes(headers, rows):
    import csv
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(headers)
    w.writerows(rows)
    return buf.getvalue().encode("utf-8")

def section_title(title, subtitle=""):
    st.markdown(f"## {title}")
    if subtitle:
        st.caption(subtitle)

# -----------------------------
# Header (Top)
# -----------------------------
st.markdown("""
### OSTİM Technical University  
### Faculty of Engineering  
### **MATH 204 – Probability and Statistics**  
### **Weeks 1–4 Homework Companion (Excel / Calc Focus)**
---
This Streamlit application supports the homework set for **Weeks 1–4**.
It is designed to help students **structure their Excel/Calc work**, generate **tables**,
create **charts**, and verify computations. Each homework section includes:
- clear engineering-oriented problem context,
- required deliverables (tables + charts + interpretation),
- interactive parameters (optional),
- reference calculations, and
- **downloadable CSV templates** to open in Excel / Google Sheets / LibreOffice Calc.
""")

st.markdown("""
## How to Use This App (for Excel-based Homework)
1. Choose a homework from the **sidebar**.
2. Use **Download CSV Template** to get a ready-to-plot table for Excel.
3. In Excel/Calc, reproduce calculations using **cell formulas** (not hard-coded numbers).
4. Create the required charts (bar/line) and add axis labels + titles.
5. Write a short engineering interpretation (what does the result mean operationally?).
""")

st.markdown("""
## Learning Outcomes (Weeks 1–4)
By completing these homeworks, you will be able to:
- Represent sample spaces and events, and check probability axioms (Week 1)
- Work with discrete random variables: PMF, CDF, mean, variance (Week 1)
- Apply conditional probability and the product (multiplicative) rule (Week 2)
- Apply total probability and Bayes’ rule for engineering decisions (Week 3)
- Work with continuous random variables and common PDFs (Week 4)
- Use Excel/Calc to compute results and produce professional engineering plots
""")

st.markdown("""
## Engineering Coverage
The contexts include **Software**, **Electronics**, **AI**, **Aerospace**, and **Mechanical** engineering,
so every student can connect probability concepts to real systems.
""")

# -----------------------------
# Sidebar Navigation
# -----------------------------
st.sidebar.title("Navigate")
page = st.sidebar.radio(
    "Homework",
    ["Overview",
     "HW1 (Week 1) – Sample Space & Axioms",
     "HW2 (Week 1) – Discrete RV (PMF/CDF)",
     "HW3 (Week 2) – Conditional Probability",
     "HW4 (Week 2) – Product Rule & Reliability",
     "HW5 (Week 3) – Bayes (AI Classifier)",
     "HW6 (Week 3) – Bayesian Update (Safety)",
     "HW7 (Week 4) – Uniform Distribution",
     "HW8 (Week 4) – Normal Distribution"]
)

# -----------------------------
# Overview
# -----------------------------
if page == "Overview":
    section_title("Overview", "What each homework is training you to do in Excel/Calc")
    st.markdown("""
- **HW1:** Build a probability table and verify axioms; create a bar chart (quality categories).  
- **HW2:** PMF/CDF table + plots; compute \(E[X]\) and \(Var(X)\) (defect counts).  
- **HW3:** Compute \(P(A\\mid B)\) and summarize in a probability table + chart (incident diagnosis).  
- **HW4:** Series vs parallel reliability; compare with a chart (system design).  
- **HW5:** Bayes for an AI flagging system; sensitivity plot vs false alarm rate.  
- **HW6:** Prior → Posterior update in safety monitoring; visualize change.  
- **HW7:** Uniform PDF plot; compute mean/variance; critique realism.  
- **HW8:** Normal PDF plot; compute precision probabilities using CDF; interpret metrology impact.
""")
    st.info("Tip: In Excel, use named cells or a clear assumptions block so your sheet is easy to audit.")

# -----------------------------
# HW1
# -----------------------------
elif page == "HW1 (Week 1) – Sample Space & Axioms":
    section_title("HW1 – Sample Space & Probability Axioms", "Manufacturing quality categories (E/G/D)")
    st.markdown("""
**Context:** A manufacturing process outputs parts categorized as:
- **E** = Excellent, **G** = Good, **D** = Defective  
Given probabilities: \(P(E)=0.5\), \(P(G)=0.3\), \(P(D)=0.2\).

**Excel Deliverables**
1) Probability table, 2) Axioms check (sum = 1, non-negativity), 3) Bar chart, 4) Short interpretation.
""")

    data = [("Excellent (E)", 0.50), ("Good (G)", 0.30), ("Defective (D)", 0.20)]
    st.write("Reference table:")
    st.table({"Category": [d[0] for d in data], "Probability": [d[1] for d in data]})

    st.markdown("**Axioms check:**")
    st.latex(r"\sum_i P(\omega_i) = 0.5 + 0.3 + 0.2 = 1.0")
    st.success("Valid probability model (non-negative and sums to 1).")

    fig, ax = plt.subplots(figsize=(5.5, 3.2))
    ax.bar([d[0] for d in data], [d[1] for d in data])
    ax.set_ylim(0, 1)
    ax.set_ylabel("Probability")
    ax.set_title("Quality Category Probability Distribution")
    plt.xticks(rotation=20, ha="right")
    st.pyplot(fig, clear_figure=True)

    csv_bytes = to_csv_bytes(["Category", "Probability"], data)
    st.download_button("Download CSV Template (HW1)", data=csv_bytes, file_name="HW1_quality_probabilities.csv", mime="text/csv")

# -----------------------------
# HW2
# -----------------------------
elif page == "HW2 (Week 1) – Discrete RV (PMF/CDF)":
    section_title("HW2 – Discrete Random Variable (PMF/CDF)", "Defect count sensor (circuit board)")
    st.markdown(r"""
**Random variable:** \(X\) = number of defects on a board.  
PMF:
\[
P(X=0)=0.40,\ P(X=1)=0.35,\ P(X=2)=0.20,\ P(X=3)=0.05
\]

**Excel Deliverables**
- PMF table
- Compute \(E[X]\) and \(Var(X)\)
- Plot PMF (bar chart) and CDF (line chart)
- Short quality-control interpretation
""")

    xs = np.array([0, 1, 2, 3], dtype=int)
    ps = np.array([0.40, 0.35, 0.20, 0.05], dtype=float)
    cdf = np.cumsum(ps)

    EX = float(np.sum(xs * ps))
    EX2 = float(np.sum((xs**2) * ps))
    VarX = EX2 - EX**2

    st.markdown("### Reference computations")
    st.latex(r"E[X]=\sum_x xP(X=x)")
    st.latex(fr"E[X]={EX:.4f}")
    st.latex(r"Var(X)=E[X^2]-\left(E[X]\right)^2")
    st.latex(fr"Var(X)={VarX:.4f}")

    # PMF plot
    fig1, ax1 = plt.subplots(figsize=(5.5, 3.0))
    ax1.bar(xs.astype(str), ps)
    ax1.set_title("PMF of X (Defect Count)")
    ax1.set_xlabel("x")
    ax1.set_ylabel("P(X=x)")
    st.pyplot(fig1, clear_figure=True)

    # CDF plot
    fig2, ax2 = plt.subplots(figsize=(5.5, 3.0))
    ax2.step(xs, cdf, where="post")
    ax2.set_ylim(0, 1.05)
    ax2.set_title("CDF of X")
    ax2.set_xlabel("x")
    ax2.set_ylabel("P(X ≤ x)")
    st.pyplot(fig2, clear_figure=True)

    rows = [(int(x), float(p), float(F)) for x, p, F in zip(xs, ps, cdf)]
    csv_bytes = to_csv_bytes(["x", "P(X=x)", "CDF=P(X<=x)"], rows)
    st.download_button("Download CSV Template (HW2)", data=csv_bytes, file_name="HW2_pmf_cdf.csv", mime="text/csv")

# -----------------------------
# HW3
# -----------------------------
elif page == "HW3 (Week 2) – Conditional Probability":
    section_title("HW3 – Conditional Probability", "Software incident diagnosis (A=latency spike, B=error rate high)")
    st.markdown(r"""
Given:
\[
P(B)=0.4,\quad P(A\cap B)=0.18
\]
Compute:
\[
P(A\mid B)=\frac{P(A\cap B)}{P(B)}
\]

**Excel Deliverables**
- Compute \(P(A\mid B)\) using formulas
- Create a summary probability table (include \(P(B)\), \(P(A\cap B)\), \(P(A\mid B)\))
- Column chart
- Interpretation as a debugging / root-cause prioritization metric
""")

    pB = st.number_input("P(B)", min_value=0.01, max_value=1.0, value=0.40, step=0.01)
    pAB = st.number_input("P(A ∩ B)", min_value=0.00, max_value=1.0, value=0.18, step=0.01)
    pA_given_B = pAB / pB

    st.latex(r"P(A\mid B)=\frac{P(A\cap B)}{P(B)}")
    st.write(f"**P(A|B) = {pA_given_B:.4f}**")

    table = [("P(B)", pB), ("P(A∩B)", pAB), ("P(A|B)", pA_given_B)]
    st.table({"Quantity": [t[0] for t in table], "Value": [t[1] for t in table]})

    fig, ax = plt.subplots(figsize=(5.5, 3.0))
    ax.bar([t[0] for t in table], [t[1] for t in table])
    ax.set_ylim(0, 1)
    ax.set_title("HW3 Summary Probabilities")
    ax.set_ylabel("Value")
    plt.xticks(rotation=15, ha="right")
    st.pyplot(fig, clear_figure=True)

    csv_bytes = to_csv_bytes(["Quantity", "Value"], table)
    st.download_button("Download CSV Template (HW3)", data=csv_bytes, file_name="HW3_conditional_table.csv", mime="text/csv")

# -----------------------------
# HW4
# -----------------------------
elif page == "HW4 (Week 2) – Product Rule & Reliability":
    section_title("HW4 – Product Rule & Reliability", "Series vs Parallel (electronics/mechanical design)")
    st.markdown(r"""
Three independent components have reliabilities \(R_1, R_2, R_3\).

**Series system:** works only if all work
\[
R_s=R_1R_2R_3
\]

**Parallel system:** works if at least one works
\[
R_p=1-(1-R_1)(1-R_2)(1-R_3)
\]

**Excel Deliverables**
- Compute \(R_s\) and \(R_p\) using formulas
- Comparison chart (Series vs Parallel)
- Short explanation of redundancy benefits
""")

    c1, c2, c3 = st.columns(3)
    with c1:
        R1 = st.number_input("R1", 0.0, 1.0, 0.95, 0.01)
    with c2:
        R2 = st.number_input("R2", 0.0, 1.0, 0.92, 0.01)
    with c3:
        R3 = st.number_input("R3", 0.0, 1.0, 0.90, 0.01)

    Rs = R1 * R2 * R3
    Rp = 1 - (1 - R1) * (1 - R2) * (1 - R3)

    st.latex(r"R_s=R_1R_2R_3")
    st.write(f"**Series reliability (Rs) = {Rs:.6f}**")
    st.latex(r"R_p=1-(1-R_1)(1-R_2)(1-R_3)")
    st.write(f"**Parallel reliability (Rp) = {Rp:.6f}**")

    fig, ax = plt.subplots(figsize=(5.5, 3.0))
    ax.bar(["Series", "Parallel"], [Rs, Rp])
    ax.set_ylim(0, 1)
    ax.set_ylabel("Reliability")
    ax.set_title("Series vs Parallel Reliability")
    st.pyplot(fig, clear_figure=True)

    csv_bytes = to_csv_bytes(["System", "Reliability"], [("Series", Rs), ("Parallel", Rp)])
    st.download_button("Download CSV Template (HW4)", data=csv_bytes, file_name="HW4_reliability_comparison.csv", mime="text/csv")

# -----------------------------
# HW5
# -----------------------------
elif page == "HW5 (Week 3) – Bayes (AI Classifier)":
    section_title("HW5 – Bayes’ Rule (AI Classifier / False Alarms)", "Compute posterior defect probability given a flag")
    st.markdown(r"""
Let \(D\) = item is defective, \(F\) = AI flags the item.
Given:
\[
P(D)=0.08,\quad P(F\mid D)=0.90,\quad P(F\mid D^c)=0.15
\]
Compute:
\[
P(F)=P(F\mid D)P(D)+P(F\mid D^c)P(D^c),
\quad
P(D\mid F)=\frac{P(F\mid D)P(D)}{P(F)}.
\]

**Excel Deliverables**
- Compute \(P(F)\) and \(P(D|F)\)
- Create a probability tree diagram (Excel shapes) OR a clear tree table
- Sensitivity plot: vary \(P(F\mid D^c)\) from 0.05 to 0.30 and plot \(P(D|F)\)
- Interpretation: false alarms and decision thresholds
""")

    col1, col2, col3 = st.columns(3)
    with col1:
        pD = st.number_input("P(D)", 0.0, 1.0, 0.08, 0.01)
    with col2:
        pF_given_D = st.number_input("P(F|D)", 0.0, 1.0, 0.90, 0.01)
    with col3:
        pF_given_Dc = st.number_input("P(F|Dᶜ)", 0.0, 1.0, 0.15, 0.01)

    pDc = 1 - pD
    pF = pF_given_D * pD + pF_given_Dc * pDc
    pD_given_F = (pF_given_D * pD) / pF if pF > 0 else float("nan")

    st.markdown("### Reference results")
    st.latex(r"P(F)=P(F\mid D)P(D)+P(F\mid D^c)P(D^c)")
    st.write(f"**P(F) = {pF:.6f}**")
    st.latex(r"P(D\mid F)=\frac{P(F\mid D)P(D)}{P(F)}")
    st.write(f"**P(D|F) = {pD_given_F:.6f}**")

    # Sensitivity plot
    st.markdown("### Sensitivity: vary false alarm rate P(F|Dᶜ)")
    xs = np.linspace(0.05, 0.30, 26)
    ys = []
    for x in xs:
        pF_x = pF_given_D * pD + x * pDc
        ys.append((pF_given_D * pD) / pF_x)

    fig, ax = plt.subplots(figsize=(6.2, 3.2))
    ax.plot(xs, ys, marker="o", linewidth=1)
    ax.set_xlabel("False alarm rate  P(F|Dᶜ)")
    ax.set_ylabel("Posterior  P(D|F)")
    ax.set_title("Sensitivity of P(D|F) to false alarms")
    ax.grid(True, alpha=0.25)
    st.pyplot(fig, clear_figure=True)

    rows = [(float(x), float(y)) for x, y in zip(xs, ys)]
    csv_bytes = to_csv_bytes(["P(F|D^c)", "P(D|F)"], rows)
    st.download_button("Download CSV Template (HW5 Sensitivity)", data=csv_bytes, file_name="HW5_sensitivity.csv", mime="text/csv")

# -----------------------------
# HW6
# -----------------------------
elif page == "HW6 (Week 3) – Bayesian Update (Safety)":
    section_title("HW6 – Bayesian Update (Safety Monitoring)", "Prior → Posterior update (aerospace/critical systems)")
    st.markdown(r"""
Let \(F\) = failure condition exists, \(S\) = sensor triggers.
Given:
\[
P(F)=0.02,\quad P(S\mid F)=0.95,\quad P(S\mid F^c)=0.10
\]
Compute:
\[
P(F\mid S)=\frac{P(S\mid F)P(F)}{P(S\mid F)P(F)+P(S\mid F^c)P(F^c)}.
\]

**Excel Deliverables**
- Prior/likelihood/posterior table
- Bar chart: prior vs posterior
- Short safety interpretation (why posterior matters)
""")

    col1, col2, col3 = st.columns(3)
    with col1:
        pF = st.number_input("P(F) prior", 0.0, 1.0, 0.02, 0.01)
    with col2:
        pS_given_F = st.number_input("P(S|F)", 0.0, 1.0, 0.95, 0.01)
    with col3:
        pS_given_Fc = st.number_input("P(S|Fᶜ)", 0.0, 1.0, 0.10, 0.01)

    pFc = 1 - pF
    denom = pS_given_F * pF + pS_given_Fc * pFc
    post = (pS_given_F * pF) / denom if denom > 0 else float("nan")

    st.latex(r"P(F\mid S)=\frac{P(S\mid F)P(F)}{P(S\mid F)P(F)+P(S\mid F^c)P(F^c)}")
    st.write(f"**Posterior P(F|S) = {post:.6f}**")

    fig, ax = plt.subplots(figsize=(5.5, 3.0))
    ax.bar(["Prior P(F)", "Posterior P(F|S)"], [pF, post])
    ax.set_ylim(0, max(0.1, post*1.3))
    ax.set_ylabel("Probability")
    ax.set_title("Bayesian Update: Prior vs Posterior")
    st.pyplot(fig, clear_figure=True)

    table = [("Prior P(F)", pF), ("P(S|F)", pS_given_F), ("P(S|F^c)", pS_given_Fc), ("Posterior P(F|S)", post)]
    csv_bytes = to_csv_bytes(["Quantity", "Value"], table)
    st.download_button("Download CSV Template (HW6 Table)", data=csv_bytes, file_name="HW6_bayes_update_table.csv", mime="text/csv")

# -----------------------------
# HW7
# -----------------------------
elif page == "HW7 (Week 4) – Uniform Distribution":
    section_title("HW7 – Uniform Distribution", "Component lifetime modeled as Uniform(a,b)")
    st.markdown(r"""
Assume lifetime \(X\sim U(a,b)\). For Week 4 homework, use \([0,2000]\) hours by default.

Uniform PDF:
\[
f(x)=\frac{1}{b-a}\quad \text{for } a\le x\le b
\]

Mean/Variance:
\[
E[X]=\frac{a+b}{2},\quad Var(X)=\frac{(b-a)^2}{12}.
\]

**Excel Deliverables**
- PDF table and plot
- Compute mean and variance
- Critique: is uniform realistic for lifetimes?
""")

    a = st.number_input("a (lower bound)", value=0.0, step=100.0)
    b = st.number_input("b (upper bound)", value=2000.0, step=100.0)
    if b <= a:
        st.error("Please ensure b > a.")
        st.stop()

    fx = 1.0 / (b - a)
    mean = (a + b) / 2.0
    var = ((b - a) ** 2) / 12.0

    st.write(f"**PDF height f(x) = {fx:.6f}**")
    st.latex(r"E[X]=\frac{a+b}{2}")
    st.write(f"**E[X] = {mean:.3f}**")
    st.latex(r"Var(X)=\frac{(b-a)^2}{12}")
    st.write(f"**Var(X) = {var:.3f}**")

    xs = np.linspace(a, b, 50)
    ys = np.full_like(xs, fx)

    fig, ax = plt.subplots(figsize=(6.2, 3.2))
    ax.plot(xs, ys)
    ax.set_title("Uniform PDF")
    ax.set_xlabel("x (hours)")
    ax.set_ylabel("f(x)")
    ax.set_ylim(0, fx * 1.5)
    ax.grid(True, alpha=0.25)
    st.pyplot(fig, clear_figure=True)

    rows = [(float(x), float(fx)) for x in xs]
    csv_bytes = to_csv_bytes(["x", "f(x)"], rows)
    st.download_button("Download CSV Template (HW7 PDF)", data=csv_bytes, file_name="HW7_uniform_pdf.csv", mime="text/csv")

# -----------------------------
# HW8
# -----------------------------
elif page == "HW8 (Week 4) – Normal Distribution":
    section_title("HW8 – Normal Distribution", "Laser measurement error ~ N(μ, σ²)")
    st.markdown(r"""
Assume measurement error \(X\sim N(\mu,\sigma^2)\) with \(\mu=0\), \(\sigma=0.05\).

**Excel Deliverables**
- Generate PDF values and plot normal curve
- Compute probabilities:
  - \(P(|X|<0.05)\)
  - \(P(|X|<0.10)\)
- Interpret in terms of measurement precision
""")

    mu = st.number_input("μ (mean)", value=0.0, step=0.01)
    sigma = st.number_input("σ (std dev)", value=0.05, step=0.01, min_value=1e-6)

    # Probability computations
    p_abs_005 = normal_cdf(0.05, mu, sigma) - normal_cdf(-0.05, mu, sigma)
    p_abs_010 = normal_cdf(0.10, mu, sigma) - normal_cdf(-0.10, mu, sigma)

    st.markdown("### Probability results")
    st.latex(r"P(|X|<a)=F(a)-F(-a)")
    st.write(f"**P(|X| < 0.05) = {p_abs_005:.6f}**")
    st.write(f"**P(|X| < 0.10) = {p_abs_010:.6f}**")

    # PDF curve
    xs = np.linspace(mu - 4*sigma, mu + 4*sigma, 200)
    ys = [normal_pdf(float(x), mu, sigma) for x in xs]

    fig, ax = plt.subplots(figsize=(6.2, 3.2))
    ax.plot(xs, ys)
    ax.set_title("Normal PDF")
    ax.set_xlabel("x (measurement error)")
    ax.set_ylabel("f(x)")
    ax.grid(True, alpha=0.25)
    st.pyplot(fig, clear_figure=True)

    rows = [(float(x), float(y)) for x, y in zip(xs, ys)]
    csv_bytes = to_csv_bytes(["x", "f(x)"], rows)
    st.download_button("Download CSV Template (HW8 PDF)", data=csv_bytes, file_name="HW8_normal_pdf.csv", mime="text/csv")

    prob_rows = [("P(|X|<0.05)", p_abs_005), ("P(|X|<0.10)", p_abs_010)]
    csv_bytes2 = to_csv_bytes(["Quantity", "Value"], prob_rows)
    st.download_button("Download CSV Template (HW8 Probabilities)", data=csv_bytes2, file_name="HW8_probabilities.csv", mime="text/csv")
