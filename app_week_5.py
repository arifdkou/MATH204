import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

# ================================================================
# HEADER — COURSE TITLE
# ================================================================
st.title("Ostim Teknik Üniversitesi – Mühendislik Fakültesi")
st.subheader("MATH 204 – Probability and Statistics")
st.write("---")

st.header("Week 5 – Discrete Random Variables & Probability Mass Function (PMF)")

# ================================================================
# SIDEBAR — ENGINEERING FIELD SELECTION
# ================================================================
st.sidebar.header("Engineering Discipline")
field = st.sidebar.selectbox(
    "Choose a field",
    [
        "Electronics Engineering",
        "Computer / Software / AI Engineering",
        "Mechanical Engineering",
        "Nanotechnology Engineering",
        "Aerospace Engineering",
    ]
)

# ================================================================
# 1. INTRODUCTION
# ================================================================
st.header("1. Introduction")

st.markdown("""
In many engineering applications, we deal with outcomes that are **countable**:  
the number of bit errors in a packet, the number of defects on a machined surface,
the number of sensor failures during a flight, or the number of defective nanoparticles
in a sample. These types of quantities are naturally modeled as **Discrete Random Variables (DRVs)**.

In Week 5, we introduce **discrete random variables** and their associated
**Probability Mass Functions (PMFs)**. The PMF assigns a probability to each possible
value of the random variable. Using the PMF, we define and compute the **expected value**
(average behavior) and the **variance** (how much the values fluctuate around the mean).
We will also see how these concepts appear in different branches of engineering.
""")

# ================================================================
# 2. THEORY – PMF, EXPECTATION, VARIANCE
# ================================================================
st.header("2. Theory: PMF, Expected Value, and Variance")

st.subheader("2.1 Discrete Random Variables")

st.markdown("""
A **Discrete Random Variable (DRV)** is a random variable that takes values from
a **countable** (finite or countably infinite) set. Typical engineering examples include:
- Number of bit errors in a data packet (Electronics)
- Number of misclassified samples in a batch (Computer / AI)
- Number of surface defects on a part (Mechanical)
- Number of defects in a thin film or nanoparticle batch (Nanotechnology)
- Number of sensor anomalies during a flight (Aerospace)
""")

st.markdown("We denote the possible values of a DRV as:")

st.latex(r"""
X \in \{x_1, x_2, x_3, \ldots\}
""")

# ---------------------------------------------------------------
# 2.2 PMF
# ---------------------------------------------------------------
st.subheader("2.2 Probability Mass Function (PMF)")

st.markdown("""
The **Probability Mass Function (PMF)** of a discrete random variable \\(X\\)
is defined as:
""")

st.latex(r"""
p_X(x) = P(X = x)
""")

st.markdown("""
This expression means: *the probability that the random variable X takes the value x*.
For a valid PMF, two fundamental properties must hold:
""")

st.markdown("**1. Non-negativity** – probabilities cannot be negative:")

st.latex(r"""
p_X(x) \ge 0 \quad \text{for all } x
""")

st.markdown("**2. Normalization** – total probability must be 1:")

st.latex(r"""
\sum_{x} p_X(x) = 1
""")

st.markdown("""
In practice, this means we must list all possible values of \\(X\\) and assign
a non-negative probability to each, such that all probabilities add up to 1.
""")

# ---------------------------------------------------------------
# 2.3 Expected Value
# ---------------------------------------------------------------
st.subheader("2.3 Expected Value (Mean)")

st.markdown("""
The **expected value** (or mean) of a discrete random variable \\(X\\) is defined as:
""")

st.latex(r"""
E[X] = \sum_{x} x \, p_X(x)
""")

st.markdown("""
This formula is a **weighted average** of the possible values:  
each value \\(x\\) is multiplied by the probability of observing that value, \\(p_X(x)\\),
and then all such products are summed. As the number of experiments grows,
the sample average of X tends to approach \\(E[X]\\).
""")

# ---------------------------------------------------------------
# 2.4 Variance and Standard Deviation
# ---------------------------------------------------------------
st.subheader("2.4 Variance and Standard Deviation")

st.markdown("""
The **variance** measures how much the values of \\(X\\) fluctuate around the mean.
One convenient formula for the variance is:
""")

st.latex(r"""
Var(X) = E[X^2] - (E[X])^2
""")

st.markdown("""
Here \\(E[X^2]\\) is the expected value of the square of \\(X\\):
""")

st.latex(r"""
E[X^2] = \sum_{x} x^2 \, p_X(x)
""")

st.markdown("""
The **standard deviation** is the square root of the variance:
""")

st.latex(r"""
\sigma = \sqrt{Var(X)}
""")

st.markdown("""
A small \\(\\sigma\\) means that most outcomes lie close to the mean; a large \\(\\sigma\\)
indicates more variability.
""")

# ================================================================
# 3. INTERACTIVE PMF SIMULATOR
# ================================================================
st.header("3. Interactive PMF Simulator")

st.markdown("""
Below, you can define a discrete random variable by specifying its possible values
and the corresponding probabilities. The application will:

- Check and plot the PMF  
- Compute \\(E[X]\\), \\(E[X^2]\\), \\(Var(X)\\), and \\(\\sigma\\)  
""")

values = st.multiselect(
    "Choose possible values of X (discrete):",
    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
    default=[0, 1, 2]
)

probs_input = st.text_input(
    "Enter probabilities for these values (comma-separated, must sum to 1):",
    "0.6, 0.3, 0.1"
)

if values:
    try:
        probs = np.array([float(p.strip()) for p in probs_input.split(",")])
        if len(probs) != len(values):
            st.error("Number of probabilities must match number of values.")
        elif np.any(probs < 0):
            st.error("All probabilities must be non-negative.")
        elif not np.isclose(np.sum(probs), 1.0):
            st.error(f"Probabilities must sum to 1. Current sum = {np.sum(probs):.4f}")
        else:
            # Plot PMF
            fig, ax = plt.subplots()
            ax.bar(values, probs)
            ax.set_xlabel("x")
            ax.set_ylabel("P(X = x)")
            ax.set_title("User-defined PMF")
            st.pyplot(fig)

            x_arr = np.array(values, dtype=float)
            mean = np.sum(x_arr * probs)
            ex2 = np.sum((x_arr ** 2) * probs)
            var = ex2 - mean**2
            std = np.sqrt(var)

            st.markdown("### Computed statistics for your PMF:")
            st.latex(rf"E[X] = {mean:.4f}")
            st.latex(rf"E[X^2] = {ex2:.4f}")
            st.latex(rf"Var(X) = {var:.4f}")
            st.latex(rf"\sigma = {std:.4f}")

            st.markdown("""
These values give you the **average behavior** (\\(E[X]\\)) and the
**variability** (\\(Var(X), \\sigma\\)) for the engineering system you have modeled.
""")
    except Exception as e:
        st.error(f"Error parsing probabilities: {e}")

# ================================================================
# 4. ENGINEERING EXAMPLES PER DISCIPLINE
# ================================================================
st.header("4. Engineering Examples (Discipline-specific)")

# --------------------------- ELECTRONICS -------------------------
if field == "Electronics Engineering":
    st.subheader("Electronics Engineering Example – Bit Error Count")

    st.markdown("""
Consider a digital communication channel. Let \\(X\\) be the **number of bit errors**
in a received packet. Suppose the PMF is given by:
""")

    st.latex(r"""
P(X=0)=0.90, \quad P(X=1)=0.08, \quad P(X=2)=0.02
""")

    st.markdown("""
This means:
- 90% of the packets arrive with no bit errors  
- 8% have exactly one bit error  
- 2% have exactly two bit errors  

First, we verify that this is a valid PMF by checking that the probabilities sum to 1.
""")

    st.latex(r"""
0.90 + 0.08 + 0.02 = 1.00
""")

    st.markdown("""
Next, we compute the expected number of bit errors per packet:
""")

    st.latex(r"""
E[X] = 0 \cdot 0.90 + 1 \cdot 0.08 + 2 \cdot 0.02 = 0.12
""")

    st.markdown("""
On average, each packet contains **0.12 bit errors**. Now we compute \\(E[X^2]\\):
""")

    st.latex(r"""
E[X^2] = 0^2 \cdot 0.90 + 1^2 \cdot 0.08 + 2^2 \cdot 0.02 = 0.16
""")

    st.markdown("Then the variance and standard deviation:")

    st.latex(r"""
Var(X) = 0.16 - (0.12)^2 = 0.1456
""")

    st.latex(r"""
\sigma = \sqrt{0.1456} \approx 0.3815
""")

    st.markdown("""
These metrics tell us that the channel is highly reliable (low mean error),
but there is still some small variability in the number of errors per packet.
Below is the PMF plotted for this example.
""")

    # PMF plot for electronics example
    x_vals = np.array([0, 1, 2])
    p_vals = np.array([0.90, 0.08, 0.02])
    fig_e, ax_e = plt.subplots()
    ax_e.bar(x_vals, p_vals)
    ax_e.set_xlabel("Number of bit errors (x)")
    ax_e.set_ylabel("P(X = x)")
    ax_e.set_title("PMF – Bit Errors per Packet")
    st.pyplot(fig_e)

# ------------------ COMPUTER / SOFTWARE / AI ---------------------
elif field == "Computer / Software / AI Engineering":
    st.subheader("Computer / Software / AI Engineering Example – Misclassification Count")

    st.markdown("""
Assume an AI classifier processes a batch of 3 samples. Each sample has an **independent**
probability of misclassification equal to 0.1. Let \\(X\\) be the **number of misclassified
samples in the batch**. Then \\(X\\) follows a binomial distribution with parameters
\\(n=3\\) and \\(p=0.1\\):
""")

    st.latex(r"""
X \in \{0,1,2,3\}
""")

    st.latex(r"""
P(X=k) = \binom{3}{k} (0.1)^k (0.9)^{3-k}
""")

    st.markdown("""
The expected number of misclassifications in each batch is:
""")

    st.latex(r"""
E[X] = n p = 3 \cdot 0.1 = 0.3
""")

    st.markdown("""
So on average, there are **0.3 misclassified samples per batch**. Most of the time
(when \\(X=0\\)), the batch is completely correct; occasionally, 1 or more samples
are misclassified. This behavior can be visualized via the PMF.
""")

    # PMF for binomial(3,0.1)
    x_vals = np.array([0, 1, 2, 3])
    from math import comb
    p_vals = np.array([comb(3, k) * (0.1**k) * (0.9**(3-k)) for k in x_vals])
    fig_c, ax_c = plt.subplots()
    ax_c.bar(x_vals, p_vals)
    ax_c.set_xlabel("Number of misclassifications (x)")
    ax_c.set_ylabel("P(X = x)")
    ax_c.set_title("PMF – Misclassification Count in a Batch of 3")
    st.pyplot(fig_c)

# ------------------------ MECHANICAL -----------------------------
elif field == "Mechanical Engineering":
    st.subheader("Mechanical Engineering Example – Surface Defect Count")

    st.markdown("""
Consider a machined metal surface where \\(X\\) is the **number of visible defects**
on a part. Suppose the PMF is:
""")

    st.latex(r"""
X \in \{0,1,2,3\}
""")

    st.latex(r"""
P(X=0)=0.70,\; P(X=1)=0.20,\; P(X=2)=0.08,\; P(X=3)=0.02
""")

    st.markdown("""
This indicates a fairly high-quality process: 70% of parts have no defects.
The expected number of defects per part is:
""")

    st.latex(r"""
E[X] = 0\cdot 0.70 + 1\cdot 0.20 + 2\cdot 0.08 + 3\cdot 0.02 = 0.42
""")

    st.markdown("""
An average of **0.42 defects per part** suggests that while most parts are defect-free,
some have one or two defects. This information can be used by quality engineers to
decide whether the process capability is acceptable or needs improvement.
The PMF below shows how the probability mass is distributed over 0, 1, 2, and 3 defects.
""")

    x_vals = np.array([0, 1, 2, 3])
    p_vals = np.array([0.70, 0.20, 0.08, 0.02])
    fig_m, ax_m = plt.subplots()
    ax_m.bar(x_vals, p_vals)
    ax_m.set_xlabel("Number of defects (x)")
    ax_m.set_ylabel("P(X = x)")
    ax_m.set_title("PMF – Surface Defect Count")
    st.pyplot(fig_m)

# ---------------------- NANOTECHNOLOGY ---------------------------
elif field == "Nanotechnology Engineering":
    st.subheader("Nanotechnology Engineering Example – Nanoparticle Defect Count")

    st.markdown("""
In a nanofabrication process, let \\(X\\) be the **number of defects** observed
in a single nanoparticle under a microscope. Suppose:
""")

    st.latex(r"""
P(X=0)=0.60,\; P(X=1)=0.30,\; P(X=2)=0.10
""")

    st.markdown("""
This PMF reflects a relatively good process: 60% of nanoparticles are defect-free,
30% have exactly one defect, and 10% have two defects.

We compute the expected defect count:
""")

    st.latex(r"""
E[X] = 0\cdot 0.60 + 1\cdot 0.30 + 2\cdot 0.10 = 0.50
""")

    st.markdown("""
On average, each nanoparticle has **0.5 defects**. This quantitative measure can be
tracked across different process conditions (e.g., changes in temperature or pressure)
to optimize the nanofabrication steps. Below, the PMF is plotted:
""")

    x_vals = np.array([0, 1, 2])
    p_vals = np.array([0.60, 0.30, 0.10])
    fig_n, ax_n = plt.subplots()
    ax_n.bar(x_vals, p_vals)
    ax_n.set_xlabel("Number of defects (x)")
    ax_n.set_ylabel("P(X = x)")
    ax_n.set_title("PMF – Nanoparticle Defect Count")
    st.pyplot(fig_n)

# ------------------------- AEROSPACE -----------------------------
elif field == "Aerospace Engineering":
    st.subheader("Aerospace Engineering Example – Sensor Failure Count")

    st.markdown("""
In an aircraft, consider \\(X\\) as the **number of sensor failures** during a single flight.
Assume:
""")

    st.latex(r"""
P(X=0)=0.95,\; P(X=1)=0.04,\; P(X=2)=0.01
""")

    st.markdown("""
So:
- 95% of flights experience no sensor failures  
- 4% experience one failure  
- 1% experience two failures  

The expected number of sensor failures per flight is:
""")

    st.latex(r"""
E[X] = 0\cdot 0.95 + 1\cdot 0.04 + 2\cdot 0.01 = 0.06
""")

    st.markdown("""
On average, there are **0.06 failures per flight**, which indicates a highly reliable
sensor system. Engineering teams can use this metric along with variance estimates
to assess system reliability and plan maintenance intervals. The PMF is shown below:
""")

    x_vals = np.array([0, 1, 2])
    p_vals = np.array([0.95, 0.04, 0.01])
    fig_a, ax_a = plt.subplots()
    ax_a.bar(x_vals, p_vals)
    ax_a.set_xlabel("Number of sensor failures (x)")
    ax_a.set_ylabel("P(X = x)")
    ax_a.set_title("PMF – Sensor Failures per Flight")
    st.pyplot(fig_a)

# ================================================================
# 5. SUMMARY
# ================================================================
st.header("5. Summary")

st.markdown("""
In this week, we introduced **Discrete Random Variables (DRVs)** and their
**Probability Mass Functions (PMFs)**. A DRV takes countable values and its PMF
assigns a probability to each possible value, subject to non-negativity and normalization.

We defined the **expected value** as a weighted average of the possible outcomes and
the **variance** (and standard deviation) as measures of how much the values of the
random variable fluctuate around the mean. Through examples in Electronics, Computer/AI,
Mechanical, Nanotechnology, and Aerospace Engineering, we saw how DRVs naturally arise
when modeling error counts, defect counts, and failure counts.

By the end of this topic, you should be able to:
- Define a discrete random variable and its PMF  
- Verify that a given function is a valid PMF  
- Compute \\(E[X]\\), \\(Var(X)\\), and \\(\\sigma\\) for a DRV  
- Interpret these quantities in real engineering contexts  
""")

st.success("Week 5 – Discrete Random Variables & PMF: Lecture note successfully integrated into Streamlit.")
