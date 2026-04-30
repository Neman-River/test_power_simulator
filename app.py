import streamlit as st
import numpy as np
from scipy import stats
import plotly.graph_objects as go

st.set_page_config(
    page_title="Statistical Power Simulator",
    page_icon="📊",
    layout="wide",
)

# ── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Global Settings")
    alpha = st.slider("Significance level (α)", 0.01, 0.10, 0.05, 0.01,
                      help="Probability of a false positive (Type I error)")
    n_trials = st.select_slider(
        "Simulations per point",
        options=[200, 500, 1000, 2000],
        value=500,
        help="More simulations → smoother estimates, slower runtime",
    )
    base_mean = st.number_input("Base mean (μ)", value=100.0, min_value=1.0)
    sigma = st.number_input("Std deviation (σ)", value=10.0, min_value=0.1)

    st.divider()
    st.markdown(
        """
**About**

Statistical **power** = probability a test correctly detects a real effect.

Rule of thumb: aim for **≥ 80 %** power.

Two main levers:
- ↑ **sample size** → ↑ power
- ↑ **effect size** → ↑ power
        """
    )

# ── Helpers ───────────────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def simulate_power(base_mean: float, sigma: float, effect: float,
                   n: int, alpha: float, n_trials: int) -> float:
    """Monte-Carlo estimate of statistical power for a two-sample t-test."""
    rng = np.random.default_rng(42)
    significant = 0
    for _ in range(n_trials):
        control   = rng.normal(base_mean, sigma, n)
        treatment = rng.normal(base_mean * (1 + effect), sigma, n)
        if stats.ttest_ind(control, treatment).pvalue < alpha:
            significant += 1
    return significant / n_trials


def power_bar_chart(x_labels: list, powers: list, x_title: str, chart_title: str):
    colors = ["#2ecc71" if p >= 0.8 else "#e74c3c" for p in powers]
    fig = go.Figure()
    fig.add_bar(
        x=x_labels,
        y=powers,
        marker_color=colors,
        text=[f"{p:.1%}" for p in powers],
        textposition="outside",
    )
    fig.add_hline(
        y=0.8,
        line_dash="dash",
        line_color="#2c3e50",
        annotation_text="80% threshold",
        annotation_position="right",
    )
    fig.update_layout(
        title=chart_title,
        xaxis_title=x_title,
        yaxis_title="Statistical Power",
        yaxis=dict(range=[0, 1.05], tickformat=".0%"),
        plot_bgcolor="white",
        paper_bgcolor="white",
        font=dict(size=13),
    )
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(showgrid=True, gridcolor="#ecf0f1")
    return fig


# ── Title ─────────────────────────────────────────────────────────────────────
st.title("Statistical Power Simulator")
st.markdown(
    "Explore how **sample size** and **effect size** drive the power of an A/B test "
    "(two-sample t-test). Adjust sliders and see results update instantly."
)
st.divider()

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(["Sample Size Explorer", "Effect Size Explorer", "Glossary"])

# ── Tab 1: Sample Size ────────────────────────────────────────────────────────
with tab1:
    st.subheader("How does sample size affect power?")
    st.markdown(
        "Fix a **small effect** and watch how collecting more data pushes power above 80%."
    )

    col_ctrl, col_chart = st.columns([1, 2])

    with col_ctrl:
        effect_pct = st.slider(
            "True effect size (%)",
            min_value=0.5, max_value=30.0, value=1.0, step=0.5,
            key="t1_effect",
            help="Relative difference between control and treatment means",
        )
        size_mode = st.radio(
            "Sample sizes input",
            ["Range slider", "Custom values"],
            horizontal=True,
            key="t1_size_mode",
        )
        if size_mode == "Range slider":
            n_min, n_max = st.slider(
                "Sample size range",
                min_value=10, max_value=5000,
                value=(10, 2000), step=10,
                key="t1_size_range",
            )
            n_steps = st.slider("Number of points", 3, 10, 5, key="t1_n_steps")
            sample_sizes = [
                int(v) for v in np.linspace(n_min, n_max, n_steps).astype(int)
            ]
            st.caption(f"Points: {sample_sizes}")
        else:
            raw = st.text_input(
                "Enter sample sizes (comma-separated)",
                value="10, 100, 500, 1000, 2000",
                key="t1_custom",
            )
            try:
                sample_sizes = sorted(
                    set(int(x.strip()) for x in raw.split(",") if x.strip())
                )
            except ValueError:
                st.warning("Use integers separated by commas, e.g. 50, 200, 1000")
                sample_sizes = []

    with col_chart:
        if not sample_sizes:
            st.info("Select at least one sample size.")
        else:
            with st.spinner("Simulating…"):
                powers = [
                    simulate_power(base_mean, sigma, effect_pct / 100,
                                   n, alpha, n_trials)
                    for n in sample_sizes
                ]

            fig = power_bar_chart(
                x_labels=[str(n) for n in sample_sizes],
                powers=powers,
                x_title="Sample size (n)",
                chart_title=f"Power vs Sample Size  |  effect = {effect_pct}%",
            )
            st.plotly_chart(fig, use_container_width=True)

            # Metric row
            sufficient = [(n, p) for n, p in zip(sample_sizes, powers) if p >= 0.8]
            min_sufficient = min(sufficient, key=lambda x: x[0]) if sufficient else None

            c1, c2, c3 = st.columns(3)
            c1.metric("Max power", f"{max(powers):.1%}")
            c2.metric("Min sample size ≥ 80%",
                      str(min_sufficient[0]) if min_sufficient else "—")
            c3.metric("Simulations run", f"{n_trials * len(sample_sizes):,}")

            # Results table
            with st.expander("Show results table"):
                rows = {
                    "n": sample_sizes,
                    "Power": [f"{p:.1%}" for p in powers],
                    "≥ 80%?": ["yes" if p >= 0.8 else "no" for p in powers],
                }
                st.dataframe(rows, use_container_width=True)

# ── Tab 2: Effect Size ────────────────────────────────────────────────────────
with tab2:
    st.subheader("How does effect size affect power?")
    st.markdown(
        "Fix the **sample size** and see how a larger real difference makes the test more sensitive."
    )

    col_ctrl2, col_chart2 = st.columns([1, 2])

    with col_ctrl2:
        fixed_n = st.slider(
            "Sample size (n per group)",
            min_value=10, max_value=2000, value=30, step=10,
            key="t2_n",
        )
        effect_mode = st.radio(
            "Effect sizes input",
            ["Range slider", "Custom values"],
            horizontal=True,
            key="t2_effect_mode",
        )
        if effect_mode == "Range slider":
            e_min, e_max = st.slider(
                "Effect size range (%)",
                min_value=0.5, max_value=50.0,
                value=(1.0, 20.0), step=0.5,
                key="t2_effect_range",
            )
            e_steps = st.slider("Number of points", 3, 10, 5, key="t2_e_steps")
            effect_options = [
                round(v, 1) for v in np.linspace(e_min, e_max, e_steps)
            ]
            st.caption(f"Points: {effect_options}")
        else:
            raw_e = st.text_input(
                "Enter effect sizes % (comma-separated)",
                value="2, 5, 12",
                key="t2_custom",
            )
            try:
                effect_options = sorted(
                    set(float(x.strip()) for x in raw_e.split(",") if x.strip())
                )
            except ValueError:
                st.warning("Use numbers separated by commas, e.g. 2, 5, 12")
                effect_options = []

    with col_chart2:
        if not effect_options:
            st.info("Select at least one effect size.")
        else:
            with st.spinner("Simulating…"):
                powers2 = [
                    simulate_power(base_mean, sigma, d / 100,
                                   fixed_n, alpha, n_trials)
                    for d in effect_options
                ]

            fig2 = power_bar_chart(
                x_labels=[f"{d:g}%" for d in effect_options],
                powers=powers2,
                x_title="Effect size (relative %)",
                chart_title=f"Power vs Effect Size  |  n = {fixed_n}",
            )
            st.plotly_chart(fig2, use_container_width=True)

            c1, c2, c3 = st.columns(3)
            c1.metric("Max power", f"{max(powers2):.1%}")
            min_detectable = next(
                (d for d, p in zip(effect_options, powers2) if p >= 0.8), None
            )
            c2.metric("Min detectable effect (≥80%)",
                      f"{min_detectable}%" if min_detectable else "—")
            c3.metric("Simulations run", f"{n_trials * len(effect_options):,}")

            with st.expander("Show results table"):
                rows2 = {
                    "Effect (%)": [f"{d:g}" for d in effect_options],
                    "Power": [f"{p:.1%}" for p in powers2],
                    "≥ 80%?": ["✅" if p >= 0.8 else "❌" for p in powers2],
                }
                st.dataframe(rows2, use_container_width=True)

# ── Tab 3: Glossary ───────────────────────────────────────────────────────────
with tab3:
    st.subheader("Glossary of Terms")
    st.markdown("All variables and concepts used in this simulator, explained.")

    terms = [
        (
            "Statistical Power (1 − β)",
            "The probability that a test **correctly rejects a false null hypothesis** — "
            "i.e. it detects a real effect when one truly exists. "
            "A power of 0.80 means the test will catch the effect 80% of the time. "
            "Industry convention: aim for **≥ 80%**.",
            "If your test has 60% power and you run it 10 times, you'd expect to miss the real effect 4 times.",
        ),
        (
            "Effect Size (%)",
            "The **relative difference** between the treatment and control group means. "
            "Defined here as `(μ_treatment − μ_control) / μ_control × 100`. "
            "A 5% effect means the treatment mean is 5% higher than the control mean. "
            "Larger effects are easier for a test to detect.",
            "Control mean = 100, treatment mean = 105 → effect size = 5%.",
        ),
        (
            "Sample Size (n)",
            "The **number of observations per group** (control and treatment each have n rows). "
            "More data reduces random noise and increases the chance of detecting a real effect. "
            "Doubling n does not double power — the relationship follows a square-root curve.",
            "Going from n=100 to n=400 roughly doubles the precision, not the power directly.",
        ),
        (
            "Significance Level (α)",
            "The **maximum acceptable false-positive rate** — the probability of rejecting the "
            "null hypothesis when it is actually true (Type I error). "
            "Common values: 0.05 (5%) or 0.01 (1%). "
            "Lower α → harder to reach significance → lower power for the same n.",
            "α = 0.05 means you accept a 5% chance of a false alarm.",
        ),
        (
            "Base Mean (μ)",
            "The **true average value of the control group**. "
            "Used to generate synthetic data. The treatment mean is computed as `μ × (1 + effect)`.",
            "μ = 100, effect = 10% → treatment mean = 110.",
        ),
        (
            "Standard Deviation (σ)",
            "Measures the **spread of individual observations** around the mean. "
            "Higher σ means more noise, which makes it harder to distinguish signal from randomness. "
            "Increasing σ reduces power for the same n and effect.",
            "σ = 2 (tight data) is much easier to analyse than σ = 20 (noisy data).",
        ),
        (
            "Simulations per Point",
            "The **number of Monte-Carlo trials** run for each (n, effect) combination. "
            "Each trial generates fresh random data and runs a t-test. "
            "Power = fraction of trials where p < α. "
            "More simulations → smoother, more accurate estimates → slower runtime.",
            "500 simulations gives ±2–3% accuracy; 2000 gives ±1%.",
        ),
        (
            "p-value",
            "The probability of observing data **at least as extreme** as the sample, "
            "assuming the null hypothesis (no effect) is true. "
            "A small p-value suggests the data is unlikely under H₀. "
            "We reject H₀ when p < α.",
            "p = 0.03 with α = 0.05 → statistically significant result.",
        ),
        (
            "Type I Error (False Positive)",
            "Rejecting H₀ when it is actually true — detecting an effect that **does not exist**. "
            "Controlled by α.",
            "Declaring a treatment works when it actually has no effect.",
        ),
        (
            "Type II Error (False Negative, β)",
            "Failing to reject H₀ when it is false — **missing a real effect**. "
            "Power = 1 − β. Lower β → higher power.",
            "Concluding a treatment doesn't work when it actually does.",
        ),
    ]

    for term, definition, example in terms:
        with st.expander(f"**{term}**"):
            st.markdown(definition)
            st.info(f"**Example:** {example}")

# ── Footer ────────────────────────────────────────────────────────────────────
st.divider()
st.caption(
    "Simulation method: Monte-Carlo two-sample t-test. "
    f"Random seed fixed at 42 for reproducibility. α = {alpha}."
)
