# Statistical Power Simulator

An interactive Streamlit app for exploring **statistical power** in A/B tests — built as a learning project.

## What it does

Statistical power is the probability that a hypothesis test correctly detects a real effect. This app lets you interactively explore two core drivers of power:

| Tab | What you explore |
|-----|-----------------|
| Sample Size Explorer | Fix an effect size, vary sample sizes → see when power crosses 80% |
| Effect Size Explorer | Fix sample size, vary effect sizes → find the minimum detectable effect |

**Sidebar controls:** significance level (α), simulations per point, base mean, std deviation.

## How it works

Each bar in the chart is a Monte-Carlo simulation: for a given `(n, effect, α)` combination, we run `n_trials` two-sample t-tests on synthetic data and measure what fraction return `p < α`. That fraction is the empirical power estimate.

```
power = P(reject H₀ | H₁ is true)
      ≈ #{significant tests} / #{total simulations}
```

## Run locally

```bash
uv run streamlit run app.py
```

## Stack

- [Streamlit](https://streamlit.io) — UI
- [SciPy](https://scipy.org) — t-test
- [NumPy](https://numpy.org) — random data generation
- [Plotly](https://plotly.com/python/) — interactive charts
- [uv](https://docs.astral.sh/uv/) — dependency management

## Key takeaways

- A **1% effect** requires ~2 000 samples per group to reach 80% power
- A **12% effect** is detectable with as few as 30 samples
- Doubling sample size does **not** double power — the relationship is non-linear
