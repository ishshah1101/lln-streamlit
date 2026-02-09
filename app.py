import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

# -----------------------------
# Page setup
# -----------------------------
st.set_page_config(page_title="LLN for Humans", layout="wide")
st.title("📈 Why Averages Become Trustworthy (Law of Large Numbers)")
st.write(
    "This interactive demo shows a simple idea:\n"
    "**When you collect more observations, the average becomes more stable.**\n\n"
    "No math required — just move sliders and watch what happens."
)

# -----------------------------
# Sidebar controls (simple language)
# -----------------------------
st.sidebar.header("Controls")

seed_mode = st.sidebar.radio(
    "Experiment mode",
    ["Repeatable (same results each time)", "Random (new results each time)"],
    index=0
)

if seed_mode == "Repeatable (same results each time)":
    seed = st.sidebar.number_input("Random seed (controls repeatability)", 0, 1_000_000, 42, 1)
    rng = np.random.default_rng(int(seed))
else:
    rng = np.random.default_rng(None)  # different each run

# Human-language label instead of "n"
n = st.sidebar.slider("How many observations do we collect?", 10, 5000, 200, 10)

# Let user pick a real-life analogy (non-technical framing)
analogy = st.sidebar.selectbox(
    "Real-world analogy",
    ["Opinion polling", "Product ratings", "Daily sales average", "Website conversion rate"],
    index=0
)

# Optional: distribution choice (still simple)
dist = st.sidebar.selectbox(
    "Random behavior style (distribution)",
    ["Normal (centered around 0)", "Uniform (even spread)", "Exponential (many small, few big)"],
    index=0
)

# -----------------------------
# Helper: sample generator (kept minimal)
# -----------------------------
def generate_samples(dist_name: str, size: int, rng: np.random.Generator) -> np.ndarray:
    if dist_name == "Normal (centered around 0)":
        return rng.normal(loc=0.0, scale=1.0, size=size)
    if dist_name == "Uniform (even spread)":
        return rng.uniform(low=-1.0, high=1.0, size=size)
    if dist_name == "Exponential (many small, few big)":
        return rng.exponential(scale=1.0, size=size)
    raise ValueError("Unknown distribution")

def true_mean(dist_name: str) -> float:
    if dist_name == "Normal (centered around 0)":
        return 0.0
    if dist_name == "Uniform (even spread)":
        return 0.0
    if dist_name == "Exponential (many small, few big)":
        return 1.0
    raise ValueError("Unknown distribution")

# -----------------------------
# Generate samples + compute summaries
# -----------------------------
samples = generate_samples(dist, n, rng)
mu = true_mean(dist)
avg_so_far = float(np.mean(samples))
abs_err = abs(avg_so_far - mu)

# -----------------------------
# Top: Explain what the user should look for (guidance)
# -----------------------------
st.info(
    "👀 What to notice:\n"
    "- With **few observations**, the average can jump around a lot.\n"
    "- With **many observations**, the average changes more slowly.\n"
    "- Randomness never disappears — it just becomes **less influential** as you collect more data."
)

# -----------------------------
# Show analogy text (non-technical meaning)
# -----------------------------
analogy_text = {
    "Opinion polling": (
        "🗳️ **Opinion polling:** If you ask only 10 people, results swing wildly. "
        "If you ask 2000 people, the average opinion becomes much more stable."
    ),
    "Product ratings": (
        "⭐ **Product ratings:** Early reviews can be misleading. "
        "As more people rate a product, the average rating becomes more trustworthy."
    ),
    "Daily sales average": (
        "💰 **Daily sales average:** One unusual day can distort the average early on. "
        "Over many days, the average settles and becomes reliable."
    ),
    "Website conversion rate": (
        "🛒 **Conversion rate:** With low traffic, conversion jumps a lot day-to-day. "
        "With more sessions, the conversion average stabilizes and is easier to trust."
    ),
}
st.write(analogy_text[analogy])

# -----------------------------
# Metrics in simple language
# -----------------------------
col1, col2, col3 = st.columns(3)
col1.metric("True average (what we'd expect in the long run)", f"{mu:.6f}")
col2.metric("Average from your collected observations", f"{avg_so_far:.6f}")
col3.metric("How far off is the average?", f"{abs_err:.6f}")

# -----------------------------
# Plots: histogram + running mean
# -----------------------------
left, right = st.columns(2)

with left:
    st.subheader("1) What values did we observe?")
    fig1 = plt.figure()
    plt.hist(samples, bins=40)
    plt.title("Histogram of observed values")
    plt.xlabel("Value")
    plt.ylabel("Count")
    st.pyplot(fig1)
    st.caption(
        "This is the *shape* of the random values you collected. "
        "With more observations, the shape looks smoother."
    )

with right:
    st.subheader("2) Running average (the main LLN visual)")
    # Running mean: average of first k observations for k=1..n
    running_mean = np.cumsum(samples) / np.arange(1, n + 1)

    fig2 = plt.figure()
    plt.plot(running_mean)
    plt.axhline(mu, linestyle="--")
    plt.title("Running average as you collect more observations")
    plt.xlabel("k (how many observations used so far)")
    plt.ylabel("Running average")
    st.pyplot(fig2)
    st.caption(
        "The dashed line is the true long-run average. "
        "The running line wiggles at first, then settles closer to the dashed line."
    )

# -----------------------------
# Stability zone (simple heuristic)
# -----------------------------
st.divider()
st.subheader("✅ Where does it start feeling 'stable'?")

# A gentle, non-mathy heuristic: show last chunk variability
window = min(200, n)
tail_std = float(np.std(running_mean[-window:], ddof=1)) if n > 5 else float("nan")

st.write(
    f"We look at the last **{window}** points of the running average and measure how much it wiggles.\n"
    f"- Wiggle score (standard deviation of the last {window} running-averages): **{tail_std:.6f}**"
)

st.write(
    "🧒 **Like you're 10:** Early on, each new number has a big influence. "
    "Later, each new number is just one drop in a big bucket."
)

# Optional: simple “interpretation” for non-technical people
if tail_std < 0.01:
    st.success("It looks fairly stable now (only small wiggles).")
elif tail_std < 0.03:
    st.warning("It’s getting more stable, but still wiggles noticeably.")
else:
    st.error("Still pretty unstable — try increasing the number of observations.")

# -----------------------------
# Why this matters (business-friendly)
# -----------------------------
st.divider()
st.markdown("### Why this matters (real world)")
st.write(
    "- **Early metrics are noisy.** Don’t overreact to small sample sizes.\n"
    "- **More data builds trust.** Averages become more stable with more observations.\n"
    "- **This is why experiments need sufficient traffic.** Otherwise results bounce around."
)


