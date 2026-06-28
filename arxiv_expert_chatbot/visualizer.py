import plotly.graph_objects as go
import plotly.express as px
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import io
import base64
import numpy as np
from nlp_utils import extract_keywords_tfidf, get_category_stats, get_year_stats


CATEGORY_LABELS = {
    "cs.AI": "Artificial Intelligence",
    "cs.LG": "Machine Learning",
    "cs.CL": "Computation & Language (NLP)",
    "cs.CV": "Computer Vision",
    "cs.NE": "Neural & Evolutionary Computing",
    "cs.IR": "Information Retrieval",
    "cs.HC": "Human-Computer Interaction",
    "cs.SE": "Software Engineering",
    "cs.DB": "Databases",
    "cs.CR": "Cryptography & Security",
}

COLOR_PALETTE = px.colors.qualitative.Vivid


def plot_topic_distribution(papers: list) -> go.Figure:
    """Pie/bar chart of paper distribution by category."""
    stats = get_category_stats(papers)
    labels = [CATEGORY_LABELS.get(k, k) for k in stats.keys()]
    values = list(stats.values())

    fig = go.Figure(go.Pie(
        labels=labels,
        values=values,
        hole=0.4,
        textinfo="label+percent",
        marker=dict(colors=COLOR_PALETTE),
        hovertemplate="<b>%{label}</b><br>Papers: %{value}<br>Share: %{percent}<extra></extra>"
    ))
    fig.update_layout(
        title=dict(text="📊 Papers by CS Sub-field", font=dict(size=18, color="#4fc3f7")),
        paper_bgcolor="#1a1d2e",
        plot_bgcolor="#1a1d2e",
        font=dict(color="#e0e0e0"),
        legend=dict(bgcolor="#1a1d2e", font=dict(color="#e0e0e0")),
        margin=dict(t=60, b=20, l=20, r=20)
    )
    return fig


def plot_publication_timeline(papers: list) -> go.Figure:
    """Line chart of paper count by year."""
    stats = get_year_stats(papers)
    years = list(stats.keys())
    counts = list(stats.values())

    fig = go.Figure(go.Scatter(
        x=years,
        y=counts,
        mode="lines+markers",
        line=dict(color="#4fc3f7", width=3),
        marker=dict(size=8, color="#4fc3f7", line=dict(color="#1a1d2e", width=2)),
        fill="tozeroy",
        fillcolor="rgba(79,195,247,0.1)",
        hovertemplate="Year: %{x}<br>Papers: %{y}<extra></extra>"
    ))
    fig.update_layout(
        title=dict(text="📅 Publication Timeline", font=dict(size=18, color="#4fc3f7")),
        paper_bgcolor="#1a1d2e",
        plot_bgcolor="#1a1d2e",
        font=dict(color="#e0e0e0"),
        xaxis=dict(gridcolor="#2a2a3e", title="Year"),
        yaxis=dict(gridcolor="#2a2a3e", title="Papers Published"),
        margin=dict(t=60, b=40, l=40, r=20)
    )
    return fig


def plot_keyword_network(papers: list) -> go.Figure:
    """Interactive keyword co-occurrence network."""
    if not papers:
        fig = go.Figure()
        fig.update_layout(
            title="No papers to visualize",
            paper_bgcolor="#1a1d2e",
            font=dict(color="#e0e0e0")
        )
        return fig

    all_texts = [f"{p.get('title', '')} {p.get('abstract', '')}" for p in papers]
    top_keywords = [kw for kw, _ in extract_keywords_tfidf(all_texts, top_n=18)]

    if not top_keywords:
        return go.Figure()

    # Create simple circular layout
    n = len(top_keywords)
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    x_pos = np.cos(angles).tolist()
    y_pos = np.sin(angles).tolist()

    pos = {kw: (x_pos[i], y_pos[i]) for i, kw in enumerate(top_keywords)}

    # Edges: keywords that co-occur in same abstract
    edge_x, edge_y = [], []
    for text in all_texts[:60]:
        text_lower = text.lower()
        present = [kw for kw in top_keywords if kw in text_lower]
        for i in range(len(present)):
            for j in range(i + 1, min(i + 3, len(present))):
                kw1, kw2 = present[i], present[j]
                if kw1 in pos and kw2 in pos:
                    x0, y0 = pos[kw1]
                    x1, y1 = pos[kw2]
                    edge_x += [x0, x1, None]
                    edge_y += [y0, y1, None]

    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        mode="lines",
        line=dict(width=0.8, color="rgba(79,195,247,0.3)"),
        hoverinfo="none"
    )

    node_trace = go.Scatter(
        x=x_pos, y=y_pos,
        mode="markers+text",
        marker=dict(
            size=20,
            color=COLOR_PALETTE[:n] if n <= len(COLOR_PALETTE) else COLOR_PALETTE * (n // len(COLOR_PALETTE) + 1),
            line=dict(width=2, color="#1a1d2e")
        ),
        text=top_keywords,
        textposition="top center",
        textfont=dict(size=10, color="#e0e0e0"),
        hovertemplate="<b>%{text}</b><extra></extra>"
    )

    fig = go.Figure(data=[edge_trace, node_trace])
    fig.update_layout(
        title=dict(text="🔗 Keyword Co-occurrence Network", font=dict(size=18, color="#4fc3f7")),
        paper_bgcolor="#1a1d2e",
        plot_bgcolor="#1a1d2e",
        font=dict(color="#e0e0e0"),
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        showlegend=False,
        margin=dict(t=60, b=20, l=20, r=20),
        height=500
    )
    return fig


def generate_wordcloud_b64(papers: list) -> str:
    """Generate a word cloud from paper titles and abstracts, return as base64 PNG."""
    text = " ".join([
        f"{p.get('title', '')} {p.get('abstract', '')}"
        for p in papers
    ])
    if not text.strip():
        return ""

    wc = WordCloud(
        width=900,
        height=400,
        background_color="#1a1d2e",
        colormap="cool",
        max_words=80,
        prefer_horizontal=0.8,
        collocations=False,
        stopwords={"the", "a", "an", "and", "or", "in", "of", "to", "is", "are",
                   "for", "with", "that", "this", "we", "our", "can", "also"}
    ).generate(text)

    buf = io.BytesIO()
    plt.figure(figsize=(10, 4), facecolor="#1a1d2e")
    plt.imshow(wc, interpolation="bilinear")
    plt.axis("off")
    plt.tight_layout(pad=0)
    plt.savefig(buf, format="png", bbox_inches="tight", dpi=120, facecolor="#1a1d2e")
    plt.close()
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8")
