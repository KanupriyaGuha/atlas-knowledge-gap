import streamlit as st
import sys
import plotly.express as px
import pandas as pd
sys.path.append('src')

from fetcher import fetch_papers
from analyzer import embed_papers, cluster_papers, identify_gaps
from graph import build_knowledge_graph, visualize_graph

# Cached fetch
@st.cache_data(show_spinner=False)
def cached_fetch(query, max_papers):
    return fetch_papers(query, max_papers)

# Page config
st.set_page_config(
    page_title="ATLAS — Knowledge Gap Intelligence",
    page_icon="🌍",
    layout="wide"
)

# Styling
st.markdown("""
    <style>
    .main-title { font-size: 3rem; font-weight: bold; color: #2E4B8F; text-align: center; }
    .subtitle { font-size: 1.2rem; color: #666; text-align: center; margin-bottom: 2rem; }
    .gap-card { background: #f8f9fa; border-radius: 10px; padding: 1rem; margin: 0.5rem 0; border-left: 4px solid #FF4B4B; }
    .gap-card-medium { border-left: 4px solid #FFA500; }
    .gap-card-low { border-left: 4px solid #27AE60; }
    </style>
""", unsafe_allow_html=True)

st.markdown('<p class="main-title">🌍 ATLAS</p>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">AI-Powered Knowledge Gap Intelligence — Discover Where Research is Needed Most</p>', unsafe_allow_html=True)

# Gap level helper — single source of truth
def get_gap_level(score):
    if score > 5.5:
        return "large"
    elif score > 3.5:
        return "medium"
    else:
        return "covered"

# Sidebar
with st.sidebar:
    st.header("Search Parameters")
    query = st.text_input(
        "Research Topic",
        value="mental health sentiment analysis NLP",
        placeholder="e.g. mental health NLP, climate change ML..."
    )
    max_papers = st.slider("Number of Papers to Analyze", 20, 100, 40)
    n_clusters = st.slider("Number of Topic Clusters", 4, 12, 5)
    analyze_btn = st.button("🔍 Analyze Knowledge Gaps", type="primary")

    st.markdown("---")
    st.markdown("**How it works:**")
    st.markdown("1. Fetches papers from arxiv")
    st.markdown("2. Embeds abstracts semantically")
    st.markdown("3. Clusters into topic groups")
    st.markdown("4. Identifies gaps by density, recency & diversity")
    st.markdown("5. Visualizes as knowledge graph")

    st.markdown("---")
    st.markdown("**Gap Score Formula:**")
    st.markdown("Gap = 0.4×(1/density) + 0.3×recency + 0.3×diversity")
    st.markdown("Higher score = more under-explored")

# Main content
if analyze_btn:
    with st.spinner("Fetching papers from arxiv..."):
        try:
            papers = cached_fetch(query, max_papers)
        except Exception:
            st.error("arXiv rate limit hit. Please wait ~30 seconds and try again.")
            st.stop()

    if not papers:
        st.error("No papers found. Try a different query.")
    else:
        st.success(f"Found {len(papers)} papers")

        with st.spinner("Generating semantic embeddings..."):
            embeddings = embed_papers(papers)

        with st.spinner("Clustering topics..."):
            labels, centers = cluster_papers(embeddings, n_clusters=n_clusters)

        with st.spinner("Identifying knowledge gaps..."):
            gaps = identify_gaps(papers, embeddings, labels, centers)

        # Knowledge Graph
        st.subheader("🗺️ Knowledge Gap Map")
        G = build_knowledge_graph(gaps, query)
        fig = visualize_graph(G, f"Knowledge Gap Map: {query}")
        st.plotly_chart(fig, use_container_width=True)

        # Year Timeline Chart
        st.subheader("📈 Research Volume Over Time")
        years_data = []
        for p in papers:
            try:
                years_data.append(int(p['published'][:4]))
            except:
                pass

        if years_data:
            year_counts = pd.Series(years_data).value_counts().sort_index()
            fig_year = px.bar(
                x=year_counts.index,
                y=year_counts.values,
                labels={'x': 'Year', 'y': 'Number of Papers'},
                title=f'Research Output Over Time: {query}',
                color=year_counts.values,
                color_continuous_scale='Blues'
            )
            fig_year.update_layout(
                plot_bgcolor='white',
                showlegend=False,
                height=300
            )
            st.plotly_chart(fig_year, use_container_width=True)

        # Gap Analysis — using get_gap_level consistently
        st.subheader("🔍 Knowledge Gap Analysis")
        col1, col2, col3 = st.columns(3)

        large_gaps = [g for g in gaps if get_gap_level(g['overall_gap']) == "large"]
        medium_gaps = [g for g in gaps if get_gap_level(g['overall_gap']) == "medium"]
        covered = [g for g in gaps if get_gap_level(g['overall_gap']) == "covered"]

        col1.metric("🔴 Large Gaps", len(large_gaps))
        col2.metric("🟡 Medium Gaps", len(medium_gaps))
        col3.metric("🟢 Well Covered", len(covered))

        # Detailed Gap Cards — using get_gap_level consistently
        st.subheader("📋 Detailed Gap Report")
        for i, gap in enumerate(gaps[:8]):
            level = get_gap_level(gap['overall_gap'])
            if level == "large":
                indicator = "🔴 Large Gap"
            elif level == "medium":
                indicator = "🟡 Medium Gap"
            else:
                indicator = "🟢 Well Covered"

            with st.expander(f"{indicator} — {' + '.join(gap['keywords'][:3])}"):
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Gap Score", gap['overall_gap'])
                c2.metric("Papers Found", gap['paper_count'])
                c3.metric("Avg Year", int(gap['avg_year']))
                c4.metric("Diversity", round(gap['diversity_score'], 2))

                st.markdown(f"**Keywords:** {', '.join(gap['keywords'])}")
                st.markdown("**Sample Papers:**")
                for p in gap['papers']:
                    st.markdown(f"- [{p['title']}]({p['url']}) ({p['published'][:4]})")

        # Raw data
        with st.expander("📄 View All Fetched Papers"):
            for p in papers:
                st.markdown(f"**{p['title']}** ({p['published'][:4]})")
                st.markdown(f"_{p['abstract'][:300]}..._")
                st.markdown("---")

else:
    st.info("👈 Enter a research topic and click Analyze to discover knowledge gaps")

    st.markdown("### 💡 Example Queries")
    examples = [
        "mental health sentiment analysis NLP",
        "large language models bias detection",
        "federated learning privacy healthcare",
        "multimodal emotion recognition",
        "low resource NLP languages"
    ]
    for ex in examples:
        st.markdown(f"- `{ex}`")