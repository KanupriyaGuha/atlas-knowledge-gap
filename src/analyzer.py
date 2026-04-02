from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from typing import List, Dict
from keybert import KeyBERT

model = SentenceTransformer('all-MiniLM-L6-v2')
kw_model = KeyBERT()


def embed_papers(papers: List[Dict]) -> np.ndarray:
    """Generate embeddings for paper abstracts."""
    texts = [f"{p['title']}. {p['abstract']}" for p in papers]
    print("Generating embeddings...")
    embeddings = model.encode(texts, show_progress_bar=True)
    return embeddings


def cluster_papers(embeddings: np.ndarray, n_clusters: int = 5) -> np.ndarray:
    """Cluster papers into topic groups."""
    n_clusters = min(n_clusters, len(embeddings))
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = kmeans.fit_predict(embeddings)
    return labels, kmeans.cluster_centers_


def extract_keywords(text: str, top_n: int = 5) -> List[str]:
    """Extract meaningful keywords using KeyBERT."""
    if not text.strip():
        return []
    keywords = kw_model.extract_keywords(
        text,
        keyphrase_ngram_range=(1, 2),
        stop_words='english',
        top_n=top_n,
        diversity=0.5
    )
    return [kw[0] for kw in keywords]


def identify_gaps(papers: List[Dict], embeddings: np.ndarray,
                  labels: np.ndarray, cluster_centers: np.ndarray) -> List[Dict]:
    """Identify knowledge gaps based on normalized cluster density, recency and diversity."""

    clusters = {}
    for i, label in enumerate(labels):
        if label not in clusters:
            clusters[label] = []
        clusters[label].append(papers[i])

    # First pass — collect raw values for normalization
    raw_data = []
    for cluster_id, cluster_papers_list in clusters.items():
        all_text = " ".join([f"{p['title']} {p['abstract']}"
                             for p in cluster_papers_list])
        keywords = extract_keywords(all_text, top_n=6)

        density = len(cluster_papers_list)

        years = []
        for p in cluster_papers_list:
            try:
                years.append(int(p['published'][:4]))
            except:
                pass
        avg_year = np.mean(years) if years else 2020

        cluster_embeddings = embeddings[labels == cluster_id]
        if len(cluster_embeddings) > 1:
            similarities = cosine_similarity(cluster_embeddings)
            avg_similarity = np.mean(similarities[np.triu_indices_from(
                similarities, k=1)])
        else:
            avg_similarity = 1.0

        diversity_score = 1 - avg_similarity

        raw_data.append({
            "cluster_id": int(cluster_id),
            "keywords": keywords,
            "paper_count": density,
            "avg_year": round(avg_year, 0),
            "diversity_score": round(diversity_score, 2),
            "papers": cluster_papers_list[:3],
        })

    # Normalize each component to 0-10 scale
    densities = [d['paper_count'] for d in raw_data]
    years_list = [d['avg_year'] for d in raw_data]
    diversities = [d['diversity_score'] for d in raw_data]

    min_d, max_d = min(densities), max(densities)
    min_y, max_y = min(years_list), max(years_list)
    min_div, max_div = min(diversities), max(diversities)

    def normalize(val, min_val, max_val, invert=False):
        if max_val == min_val:
            return 5.0
        normalized = (val - min_val) / (max_val - min_val) * 10
        return round(10 - normalized if invert else normalized, 2)

    gaps = []
    for d in raw_data:
        # Higher density = lower gap (invert=True)
        density_gap = normalize(d['paper_count'], min_d, max_d, invert=True)
        # Older avg year = bigger gap (invert=True)
        recency_gap = normalize(d['avg_year'], min_y, max_y, invert=True)
        # Higher diversity = bigger gap
        diversity_gap = normalize(d['diversity_score'], min_div, max_div)

        overall_gap = round(
            density_gap * 0.4 + recency_gap * 0.3 + diversity_gap * 0.3, 2
        )

        gaps.append({
            "cluster_id": d['cluster_id'],
            "keywords": d['keywords'],
            "paper_count": d['paper_count'],
            "gap_score": density_gap,
            "recency_gap": recency_gap,
            "diversity_score": d['diversity_score'],
            "overall_gap": overall_gap,
            "papers": d['papers'],
            "avg_year": d['avg_year']
        })

    gaps.sort(key=lambda x: x['overall_gap'], reverse=True)
    return gaps


if __name__ == "__main__":
    from fetcher import fetch_papers
    papers = fetch_papers("mental health sentiment analysis NLP", max_results=30)
    embeddings = embed_papers(papers)
    labels, centers = cluster_papers(embeddings)
    gaps = identify_gaps(papers, embeddings, labels, centers)
    for g in gaps[:3]:
        print(f"\nCluster: {g['keywords']}")
        print(f"Papers: {g['paper_count']} | Gap Score: {g['overall_gap']}")