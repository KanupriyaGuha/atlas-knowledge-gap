import arxiv
import time
from typing import List, Dict

def fetch_papers(query: str, max_results: int = 50) -> List[Dict]:
    """Fetch papers from arxiv with rate limit handling and relevance filtering."""

    client = arxiv.Client(
        page_size=10,
        delay_seconds=5.0,
        num_retries=5
    )

    search = arxiv.Search(
        query=query,
        max_results=min(max_results * 2, 100),  # fetch more to compensate for filtering
        sort_by=arxiv.SortCriterion.Relevance
    )

    query_words = query.lower().split()
    papers = []
    retries = 0
    max_retries = 3

    while retries < max_retries:
        try:
            for result in client.results(search):
                abstract_lower = result.summary.lower()
                title_lower = result.title.lower()

                # relevance score — count query words in abstract + title
                relevance = sum(
                    word in abstract_lower or word in title_lower
                    for word in query_words
                )

                # filter — at least 2 query words must appear
                if relevance < 2:
                    continue

                papers.append({
                    "id": result.entry_id,
                    "title": result.title,
                    "abstract": result.summary,
                    "authors": [str(a) for a in result.authors],
                    "published": str(result.published.date()),
                    "categories": result.categories,
                    "url": result.pdf_url,
                    "relevance_score": relevance
                })

                time.sleep(0.5)

                # stop once we have enough relevant papers
                if len(papers) >= max_results:
                    break

            break

        except Exception as e:
            retries += 1
            wait_time = 30 * retries
            print(f"Rate limited. Waiting {wait_time} seconds... (attempt {retries}/{max_retries})")
            time.sleep(wait_time)

    # sort by relevance — most relevant papers first
    papers.sort(key=lambda x: x['relevance_score'], reverse=True)

    print(f"Fetched {len(papers)} papers for query: {query}")
    return papers