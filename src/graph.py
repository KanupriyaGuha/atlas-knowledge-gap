import networkx as nx
import plotly.graph_objects as go
import numpy as np
from typing import List, Dict

def build_knowledge_graph(gaps: List[Dict], query: str) -> nx.Graph:
    G = nx.Graph()
    G.add_node("ROOT",
               label=query,
               node_type="root",
               size=40,
               color="#2E4B8F")

    for gap in gaps:
        keywords = gap['keywords'][:3]
        cluster_label = " | ".join(k.title() for k in keywords)
        node_id = f"cluster_{gap['cluster_id']}"

        if gap['overall_gap'] > 7:
            color = "#E74C3C"
        elif gap['overall_gap'] > 4:
            color = "#F39C12"
        else:
            color = "#27AE60"

        G.add_node(node_id,
                   label=cluster_label,
                   node_type="cluster",
                   size=20 + gap['paper_count'] * 2,
                   color=color,
                   gap_score=gap['overall_gap'],
                   paper_count=gap['paper_count'],
                   keywords=gap['keywords'],
                   avg_year=gap['avg_year'])

        G.add_edge("ROOT", node_id, weight=1)

        if gap['overall_gap'] > 4:
            for keyword in gap['keywords'][:2]:
                kw_id = f"kw_{keyword}_{gap['cluster_id']}"
                G.add_node(kw_id,
                           label=keyword.title(),
                           node_type="keyword",
                           size=10,
                           color="#9B59B6")
                G.add_edge(node_id, kw_id, weight=0.5)

    return G

def visualize_graph(G: nx.Graph, title: str = "ATLAS Knowledge Gap Map") -> go.Figure:
    """Create interactive Plotly visualization of knowledge graph."""
    pos = nx.spring_layout(G, k=0.8, iterations=100, seed=42)
    
    # Create edge traces
    edge_traces = []
    for edge in G.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_traces.append(go.Scatter(
            x=[x0, x1, None],
            y=[y0, y1, None],
            mode='lines',
            line=dict(width=1.5, color='rgba(150,150,150,0.4)'),
            hoverinfo='none',
            showlegend=False
        ))
    
    # Separate node types
    root_nodes, cluster_nodes, keyword_nodes = [], [], []
    for node in G.nodes():
        t = G.nodes[node]['node_type']
        if t == 'root': root_nodes.append(node)
        elif t == 'cluster': cluster_nodes.append(node)
        else: keyword_nodes.append(node)
    
    node_traces = []

    # Root node
    for node in root_nodes:
        x, y = pos[node]
        node_traces.append(go.Scatter(
            x=[x], y=[y],
            mode='markers+text',
            marker=dict(size=50, color='#2E4B8F',
                       line=dict(width=3, color='white')),
            text=[G.nodes[node]['label']],
            textposition="top center",
            textfont=dict(size=13, color='white'),
            hoverinfo='text',
            hovertext=G.nodes[node]['label'],
            showlegend=False
        ))

    # Cluster nodes
    for node in cluster_nodes:
        x, y = pos[node]
        data = G.nodes[node]
        size = max(45, min(80, 30 + data['paper_count'] * 3))
        node_traces.append(go.Scatter(
            x=[x], y=[y],
            mode='markers+text',
            marker=dict(size=size, color=data['color'],
                       line=dict(width=2, color='white')),
            text=[data['label']],
            textposition="top center",
            textfont=dict(size=11, color='#222222'),
            hoverinfo='text',
            hovertext=(f"<b>{data['label']}</b><br>"
                      f"Papers: {data['paper_count']}<br>"
                      f"Gap Score: {data['gap_score']}<br>"
                      f"Avg Year: {data['avg_year']}<br>"
                      f"Keywords: {', '.join(data['keywords'])}"),
            showlegend=False
        ))

    # Keyword nodes — smaller, purple
    kw_x, kw_y, kw_text, kw_hover = [], [], [], []
    for node in keyword_nodes:
        x, y = pos[node]
        kw_x.append(x)
        kw_y.append(y)
        kw_text.append(G.nodes[node]['label'])
        kw_hover.append(G.nodes[node]['label'])

    if kw_x:
        node_traces.append(go.Scatter(
            x=kw_x, y=kw_y,
            mode='markers+text',
            marker=dict(size=18, color='#9B59B6',
                       line=dict(width=1.5, color='white')),
            text=kw_text,
            textposition="bottom center",
            textfont=dict(size=9, color='#555555'),
            hovertext=kw_hover,
            hoverinfo='text',
            showlegend=False
        ))

    # Legend
    legend_annotations = [
        dict(x=0.01, y=0.99, xref="paper", yref="paper",
             text="<b>ATLAS Knowledge Gap Map</b>",
             showarrow=False, font=dict(size=14, color='#2E4B8F'),
             align="left"),
        dict(x=0.01, y=0.93, xref="paper", yref="paper",
             text="🔴 Large Gap  🟡 Medium Gap  🟢 Well Covered  🟣 Keywords",
             showarrow=False, font=dict(size=11, color='#444444'),
             align="left"),
    ]

    fig = go.Figure(
        data=edge_traces + node_traces,
        layout=go.Layout(
            title=dict(text=title, font=dict(size=18, color='#2E4B8F'),
                      x=0.5, xanchor='center'),
            showlegend=False,
            hovermode='closest',
            margin=dict(b=40, l=40, r=40, t=60),
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            plot_bgcolor='#FAFAFA',
            paper_bgcolor='#FAFAFA',
            annotations=legend_annotations,
            height=650
        )
    )

    return fig