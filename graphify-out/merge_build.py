import json, glob
from pathlib import Path

# Step B3
chunks = sorted(glob.glob('graphify-out/.graphify_chunk_*.json'))
all_nodes, all_edges, all_hyperedges = [], [], []
total_in, total_out = 0, 0
for c in chunks:
    d = json.loads(Path(c).read_text(encoding="utf-8-sig"))
    all_nodes += d.get('nodes', [])
    all_edges += d.get('edges', [])
    all_hyperedges += d.get('hyperedges', [])
    total_in += d.get('input_tokens', 0)
    total_out += d.get('output_tokens', 0)

Path('graphify-out/.graphify_semantic_new.json').write_text(json.dumps({
    'nodes': all_nodes, 'edges': all_edges, 'hyperedges': all_hyperedges,
    'input_tokens': total_in, 'output_tokens': total_out,
}, indent=2, ensure_ascii=False), encoding="utf-8")
print(f'Merged {len(chunks)} chunks: {total_in:,} in / {total_out:,} out tokens')

new = json.loads(Path('graphify-out/.graphify_semantic_new.json').read_text(encoding="utf-8")) if Path('graphify-out/.graphify_semantic_new.json').exists() else {'nodes':[],'edges':[],'hyperedges':[]}
from graphify.cache import save_semantic_cache
saved = save_semantic_cache(new.get('nodes', []), new.get('edges', []), new.get('hyperedges', []), root=".")
print(f'Cached {saved} files')

cached = json.loads(Path('graphify-out/.graphify_cached.json').read_text(encoding="utf-8-sig")) if Path('graphify-out/.graphify_cached.json').exists() else {'nodes':[],'edges':[],'hyperedges':[]}

all_nodes = cached['nodes'] + new.get('nodes', [])
all_edges = cached['edges'] + new.get('edges', [])
all_hyperedges = cached.get('hyperedges', []) + new.get('hyperedges', [])
seen = set()
deduped = []
for n in all_nodes:
    if n['id'] not in seen:
        seen.add(n['id'])
        deduped.append(n)

merged = {
    'nodes': deduped,
    'edges': all_edges,
    'hyperedges': all_hyperedges,
    'input_tokens': new.get('input_tokens', 0),
    'output_tokens': new.get('output_tokens', 0),
}
Path('graphify-out/.graphify_semantic.json').write_text(json.dumps(merged, indent=2, ensure_ascii=False), encoding="utf-8")
print(f'Extraction complete - {len(deduped)} nodes, {len(all_edges)} edges')

# Part C
ast = json.loads(Path('graphify-out/.graphify_ast.json').read_text(encoding="utf-8-sig"))
sem = json.loads(Path('graphify-out/.graphify_semantic.json').read_text(encoding="utf-8-sig"))

seen = {n['id'] for n in ast['nodes']}
merged_nodes = list(ast['nodes'])
for n in sem['nodes']:
    if n['id'] not in seen:
        merged_nodes.append(n)
        seen.add(n['id'])

merged_edges = ast['edges'] + sem['edges']
merged_hyperedges = sem.get('hyperedges', [])
merged = {
    'nodes': merged_nodes,
    'edges': merged_edges,
    'hyperedges': merged_hyperedges,
    'input_tokens': sem.get('input_tokens', 0),
    'output_tokens': sem.get('output_tokens', 0),
}
Path('graphify-out/.graphify_extract.json').write_text(json.dumps(merged, indent=2, ensure_ascii=False), encoding="utf-8")
total = len(merged_nodes)
edges = len(merged_edges)
print(f'Merged: {total} nodes, {edges} edges')

# Step 4
from graphify.build import build_from_json
from graphify.cluster import cluster, score_all
from graphify.analyze import god_nodes, surprising_connections
from graphify.export import generate_html

print("Building graph...")
extract_dict = json.loads(Path("graphify-out/.graphify_extract.json").read_text(encoding="utf-8-sig"))
G = build_from_json(extract_dict)
print("Clustering and scoring...")
communities = cluster(G)
score_all(G, communities)
gods = god_nodes(G)
surprising = surprising_connections(G)
print(f"Graph built: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
Path("graphify-out/graph.json").write_text(json.dumps({
    "nodes": [{"id": n, **d} for n, d in G.nodes(data=True)],
    "edges": [{"source": u, "target": v, **d} for u, v, d in G.edges(data=True)],
}, ensure_ascii=False), encoding="utf-8")
generate_html(G, communities, Path("graphify-out/index.html"))
print("Graph HTML saved to graphify-out/index.html")

from graphify.report import generate_report
Path("graphify-out/GRAPH_REPORT.md").write_text(generate_report(G, gods, surprising), encoding="utf-8")
print(f"Report saved to graphify-out/GRAPH_REPORT.md")
