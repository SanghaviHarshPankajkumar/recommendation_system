from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd


NODE_TYPE_LABELS = {
    "question": "Question",
    "bundle": "Question bundle",
    "explanation": "Explanation",
    "part": "Curriculum part",
    "skill": "Anonymous skill tag",
    "lecture": "Lecture",
    "module": "Module",
    "module_presentation": "Module offering",
    "assessment": "Assessment",
    "assessment_type": "Assessment type",
    "vle_activity": "Learning resource",
    "activity_type": "Resource type",
}

EDGE_TYPE_LABELS = {
    "belongs_to_bundle": "question belongs to bundle",
    "belongs_to_part": "content belongs to curriculum part",
    "explained_by": "question has explanation",
    "tests": "question tests skill tag",
    "teaches": "lecture teaches skill tag",
    "empirical_prerequisite_candidate": "candidate prerequisite between skill tags",
    "presentation_of": "offering belongs to module",
    "belongs_to_presentation": "resource/assessment belongs to offering",
    "has_activity_type": "resource has activity type",
    "has_assessment_type": "assessment has type",
}


def _human_node(
    node_id: str,
    row: dict[str, object],
    info: dict[str, dict[str, object]],
    outgoing: dict[str, dict[str, list[str]]],
    incoming: dict[str, dict[str, list[str]]],
) -> tuple[str, str, list[str]]:
    node_type = str(row["node_type"])
    raw_label = str(row["label"])
    attributes = json.loads(row["attributes_json"]) if isinstance(row["attributes_json"], str) else {}

    def targets(edge_type: str) -> list[str]:
        return outgoing.get(node_id, {}).get(edge_type, [])

    def sources(edge_type: str) -> list[str]:
        return incoming.get(node_id, {}).get(edge_type, [])

    def labels(node_ids: list[str]) -> list[str]:
        return [str(info[value]["label"]) for value in node_ids if value in info]

    if node_type == "question":
        bundles = labels(targets("belongs_to_bundle"))
        explanations = labels(targets("explained_by"))
        skills = [value.replace("Skill ", "") for value in labels(targets("tests"))]
        parts: list[str] = []
        for bundle_id in targets("belongs_to_bundle"):
            parts.extend(labels(outgoing.get(bundle_id, {}).get("belongs_to_part", [])))
        part = parts[0] if parts else "Part unavailable"
        facts = [f"Correct option: {str(attributes.get('correct_answer', 'unavailable')).upper()}"]
        if bundles:
            facts.append(f"Question bundle: {bundles[0]}")
        if explanations:
            facts.append(f"Explanation record: {explanations[0]}")
        if skills:
            facts.append(f"Anonymous expert skill tags: {', '.join(skills)}")
        return (
            f"Question {raw_label} - {part}",
            f"EdNet question in {part}. The public dataset does not include its question wording.",
            facts,
        )
    if node_type == "skill":
        tag = raw_label.replace("Skill ", "")
        question_count = len(sources("tests"))
        lecture_count = len(sources("teaches"))
        prerequisite_in = len(sources("empirical_prerequisite_candidate"))
        prerequisite_out = len(targets("empirical_prerequisite_candidate"))
        return (
            f"Anonymous skill tag {tag}",
            "Expert-assigned EdNet skill tag. The public dataset supplies the numeric tag but no skill name.",
            [
                f"Linked questions: {question_count}",
                f"Linked lectures: {lecture_count}",
                f"Candidate prerequisites: {prerequisite_in} incoming, {prerequisite_out} outgoing",
            ],
        )
    if node_type == "lecture":
        parts = labels(targets("belongs_to_part"))
        skills = [value.replace("Skill ", "") for value in labels(targets("teaches"))]
        length = attributes.get("video_length")
        duration = "Duration unavailable" if str(length) == "-1" else f"Duration: {round(int(length) / 60000, 1)} minutes"
        return (
            f"Lecture {raw_label}" + (f" - {parts[0]}" if parts else ""),
            "EdNet video lecture record; lecture title and transcript are not included publicly.",
            [duration, f"Anonymous expert skill tags: {', '.join(skills) or 'unavailable'}"],
        )
    if node_type == "bundle":
        parts = labels(targets("belongs_to_part"))
        return (
            f"Question bundle {raw_label}" + (f" - {parts[0]}" if parts else ""),
            "A group of EdNet questions that may share a passage, image, or listening material.",
            [f"Questions in bundle: {len(sources('belongs_to_bundle'))}"],
        )
    if node_type == "explanation":
        return (
            f"Explanation {raw_label}",
            "EdNet explanation record. The explanation text is not included publicly.",
            [f"Explains questions: {len(sources('explained_by'))}"],
        )
    if node_type == "part":
        return (
            f"Curriculum {raw_label}",
            "High-level EdNet curriculum section.",
            [f"Connected bundles/lectures: {len(sources('belongs_to_part'))}"],
        )
    if node_type == "module_presentation":
        modules = labels(targets("presentation_of"))
        return (
            f"Module {modules[0] if modules else raw_label.split()[0]} - {raw_label.split()[-1]} offering",
            "A specific OULAD module delivery period.",
            [f"Length: {attributes.get('length_days', 'unavailable')} days"],
        )
    if node_type == "module":
        return (
            f"Module {raw_label}",
            "An anonymized Open University module code.",
            [f"Offerings in graph: {len(sources('presentation_of'))}"],
        )
    if node_type == "assessment":
        assessment_types = labels(targets("has_assessment_type"))
        presentations = labels(targets("belongs_to_presentation"))
        assessment_type = assessment_types[0] if assessment_types else "Unknown type"
        facts = [f"Weight: {attributes.get('weight', 'unavailable')}%"]
        facts.append(f"Scheduled day: {attributes.get('date', 'final assessment')}" if attributes.get("date") else "Scheduled at end of module")
        if presentations:
            facts.append(f"Module offering: {presentations[0]}")
        return (
            f"{assessment_type} assessment {raw_label}",
            "OULAD assessment record; assessment questions and title are not published.",
            facts,
        )
    if node_type == "assessment_type":
        return (
            f"Assessment type: {raw_label}",
            "OULAD assessment category (for example TMA, CMA, or Exam).",
            [f"Assessments of this type: {len(sources('has_assessment_type'))}"],
        )
    if node_type == "vle_activity":
        activity_types = labels(targets("has_activity_type"))
        presentations = labels(targets("belongs_to_presentation"))
        activity_type = activity_types[0] if activity_types else "learning resource"
        facts = [f"Resource type: {activity_type}"]
        if presentations:
            facts.append(f"Module offering: {presentations[0]}")
        if attributes.get("week_from"):
            facts.append(f"Available weeks: {attributes['week_from']} to {attributes.get('week_to', 'unavailable')}")
        return (
            f"{activity_type.replace('_', ' ').title()} resource {raw_label}",
            "An OULAD virtual-learning-environment resource; its page title/content is not published.",
            facts,
        )
    if node_type == "activity_type":
        return (
            f"Resource type: {raw_label.replace('_', ' ')}",
            "Published OULAD category describing how a learning resource is used.",
            [f"Resources of this type: {len(sources('has_activity_type'))}"],
        )
    return raw_label, f"{NODE_TYPE_LABELS.get(node_type, node_type)} record.", []


def balanced_sample(nodes: pd.DataFrame, edges: pd.DataFrame, seed_type: str, limit: int) -> dict[str, object]:
    info = nodes.set_index("node_id").to_dict("index")
    adjacency: dict[str, set[str]] = defaultdict(set)
    outgoing: dict[str, dict[str, list[str]]] = defaultdict(lambda: defaultdict(list))
    incoming: dict[str, dict[str, list[str]]] = defaultdict(lambda: defaultdict(list))
    for source, target, edge_type in edges[["source_id", "target_id", "edge_type"]].itertuples(index=False):
        source, target = str(source), str(target)
        adjacency[source].add(target)
        adjacency[target].add(source)
        outgoing[source][str(edge_type)].append(target)
        incoming[target][str(edge_type)].append(source)
    candidates = [node_id for node_id, row in info.items() if row["node_type"] == seed_type]
    seed = max(candidates, key=lambda node_id: len(adjacency[node_id]))
    selected = [seed]
    selected_set = {seed}
    frontier = set(adjacency[seed])
    type_counts = Counter([info[seed]["node_type"]])
    while frontier and len(selected) < limit:
        def priority(node_id: str) -> tuple[float, int, str]:
            node_type = str(info[node_id]["node_type"])
            return (type_counts[node_type], -len(adjacency[node_id]), node_id)

        node_id = min(frontier, key=priority)
        frontier.remove(node_id)
        if node_id in selected_set:
            continue
        selected.append(node_id)
        selected_set.add(node_id)
        type_counts[str(info[node_id]["node_type"])] += 1
        frontier.update(neighbor for neighbor in adjacency[node_id] if neighbor not in selected_set)
    sample_edges = edges[
        edges["source_id"].astype(str).isin(selected_set)
        & edges["target_id"].astype(str).isin(selected_set)
    ].copy()
    sample_nodes = nodes[nodes["node_id"].astype(str).isin(selected_set)].copy()
    sample_degree = Counter(sample_edges["source_id"].astype(str)) + Counter(sample_edges["target_id"].astype(str))
    node_records = []
    for row in sample_nodes.itertuples(index=False):
        attributes = json.loads(row.attributes_json) if isinstance(row.attributes_json, str) else {}
        human_label, description, facts = _human_node(str(row.node_id), info[str(row.node_id)], info, outgoing, incoming)
        node_records.append(
            {
                "id": str(row.node_id),
                "type": str(row.node_type),
                "label": human_label,
                "typeLabel": NODE_TYPE_LABELS.get(str(row.node_type), str(row.node_type).replace("_", " ").title()),
                "description": description,
                "facts": facts,
                "degree": int(sample_degree[str(row.node_id)]),
                "attributes": attributes,
            }
        )
    edge_records = [
        {
            "source": str(row.source_id),
            "target": str(row.target_id),
            "type": str(row.edge_type),
            "typeLabel": EDGE_TYPE_LABELS.get(str(row.edge_type), str(row.edge_type).replace("_", " ")),
            "provenance": str(row.provenance),
            "confidence": float(row.confidence),
        }
        for row in sample_edges.itertuples(index=False)
    ]
    return {
        "seed": seed,
        "nodes": node_records,
        "edges": edge_records,
        "nodeTypes": dict(Counter(record["type"] for record in node_records)),
        "edgeTypes": dict(Counter(record["type"] for record in edge_records)),
    }


def load_samples(project_root: Path) -> dict[str, object]:
    graph_root = project_root / "outputs" / "phase4_graphs"
    manifest = json.loads((graph_root / "manifest.json").read_text(encoding="utf-8"))
    result: dict[str, object] = {}
    for dataset, seed_type in (("ednet", "skill"), ("oulad", "module_presentation")):
        graph_dir = graph_root / dataset
        nodes = pd.read_csv(graph_dir / "nodes.csv.gz")
        edges = pd.read_csv(graph_dir / "edges_explicit.csv.gz")
        prerequisite = graph_dir / "edges_prerequisite_dag.csv.gz"
        if prerequisite.exists():
            edges = pd.concat([edges, pd.read_csv(prerequisite)], ignore_index=True)
        sample = balanced_sample(nodes, edges, seed_type, 100)
        sample["fullNodes"] = int(len(nodes))
        sample["fullEdges"] = int(len(edges))
        sample["validationPassed"] = bool(manifest[dataset]["validation"]["passed"])
        result[dataset] = sample
    return result


def knowledge_graph_fragment(samples: dict[str, object]) -> str:
    payload = json.dumps(samples, separators=(",", ":"))
    return f'''<div id="actual-kg-view">
  <div class="viz-controls" aria-label="Dataset selector">
    <button type="button" class="btn btn-primary" data-dataset="ednet" aria-pressed="true">EdNet sample</button>
    <button type="button" class="btn" data-dataset="oulad" aria-pressed="false">OULAD sample</button>
  </div>
  <div class="viz-grid kgv-stats" aria-live="polite">
    <div class="card viz-stat"><span class="text-muted">Full graph nodes</span><span class="viz-stat-value" data-stat="nodes"></span></div>
    <div class="card viz-stat"><span class="text-muted">Full graph edges</span><span class="viz-stat-value" data-stat="edges"></span></div>
    <div class="card viz-stat"><span class="text-muted">Displayed connected sample</span><span class="viz-stat-value" data-stat="sample"></span></div>
  </div>
  <div class="kgv-legend viz-row" aria-label="Node type legend"></div>
  <div class="kgv-stage">
    <svg class="kgv-svg" viewBox="0 0 960 620" role="img" aria-label="Connected sample from the validated educational knowledge graph"></svg>
  </div>
  <div class="card kgv-detail" aria-live="polite">
    <span class="text-muted">Selected node</span>
    <strong data-detail="label">Select a node to inspect its meaning</strong>
    <span data-detail="description"></span>
    <span data-detail="facts" class="text-small"></span>
    <code data-detail="id" class="text-small"></code>
  </div>
  <div class="text-small text-muted kgv-edge-note"></div>
</div>
<style>
#actual-kg-view .kgv-stats {{ margin-block: 12px; }}
#actual-kg-view .kgv-legend {{ margin-block: 8px; gap: 12px; }}
#actual-kg-view .kgv-legend-item {{ display: inline-flex; align-items: center; gap: 5px; }}
#actual-kg-view .kgv-dot {{ width: 10px; height: 10px; border-radius: 50%; display: inline-block; background: var(--series-color); }}
#actual-kg-view .kgv-stage {{ width: 100%; min-height: 430px; }}
#actual-kg-view .kgv-svg {{ width: 100%; height: auto; display: block; color: var(--foreground); }}
#actual-kg-view .kgv-edge {{ stroke: var(--border); stroke-opacity: .52; stroke-width: 1.2; }}
#actual-kg-view .kgv-edge.is-prerequisite {{ stroke: var(--viz-series-6); stroke-opacity: .9; stroke-width: 2; stroke-dasharray: 5 4; }}
#actual-kg-view .kgv-node {{ stroke: var(--background); stroke-width: 1.5; cursor: pointer; }}
#actual-kg-view .kgv-node.is-selected {{ stroke: var(--foreground); stroke-width: 3; }}
#actual-kg-view .kgv-label {{ fill: var(--foreground); font-size: 11px; pointer-events: none; paint-order: stroke; stroke: var(--background); stroke-width: 3px; stroke-linejoin: round; }}
#actual-kg-view .kgv-detail {{ display: grid; gap: 3px; margin-top: 8px; }}
#actual-kg-view .kgv-edge-note {{ margin-top: 8px; }}
@media (max-width: 520px) {{ #actual-kg-view .kgv-stage {{ min-height: 360px; }} }}
</style>
<script src="https://cdn.jsdelivr.net/npm/d3@7.9.0/dist/d3.min.js"></script>
<script>
(() => {{
  const root = document.getElementById('actual-kg-view');
  const datasets = {payload};
  const colors = ['var(--viz-series-1)','var(--viz-series-2)','var(--viz-series-3)','var(--viz-series-4)','var(--viz-series-5)','var(--viz-series-6)'];
  const svg = d3.select(root.querySelector('.kgv-svg'));
  const width = 960, height = 620;
  let simulation;
  function render(datasetName) {{
    const data = datasets[datasetName];
    if (simulation) simulation.stop();
    svg.selectAll('*').remove();
    root.querySelector('[data-stat="nodes"]').textContent = data.fullNodes.toLocaleString();
    root.querySelector('[data-stat="edges"]').textContent = data.fullEdges.toLocaleString();
    root.querySelector('[data-stat="sample"]').textContent = `${{data.nodes.length}} nodes | ${{data.edges.length}} edges`;
    const edgeLabels = new Map(data.edges.map(edge => [edge.type, edge.typeLabel]));
    root.querySelector('.kgv-edge-note').textContent = `${{Object.entries(data.edgeTypes).map(([key,value]) => `${{edgeLabels.get(key) || key}}: ${{value}}`).join(' | ')}}`;
    root.querySelector('[data-detail="label"]').textContent = 'Select a node to inspect its meaning';
    root.querySelector('[data-detail="description"]').textContent = '';
    root.querySelector('[data-detail="facts"]').textContent = '';
    root.querySelector('[data-detail="id"]').textContent = '';
    const types = Object.keys(data.nodeTypes).sort();
    const typeColor = new Map(types.map((type, index) => [type, colors[index % colors.length]]));
    const legend = root.querySelector('.kgv-legend');
    const typeLabels = new Map(data.nodes.map(node => [node.type, node.typeLabel]));
    legend.innerHTML = types.map(type => `<span class="kgv-legend-item text-small"><span class="kgv-dot" style="--series-color:${{typeColor.get(type)}}"></span>${{typeLabels.get(type) || type}} (${{data.nodeTypes[type]}})</span>`).join('');
    const nodes = data.nodes.map(node => ({{...node}}));
    const links = data.edges.map(edge => ({{...edge}}));
    const zoomLayer = svg.append('g');
    svg.call(d3.zoom().scaleExtent([.45, 4]).on('zoom', event => zoomLayer.attr('transform', event.transform)));
    const link = zoomLayer.append('g').selectAll('line').data(links).join('line')
      .attr('class', d => `kgv-edge${{d.type === 'empirical_prerequisite_candidate' ? ' is-prerequisite' : ''}}`);
    const node = zoomLayer.append('g').selectAll('circle').data(nodes).join('circle')
      .attr('class','kgv-node').attr('r', d => 5 + Math.min(Math.sqrt(d.degree), 7)).attr('fill', d => typeColor.get(d.type));
    const labelNodes = new Set([...nodes].sort((a,b) => b.degree-a.degree).slice(0,12).map(d => d.id));
    const labels = zoomLayer.append('g').selectAll('text').data(nodes.filter(d => labelNodes.has(d.id))).join('text')
      .attr('class','kgv-label').text(d => d.label).attr('dx',9).attr('dy',4);
    node.on('click', (event,d) => {{
      node.classed('is-selected', n => n.id === d.id);
      root.querySelector('[data-detail="label"]').textContent = d.label;
      root.querySelector('[data-detail="description"]').textContent = d.description;
      root.querySelector('[data-detail="facts"]').textContent = `${{d.typeLabel}} | ${{d.facts.join(' | ')}} | displayed connections: ${{d.degree}}`;
      root.querySelector('[data-detail="id"]').textContent = `Dataset identifier: ${{d.id}}`;
    }}).call(d3.drag().on('start',(event,d)=>{{if(!event.active)simulation.alphaTarget(.3).restart();d.fx=d.x;d.fy=d.y;}})
      .on('drag',(event,d)=>{{d.fx=event.x;d.fy=event.y;}}).on('end',(event,d)=>{{if(!event.active)simulation.alphaTarget(0);d.fx=null;d.fy=null;}}));
    simulation = d3.forceSimulation(nodes)
      .force('link', d3.forceLink(links).id(d=>d.id).distance(d=>d.type === 'empirical_prerequisite_candidate' ? 110 : 64).strength(.55))
      .force('charge', d3.forceManyBody().strength(-135))
      .force('center', d3.forceCenter(width/2,height/2))
      .force('collision', d3.forceCollide().radius(d=>9+Math.min(Math.sqrt(d.degree),7)))
      .on('tick',()=>{{
        link.attr('x1',d=>d.source.x).attr('y1',d=>d.source.y).attr('x2',d=>d.target.x).attr('y2',d=>d.target.y);
        node.attr('cx',d=>d.x).attr('cy',d=>d.y);
        labels.attr('x',d=>d.x).attr('y',d=>d.y);
      }});
  }}
  root.querySelectorAll('[data-dataset]').forEach(button => button.addEventListener('click', () => {{
    root.querySelectorAll('[data-dataset]').forEach(peer => {{ const active=peer===button; peer.setAttribute('aria-pressed',String(active)); peer.classList.toggle('btn-primary',active); }});
    render(button.dataset.dataset);
  }}));
  render('ednet');
}})();
</script>'''


def state_model_fragment(project_root: Path) -> str:
    manifest = json.loads((project_root / "outputs" / "phase6_model" / "manifest.json").read_text(encoding="utf-8"))
    payload = json.dumps(
        {
            dataset: {
                "parameters": values["parameters"],
                "nodes": values["graph"]["nodes"],
                "edges": values["graph"]["directed_edges"],
                "relations": values["graph"]["relations_including_reverse"],
                "items": values["model_config"]["item_vocab_size"],
                "concepts": values["model_config"]["concept_vocab_size"],
            }
            for dataset, values in manifest["datasets"].items()
        },
        separators=(",", ":"),
    )
    return f'''<div id="student-state-architecture">
  <div class="viz-controls" aria-label="Dataset selector">
    <button type="button" class="btn btn-primary" data-model-dataset="ednet" aria-pressed="true">EdNet model</button>
    <button type="button" class="btn" data-model-dataset="oulad" aria-pressed="false">OULAD model</button>
  </div>
  <div class="viz-grid ssv-stats" aria-live="polite">
    <div class="card viz-stat"><span class="text-muted">Trainable parameters</span><span class="viz-stat-value" data-model-stat="parameters"></span></div>
    <div class="card viz-stat"><span class="text-muted">Graph passed to R-GCN</span><span class="viz-stat-value" data-model-stat="graph"></span></div>
    <div class="card viz-stat"><span class="text-muted">Output state</span><span class="viz-stat-value">64 dimensions</span></div>
  </div>
  <svg class="ssv-svg" viewBox="0 0 1080 590" role="img" aria-label="Knowledge-aware student-state model architecture">
    <defs><marker id="ssv-arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0 0L10 5L0 10Z"></path></marker></defs>
    <g class="ssv-links">
      <path d="M190 105H265"></path><path d="M415 105H490"></path><path d="M640 105H715"></path>
      <path d="M190 330H285"></path><path d="M435 330H535"></path><path d="M665 330H715"></path>
      <path d="M790 145V290"></path><path d="M565 145V390H600V420"></path><path d="M665 460H690V350H715"></path>
      <path d="M865 330H930V170"></path><path d="M865 330H930V290"></path><path d="M865 330H930V410"></path>
    </g>
    <g class="ssv-block ssv-b1" transform="translate(40 65)"><rect width="150" height="80"></rect><text x="75" y="28">Learner history</text><text x="75" y="50">127 causal inputs</text><text x="75" y="68">items · actions · time</text></g>
    <g class="ssv-block ssv-b2" transform="translate(265 65)"><rect width="150" height="80"></rect><text x="75" y="28">Feature embeddings</text><text x="75" y="50">item · concept · action</text><text x="75" y="68">correctness · context</text></g>
    <g class="ssv-block ssv-b3" transform="translate(490 65)"><rect width="150" height="80"></rect><text x="75" y="28">Causal Transformer</text><text x="75" y="50">2 layers · 4 heads</text><text x="75" y="68">future masked</text></g>
    <g class="ssv-block ssv-b4" transform="translate(715 65)"><rect width="150" height="80"></rect><text x="75" y="28">History state</text><text x="75" y="50">64 dimensions</text><text x="75" y="68">per time step</text></g>
    <g class="ssv-block ssv-b5" transform="translate(40 290)"><rect width="150" height="80"></rect><text x="75" y="28">Actual Phase 4 graph</text><text x="75" y="50" data-block="graph-nodes"></text><text x="75" y="68" data-block="graph-relations"></text></g>
    <g class="ssv-block ssv-b6" transform="translate(285 290)"><rect width="150" height="80"></rect><text x="75" y="28">Relational GNN</text><text x="75" y="50">2 message layers</text><text x="75" y="68">relation + reverse edges</text></g>
    <g class="ssv-block ssv-b7" transform="translate(535 290)"><rect width="130" height="80"></rect><text x="65" y="28">Graph context</text><text x="65" y="50">item + concept</text><text x="65" y="68">node embeddings</text></g>
    <g class="ssv-block ssv-b8" transform="translate(715 290)"><rect width="150" height="80"></rect><text x="75" y="28">Gated fusion</text><text x="75" y="50">history + graph</text><text x="75" y="68">+ mastery</text></g>
    <g class="ssv-block ssv-b9" transform="translate(535 420)"><rect width="130" height="80"></rect><text x="65" y="28">Mastery head</text><text x="65" y="50" data-block="concepts"></text><text x="65" y="68">probability per concept</text></g>
    <g class="ssv-block ssv-output" transform="translate(930 130)"><rect width="125" height="80"></rect><text x="62" y="30">Next item</text><text x="62" y="52" data-block="items"></text><text x="62" y="68">tied logits</text></g>
    <g class="ssv-block ssv-output" transform="translate(930 250)"><rect width="125" height="80"></rect><text x="62" y="30">Next action</text><text x="62" y="54">classification</text></g>
    <g class="ssv-block ssv-output" transform="translate(930 370)"><rect width="125" height="80"></rect><text x="62" y="30">Correctness</text><text x="62" y="54">binary logit</text></g>
  </svg>
  <div class="text-small text-muted">All arrows show implemented tensor flow. The Transformer supplies history and mastery; the validated graph supplies item/concept context; learned gates produce the state later consumed by the RL environment.</div>
</div>
<style>
#student-state-architecture .ssv-stats {{ margin-block: 12px; }}
#student-state-architecture .ssv-svg {{ width: 100%; height: auto; display: block; color: var(--foreground); }}
#student-state-architecture .ssv-links path {{ fill: none; stroke: var(--muted-foreground); stroke-width: 2; marker-end: url(#ssv-arrow); }}
#student-state-architecture #ssv-arrow path {{ fill: var(--muted-foreground); }}
#student-state-architecture .ssv-block rect {{ stroke: var(--border); stroke-width: 1.5; fill-opacity: .18; rx: 10; }}
#student-state-architecture .ssv-block text {{ fill: var(--foreground); text-anchor: middle; font-size: 12px; font-weight: 400; }}
#student-state-architecture .ssv-block text:first-of-type {{ font-weight: 500; }}
#student-state-architecture .ssv-b1 rect,#student-state-architecture .ssv-b2 rect {{ fill: var(--viz-series-1); }}
#student-state-architecture .ssv-b3 rect,#student-state-architecture .ssv-b4 rect {{ fill: var(--viz-series-2); }}
#student-state-architecture .ssv-b5 rect,#student-state-architecture .ssv-b6 rect,#student-state-architecture .ssv-b7 rect {{ fill: var(--viz-series-3); }}
#student-state-architecture .ssv-b8 rect {{ fill: var(--viz-series-4); }}
#student-state-architecture .ssv-b9 rect {{ fill: var(--viz-series-5); }}
#student-state-architecture .ssv-output rect {{ fill: var(--viz-series-6); }}
</style>
<script>
(() => {{
  const root = document.getElementById('student-state-architecture');
  const datasets = {payload};
  function render(name) {{
    const data = datasets[name];
    root.querySelector('[data-model-stat="parameters"]').textContent = data.parameters.toLocaleString();
    root.querySelector('[data-model-stat="graph"]').textContent = `${{data.nodes.toLocaleString()}} nodes · ${{data.edges.toLocaleString()}} directed edges`;
    root.querySelector('[data-block="graph-nodes"]').textContent = `${{data.nodes.toLocaleString()}} nodes`;
    root.querySelector('[data-block="graph-relations"]').textContent = `${{data.relations}} relation directions`;
    root.querySelector('[data-block="concepts"]').textContent = `${{data.concepts.toLocaleString()}} concept tokens`;
    root.querySelector('[data-block="items"]').textContent = `${{data.items.toLocaleString()}} item tokens`;
  }}
  root.querySelectorAll('[data-model-dataset]').forEach(button => button.addEventListener('click', () => {{
    root.querySelectorAll('[data-model-dataset]').forEach(peer => {{ const active=peer===button; peer.setAttribute('aria-pressed',String(active)); peer.classList.toggle('btn-primary',active); }});
    render(button.dataset.modelDataset);
  }}));
  render('ednet');
}})();
</script>'''


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    project_root = args.project_root.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    samples = load_samples(project_root)
    (output_dir / "knowledge-graph-samples.json").write_text(json.dumps(samples, indent=2), encoding="utf-8")
    (output_dir / "actual-knowledge-graph.html").write_text(knowledge_graph_fragment(samples), encoding="utf-8")
    (output_dir / "human-readable-knowledge-graph.html").write_text(knowledge_graph_fragment(samples), encoding="utf-8")
    (output_dir / "student-state-model.html").write_text(state_model_fragment(project_root), encoding="utf-8")


if __name__ == "__main__":
    main()
