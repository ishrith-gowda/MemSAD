"""
graph-structured memory system for llm agents.

motivation:
    flat vector stores treat all memory entries as independent points.
    production systems (memgpt graph, zep, memoryos) structure memory as
    entity–relation graphs: nodes are concepts/entities extracted from memory,
    edges represent semantic or temporal relations between them.

    this structure creates a different attack surface:
        (1) hub insertion: an adversarial node with high-degree connectivity
            becomes a hub that is retrieved in many query contexts.
        (2) edge hijacking: adversarial edges link a benign node to a poison
            payload, so retrieval that starts from the benign node traverses
            to the adversary's content.
        (3) subgraph poisoning: a cluster of coordinated adversarial nodes
            that re-enforce each other, making subgraph-based retrieval
            reliably surface malicious content.

system design:
    - nodes: memory entries (text strings with a type: "fact" | "event" | "task")
    - edges: semantic similarity (cosine similarity > edge_threshold)
    - retrieval: two-phase
        phase 1: embed query, find top-k nodes by cosine similarity (anchor nodes)
        phase 2: expand to neighbours via edge traversal (depth = hop_depth)
      the final context is the union of anchor nodes + their graph neighbourhood

    this retrieval pattern is analogous to graph-rag and entity-centric memory
    systems used in production (e.g., microsoft graphrag, memoryos).

attacks modelled:
    - hub insertion: insert a poison node with many edges to existing benign nodes
    - edge hijacking: add edges from high-connectivity benign nodes to poison node
    - subgraph cluster: insert a clique of k poison nodes fully connected to each
      other, making the cluster highly reachable from any anchor node

all comments are lowercase.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

# ---------------------------------------------------------------------------
# node type
# ---------------------------------------------------------------------------


@dataclass
class MemoryNode:
    """
    a single node in the graph memory system.

    fields:
        node_id: unique integer identifier
        text: memory entry string
        node_type: one of "fact" | "event" | "task" | "poison_hub" | "poison_edge"
            | "poison_cluster"
        is_adversarial: True if this is an attacker-controlled node
        embedding: pre-computed float32 unit-norm embedding vector
        metadata: arbitrary extra fields (creation time, source agent, etc.)
    """

    node_id: int
    text: str
    node_type: str = "fact"
    is_adversarial: bool = False
    embedding: np.ndarray | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# graph memory system
# ---------------------------------------------------------------------------


class GraphMemorySystem:
    """
    entity–relation graph memory with semantic edge construction.

    graph structure:
        - nodes are MemoryNode objects stored in self.nodes
        - edges are stored as an adjacency set {(i, j): similarity}
        - edges are added when cosine_similarity(node_i, node_j) >= edge_threshold

    retrieval:
        1. embed query with sentence-transformers
        2. compute cosine similarity to all nodes → top-k anchor nodes
        3. expand via bfs up to hop_depth edges
        4. return union of anchor + neighbourhood texts (deduplicated)

    attack surface:
        - insert_hub_attack(): add node with forced edges to top benign nodes
        - insert_edge_hijack(): add adversarial edges from benign hubs to poison node
        - insert_subgraph_cluster(): add k coordinated poison nodes with full edges

    defense:
        - degree_anomaly_score(): flag nodes with abnormally high degree (hub signal)
        - adjacency_contamination_score(): fraction of neighbours that are adversarial
    """

    def __init__(
        self,
        encoder_model: str = "all-MiniLM-L6-v2",
        edge_threshold: float = 0.5,
        top_k: int = 5,
        hop_depth: int = 1,
    ) -> None:
        """
        args:
            encoder_model: sentence-transformers model for node embeddings
            edge_threshold: cosine similarity >= this creates an edge
            top_k: number of anchor nodes retrieved per query
            hop_depth: BFS depth for graph expansion (1 = immediate neighbours)
        """
        from sentence_transformers import SentenceTransformer

        self._st = SentenceTransformer(encoder_model)
        self._dim = self._st.get_sentence_embedding_dimension()
        self.edge_threshold = edge_threshold
        self.top_k = top_k
        self.hop_depth = hop_depth

        self.nodes: list[MemoryNode] = []
        # adjacency: {node_id: set of connected node_ids}
        self._adj: dict[int, set[int]] = {}
        self._next_id: int = 0

    # ------------------------------------------------------------------
    # ingestion
    # ------------------------------------------------------------------

    def add_node(
        self,
        text: str,
        node_type: str = "fact",
        is_adversarial: bool = False,
        metadata: dict[str, Any] | None = None,
        forced_edges: list[int] | None = None,
    ) -> int:
        """
        add a new memory node and auto-construct edges to similar existing nodes.

        args:
            text: memory entry text
            node_type: semantic type label
            is_adversarial: whether this is an attack-controlled node
            metadata: optional metadata dict
            forced_edges: node_ids to unconditionally connect (for hub attacks)

        returns:
            node_id of the newly created node
        """
        vec = self._encode([text])[0]
        nid = self._next_id
        self._next_id += 1

        node = MemoryNode(
            node_id=nid,
            text=text,
            node_type=node_type,
            is_adversarial=is_adversarial,
            embedding=vec,
            metadata=metadata or {},
        )
        self.nodes.append(node)
        self._adj[nid] = set()

        # auto-construct semantic edges to all existing nodes
        if len(self.nodes) > 1:
            existing_vecs = np.stack([n.embedding for n in self.nodes[:-1]])
            sims = existing_vecs @ vec
            for i, sim in enumerate(sims):
                if float(sim) >= self.edge_threshold:
                    existing_id = self.nodes[i].node_id
                    self._adj[nid].add(existing_id)
                    self._adj.setdefault(existing_id, set()).add(nid)

        # forced edges (used by hub attack to ensure reachability)
        if forced_edges:
            for eid in forced_edges:
                self._adj[nid].add(eid)
                self._adj.setdefault(eid, set()).add(nid)

        return nid

    def add_forced_edge(self, id_a: int, id_b: int) -> None:
        """
        add a directed edge between two existing nodes regardless of similarity.

        used by edge-hijack attacks to link benign hubs to poison nodes.
        """
        self._adj.setdefault(id_a, set()).add(id_b)
        self._adj.setdefault(id_b, set()).add(id_a)

    def add_batch(
        self,
        texts: list[str],
        node_type: str = "fact",
        is_adversarial: bool = False,
    ) -> list[int]:
        """
        add multiple nodes efficiently.

        builds all embeddings in one forward pass, then inserts nodes
        and constructs edges against the cumulative index.
        """
        vecs = self._encode(texts)
        new_ids = []
        for i, (text, vec) in enumerate(zip(texts, vecs)):
            nid = self._next_id
            self._next_id += 1
            node = MemoryNode(
                node_id=nid,
                text=text,
                node_type=node_type,
                is_adversarial=is_adversarial,
                embedding=vec,
            )
            self.nodes.append(node)
            self._adj[nid] = set()

            # connect to all previously added nodes
            if len(self.nodes) > 1:
                prev_vecs = np.stack([n.embedding for n in self.nodes[:-1]])
                sims = prev_vecs @ vec
                for j, sim in enumerate(sims):
                    if float(sim) >= self.edge_threshold:
                        eid = self.nodes[j].node_id
                        self._adj[nid].add(eid)
                        self._adj.setdefault(eid, set()).add(nid)
            new_ids.append(nid)
        return new_ids

    # ------------------------------------------------------------------
    # retrieval
    # ------------------------------------------------------------------

    def retrieve(self, query: str) -> list[tuple[MemoryNode, str, float]]:
        """
        two-phase graph retrieval for a query.

        phase 1: top-k anchor nodes by cosine similarity
        phase 2: bfs expansion up to hop_depth edges

        returns:
            list of (node, source, score) where source is "anchor" or "graph"
            sorted by score descending.
        """
        if not self.nodes:
            return []

        q_vec = self._encode([query])[0]
        all_vecs = np.stack([n.embedding for n in self.nodes])
        sims = all_vecs @ q_vec

        top_ids = np.argsort(sims)[::-1][: min(self.top_k, len(sims))]
        anchor_set: set[int] = {int(i) for i in top_ids}

        # bfs expansion
        expanded: set[int] = set(anchor_set)
        frontier = set(anchor_set)
        for _ in range(self.hop_depth):
            next_frontier: set[int] = set()
            for nid in frontier:
                for neighbour in self._adj.get(nid, set()):
                    if neighbour not in expanded:
                        expanded.add(neighbour)
                        next_frontier.add(neighbour)
            frontier = next_frontier

        # build result
        id_to_node = {n.node_id: n for n in self.nodes}
        results: list[tuple[MemoryNode, str, float]] = []
        for nid in expanded:
            node = id_to_node[nid]
            sim = float(sims[nid]) if nid < len(sims) else 0.0
            source = "anchor" if nid in anchor_set else "graph"
            results.append((node, source, sim))

        results.sort(key=lambda x: x[2], reverse=True)
        return results

    def poison_retrieved(self, query: str) -> bool:
        """return True if any retrieved node is adversarial."""
        return any(n.is_adversarial for n, _, _ in self.retrieve(query))

    # ------------------------------------------------------------------
    # attack methods
    # ------------------------------------------------------------------

    def insert_hub_attack(
        self,
        poison_text: str,
        n_forced_connections: int = 10,
    ) -> int:
        """
        hub insertion attack: add a high-degree adversarial node.

        connects the poison node to the top-n_forced_connections benign nodes
        by degree (most connected benign nodes first — high degree = high
        centrality = frequently traversed).

        args:
            poison_text: adversarial entry text
            n_forced_connections: how many benign nodes to force-connect to

        returns:
            node_id of the poison hub node
        """
        benign_nodes = [n for n in self.nodes if not n.is_adversarial]
        # sort by current degree (highest first)
        benign_nodes.sort(
            key=lambda n: len(self._adj.get(n.node_id, set())), reverse=True
        )
        forced = [n.node_id for n in benign_nodes[:n_forced_connections]]
        nid = self.add_node(
            text=poison_text,
            node_type="poison_hub",
            is_adversarial=True,
            forced_edges=forced,
        )
        return nid

    def insert_edge_hijack(
        self,
        poison_text: str,
        n_hijack_targets: int = 5,
    ) -> int:
        """
        edge-hijack attack: add adversarial edges from high-centrality benign nodes.

        first inserts the poison node normally (semantic edges only), then
        adds forced edges from the top-n_hijack_targets benign hub nodes.

        args:
            poison_text: adversarial entry text
            n_hijack_targets: how many high-degree benign hubs to hijack

        returns:
            node_id of the poison node
        """
        nid = self.add_node(
            text=poison_text,
            node_type="poison_edge",
            is_adversarial=True,
        )
        benign_nodes = [n for n in self.nodes if not n.is_adversarial]
        benign_nodes.sort(
            key=lambda n: len(self._adj.get(n.node_id, set())), reverse=True
        )
        for bn in benign_nodes[:n_hijack_targets]:
            self.add_forced_edge(bn.node_id, nid)
        return nid

    def insert_subgraph_cluster(
        self,
        poison_texts: list[str],
    ) -> list[int]:
        """
        subgraph cluster attack: insert a fully connected clique of poison nodes.

        each pair of poison nodes is force-connected, making the cluster
        a highly cohesive subgraph.  once any cluster node is retrieved as
        an anchor, bfs expansion surfaces all other cluster nodes within
        hop_depth steps.

        args:
            poison_texts: list of adversarial texts forming the cluster

        returns:
            list of node_ids for the inserted cluster
        """
        nids = []
        for text in poison_texts:
            nid = self.add_node(
                text=text,
                node_type="poison_cluster",
                is_adversarial=True,
            )
            # force edges to all previously inserted cluster nodes
            for prev_nid in nids:
                self.add_forced_edge(nid, prev_nid)
            nids.append(nid)
        return nids

    # ------------------------------------------------------------------
    # defense: degree-based anomaly detection
    # ------------------------------------------------------------------

    def degree_anomaly_score(
        self,
        node_id: int,
        sigma: float = 2.0,
    ) -> dict[str, Any]:
        """
        flag nodes with abnormally high degree as potential hub attacks.

        hub-attack nodes have degree >> benign nodes (forced connections).
        this function computes whether a node's degree is an outlier under
        the benign degree distribution.

        args:
            node_id: node to evaluate
            sigma: threshold in standard deviations above benign mean

        returns:
            dict with degree, mean_benign_degree, std, threshold, is_anomalous
        """
        benign_degrees = [
            len(self._adj.get(n.node_id, set()))
            for n in self.nodes
            if not n.is_adversarial and n.node_id != node_id
        ]
        if not benign_degrees:
            return {
                "degree": 0,
                "is_anomalous": False,
                "mean_benign_degree": 0.0,
                "std": 0.0,
                "threshold": 0.0,
            }

        mean_deg = float(np.mean(benign_degrees))
        std_deg = float(np.std(benign_degrees)) + 1e-6
        threshold = mean_deg + sigma * std_deg
        degree = len(self._adj.get(node_id, set()))

        return {
            "degree": degree,
            "mean_benign_degree": mean_deg,
            "std": std_deg,
            "threshold": threshold,
            "is_anomalous": degree > threshold,
        }

    def adjacency_contamination(self, node_id: int) -> float:
        """
        fraction of a node's neighbours that are adversarial.

        high contamination (> 0.3) indicates the node is embedded in an
        adversarial neighbourhood — a signal for subgraph or edge-hijack attacks.

        returns:
            float in [0, 1]; 0.0 if node has no neighbours
        """
        neighbours = self._adj.get(node_id, set())
        if not neighbours:
            return 0.0
        id_to_adv = {n.node_id: n.is_adversarial for n in self.nodes}
        adv_count = sum(1 for nid in neighbours if id_to_adv.get(nid, False))
        return adv_count / len(neighbours)

    def evaluate_attacks(
        self,
        victim_queries: list[str],
    ) -> dict[str, Any]:
        """
        compute asr_r (fraction of queries that retrieve any adversarial node).

        returns:
            dict with asr_r, n_queries, n_adversarial_nodes, mean_degree_poison,
            mean_degree_benign, mean_contamination_benign
        """
        n_hit = sum(1 for q in victim_queries if self.poison_retrieved(q))
        asr_r = n_hit / max(len(victim_queries), 1)

        adv_nodes = [n for n in self.nodes if n.is_adversarial]
        benign_nodes = [n for n in self.nodes if not n.is_adversarial]

        mean_deg_adv = (
            float(np.mean([len(self._adj.get(n.node_id, set())) for n in adv_nodes]))
            if adv_nodes
            else 0.0
        )
        mean_deg_ben = (
            float(np.mean([len(self._adj.get(n.node_id, set())) for n in benign_nodes]))
            if benign_nodes
            else 0.0
        )
        mean_cont_ben = (
            float(
                np.mean([self.adjacency_contamination(n.node_id) for n in benign_nodes])
            )
            if benign_nodes
            else 0.0
        )

        return {
            "asr_r": asr_r,
            "n_queries": len(victim_queries),
            "n_adversarial_nodes": len(adv_nodes),
            "n_benign_nodes": len(benign_nodes),
            "mean_degree_poison": mean_deg_adv,
            "mean_degree_benign": mean_deg_ben,
            "mean_contamination_benign": mean_cont_ben,
        }

    def graph_stats(self) -> dict[str, Any]:
        """
        return summary statistics of the graph.

        returns:
            dict with n_nodes, n_edges, mean_degree, max_degree, n_adversarial,
            density (actual edges / max possible edges)
        """
        n = len(self.nodes)
        n_edges = sum(len(v) for v in self._adj.values()) // 2
        degrees = [len(self._adj.get(nd.node_id, set())) for nd in self.nodes]
        max_possible = n * (n - 1) // 2
        return {
            "n_nodes": n,
            "n_edges": n_edges,
            "mean_degree": float(np.mean(degrees)) if degrees else 0.0,
            "max_degree": max(degrees) if degrees else 0,
            "n_adversarial": sum(1 for nd in self.nodes if nd.is_adversarial),
            "density": n_edges / max_possible if max_possible > 0 else 0.0,
        }

    # ------------------------------------------------------------------
    # internal helpers
    # ------------------------------------------------------------------

    def _encode(self, texts: list[str]) -> np.ndarray:
        """encode texts to normalised float32 vectors."""
        return self._st.encode(
            texts, normalize_embeddings=True, show_progress_bar=False
        ).astype(np.float32)
