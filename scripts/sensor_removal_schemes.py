#!/usr/bin/env python3
"""
Generate sensor removal schemes for fault analysis.
Creates multiple schemes based on network topology to study the effect
of different sensor failure patterns on model performance.
"""

import pandas as pd
import networkx as nx
import numpy as np
import json
from pathlib import Path
from collections import defaultdict
import matplotlib.pyplot as plt

def load_rel_file(rel_path):
    """Load the .rel file and return edges dataframe."""
    df = pd.read_csv(rel_path)
    return df

def create_graph(df):
    """Create a NetworkX graph from the relationship data."""
    G = nx.DiGraph()
    all_nodes = set(df['origin_id'].unique()) | set(df['destination_id'].unique())
    G.add_nodes_from(all_nodes)
    for _, row in df.iterrows():
        G.add_edge(row['origin_id'], row['destination_id'], weight=row['cost'])
    return G

def get_undirected_graph(G):
    """Convert to undirected for certain analyses."""
    return G.to_undirected()

# =============================================================================
# SCHEME 1: Hub Removal (High-Degree Nodes)
# =============================================================================
def scheme_hub_removal(G, percentages=[10, 20, 30]):
    """
    Remove the most connected sensors (hubs).
    These are critical nodes that connect many parts of the network.
    """
    schemes = {}
    degrees = dict(G.degree())
    sorted_nodes = sorted(degrees.items(), key=lambda x: x[1], reverse=True)
    
    for pct in percentages:
        n_remove = max(1, int(len(sorted_nodes) * pct / 100))
        nodes_to_remove = [n for n, d in sorted_nodes[:n_remove]]
        schemes[f'hub_top_{pct}pct'] = {
            'sensors': nodes_to_remove,
            'description': f'Remove top {pct}% highest-degree hub sensors ({n_remove} sensors)',
            'rationale': 'Hubs are critical for network connectivity; their failure has cascading effects'
        }
    
    return schemes

# =============================================================================
# SCHEME 2: Peripheral Removal (Low-Degree Nodes)
# =============================================================================
def scheme_peripheral_removal(G, percentages=[30, 40, 50, 60]):
    """
    Remove sensors with lowest connectivity (peripheral/leaf nodes).
    These are edge sensors with few connections.
    """
    schemes = {}
    degrees = dict(G.degree())
    sorted_nodes = sorted(degrees.items(), key=lambda x: x[1])
    
    for pct in percentages:
        n_remove = max(1, int(len(sorted_nodes) * pct / 100))
        nodes_to_remove = [n for n, d in sorted_nodes[:n_remove]]
        schemes[f'peripheral_{pct}pct'] = {
            'sensors': nodes_to_remove,
            'description': f'Remove {pct}% lowest-degree peripheral sensors ({n_remove} sensors)',
            'rationale': 'Peripheral sensors have limited local connectivity; tests isolated failures'
        }
    
    return schemes

# =============================================================================
# SCHEME 3: Chain Removal (Linear Sequences)
# =============================================================================
def scheme_chain_removal(G):
    """
    Identify and remove linear chains of sensors.
    Chains are sequences where each node has degree 2 (except endpoints).
    """
    schemes = {}
    G_undirected = get_undirected_graph(G)
    degrees = dict(G_undirected.degree())
    
    # Find chain nodes (degree 2 in undirected graph)
    chain_nodes = [n for n, d in degrees.items() if d == 2]
    
    # Group into connected chains
    chain_subgraph = G_undirected.subgraph(chain_nodes)
    chains = list(nx.connected_components(chain_subgraph))
    chains = sorted(chains, key=len, reverse=True)
    
    # Create scheme for longest chain only
    # (interconnected chain schemes handle multi-chain removal better)
    if chains:
        longest_chain = list(chains[0])
        schemes['chain_longest'] = {
            'sensors': longest_chain,
            'description': f'Remove longest linear chain ({len(longest_chain)} sensors)',
            'rationale': 'Tests impact of losing a complete linear segment of the network'
        }
    
    return schemes

# =============================================================================
# SCHEME 3b: Interconnected Chain Removal
# =============================================================================
def scheme_interconnected_chain_removal(G):
    """
    Remove pairs and triplets of chains that are connected through hub nodes.
    
    A chain is a sequence of degree-2 nodes. Chains are "interconnected" when 
    their endpoint hubs are either the same OR directly adjacent to each other.
    This captures chains that form continuous corridors through the network,
    separated only by hub junctions.
    
    For example: [Chain A] -- Hub1 -- Hub2 -- [Chain B]
    Chains A and B are interconnected through the hub path Hub1-Hub2.
    """
    schemes = {}
    G_undirected = get_undirected_graph(G)
    degrees = dict(G_undirected.degree())
    
    # Find chain nodes (degree 2) and hub nodes (degree >= 3)
    chain_nodes = set(n for n, d in degrees.items() if d == 2)
    hub_nodes = set(n for n, d in degrees.items() if d >= 3)
    
    # Group chain nodes into connected chains
    chain_subgraph = G_undirected.subgraph(chain_nodes)
    chains = list(nx.connected_components(chain_subgraph))
    chains = sorted(chains, key=len, reverse=True)  # Sort by size for consistent ordering
    
    if not chains:
        return schemes
    
    # For each chain, find which hub nodes it connects to
    chain_to_hubs = {}  # chain_id -> set of connected hub nodes
    chain_nodes_map = {}  # chain_id -> list of nodes in chain
    hub_to_chains = defaultdict(set)  # hub -> set of chain_ids connected to it
    
    for idx, chain in enumerate(chains):
        chain_nodes_map[idx] = list(chain)
        connected_hubs = set()
        for node in chain:
            for neighbor in G_undirected.neighbors(node):
                if neighbor in hub_nodes:
                    connected_hubs.add(neighbor)
                    hub_to_chains[neighbor].add(idx)
        chain_to_hubs[idx] = connected_hubs
    
    # Build a graph of chain connectivity
    # Two chains are connected if:
    # 1. They share the same hub (direct connection), OR
    # 2. Their endpoint hubs are directly adjacent (one-hop through hub-hub edge)
    chain_graph = nx.Graph()
    chain_graph.add_nodes_from(range(len(chains)))
    
    for i in range(len(chains)):
        for j in range(i + 1, len(chains)):
            # Check for shared hubs (direct connection)
            shared_hubs = chain_to_hubs[i] & chain_to_hubs[j]
            
            # Check for adjacent hubs (hubs from chain i that connect to hubs from chain j)
            connecting_hub_pairs = set()
            for hub_i in chain_to_hubs[i]:
                for hub_j in chain_to_hubs[j]:
                    if hub_i == hub_j:
                        continue  # Already counted in shared_hubs
                    if G_undirected.has_edge(hub_i, hub_j):
                        # These hubs are adjacent - chains are connected through them
                        connecting_hub_pairs.add((hub_i, hub_j))
            
            if shared_hubs or connecting_hub_pairs:
                # Collect all hubs involved in the connection
                all_connecting_hubs = set(shared_hubs)
                for hub_i, hub_j in connecting_hub_pairs:
                    all_connecting_hubs.add(hub_i)
                    all_connecting_hubs.add(hub_j)
                
                chain_graph.add_edge(i, j, 
                                    shared_hubs=shared_hubs,
                                    connecting_hub_pairs=connecting_hub_pairs,
                                    all_hubs=all_connecting_hubs)
    
    # Find all pairs of connected chains
    pairs = []
    for i, j in chain_graph.edges():
        all_hubs = chain_graph[i][j]['all_hubs']
        nodes = chain_nodes_map[i] + chain_nodes_map[j] + list(all_hubs)
        # Remove duplicates while preserving order
        seen = set()
        unique_nodes = []
        for n in nodes:
            if n not in seen:
                seen.add(n)
                unique_nodes.append(n)
        
        pairs.append({
            'chain_ids': (i, j),
            'nodes': unique_nodes,
            'chain_sizes': (len(chain_nodes_map[i]), len(chain_nodes_map[j])),
            'hub_count': len(all_hubs),
            'total_size': len(unique_nodes)
        })
    
    # Sort pairs by total size (descending)
    pairs = sorted(pairs, key=lambda x: x['total_size'], reverse=True)
    
    # Find all triplets of connected chains
    # A triplet is valid if all three chains form a connected subgraph (path or triangle)
    triplets = []
    visited_triplets = set()
    
    for i in chain_graph.nodes():
        neighbors_i = set(chain_graph.neighbors(i))
        for j in neighbors_i:
            if j <= i:
                continue
            neighbors_j = set(chain_graph.neighbors(j))
            # Find third chain connected to either i or j
            potential_k = (neighbors_i | neighbors_j) - {i, j}
            for k in potential_k:
                if k <= j:
                    continue
                triplet_key = tuple(sorted([i, j, k]))
                if triplet_key in visited_triplets:
                    continue
                visited_triplets.add(triplet_key)
                
                # Collect all hubs involved
                all_hubs = set()
                edges_in_triplet = []
                if chain_graph.has_edge(i, j):
                    all_hubs |= chain_graph[i][j]['all_hubs']
                    edges_in_triplet.append((i, j))
                if chain_graph.has_edge(j, k):
                    all_hubs |= chain_graph[j][k]['all_hubs']
                    edges_in_triplet.append((j, k))
                if chain_graph.has_edge(i, k):
                    all_hubs |= chain_graph[i][k]['all_hubs']
                    edges_in_triplet.append((i, k))
                
                # Need at least 2 edges to form a connected triplet
                if len(edges_in_triplet) < 2:
                    continue
                
                nodes = (chain_nodes_map[i] + chain_nodes_map[j] + 
                        chain_nodes_map[k] + list(all_hubs))
                # Remove duplicates
                seen = set()
                unique_nodes = []
                for n in nodes:
                    if n not in seen:
                        seen.add(n)
                        unique_nodes.append(n)
                
                triplets.append({
                    'chain_ids': (i, j, k),
                    'nodes': unique_nodes,
                    'chain_sizes': (len(chain_nodes_map[i]), len(chain_nodes_map[j]), len(chain_nodes_map[k])),
                    'hub_count': len(all_hubs),
                    'total_size': len(unique_nodes)
                })
    
    # Sort triplets by total size (descending)
    triplets = sorted(triplets, key=lambda x: x['total_size'], reverse=True)
    
    # Create schemes for pairs
    # Ensure 2nd pair doesn't share any chains with 1st pair
    if len(pairs) >= 1:
        p = pairs[0]
        schemes['interconnected_chain_pair_1st'] = {
            'sensors': p['nodes'],
            'description': f"Remove longest interconnected chain pair (chains of {p['chain_sizes'][0]} + {p['chain_sizes'][1]} + {p['hub_count']} hubs = {p['total_size']} total)",
            'rationale': 'Tests failure of two connected corridor segments with their junction hubs'
        }
        
        # Find the next largest pair that doesn't share any chains with the first
        first_pair_chains = set(p['chain_ids'])
        for p2 in pairs[1:]:
            second_pair_chains = set(p2['chain_ids'])
            if not (first_pair_chains & second_pair_chains):  # No overlap
                schemes['interconnected_chain_pair_2nd'] = {
                    'sensors': p2['nodes'],
                    'description': f"Remove 2nd longest interconnected chain pair (chains of {p2['chain_sizes'][0]} + {p2['chain_sizes'][1]} + {p2['hub_count']} hubs = {p2['total_size']} total)",
                    'rationale': 'Alternative interconnected chain pair failure (non-overlapping with 1st)'
                }
                break
    
    # Create schemes for triplets
    # Ensure 2nd triplet doesn't share any chains with 1st triplet
    if len(triplets) >= 1:
        t = triplets[0]
        schemes['interconnected_chain_triplet_1st'] = {
            'sensors': t['nodes'],
            'description': f"Remove longest interconnected chain triplet (chains of {t['chain_sizes'][0]} + {t['chain_sizes'][1]} + {t['chain_sizes'][2]} + {t['hub_count']} hubs = {t['total_size']} total)",
            'rationale': 'Tests failure of three connected corridor segments forming a network branch'
        }
        
        # Find the next largest triplet that doesn't share any chains with the first
        first_triplet_chains = set(t['chain_ids'])
        for t2 in triplets[1:]:
            second_triplet_chains = set(t2['chain_ids'])
            if not (first_triplet_chains & second_triplet_chains):  # No overlap
                schemes['interconnected_chain_triplet_2nd'] = {
                    'sensors': t2['nodes'],
                    'description': f"Remove 2nd longest interconnected chain triplet (chains of {t2['chain_sizes'][0]} + {t2['chain_sizes'][1]} + {t2['chain_sizes'][2]} + {t2['hub_count']} hubs = {t2['total_size']} total)",
                    'rationale': 'Alternative interconnected chain triplet failure (non-overlapping with 1st)'
                }
                break
    
    return schemes

# =============================================================================
# SCHEME 4: Critical Disconnection (Bridges + Floating Chains)
# =============================================================================
def scheme_critical_disconnection(G):
    """
    Combined scheme that removes critical hub nodes to maximize network disconnection:
    
    1. Critical bridge hub nodes (degree >= 3) - junction points that start dead ends
    2. Endpoint hubs of all chains with 5+ nodes - removing both endpoints makes 
       the chain "float" (disconnect from the network)
    
    This is a comprehensive disconnection attack targeting network vulnerabilities.
    """
    schemes = {}
    G_undirected = get_undirected_graph(G)
    degrees = dict(G_undirected.degree())
    
    # =========================================================================
    # Part 1: Find critical bridge hub nodes (degree >= 3)
    # =========================================================================
    bridges = list(nx.bridges(G_undirected))
    
    bridge_hub_nodes = set()
    for u, v in bridges:
        if degrees[u] >= 3:
            bridge_hub_nodes.add(u)
        if degrees[v] >= 3:
            bridge_hub_nodes.add(v)
    
    # =========================================================================
    # Part 2: Find endpoint hubs for all chains with 5+ nodes
    # =========================================================================
    chain_nodes = set(n for n, d in degrees.items() if d == 2)
    hub_nodes = set(n for n, d in degrees.items() if d >= 3)
    
    # Group chain nodes into connected chains
    chain_subgraph = G_undirected.subgraph(chain_nodes)
    chains = list(nx.connected_components(chain_subgraph))
    
    # Filter to chains with 5+ nodes
    long_chains = [c for c in chains if len(c) >= 5]
    
    # Collect all endpoint hubs needed to float all long chains
    floating_endpoint_hubs = set()
    chains_to_float = []
    total_chain_nodes_affected = 0
    
    for chain in long_chains:
        # Find all hubs connected to this chain
        endpoint_hubs = set()
        for node in chain:
            for neighbor in G_undirected.neighbors(node):
                if neighbor in hub_nodes:
                    endpoint_hubs.add(neighbor)
        
        # A chain can "float" if it has exactly 2 endpoint hubs
        if len(endpoint_hubs) == 2:
            floating_endpoint_hubs.update(endpoint_hubs)
            chains_to_float.append(len(chain))
            total_chain_nodes_affected += len(chain)
    
    # =========================================================================
    # Combine both sets of critical nodes
    # =========================================================================
    all_critical_nodes = bridge_hub_nodes | floating_endpoint_hubs
    
    if all_critical_nodes:
        # Sort by degree (most critical first)
        all_critical_sorted = sorted(all_critical_nodes, key=lambda x: degrees[x], reverse=True)
        
        # Build description
        n_bridge = len(bridge_hub_nodes)
        n_floating = len(floating_endpoint_hubs)
        n_chains = len(chains_to_float)
        
        schemes['critical_disconnection'] = {
            'sensors': all_critical_sorted,
            'description': f'Remove {len(all_critical_sorted)} critical hubs ({n_bridge} bridge nodes + endpoints for {n_chains} chains totaling {total_chain_nodes_affected} nodes)',
            'rationale': f'Combined attack: bridge hubs disconnect dead-end branches, floating chain endpoints disconnect {n_chains} chains of 5+ nodes'
        }
    
    return schemes


# =============================================================================
# SCHEME 5: Cluster/Community Removal
# =============================================================================
def scheme_cluster_removal(G):
    """
    Identify and remove densely connected clusters/communities.
    Uses community detection to find tightly-coupled sensor groups.
    """
    schemes = {}
    G_undirected = get_undirected_graph(G)
    
    # Use greedy modularity for community detection
    try:
        communities = list(nx.community.greedy_modularity_communities(G_undirected))
        communities = sorted(communities, key=len, reverse=True)
        
        if len(communities) >= 1:
            # Largest community
            largest = list(communities[0])
            schemes['cluster_largest'] = {
                'sensors': largest,
                'description': f'Remove largest sensor cluster ({len(largest)} sensors)',
                'rationale': 'Tests impact of losing an entire densely-connected region'
            }
        
        if len(communities) >= 2:
            # Second largest
            second = list(communities[1])
            schemes['cluster_second'] = {
                'sensors': second,
                'description': f'Remove second-largest cluster ({len(second)} sensors)',
                'rationale': 'Alternative cluster failure scenario'
            }
            
        if len(communities) >= 3:
            # Third largest community
            third = list(communities[2])
            schemes['cluster_third'] = {
                'sensors': third,
                'description': f'Remove third-largest cluster ({len(third)} sensors)',
                'rationale': 'Alternative cluster failure scenario'
            }
            
    except Exception as e:
        print(f"Community detection failed: {e}")
    
    return schemes

# =============================================================================
# SCHEME 6: Betweenness Centrality Removal
# =============================================================================
def scheme_betweenness_removal(G, percentages=[10, 20]):
    """
    Remove nodes with highest betweenness centrality.
    These nodes lie on many shortest paths between other nodes.
    """
    schemes = {}
    betweenness = nx.betweenness_centrality(G)
    sorted_nodes = sorted(betweenness.items(), key=lambda x: x[1], reverse=True)
    
    for pct in percentages:
        n_remove = max(1, int(len(sorted_nodes) * pct / 100))
        nodes_to_remove = [n for n, b in sorted_nodes[:n_remove]]
        schemes[f'betweenness_top_{pct}pct'] = {
            'sensors': nodes_to_remove,
            'description': f'Remove top {pct}% betweenness-central sensors ({n_remove} sensors)',
            'rationale': 'High betweenness nodes control information flow through the network'
        }
    
    return schemes

# =============================================================================
# SCHEME 8: Random Removal (Baseline)
# =============================================================================
def scheme_random_removal(G, percentages=[10, 20, 30, 40, 50], seed=42):
    """
    Random sensor removal as baseline comparison.
    """
    schemes = {}
    np.random.seed(seed)
    nodes = list(G.nodes())
    
    for pct in percentages:
        n_remove = max(1, int(len(nodes) * pct / 100))
        random_nodes = list(np.random.choice(nodes, size=n_remove, replace=False))
        schemes[f'random_{pct}pct'] = {
            'sensors': random_nodes,
            'description': f'Randomly remove {pct}% of sensors ({n_remove} sensors)',
            'rationale': 'Baseline for comparison with structural removal schemes'
        }
    
    return schemes

# =============================================================================
# SCHEME 9: Floating Hub Removal (Isolate Hub Sensors)
# =============================================================================
def scheme_floating_hub_removal(G, center_nodes=None):
    """
    Remove all neighbors of a hub sensor, making the hub "float" (isolated).
    The hub itself is kept - this tests how models predict a sensor when all
    its surrounding context has been removed.
    """
    schemes = {}
    G_undirected = get_undirected_graph(G)
    degrees = dict(G.degree())
    
    # If no center nodes specified, use top hubs
    if center_nodes is None:
        sorted_by_degree = sorted(degrees.items(), key=lambda x: x[1], reverse=True)
        center_nodes = [n for n, d in sorted_by_degree[:3]]
    
    for center in center_nodes:
        neighbors = list(G_undirected.neighbors(center))
        # Only remove the neighbors, NOT the hub itself
        schemes[f'floating_hub_{center}'] = {
            'sensors': neighbors,
            'description': f'Remove {len(neighbors)} neighbors of hub {center} (hub kept, made to float)',
            'rationale': f'Tests prediction of isolated hub {center} when all its connections are removed'
        }
    
    return schemes

# =============================================================================
# SCHEME 10: Path Disruption (End-to-End)
# =============================================================================
def scheme_path_disruption(G):
    """
    Remove sensors along critical paths between distant nodes.
    Finds the longest path, then finds a second longest path that doesn't
    share any nodes with the first (independent paths).
    
    Paths must start and end at "endpoint" nodes - either leaf nodes (degree 1)
    or hub nodes (degree >= 3). This prevents paths from ending in the middle
    of a chain (degree 2 nodes).
    """
    schemes = {}
    G_undirected = get_undirected_graph(G)
    degrees = dict(G_undirected.degree())
    
    # Only consider endpoint nodes: leaf (degree 1) or hub (degree >= 3)
    # Chain nodes (degree 2) should not be path endpoints
    endpoint_nodes = [n for n, d in degrees.items() if d != 2]
    
    # Find all long paths between endpoint nodes
    all_paths = []
    
    for i, source in enumerate(endpoint_nodes):
        for target in endpoint_nodes[i+1:]:  # Avoid duplicates
            try:
                path = nx.shortest_path(G_undirected, source, target)
                if len(path) >= 10:
                    all_paths.append((len(path), path))
            except:
                pass
    
    # Sort by length (descending)
    all_paths = sorted(all_paths, key=lambda x: x[0], reverse=True)
    
    if all_paths:
        # First: longest path
        longest_path = all_paths[0][1]
        schemes['path_longest'] = {
            'sensors': longest_path,
            'description': f'Remove sensors along longest path ({len(longest_path)} sensors)',
            'rationale': 'Disrupts communication between most distant network parts'
        }
        
        # Second: find the longest path that doesn't share any nodes with the first
        longest_path_set = set(longest_path)
        for length, path in all_paths[1:]:
            path_set = set(path)
            if not (path_set & longest_path_set):  # No overlap
                schemes['path_second'] = {
                    'sensors': path,
                    'description': f'Remove sensors along 2nd longest independent path ({len(path)} sensors)',
                    'rationale': 'Alternative path disruption (no overlap with longest path)'
                }
                break
    
    return schemes

# =============================================================================
# Visualization
# =============================================================================
def visualize_schemes(G, all_schemes, output_dir):
    """Create visualizations of each removal scheme."""
    G_undirected = get_undirected_graph(G)
    pos = nx.kamada_kawai_layout(G, scale=2.0)
    
    # Create a summary figure
    n_schemes = len(all_schemes)
    n_cols = 4
    n_rows = (n_schemes + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(24, 6 * n_rows))
    axes = axes.flatten() if n_schemes > 1 else [axes]
    
    for idx, (scheme_name, scheme_data) in enumerate(all_schemes.items()):
        ax = axes[idx]
        
        removed_sensors = set(scheme_data['sensors'])
        remaining_sensors = set(G.nodes()) - removed_sensors
        
        # Check if this is a floating_hub scheme - extract the hub node to highlight
        floating_hub_node = None
        if scheme_name.startswith('floating_hub_'):
            try:
                floating_hub_node = int(scheme_name.split('_')[-1])
            except:
                pass
        
        # Draw all edges faintly
        nx.draw_networkx_edges(G, pos, ax=ax, alpha=0.1, edge_color='gray', arrows=False)
        
        # Draw remaining nodes (excluding floating hub if present)
        regular_remaining = remaining_sensors - {floating_hub_node} if floating_hub_node else remaining_sensors
        nx.draw_networkx_nodes(G, pos, nodelist=list(regular_remaining), 
                               ax=ax, node_color='#2E86AB', node_size=30, alpha=0.6)
        
        # Draw floating hub node in GREEN (if this is a floating_hub scheme)
        if floating_hub_node is not None and floating_hub_node in remaining_sensors:
            nx.draw_networkx_nodes(G, pos, nodelist=[floating_hub_node],
                                   ax=ax, node_color='#2ECC71', node_size=120, alpha=1.0,
                                   edgecolors='darkgreen', linewidths=2)
        
        # Draw removed nodes (highlighted in red)
        nx.draw_networkx_nodes(G, pos, nodelist=list(removed_sensors),
                               ax=ax, node_color='#E63946', node_size=80, alpha=0.9,
                               edgecolors='black', linewidths=1)
        
        ax.set_title(f"{scheme_name}\n({len(removed_sensors)} sensors)", fontsize=10)
        ax.axis('off')
    
    # Hide unused axes
    for idx in range(len(all_schemes), len(axes)):
        axes[idx].axis('off')
    
    plt.suptitle('Sensor Removal Schemes\n(Red = Removed, Green = Floating Hub)', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_dir / 'removal_schemes_overview.png', dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"Scheme visualization saved to: {output_dir / 'removal_schemes_overview.png'}")

def generate_all_schemes(G):
    """Generate all removal schemes."""
    all_schemes = {}
    
    print("\nGenerating removal schemes...")
    
    # Generate each type of scheme
    schemes_generators = [
        ("Hub Removal", scheme_hub_removal, [G]),
        ("Peripheral Removal", scheme_peripheral_removal, [G]),
        ("Chain Removal", scheme_chain_removal, [G]),
        ("Interconnected Chain Removal", scheme_interconnected_chain_removal, [G]),
        ("Critical Disconnection", scheme_critical_disconnection, [G]),
        ("Cluster Removal", scheme_cluster_removal, [G]),
        ("Betweenness Removal", scheme_betweenness_removal, [G]),
        ("Random Removal", scheme_random_removal, [G]),
        ("Floating Hub Removal", scheme_floating_hub_removal, [G]),
        ("Path Disruption", scheme_path_disruption, [G]),
    ]
    
    for name, generator, args in schemes_generators:
        try:
            schemes = generator(*args)
            all_schemes.update(schemes)
            print(f"  ✓ {name}: {len(schemes)} scheme(s)")
        except Exception as e:
            print(f"  ✗ {name}: Failed - {e}")
    
    return all_schemes

def save_schemes(all_schemes, output_path):
    """Save schemes to JSON file."""
    # Convert to serializable format
    serializable = {}
    for name, data in all_schemes.items():
        serializable[name] = {
            'sensors': [int(s) for s in data['sensors']],
            'count': len(data['sensors']),
            'description': data['description'],
            'rationale': data['rationale']
        }
    
    with open(output_path, 'w') as f:
        json.dump(serializable, f, indent=2)
    
    print(f"\nSchemes saved to: {output_path}")

def print_scheme_summary(all_schemes):
    """Print a summary of all schemes."""
    print("\n" + "="*80)
    print("SENSOR REMOVAL SCHEMES SUMMARY")
    print("="*80)
    
    # Group by category
    categories = defaultdict(list)
    for name, data in all_schemes.items():
        if 'hub' in name:
            categories['Hub-Based'].append((name, data))
        elif 'peripheral' in name:
            categories['Peripheral'].append((name, data))
        elif 'chain' in name and 'floating' not in name:
            categories['Chain-Based'].append((name, data))
        elif 'bridge' in name or 'articulation' in name or 'floating' in name or 'disconnection' in name:
            categories['Connectivity-Critical'].append((name, data))
        elif 'cluster' in name:
            categories['Cluster-Based'].append((name, data))
        elif 'betweenness' in name:
            categories['Centrality-Based'].append((name, data))
        elif 'random' in name:
            categories['Random (Baseline)'].append((name, data))
        elif 'neighborhood' in name or 'floating_hub' in name:
            categories['Localized Failure'].append((name, data))
        elif 'path' in name:
            categories['Path-Based'].append((name, data))
        else:
            categories['Other'].append((name, data))
    
    for category, schemes in categories.items():
        print(f"\n{'─'*40}")
        print(f"📁 {category}")
        print('─'*40)
        for name, data in schemes:
            print(f"\n  🔹 {name}")
            print(f"     Sensors: {data['sensors'][:10]}{'...' if len(data['sensors']) > 10 else ''}")
            print(f"     Count: {len(data['sensors'])} sensors")
            print(f"     {data['description']}")
    
    print("\n" + "="*80)
    print(f"Total schemes generated: {len(all_schemes)}")
    print("="*80)

def main():
    # Paths
    repo_root = Path(__file__).resolve().parents[1]
    rel_path = repo_root / "raw_data" / "PEMSD8" / "PEMSD8.rel"
    output_dir = repo_root / "raw_data" / "PEMSD8"
    
    print(f"Loading relationship data from: {rel_path}")
    
    # Load data and create graph
    df = load_rel_file(rel_path)
    G = create_graph(df)
    print(f"Created graph with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges")
    
    # Generate all schemes
    all_schemes = generate_all_schemes(G)
    
    # Print summary
    print_scheme_summary(all_schemes)
    
    # Save to JSON
    save_schemes(all_schemes, output_dir / "sensor_removal_schemes.json")
    
    # Create visualization
    visualize_schemes(G, all_schemes, output_dir)
    
    print("\n✅ Done! Files generated:")
    print(f"   - {output_dir / 'sensor_removal_schemes.json'}")
    print(f"   - {output_dir / 'removal_schemes_overview.png'}")

if __name__ == "__main__":
    main()


