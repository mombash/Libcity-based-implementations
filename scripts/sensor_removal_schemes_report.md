# Sensor Removal Schemes Report

This document describes the various sensor removal schemes implemented in `sensor_removal_schemes.py`. Each scheme represents a different fault analysis strategy designed to study how different sensor failure patterns affect model performance in traffic prediction tasks.

## Overview

The script generates multiple removal schemes based on network topology analysis of sensor networks. These schemes simulate different types of sensor failures to understand model robustness and identify critical network vulnerabilities.

## Scheme Categories

### 1. Hub-Based Removal Schemes

#### **Scheme 1: Hub Removal (High-Degree Nodes)**
- **Motivation**: Hubs are critical nodes that connect many parts of the network. Their failure has cascading effects on network connectivity and information flow.
- **Strategy**: Removes the most connected sensors based on degree centrality (node degree in the directed graph). Targets top 10%, 20%, and 30% of highest-degree hub sensors.
- **Generated Schemes**: `hub_top_10pct`, `hub_top_20pct`, `hub_top_30pct`

#### **Scheme 9: Floating Hub Removal (Isolate Hub Sensors)**
- **Motivation**: Tests how models predict sensor values when all surrounding context has been removed. The hub sensor becomes "floating" (isolated) while remaining in the network.
- **Strategy**: For each of the top 3 hub sensors (by degree), removes ALL of its neighbors while keeping the hub itself. This isolates the hub sensor, forcing the model to predict it without any connected context.
- **Generated Schemes**: `floating_hub_{center_node}` for each of the top 3 hubs

### 2. Peripheral-Based Removal Schemes

#### **Scheme 2: Peripheral Removal (Low-Degree Nodes)**
- **Motivation**: Peripheral sensors have limited local connectivity and represent edge cases where failures are more isolated. Tests how models handle sensors with minimal network integration.
- **Strategy**: Removes sensors with the lowest connectivity (degree), targeting 30%, 40%, 50%, and 60% of the lowest-degree peripheral sensors.
- **Generated Schemes**: `peripheral_30pct`, `peripheral_40pct`, `peripheral_50pct`, `peripheral_60pct`

### 3. Chain-Based Removal Schemes

#### **Scheme 3: Chain Removal (Linear Sequences)**
- **Motivation**: Tests the impact of losing a complete linear segment of the network. Chains represent corridors or pathways where sensors are sequentially connected.
- **Strategy**: Identifies linear chains of sensors (sequences where each node has degree 2 except endpoints). Removes only the longest chain in the network.
- **Generated Schemes**: `chain_longest`

#### **Scheme 3b: Interconnected Chain Removal**
- **Motivation**: Captures chains that form continuous corridors through the network, separated only by hub junctions. Tests failure of interconnected network segments that work together as a functional unit.
- **Strategy**:
  - Identifies all chains (degree-2 nodes)
  - Builds a "chain connectivity graph" where chains are connected if they share hubs OR their endpoint hubs are directly adjacent
  - Removes pairs and triplets of interconnected chains along with their connecting hub nodes
  - Ensures non-overlapping schemes by selecting alternative interconnected groups
- **Generated Schemes**: `interconnected_chain_pair_1st`, `interconnected_chain_pair_2nd`, `interconnected_chain_triplet_1st`, `interconnected_chain_triplet_2nd`

### 4. Connectivity-Critical Removal Schemes

#### **Scheme 4: Critical Disconnection (Bridges + Floating Chains)**
- **Motivation**: Comprehensive disconnection attack targeting network vulnerabilities. Combines bridge removal (which creates dead-end branches) with floating chain endpoints (which disconnects entire corridor segments).
- **Strategy**:
  - **Part 1**: Identifies critical bridge hub nodes (degree ≥ 3) that are part of network bridges (edges whose removal increases connected components)
  - **Part 2**: Finds endpoint hubs of all chains with 5+ nodes; removing these endpoints makes the chains "float" (disconnect from the network)
  - Combines both sets of critical nodes for maximum disconnection impact
- **Generated Schemes**: `critical_disconnection`

### 5. Cluster-Based Removal Schemes

#### **Scheme 5: Cluster/Community Removal**
- **Motivation**: Tests the impact of losing an entire densely-connected region. Communities represent functionally related sensor groups that work together.
- **Strategy**: Uses greedy modularity community detection to identify tightly-coupled sensor groups. Removes the largest, second-largest, and third-largest communities.
- **Generated Schemes**: `cluster_largest`, `cluster_second`, `cluster_third`

### 6. Centrality-Based Removal Schemes

#### **Scheme 6: Betweenness Centrality Removal**
- **Motivation**: High betweenness centrality nodes control information flow through the network, acting as critical intermediaries. Their removal disrupts communication pathways.
- **Strategy**: Calculates betweenness centrality (fraction of shortest paths passing through each node) and removes the top 10% and 20% most central nodes.
- **Generated Schemes**: `betweenness_top_10pct`, `betweenness_top_20pct`

### 7. Path-Based Removal Schemes

#### **Scheme 10: Path Disruption (End-to-End)**
- **Motivation**: Disrupts communication between the most distant network parts. Paths represent critical communication routes that span the entire network.
- **Strategy**:
  - Finds all paths between "endpoint nodes" (degree 1 or degree ≥ 3 nodes, excluding degree-2 chain nodes)
  - Identifies the longest path and a second longest path with no overlapping nodes
  - Removes all sensors along these critical end-to-end pathways
- **Generated Schemes**: `path_longest`, `path_second`

### 8. Random Removal (Baseline)

#### **Scheme 8: Random Removal**
- **Motivation**: Provides a baseline for comparison with structural removal schemes. Random failures represent non-targeted, unpredictable sensor outages.
- **Strategy**: Randomly selects sensors for removal at various percentages (10%, 20%, 30%, 40%, 50%) using a fixed seed for reproducibility.
- **Generated Schemes**: `random_10pct`, `random_20pct`, `random_30pct`, `random_40pct`, `random_50pct`

## Scheme Generation Process

The `generate_all_schemes()` function orchestrates the creation of all schemes:

1. **Graph Construction**: Loads sensor relationship data and creates a directed NetworkX graph
2. **Scheme Generation**: Applies each scheme generator function to create removal strategies
3. **Output**: Saves schemes to JSON format and generates visualization plots

## Output Files

- **`sensor_removal_schemes.json`**: Contains all removal schemes with sensor lists, descriptions, and rationales
- **`removal_schemes_overview.png`**: Visualization showing removed sensors (red) and remaining network for each scheme

## Applications

These schemes are designed for:
- **Fault Analysis**: Understanding how different failure patterns affect prediction accuracy
- **Model Robustness Testing**: Evaluating model performance under various sensor failure scenarios
- **Network Vulnerability Assessment**: Identifying critical sensors and network structures
- **Comparative Analysis**: Benchmarking structural failures against random failures

## Technical Notes

- All schemes operate on undirected versions of the graph for connectivity analysis
- Degree calculations use the original directed graph structure
- Community detection uses NetworkX's greedy modularity algorithm
- Centrality measures are computed on the directed graph
- Schemes are designed to be non-overlapping where possible to provide independent test cases


