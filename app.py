"""
Hamming Weight Tree GUI Application
This Streamlit application provides a visual interface for interacting with the Hamming Weight Tree data structure.
It allows users to:
1. Configure tree parameters
2. Insert and search binary codes
3. Visualize the tree structure
4. Monitor performance metrics
"""

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
from hamming_weight_tree import HammingWeightTree
import time
import importlib
import random

# Set page config must be the first Streamlit command
st.set_page_config(page_title="Hamming Weight Tree", layout="wide")

def create_tree_visualization(tree, tree_index=0):
    """
    Create a networkx graph visualization of the tree structure using a tidy tree layout (Reingold-Tilford style)
    with custom child ordering: for each parent, children are ordered left, middle, right, and each parent is centered above its children.
    This ensures a wide, balanced, non-crossing tree with the desired child order at each level.
    
    Parameters:
        tree: HammingWeightTree instance to visualize
        tree_index: Index of the tree to visualize
        
    Returns:
        matplotlib figure showing the tree structure
    """
    plt.clf()
    G = nx.DiGraph()

    # Build the tree and collect node info
    node_info = {}
    edges = []
    max_depth = 5
    root = tree.trees[tree_index]['root']

    def build_graph(node, parent_id=None, pattern=None, depth=0):
        if depth > max_depth:
            return
        node_id = id(node)
        
        # Format pattern display for better readability
        pattern_obj = node.get('pattern', None)
        if pattern_obj is None:
            pattern_display = "None"
        else:
            # Ensure pattern displays as a proper tuple
            if isinstance(pattern_obj, tuple) and len(pattern_obj) == 1:
                # Convert single-value tuple to two-value tuple for display
                pattern_display = f"({pattern_obj[0]}, 0)"
            else:
                pattern_display = str(pattern_obj)
        
        if node['is_leaf']:
            num_codes = len(node['codes'])
            if num_codes > 0:
                sample_code = node['codes'][0][1]
                label = f"Leaf\nPattern: {pattern_display}\nCodes: {num_codes}\nSample: {sample_code}"
            else:
                label = f"Leaf\nPattern: {pattern_display}\nCodes: 0"
            color = '#98FB98'
        else:
            num_children = len(node['children'])
            label = f"Node\nPattern: {pattern_display}\nChildren: {num_children}"
            color = '#87CEEB'
        node_info[node_id] = (label, color, depth)
        if parent_id is not None:
            # Format edge label
            if pattern and isinstance(pattern, tuple) and len(pattern) == 1:
                edge_pattern = f"({pattern[0]}, 0)"
            else:
                edge_pattern = str(pattern)
            edge_label = f"Pattern: {edge_pattern}"
            edges.append((parent_id, node_id, edge_label))
        if not node['is_leaf']:
            # Sort children by Qd pattern for consistent visualization
            children = [(str(k), k, node['children'][k]) for k in sorted(list(node['children'].keys()), key=str)]
            for str_pattern, pattern_key, child in children:
                build_graph(child, node_id, pattern_key, depth+1)
    build_graph(root)

    # Update the layout function to consider prefix-based structure
    def layout(node, depth=0, x=0, next_x=[0]):
        node_id = id(node)
        if node['is_leaf'] or not node['children'] or depth > max_depth:
            pos = {node_id: (next_x[0], -depth)}
            next_x[0] += 1
            return pos, next_x[0] - 1, next_x[0] - 1
        
        # Sort children by pattern
        children = [(k, node['children'][k]) for k in sorted(list(node['children'].keys()))]
        child_pos = {}
        child_xs = []
        min_x = float('inf')
        max_x = float('-inf')
        
        for k, child in children:
            pos, cmin, cmax = layout(child, depth+1, x, next_x)
            child_pos.update(pos)
            child_xs.append((cmin, cmax))
            min_x = min(min_x, cmin)
            max_x = max(max_x, cmax)
        
        # Center this node above its children
        center = (min_x + max_x) / 2 if len(child_xs) > 0 else next_x[0]
        child_pos[node_id] = (center, -depth)
        return child_pos, min_x, max_x

    pos, _, _ = layout(root)

    # Build the graph
    for node_id, (label, color, depth) in node_info.items():
        G.add_node(node_id, label=label, color=color)
    for parent_id, node_id, pattern in edges:
        G.add_edge(parent_id, node_id, label=pattern)

    if len(G.nodes) == 0:
        plt.figure(figsize=(10, 6))
        plt.text(0.5, 0.5, "Empty Tree", horizontalalignment='center', fontsize=12)
        plt.axis('off')
        return plt

    plt.figure(figsize=(32, 16))
    nx.draw_networkx_nodes(G, pos, 
                          node_color=[G.nodes[node]['color'] for node in G.nodes()],
                          node_size=3500, alpha=0.9)
    nx.draw_networkx_edges(G, pos, edge_color='gray', arrows=True, arrowsize=18)
    edge_labels = nx.get_edge_attributes(G, 'label')
    nx.draw_networkx_edge_labels(G, pos, edge_labels, font_size=12)
    labels = {node: G.nodes[node]['label'] for node in G.nodes()}
    nx.draw_networkx_labels(G, pos, labels, font_size=14)
    plt.title("Hamming Weight Tree Visualization", pad=20, fontsize=18)
    plt.axis('off')
    plt.tight_layout(pad=2.0)
    return plt

def main():
    """
    Main application function that sets up the Streamlit interface.
    Handles:
    1. Configuration parameters
    2. Tree operations (insert, search)
    3. Visualization
    4. Statistics display
    """
    # Initialize session state for persistent storage
    if 'initialized' not in st.session_state:
        st.session_state.initialized = True
        st.session_state.tree = None
        st.session_state.last_params = None
        st.session_state.needs_rerun = False
        st.session_state.search_in_progress = False
    
    # Sidebar configuration
    st.sidebar.header("Configuration")
    params = {
        'n': st.sidebar.slider("Code Length (n)", 4, 16, 8),
        't': st.sidebar.slider("Node Threshold (t)", 2, 10, 4),
        'd': st.sidebar.slider("Number of Substrings (d)", 1, 2, 2),
        'forest_size': st.sidebar.slider("Forest Size", 1, 5, 1)
    }
    distance_type = st.sidebar.radio("Distance Type", ["Hamming", "Angular"])
    if st.sidebar.button("Random Bit Reorder"):
        order = list(range(params['n']))
        random.shuffle(order)
        # Export all codes, reorder, and rebuild tree
        all_codes = st.session_state.tree.export_all_codes()
        def reorder_code(code):
            return ''.join(code[i] for i in order)
        reordered_codes = [(id, reorder_code(code)) for id, code in all_codes]
        st.session_state.tree.rebuild_tree(reordered_codes)
        st.success(f"Bits reordered: {order}")
        st.session_state.needs_rerun = True
    
    # Create or recreate tree if parameters change
    if (st.session_state.tree is None or 
        st.session_state.last_params != params):
        try:
            st.session_state.tree = HammingWeightTree(**params)
            st.session_state.last_params = params.copy()
            st.info(f"Tree recreated with new parameters: forest_size={params['forest_size']}")
        except Exception as e:
            st.error(f"Error creating tree: {str(e)}")
            return
    
    # Split interface into two columns
    col1, col2 = st.columns(2)
    
    # Left column: Operations
    with col1:
        st.header("Operations")
        
        # Sample data loading
        if st.button("Load Sample Data"):
            try:
                with st.spinner("Loading sample data..."):
                    sample_codes = [
                        "1" * params['n'],
                        "0" * params['n'],
                        "10" * (params['n']//2) + "0" * (params['n']%2)
                    ]
                    
                    for i, code in enumerate(sample_codes):
                        st.session_state.tree.insert(code, f"sample_{i}")
                    
                st.success(f"Inserted {len(sample_codes)} codes")
                st.session_state.needs_rerun = True
            except Exception as e:
                st.error(f"Error: {str(e)}")
        
        # Code insertion interface
        st.subheader("Insert Binary Code")
        code_input = st.text_input("Enter binary code", "1" * params['n'])
        if st.button("Insert"):
            if len(code_input) == params['n'] and all(c in '01' for c in code_input):
                st.session_state.tree.insert(code_input)
                st.success(f"Code {code_input} inserted successfully!")
                st.session_state.needs_rerun = True
            else:
                st.error(f"Invalid code! Must be {params['n']} bits long and contain only 0s and 1s.")
        
        # Delete code interface
        st.subheader("Delete Binary Code")
        delete_code = st.text_input("Enter code to delete", "1" * params['n'])
        if st.button("Delete"):
            if len(delete_code) == params['n'] and all(c in '01' for c in delete_code):
                if st.session_state.tree.delete(delete_code):
                    st.success(f"Code {delete_code} deleted successfully!")
                    st.session_state.needs_rerun = True
                else:
                    st.warning(f"Code {delete_code} not found in the tree.")
            else:
                st.error(f"Invalid code! Must be {params['n']} bits long and contain only 0s and 1s.")
        
        # Search interface
        st.subheader("Search")
        search_type = st.radio("Search Type", ["r-neighbor", "k-NN"])
        
        if search_type == "r-neighbor":
            r = st.slider("Radius (r)", 0, params['n'], 1)
            query = st.text_input("Query code", "1" * params['n'])
            if st.button("Search"):
                if len(query) == params['n'] and all(c in '01' for c in query):
                    start_time = time.time()
                    if distance_type == "Hamming":
                        results = st.session_state.tree.search_r(query, r)
                    else:
                        # For angular distance, scale r to be between 0 and 1
                        angular_r = r / params['n']
                        results = st.session_state.tree.search_r_angular(query, angular_r)
                    end_time = time.time()
                    st.write(f"Search completed in {end_time - start_time:.4f} seconds")
                    if results:
                        df = pd.DataFrame(results, columns=["ID", "Code", "Distance"])
                        # Verify all results have distance <= r
                        if distance_type == "Hamming":
                            # Add angular distance for comparison
                            df["Angular Distance"] = df["Code"].apply(lambda c: st.session_state.tree.angular_distance(query, c))
                        else:
                            # Add hamming distance for comparison
                            df["Hamming Distance"] = df["Code"].apply(lambda c: st.session_state.tree.hamming_distance(query, c))
                        st.dataframe(df)
                    else:
                        st.info("No results found")
                else:
                    st.error(f"Invalid query code! Must be {params['n']} bits long and contain only 0s and 1s.")
        else:
            k = st.slider("k", 1, 5, 3)
            query = st.text_input("Query code", "1" * params['n'])
            if st.button("Search"):
                if len(query) == params['n'] and all(c in '01' for c in query):
                    st.session_state.search_in_progress = True
                    results_placeholder = st.empty()
                    results_placeholder.info("Searching... This may take a moment.")
                    try:
                        start_time = time.time()
                        if distance_type == "Hamming":
                            results = st.session_state.tree.search_knn(query, k)
                        else:
                            results = st.session_state.tree.search_knn_angular(query, k)
                        end_time = time.time()
                        results_placeholder.empty()
                        st.write(f"Search completed in {end_time - start_time:.4f} seconds")
                        if results:
                            df = pd.DataFrame(results, columns=["ID", "Code", "Distance"])
                            # Add comparative distance
                            if distance_type == "Hamming":
                                # Add angular distance for comparison
                                df["Angular Distance"] = df["Code"].apply(lambda c: st.session_state.tree.angular_distance(query, c))
                                # Verify results are actually nearest
                                for i, row in df.iterrows():
                                    if row["Code"] == query:
                                        st.success(f"Found exact match: {row['Code']}")
                            else:
                                # Add hamming distance for comparison
                                df["Hamming Distance"] = df["Code"].apply(lambda c: st.session_state.tree.hamming_distance(query, c))
                                # Verify results are actually nearest
                                for i, row in df.iterrows():
                                    if row["Code"] == query:
                                        st.success(f"Found exact match: {row['Code']}")
                            st.dataframe(df)
                        else:
                            st.info("No results found")
                    except Exception as e:
                        st.error(f"Search error: {str(e)}")
                    finally:
                        st.session_state.search_in_progress = False
                else:
                    st.error(f"Invalid query code! Must be {params['n']} bits long and contain only 0s and 1s.")
    
    # Right column: Visualization and Statistics
    with col2:
        st.header("Visualization")
        
        # Tree structure visualization
        st.subheader("Tree Structure")
        # Allow user to select which tree in the forest to visualize
        forest_size = len(st.session_state.tree.trees)
        tree_index = 0
        if forest_size > 1:
            tree_index = st.selectbox("Select tree to visualize", list(range(forest_size)), format_func=lambda i: f"Tree {i}")
        try:
            fig = create_tree_visualization(st.session_state.tree, tree_index=tree_index)
            st.pyplot(fig)
        except Exception as e:
            st.error(f"Error creating visualization: {str(e)}")
            plt.clf()
        
        # Performance statistics display
        st.subheader("Performance Statistics")
        stats = st.session_state.tree.get_stats()
        
        # Create statistics table
        stats_data = {
            'Metric': ['Total Nodes', 'Total Codes', 'Cache Size', 'Uptime (s)',
                      'Insertions', 'Deletions', 'Searches', 'Cache Hits', 'Cache Misses', 'Forest Size'],
            'Value': [
                str(stats.get('total_nodes', 0)),
                str(stats.get('total_codes', 0)),
                str(stats.get('cache_size', 0)),
                f"{stats.get('uptime', 0):.2f}",
                str(stats.get('insertions', 0)),
                str(stats.get('deletions', 0)),
                str(stats.get('searches', 0)),
                str(stats.get('cache_hits', 0)),
                str(stats.get('cache_misses', 0)),
                str(params['forest_size'])
            ]
        }
        stats_df = pd.DataFrame(stats_data)
        st.dataframe(stats_df)
        
        # Tree distribution visualization
        if 'tree_distribution' in stats:
            st.subheader("Tree Distribution")
            tree_dist = stats['tree_distribution']
            tree_data = pd.DataFrame({
                'Tree': [f"Tree {i}" for i in range(len(tree_dist))],
                'Codes': tree_dist
            })
            st.bar_chart(tree_data.set_index('Tree'))
        
        # Performance metrics visualization
        st.subheader("Performance Metrics")
        metrics = ['Insertions', 'Deletions', 'Searches', 'Cache Hits', 'Cache Misses']
        values = [
            int(stats.get('insertions', 0)),
            int(stats.get('deletions', 0)),
            int(stats.get('searches', 0)),
            int(stats.get('cache_hits', 0)),
            int(stats.get('cache_misses', 0))
        ]
        
        plt.clf()
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.bar(metrics, values)
        plt.xticks(rotation=45)
        plt.title("Operation Counts")
        st.pyplot(fig)
    
    # Handle page rerun if needed
    if st.session_state.needs_rerun:
        st.session_state.needs_rerun = False
        st.rerun()

if __name__ == "__main__":
    main()