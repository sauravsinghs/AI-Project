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

# Set page config must be the first Streamlit command
st.set_page_config(page_title="Hamming Weight Tree", layout="wide")

def create_tree_visualization(tree):
    """
    Create a networkx graph visualization of the tree structure.
    
    Parameters:
        tree: HammingWeightTree instance to visualize
        
    Returns:
        matplotlib figure showing the tree structure
    """
    # Clear any existing plots
    plt.clf()
    
    G = nx.DiGraph()
    
    def add_nodes_edges(node, parent_id=None, level=0, max_depth=3):
        """
        Recursively add nodes and edges to the graph.
        Limits visualization depth to prevent overcrowding.
        
        Parameters:
            node: Current tree node
            parent_id: ID of parent node
            level: Current depth in tree
            max_depth: Maximum depth to visualize
        """
        if level > max_depth:  # Stop at max depth
            return None
            
        node_id = id(node)
        G.add_node(node_id)
        
        # Add node attributes with visual formatting
        if node['is_leaf']:
            # Format leaf node label showing stored codes
            codes = node['codes'][:3]  # Only show first 3 codes
            if len(node['codes']) > 3:
                codes_str = f"{len(node['codes'])} codes:\\n"
                codes_str += "\\n".join([code[1] for code in codes])
                codes_str += f"\\n+{len(node['codes'])-3} more"
            else:
                codes_str = "\\n".join([code[1] for code in codes])
            
            label = f"Leaf\\n{codes_str}"
            color = '#98FB98'  # Light green for leaf nodes
        else:
            # Format internal node label showing patterns
            patterns = list(node['children'].keys())[:2]
            if len(node['children']) > 2:
                patterns_str = f"{len(node['children'])} children:\\n"
                patterns_str += "\\n".join([str(p) for p in patterns])
                patterns_str += f"\\n+{len(patterns)-2} more"
            else:
                patterns_str = "\\n".join([str(p) for p in patterns])
            
            label = f"Node\\n{patterns_str}"
            color = '#87CEEB'  # Light blue for internal nodes
        
        G.nodes[node_id]['label'] = label
        G.nodes[node_id]['color'] = color
        G.nodes[node_id]['level'] = level
        
        if parent_id is not None:
            G.add_edge(parent_id, node_id)
        
        # Process children with depth limit
        if not node['is_leaf'] and level < max_depth:
            # Only process first 4 children for visualization clarity
            for pattern in list(node['children'].keys())[:4]:
                child = node['children'][pattern]
                child_id = add_nodes_edges(child, node_id, level + 1, max_depth)
                if child_id:
                    G.edges[node_id, child_id]['label'] = str(pattern)[:10]
        
        return node_id

    # Start visualization from root of first tree
    add_nodes_edges(tree.trees[0]['root'])
    
    # Handle empty tree case
    if len(G.nodes) == 0:
        plt.figure(figsize=(10, 6))
        plt.text(0.5, 0.5, "Empty Tree", horizontalalignment='center', fontsize=12)
        plt.axis('off')
        return plt
    
    # Create and configure the plot
    plt.figure(figsize=(10, 6))
    pos = nx.spring_layout(G, k=1.5, iterations=30)
    
    # Draw nodes
    nx.draw_networkx_nodes(G, pos, 
                          node_color=[G.nodes[node]['color'] for node in G.nodes()],
                          node_size=2500, alpha=0.8)
    
    # Draw edges and labels
    nx.draw_networkx_edges(G, pos, edge_color='gray', arrows=True, arrowsize=15)
    edge_labels = nx.get_edge_attributes(G, 'label')
    nx.draw_networkx_edge_labels(G, pos, edge_labels, font_size=7)
    labels = {node: G.nodes[node]['label'] for node in G.nodes()}
    nx.draw_networkx_labels(G, pos, labels, font_size=7)
    
    plt.title("Tree Structure", pad=20, fontsize=12)
    plt.axis('off')
    plt.tight_layout(pad=1.5)
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
        
        # Search interface
        st.subheader("Search")
        search_type = st.radio("Search Type", ["r-neighbor", "k-NN"])
        
        if search_type == "r-neighbor":
            # Radius search interface
            r = st.slider("Radius (r)", 0, params['n'], 1)
            query = st.text_input("Query code", "1" * params['n'])
            if st.button("Search"):
                if len(query) == params['n'] and all(c in '01' for c in query):
                    start_time = time.time()
                    results = st.session_state.tree.search_r(query, r)
                    end_time = time.time()
                    
                    st.write(f"Search completed in {end_time - start_time:.4f} seconds")
                    if results:
                        df = pd.DataFrame(results, columns=["ID", "Code", "Distance"])
                        st.dataframe(df)
                    else:
                        st.info("No results found")
                else:
                    st.error(f"Invalid query code! Must be {params['n']} bits long and contain only 0s and 1s.")
        else:
            # k-NN search interface
            k = st.slider("k", 1, 5, 3)
            query = st.text_input("Query code", "1" * params['n'])
            
            if st.button("Search"):
                if len(query) == params['n'] and all(c in '01' for c in query):
                    st.session_state.search_in_progress = True
                    results_placeholder = st.empty()
                    results_placeholder.info("Searching... This may take a moment.")
                    
                    try:
                        start_time = time.time()
                        results = st.session_state.tree.search_knn(query, k)
                        end_time = time.time()
                        
                        results_placeholder.empty()
                        st.write(f"Search completed in {end_time - start_time:.4f} seconds")
                        if results:
                            df = pd.DataFrame(results, columns=["ID", "Code", "Distance"])
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
        try:
            fig = create_tree_visualization(st.session_state.tree)
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
                      'Insertions', 'Searches', 'Cache Hits', 'Cache Misses', 'Forest Size'],
            'Value': [
                str(stats.get('total_nodes', 0)),
                str(stats.get('total_codes', 0)),
                str(stats.get('cache_size', 0)),
                f"{stats.get('uptime', 0):.2f}",
                str(stats.get('insertions', 0)),
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
        metrics = ['Insertions', 'Searches', 'Cache Hits', 'Cache Misses']
        values = [
            int(stats.get('insertions', 0)),
            int(stats.get('searches', 0)),
            int(stats.get('cache_hits', 0)),
            int(stats.get('cache_misses', 0))
        ]
        
        plt.clf()
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.bar(metrics, values)
        plt.xticks(rotation=45)
        st.pyplot(fig)
    
    # Handle page rerun if needed
    if st.session_state.needs_rerun:
        st.session_state.needs_rerun = False
        st.rerun()

if __name__ == "__main__":
    main()