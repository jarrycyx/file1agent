import os
import graphviz
from loguru import logger
from typing import Dict, List, Optional, Tuple, Set

try:
    import graphviz

    GRAPHVIZ_AVAILABLE = True
except ImportError:
    GRAPHVIZ_AVAILABLE = False
    logger.warning("graphviz not installed. Cannot visualize graph.")

def visualize_graph(file_relationships: Dict[str, List[str]], all_files: List[str], save_fig_path: str) -> None:
    """
    Visualize the file relationship graph using graphviz

    Args:
        file_relationships: Dictionary mapping file paths to list of referenced file paths
        all_files: List of all files in the project (including binary files)
    """
    if not GRAPHVIZ_AVAILABLE:
        logger.error("graphviz is not available. Cannot visualize graph.")
        return
    try:
        # Create a directed graph with optimized layout
        dot = graphviz.Digraph(
            comment="File Relationship Graph",
            graph_attr={
                "rankdir": "LR",  # Left to right layout instead of top to bottom
                "splines": "ortho",  # Use orthogonal lines for cleaner look
                "nodesep": "0.8",  # Increase separation between nodes
                "ranksep": "1.0",  # Increase separation between ranks
                "fontname": "Arial",
                "fontsize": "12",
                "concentrate": "true",  # Merge parallel edges
            },
            node_attr={
                "shape": "box",
                "style": "rounded,filled",
                "fillcolor": "lightblue",
                "fontname": "Arial",
                "fontsize": "10",
            },
            edge_attr={"fontname": "Arial", "fontsize": "9"},
        )

        # Then, add edges for file relationships
        for file_path, referenced_files in file_relationships.items():
            filename = os.path.basename(file_path)
            dot.node(file_path, filename)
            # Add edges to referenced files
            for ref_file in referenced_files:
                # Add an edge from the current file to the referenced file
                dot.edge(file_path, ref_file)

        # Save the graph
        dot.render(save_fig_path, format="png", cleanup=True)

        logger.info(f"File relationship graph saved to {save_fig_path}.png")
    except Exception as e:
        logger.error(f"Error visualizing graph: {e}")
        return
