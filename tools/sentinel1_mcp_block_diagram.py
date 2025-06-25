from dis import Positions
import os
from turtle import position
from graphviz import Digraph
import graphviz

# Set this to your actual Graphviz bin path if needed
GRAPHVIZ_BIN_PATH = r"C:\Program Files\Graphviz\bin"

if GRAPHVIZ_BIN_PATH not in os.environ["PATH"]:
    os.environ["PATH"] += os.pathsep + GRAPHVIZ_BIN_PATH

# Optionally, set the engine path directly for graphviz >=0.20
graphviz.set_default_engine('dot')

# Create a new diagram with improved settings
diagram = Digraph('Sentinel1_MCP_Workflow', filename='sentinel1_mcp_workflow', format='png')

# Set global graph attributes for better layout and size
diagram.attr(
    rankdir='LR',      # Left to right layout
    size='12,8',       # Back to original size as requested
    dpi='300',         # Higher resolution
    fontname='Arial',
    fontsize='16',     # Increased font size to 16
    nodesep='0.6',     # Reduced horizontal spacing
    ranksep='0.8',     # Reduced vertical spacing
    splines='polyline',  # Polyline edges (work better with dashed lines)
    overlap='false',   # Prevent node overlap
    concentrate='true' # Merge edges where possible
)

# Define node styles with improved formatting
node_attrs = {
    'fontname': 'Arial',
    'fontsize': '16',     # Increased font size to 16
    'penwidth': '2',
    'margin': '0.2,0.1',  # Reduced margin for more compact nodes
    'height': '1.0',      # Increased height
    'width': '2.0'        # Increased width
}

# Create nodes with better styling
diagram.node('Admin', 'Admin Agent', 
           shape='box3d', 
           style='filled,rounded', 
           fillcolor='#4285F4', 
           fontcolor='white',
           **node_attrs)

diagram.node('MCP', 'MCP Server', 
           shape='cylinder', 
           style='filled', 
           fillcolor='#FBBC05', 
           fontcolor='black',
           **node_attrs)

# Worker agent nodes with distinct colors
diagram.node('Tile', 'Sentinel1TileCoverageAgent\n(Tile Coverage)', 
           shape='box', 
           style='filled,rounded', 
           fillcolor='#34A853', 
           fontcolor='white',
           **node_attrs)

diagram.node('Extract', 'Sentinel1DataExtractionAgent\n(Data Extraction)', 
           shape='box', 
           style='filled,rounded', 
           fillcolor='#EA4335', 
           fontcolor='white',
           **node_attrs)

diagram.node('Train', 'Sentinel1ModelTrainingAgent\n(Model Training)', 
           shape='box', 
           style='filled,rounded', 
           fillcolor='#673AB7', 
           fontcolor='white',
           **node_attrs)

diagram.node('Infer', 'Sentinel1InferenceAgent\n(Inference)', 
           shape='box', 
           style='filled,rounded', 
           fillcolor='#00ACC1', 
           fontcolor='white',
           **node_attrs)

# Define edge styles
edge_attrs = {
    'fontname': 'Arial',
    'fontsize': '16',     # Increased font size to 16
    'penwidth': '2',
    'fontcolor': '#333333',
    'arrowsize': '1.2',
    'arrowhead': 'normal',
    'arrowtail': 'none',
    'minlen': '1'         # Reduced minimum edge length for more compact layout
}

# Task assignment edges (MCP to agents)
diagram.edge('Admin', 'MCP', 
           label=' Task/Memory\nManagement ', 
           color='#4285F4', 
           **edge_attrs)

diagram.edge('MCP', 'Tile', 
           label=' 1. Assign Tile\nCoverage Task ', 
           color='#34A853', 
           **edge_attrs)

diagram.edge('Tile', 'MCP', 
           xlabel=' Tile Coverage\nResult ', 
           style='dashed', 
           color='#34A853', 
           **edge_attrs)

diagram.edge('MCP', 'Extract', 
           label=' 2. Assign Data\nExtraction Task ', 
           color='#EA4335', 
           **edge_attrs)

diagram.edge('Extract', 'MCP', 
           xlabel=' Extracted\nData ', 
           style='dashed', 
           color='#EA4335', 
           **edge_attrs)

diagram.edge('MCP', 'Train', 
           label=' 3. Assign Model\nTraining Task ', 
           color='#673AB7', 
           **edge_attrs)

diagram.edge('Train', 'MCP', 
           xlabel=' Trained\nModel ', 
           style='dashed', 
           color='#673AB7', 
           **edge_attrs)

diagram.edge('MCP', 'Infer', 
           label=' 4. Assign\nInference Task ', 
           color='#00ACC1', 
           **edge_attrs)

diagram.edge('Infer', 'MCP', 
           xlabel=' Inference\nResults ', 
           style='dashed', 
           color='#00ACC1', 
           **edge_attrs)

# Data flow arrows between agents (workflow sequence)
diagram.edge('Tile', 'Extract', 
           label=' GeoJSON/\nTile Info ', 
           color='#000000', 
           style='bold', 
           **edge_attrs)

diagram.edge('Extract', 'Train', 
           label=' Extracted\nData ', 
           color='#000000', 
           style='bold', 
           **edge_attrs)

diagram.edge('Train', 'Infer', 
           label=' Trained\nModel ', 
           color='#000000', 
           style='bold', 
           **edge_attrs)

# Add a title to the diagram
diagram.attr(label='\n\nMCP Server Architecture for Sentinel-1 Crop Type Classification using PSETAE\n', labelloc='l', fontsize='24', fontname='Arial Bold', fontcolor='black')
# Render the diagram with higher quality settings
diagram.render(cleanup=True, view=False)
print('Block diagram generated as sentinel1_mcp_workflow.png')
print('The diagram should be clearer and more visually appealing now.')
