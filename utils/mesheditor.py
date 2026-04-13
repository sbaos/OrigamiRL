import tkinter as tk
from tkinter import simpledialog, messagebox, filedialog
import json
import os
from datetime import datetime
import math

class SegmentAssignerApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Line Segment Value Assigner")
        
        self.canvas_width = 600
        self.canvas_height = 600
        self.margin = 100
        
        self.rows = 9
        self.cols = 9
        
        # Preset values with symbolic labels
        self.preset_values = [
            (-math.pi, "-π"),
            (-math.pi*3/4, "-3π/4"),
            (-math.pi/2, "-π/2"),
            (-math.pi/4, "-π/4"),
            (0.0, "0"),
            (math.pi/4, "π/4"),
            (math.pi/2, "π/2"),
            (math.pi*3/4, "3π/4"),
            (math.pi, "π"),
        ]
        
        sorted_vals = sorted([v for v, _ in self.preset_values])
        self.min_val = sorted_vals[0]
        self.max_val = sorted_vals[-1]
        
        n = len(sorted_vals)
        if n % 2 == 1:
            self.mid_val = sorted_vals[n // 2]
        else:
            self.mid_val = (sorted_vals[n // 2 - 1] + sorted_vals[n // 2]) / 2.0
            
        self.current_tool_value = None
        self.current_tool_label = "None"  # Display label for current value
        
        # Grid structure
        self.nodes = {}  # node_id -> (x, y) canvas position
        self.init_edges = {}  # edge_key (u, v) sorted -> True, init grid edges
        self.border_edges = set()  # edges on the grid perimeter (fixed value 0)
        self.imported_boundary_nodes = set()  # Boundary nodes loaded from import
        self.imported_border_points = set()  # Border points (on grid perimeter) loaded from import
        self.imported_no_direction_edges = set()  # Edges with no direction from import
        
        # Edge management - directed edges with values
        self.active_edges = {}  # edge_key (u, v) -> value, directed from u to v
        
        # Node connection tracking
        self.node_out_edges = {}  # node_id -> set of (u, v) outgoing edge keys
        self.node_in_edges = {}   # node_id -> set of (u, v) incoming edge keys
        
        # Selection state
        self.first_selected_node = None
        self.second_selected_node = None
        
        # Canvas element tracking
        self.line_ids = {}
        self.edge_to_line_id = {}
        self.text_ids = {}
        self.node_oval_ids = {}
        self.arrow_ids = {}

        # Create main frame
        main_frame = tk.Frame(root)
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        self.canvas = tk.Canvas(main_frame, width=self.canvas_width, height=self.canvas_height, bg="white")
        self.canvas.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        # Tool frame for value selection
        tool_frame = tk.Frame(root)
        tool_frame.pack(side=tk.TOP, fill=tk.X, pady=5)
        
        tk.Label(tool_frame, text="Step 1 - Select Value: ").pack(side=tk.LEFT, padx=5)
        
        self.tool_buttons = {}
        
        # Preset value buttons with symbolic labels
        for val, label in self.preset_values:
            btn = tk.Button(tool_frame, text=label, width=5, command=lambda v=val, l=label: self.set_tool(v, l))
            btn.pack(side=tk.LEFT, padx=2)
            self.tool_buttons[val] = btn
        
        # Custom value button
        custom_btn = tk.Button(tool_frame, text="Custom...", command=self.set_custom_value, bg="#ffffcc")
        custom_btn.pack(side=tk.LEFT, padx=5)
        self.custom_btn = custom_btn
            
        self.update_tool_buttons()

        # Current value display
        value_display_frame = tk.Frame(root)
        value_display_frame.pack(side=tk.TOP, fill=tk.X, pady=2)
        
        self.current_value_label = tk.Label(value_display_frame, 
                                            text="Current Value: None (select a value first)", 
                                            font=("Arial", 11, "bold"), fg="#0066cc")
        self.current_value_label.pack(side=tk.LEFT, padx=10)

        # Selection status frame
        selection_frame = tk.Frame(root)
        selection_frame.pack(side=tk.TOP, fill=tk.X, pady=5)
        
        tk.Label(selection_frame, text="Step 2 - Connect Nodes: ", font=("Arial", 10)).pack(side=tk.LEFT, padx=5)
        
        self.selection_label = tk.Label(selection_frame, text="Click first node...", font=("Arial", 11, "bold"), fg="#006600")
        self.selection_label.pack(side=tk.LEFT, padx=10)
        
        self.clear_selection_btn = tk.Button(selection_frame, text="Clear Selection", command=self.clear_selection, bg="#ffcccc")
        self.clear_selection_btn.pack(side=tk.LEFT, padx=5)

        # Remove actions frame
        remove_frame = tk.Frame(root)
        remove_frame.pack(side=tk.TOP, fill=tk.X, pady=5)
        
        tk.Label(remove_frame, text="Remove: ", font=("Arial", 10)).pack(side=tk.LEFT, padx=5)
        
        # Remove mode toggle
        self.remove_mode = False
        self.remove_mode_btn = tk.Button(remove_frame, text="Remove Mode: OFF", command=self.toggle_remove_mode, bg="#f0f0f0", width=15)
        self.remove_mode_btn.pack(side=tk.LEFT, padx=3)
        
        # Remove last edge button
        self.remove_last_btn = tk.Button(remove_frame, text="Undo Last", command=self.remove_last_edge, bg="#ffdddd")
        self.remove_last_btn.pack(side=tk.LEFT, padx=3)

        # File actions frame (same row)
        file_frame = tk.Frame(root)
        file_frame.pack(side=tk.TOP, fill=tk.X, pady=5)
        
        tk.Label(file_frame, text="File: ", font=("Arial", 10)).pack(side=tk.LEFT, padx=5)

        self.reset_btn = tk.Button(file_frame, text="Reset All", command=self.reset_values, bg="#ffdddd", padx=5)
        self.reset_btn.pack(side=tk.LEFT, padx=3)

        self.import_btn = tk.Button(file_frame, text="Import", command=self.import_data_json, bg="#ddddff", padx=5)
        self.import_btn.pack(side=tk.LEFT, padx=3)
        
        self.quick_export_btn = tk.Button(file_frame, text="Quick Export", command=self.export_data_json, bg="#cceecc", padx=5)
        self.quick_export_btn.pack(side=tk.LEFT, padx=3)

        self.export_btn = tk.Button(file_frame, text="Export JSON...", command=self.export_data_json_dialog, bg="#ddffdd", padx=10)
        self.export_btn.pack(side=tk.LEFT, padx=3)
        
        # Edge history for undo
        self.edge_history = []

        self.generate_grid_topology()
        self.draw_network()

    def get_color_for_value(self, value):
        """
        Get color for edge value with opacity effect:
        - -999 (unconnected): gray
        - 0: gray
        - -pi: full red (FF0000)
        - pi: full blue (0000FF)
        - Values between use opacity (blend with white)
        """
        import math
        
        if value == -999:
            return "#888888"  # Gray for unconnected
        if value == 0:
            return "#888888"  # Gray for zero
        
        PI = math.pi
        
        if value > 0:
            # Blue for positive: blend from white to full blue based on value/pi
            opacity = min(abs(value) / PI, 1.0)
            # Blend with white (255, 255, 255) to (0, 0, 255)
            r = int(255 * (1 - opacity))
            g = int(255 * (1 - opacity))
            b = 255
            return f"#{r:02x}{g:02x}{b:02x}"
        else:
            # Red for negative: blend from white to full red based on |value|/pi
            opacity = min(abs(value) / PI, 1.0)
            # Blend with white (255, 255, 255) to (255, 0, 0)
            r = 255
            g = int(255 * (1 - opacity))
            b = int(255 * (1 - opacity))
            return f"#{r:02x}{g:02x}{b:02x}"

    def set_tool(self, value, label=None):
        """Set the current tool value"""
        self.current_tool_value = value
        self.current_tool_label = label if label else f"{value:.4f}"
        self.update_tool_buttons()
        self.update_value_display()
    
    def set_custom_value(self):
        """Open dialog to enter a custom value"""
        value = simpledialog.askfloat("Custom Value", 
                                       "Enter custom angle value (in radians):\n\nHint: π ≈ 3.14159, π/2 ≈ 1.5708",
                                       initialvalue=0.0, parent=self.root,
                                       minvalue=-math.pi, maxvalue=math.pi)
        if value is not None:
            self.set_tool(value, f"Custom ({value:.4f})")
    
    def update_value_display(self):
        """Update the current value display label"""
        if self.current_tool_value is not None:
            self.current_value_label.config(
                text=f"Current Value: {self.current_tool_label} = {self.current_tool_value:.4f} rad",
                fg="#006600"
            )
        else:
            self.current_value_label.config(
                text="Current Value: None (select a value first)",
                fg="#cc0000"
            )

    def update_tool_buttons(self):
        for val, btn in self.tool_buttons.items():
            if val == self.current_tool_value:
                btn.config(relief=tk.SUNKEN, bg="#aaddaa")
            else:
                btn.config(relief=tk.RAISED, bg="#f0f0f0")

    def generate_grid_topology(self):
        """Generate the initial grid with nodes and edges"""
        available_width = self.canvas_width - 2 * self.margin
        available_height = self.canvas_height - 2 * self.margin
        dx = available_width / (self.cols - 1)
        dy = available_height / (self.rows - 1)

        # Create nodes
        for r in range(self.rows):
            for c in range(self.cols):
                node_id = r * self.cols + c
                x = self.margin + c * dx
                y = self.margin + r * dy
                self.nodes[node_id] = (x, y)
                self.node_out_edges[node_id] = set()
                self.node_in_edges[node_id] = set()

        # Create init edges (potential connections)
        for r in range(self.rows):
            for c in range(self.cols):
                u = r * self.cols + c
                
                # Horizontal edge
                if c < self.cols - 1:
                    v = r * self.cols + (c + 1)
                    self.add_init_edge(u, v)
                    # Border: top row (r=0) or bottom row (r=rows-1)
                    if r == 0 or r == self.rows - 1:
                        self.add_border_edge(u, v)

                # Vertical edge
                if r < self.rows - 1:
                    v = (r + 1) * self.cols + c
                    self.add_init_edge(u, v)
                    # Border: left column (c=0) or right column (c=cols-1)
                    if c == 0 or c == self.cols - 1:
                        self.add_border_edge(u, v)

                # Diagonal \ edge (top-left to bottom-right)
                if r < self.rows - 1 and c < self.cols - 1 and (c+r) % 2 == 0:
                    v = (r + 1) * self.cols + (c + 1)
                    self.add_init_edge(u, v)

                # Diagonal / edge (top-right to bottom-left)
                if r + 1 < self.rows and c > 0 and (c+r) % 2 == 0:
                    v = (r + 1) * self.cols + (c - 1)
                    self.add_init_edge(u, v)

    def add_init_edge(self, u, v):
        """Add an init edge (potential connection) to the grid"""
        edge_key = tuple(sorted((u, v)))
        if edge_key not in self.init_edges:
            self.init_edges[edge_key] = True

    def add_border_edge(self, u, v):
        """Mark an edge as a border edge (fixed value 0)"""
        edge_key = tuple(sorted((u, v)))
        self.border_edges.add(edge_key)

    def is_border_edge(self, u, v):
        """Check if an edge is a border edge"""
        edge_key = tuple(sorted((u, v)))
        return edge_key in self.border_edges

    def can_connect(self, u, v):
        """Check if two nodes can be connected (must have init edge)"""
        edge_key = tuple(sorted((u, v)))
        return edge_key in self.init_edges

    def add_active_edge(self, u, v, value):
        """Add a directed active edge from u to v with a value, or update if exists"""
        # Border edges cannot be changed
        if self.is_border_edge(u, v):
            return False
            
        if not self.can_connect(u, v):
            messagebox.showwarning("Invalid Connection", f"Nodes {u} and {v} are not adjacent in the grid.")
            return False
        
        # Check if edge already exists in same direction - update value
        if (u, v) in self.active_edges:
            self.active_edges[(u, v)] = value
            return True
        
        # Check if edge exists in opposite direction - update that one
        if (v, u) in self.active_edges:
            self.active_edges[(v, u)] = value
            return True
        
        # Add new directed edge
        self.active_edges[(u, v)] = value
        self.node_out_edges[u].add((u, v))
        self.node_in_edges[v].add((u, v))
        
        # Track in history for undo
        self.edge_history.append((u, v, value))
        
        return True

    def remove_active_edge(self, u, v):
        """Remove a directed active edge"""
        if (u, v) in self.active_edges:
            del self.active_edges[(u, v)]
            self.node_out_edges[u].discard((u, v))
            self.node_in_edges[v].discard((u, v))
            return True
        return False

    def is_boundary_node(self, node_id):
        """A boundary node has no outgoing edges AND at least 1 incoming edge"""
        return len(self.node_out_edges[node_id]) == 0 and len(self.node_in_edges[node_id]) >= 1

    def get_boundary_nodes(self):
        """Get all boundary nodes (no outgoing edges, at least 1 incoming edge)"""
        return [node_id for node_id in self.nodes if self.is_boundary_node(node_id)]
    
    def get_border_points(self):
        """Get all border points (nodes on the perimeter of the grid)"""
        border_points = []
        for node_id in self.nodes:
            r, c = self.get_node_rc(node_id)
            # A node is on the border if it's on first/last row or first/last column
            if r == 0 or r == self.rows - 1 or c == 0 or c == self.cols - 1:
                border_points.append(node_id)
        return border_points

    def toggle_remove_mode(self):
        """Toggle remove mode on/off"""
        self.remove_mode = not self.remove_mode
        if self.remove_mode:
            self.remove_mode_btn.config(text="Remove Mode: ON", bg="#ff6666")
            self.current_value_label.config(text="REMOVE MODE - Click on edges to remove them", fg="#cc0000")
        else:
            self.remove_mode_btn.config(text="Remove Mode: OFF", bg="#f0f0f0")
            self.update_value_display()
        self.clear_selection()

    def remove_last_edge(self):
        """Remove the last added edge"""
        if not self.edge_history:
            return
        
        u, v, value = self.edge_history.pop()
        if self.remove_active_edge(u, v):
            self.draw_network()

    def remove_edges_from_node_dialog(self):
        """Open dialog to remove all edges from/to a specific node"""
        node_id = simpledialog.askinteger("Remove Edges", 
            f"Enter node ID (0-{len(self.nodes)-1}):\n\nThis will remove all outgoing AND incoming edges for this node.",
            parent=self.root, minvalue=0, maxvalue=len(self.nodes)-1)
        
        if node_id is not None:
            self.remove_edges_from_node(node_id)

    def remove_edges_from_node(self, node_id):
        """Remove all edges connected to a node (both outgoing and incoming)"""
        if node_id not in self.nodes:
            messagebox.showwarning("Invalid Node", f"Node {node_id} does not exist.")
            return
        
        edges_to_remove = []
        
        # Collect outgoing edges
        for edge in list(self.node_out_edges[node_id]):
            edges_to_remove.append(edge)
        
        # Collect incoming edges
        for edge in list(self.node_in_edges[node_id]):
            edges_to_remove.append(edge)
        
        if not edges_to_remove:
            return
        
        # Remove all collected edges
        for u, v in edges_to_remove:
            self.remove_active_edge(u, v)
        
        self.draw_network()

    def clear_selection(self):
        """Clear the current node selection"""
        self.first_selected_node = None
        self.second_selected_node = None
        self.update_selection_label()
        self.draw_network()

    def update_selection_label(self):
        """Update the selection status label"""
        if self.first_selected_node is None:
            self.selection_label.config(text="Click first node...")
        elif self.second_selected_node is None:
            self.selection_label.config(text=f"First: Node {self.first_selected_node} → Click second node...")
        else:
            self.selection_label.config(text=f"Edge: {self.first_selected_node} → {self.second_selected_node}")

    def on_node_click(self, node_id):
        """Handle node click for edge creation or removal"""
        # Handle remove mode
        if self.remove_mode:
            self.remove_edges_from_node(node_id)
            return
        
        # Check if value is selected first
        if self.current_tool_value is None:
            messagebox.showinfo("Select Value First", 
                "Please select a value (Step 1) before connecting nodes.\n\n"
                "Click one of the preset buttons (π, π/2, etc.) or use 'Custom...' to enter a value.")
            return
        
        if self.first_selected_node is None:
            # Select first node
            self.first_selected_node = node_id
            self.second_selected_node = None
            self.update_selection_label()
            self.draw_network()
        elif self.first_selected_node == node_id:
            # Clicking the same node - deselect
            self.first_selected_node = None
            self.update_selection_label()
            self.draw_network()
        else:
            # Select second node
            self.second_selected_node = node_id
            self.update_selection_label()
            
            # Check if we can connect
            if not self.can_connect(self.first_selected_node, self.second_selected_node):
                messagebox.showwarning("Invalid Connection", 
                    f"Nodes {self.first_selected_node} and {self.second_selected_node} are not adjacent in the grid.")
                self.clear_selection()
                return
            
            # Use the pre-selected value
            value = self.current_tool_value
            
            # Try to add the edge
            if self.add_active_edge(self.first_selected_node, self.second_selected_node, value):
                self.draw_network()
            
            self.clear_selection()

    def draw_network(self):
        """Draw the entire network on the canvas"""
        self.canvas.delete("all")
        self.line_ids.clear()
        self.edge_to_line_id.clear()
        self.text_ids.clear()
        self.node_oval_ids.clear()
        self.arrow_ids.clear()

        # Draw init edges (light gray, thin) and border edges (black, solid)
        for edge_key in self.init_edges:
            u, v = edge_key
            x1, y1 = self.nodes[u]
            x2, y2 = self.nodes[v]
            
            if self.is_border_edge(u, v):
                # Border edge: black, solid, thicker
                line_id = self.canvas.create_line(x1, y1, x2, y2, width=3, fill="#000000")
            else:
                # Regular init edge: light gray, dashed
                line_id = self.canvas.create_line(x1, y1, x2, y2, width=1, fill="#dddddd", dash=(2, 4))
            self.line_ids[line_id] = edge_key

        # Draw active edges (colored, with arrows)
        for (u, v), value in self.active_edges.items():
            x1, y1 = self.nodes[u]
            x2, y2 = self.nodes[v]
            
            mid_x = (x1 + x2) / 2
            mid_y = (y1 + y2) / 2

            line_color = self.get_color_for_value(value)

            # Check if edge has no direction (from import or value is -999/0)
            edge_key = tuple(sorted((u, v)))
            has_no_direction = edge_key in self.imported_no_direction_edges or value == -999 or value == 0
            
            if has_no_direction:
                # No arrow for edges without direction
                line_id = self.canvas.create_line(x1, y1, x2, y2, width=3, fill=line_color, 
                                                   activefill="orange")
            else:
                # Arrow for directed edges
                line_id = self.canvas.create_line(x1, y1, x2, y2, width=3, fill=line_color, 
                                                   activefill="orange", arrow=tk.LAST, arrowshape=(12, 15, 5))
            
            # Draw value text
            text_id = self.canvas.create_text(mid_x, mid_y, text=f"{value:.2f}", 
                                               fill="black", font=("Arial", 9, "bold"))
            
            # Background for text
            bg_rect_id = self.canvas.create_rectangle(self.canvas.bbox(text_id), fill="white", outline="")
            self.canvas.tag_lower(bg_rect_id, text_id)
            
            self.edge_to_line_id[(u, v)] = line_id
            self.text_ids[(u, v)] = text_id
            
            # Bind click to edit/remove edge
            self.canvas.tag_bind(line_id, "<Button-1>", lambda event, e=(u, v): self.edit_active_edge(e))
            self.canvas.tag_bind(text_id, "<Button-1>", lambda event, e=(u, v): self.edit_active_edge(e))
            self.canvas.tag_bind(bg_rect_id, "<Button-1>", lambda event, e=(u, v): self.edit_active_edge(e))
            
            # Right-click to remove
            self.canvas.tag_bind(line_id, "<Button-2>", lambda event, e=(u, v): self.remove_edge_prompt(e))
            self.canvas.tag_bind(line_id, "<Button-3>", lambda event, e=(u, v): self.remove_edge_prompt(e))

        # Draw nodes
        r = 12  # node radius
        border_points = self.get_border_points()
        for node_id, (cx, cy) in self.nodes.items():
            # Determine node color based on state
            if node_id == self.first_selected_node:
                fill_color = "#ff6600"  # Orange for first selected
                outline_color = "#000000"
                outline_width = 3
            elif node_id in self.imported_boundary_nodes:
                # Imported boundary nodes: use boundary_nodes list from imported JSON directly
                fill_color = "#cc44cc"  # Purple/magenta for imported boundary
                outline_color = "#990099"
                outline_width = 3
            elif node_id in border_points or node_id in self.imported_border_points:
                # Border points: nodes on the perimeter of the grid
                fill_color = "#333333"  # Dark gray/black for border points
                outline_color = "#000000"
                outline_width = 2
            elif self.is_boundary_node(node_id) and len(self.active_edges) > 0:
                fill_color = "#ff9999"  # Light red for boundary nodes (no outgoing)
                outline_color = "#cc0000"
                outline_width = 2
            else:
                fill_color = "#4a90e2"  # Default blue
                outline_color = "black"
                outline_width = 1
            
            oval_id = self.canvas.create_oval(cx - r, cy - r, cx + r, cy + r, 
                                               fill=fill_color, outline=outline_color, width=outline_width)
            text_id = self.canvas.create_text(cx, cy, text=str(node_id), fill="white", font=("Arial", 8, "bold"))
            
            self.node_oval_ids[node_id] = oval_id
            
            # Bind node click
            self.canvas.tag_bind(oval_id, "<Button-1>", lambda event, nid=node_id: self.on_node_click(nid))
            self.canvas.tag_bind(text_id, "<Button-1>", lambda event, nid=node_id: self.on_node_click(nid))

        # Draw legend
        self.draw_legend()

    def draw_legend(self):
        """Draw a legend on the canvas"""
        legend_x = 10
        legend_y = self.canvas_height - 100
        
        self.canvas.create_text(legend_x, legend_y, text="Legend:", anchor=tk.NW, font=("Arial", 10, "bold"))
        
        # Blue node - regular
        self.canvas.create_oval(legend_x, legend_y + 20, legend_x + 10, legend_y + 30, fill="#4a90e2")
        self.canvas.create_text(legend_x + 15, legend_y + 25, text="Regular Node", anchor=tk.W, font=("Arial", 9))
        
        # Black node - border point
        self.canvas.create_oval(legend_x + 110, legend_y + 20, legend_x + 120, legend_y + 30, fill="#333333", outline="#000000", width=2)
        self.canvas.create_text(legend_x + 125, legend_y + 25, text="Border Point", anchor=tk.W, font=("Arial", 9))
        
        # Light red node - boundary
        self.canvas.create_oval(legend_x + 220, legend_y + 20, legend_x + 230, legend_y + 30, fill="#ff9999", outline="#cc0000")
        self.canvas.create_text(legend_x + 235, legend_y + 25, text="Boundary", anchor=tk.W, font=("Arial", 9))
        
        # Orange node - selected
        self.canvas.create_oval(legend_x + 310, legend_y + 20, legend_x + 320, legend_y + 30, fill="#ff6600")
        self.canvas.create_text(legend_x + 325, legend_y + 25, text="Selected", anchor=tk.W, font=("Arial", 9))
        
        # Imported boundary nodes - purple/magenta
        self.canvas.create_oval(legend_x, legend_y + 40, legend_x + 10, legend_y + 50, fill="#cc44cc", outline="#990099", width=2)
        self.canvas.create_text(legend_x + 15, legend_y + 45, text="Imported Boundary", anchor=tk.W, font=("Arial", 9))
        
        # Active edges count and imported boundary count
        imported_count = len(self.imported_boundary_nodes)
        self.canvas.create_text(legend_x, legend_y + 65, 
                                text=f"Active Edges: {len(self.active_edges)} | Boundary: {len(self.get_boundary_nodes())} | Imported Boundary: {imported_count}", 
                                anchor=tk.NW, font=("Arial", 10))

    def edit_active_edge(self, edge):
        """Edit an active edge value or remove if in remove mode"""
        u, v = edge
        
        # In remove mode, just remove the edge
        if self.remove_mode:
            self.remove_active_edge(u, v)
            self.draw_network()
            return
        
        current_val = self.active_edges.get(edge, 0.0)
        
        if self.current_tool_value is not None:
            new_val = self.current_tool_value
        else:
            new_val = simpledialog.askfloat("Input", f"Enter value for edge {u} → {v}:", 
                                            initialvalue=current_val, parent=self.root)
        
        if new_val is not None:
            self.active_edges[edge] = new_val
            self.draw_network()

    def remove_edge_prompt(self, edge):
        """Remove an edge (right-click)"""
        u, v = edge
        self.remove_active_edge(u, v)
        self.draw_network()

    def get_node_rc(self, node_id):
        return node_id // self.cols, node_id % self.cols

    def get_node_id(self, r, c):
        if 0 <= r < self.rows and 0 <= c < self.cols:
            return r * self.cols + c
        return None

    def calculate_triangle_attributes(self, u, v):
        edge_type = ""
        ur, uc = self.get_node_rc(u)
        vr, vc = self.get_node_rc(v)
        
        # Store original direction
        original_u, original_v = u, v
        
        if u > v:
            u, v = v, u
            ur, uc = self.get_node_rc(u)
            vr, vc = self.get_node_rc(v)

        idx1, idx2, idx3, idx4 = None, None, u, v

        if ur == vr and abs(uc - vc) == 1:
            edge_type = "Horizontal"
            # Cell Below (ur, uc)
            if (ur + uc) % 2 == 0: # Diagonal \
                idx1 = self.get_node_id(ur + 1, uc + 1)
            else: # Diagonal /
                idx1 = self.get_node_id(ur + 1, uc)
            
            # Cell Above (ur-1, uc)
            if (ur - 1 + uc) % 2 == 0: # Diagonal \
                idx2 = self.get_node_id(ur - 1, uc)
            else: # Diagonal /
                idx2 = self.get_node_id(ur - 1, uc + 1)
            
        elif uc == vc and abs(ur - vr) == 1:
            edge_type = "Vertical"
            # Cell Right (ur, uc)
            if (ur + uc) % 2 == 0: # Diagonal \
                idx2 = self.get_node_id(ur + 1, uc + 1)
            else: # Diagonal /
                idx2 = self.get_node_id(ur, uc + 1)

            # Cell Left (ur, uc-1)
            if (ur + uc - 1) % 2 == 0: # Diagonal \
                idx1 = self.get_node_id(ur, uc - 1)
            else: # Diagonal /
                idx1 = self.get_node_id(ur + 1, uc - 1)

        elif uc + 1 == vc and ur + 1 == vr:
            edge_type = "Diagonal\\"
            idx1 = self.get_node_id(ur + 1, uc)
            idx2 = self.get_node_id(ur, uc + 1)
            
        elif uc - 1 == vc and ur + 1 == vr:
            edge_type = "Diagonal/"
            idx1 = self.get_node_id(ur, uc - 1)
            idx2 = self.get_node_id(ur + 1 , uc)

        return (idx1, idx2, idx3, idx4), edge_type

    def reset_values(self):
        if messagebox.askyesno("Confirm Reset", "Are you sure you want to reset all edges and connections?"):
            self.active_edges.clear()
            for node_id in self.nodes:
                self.node_out_edges[node_id] = set()
                self.node_in_edges[node_id] = set()
            self.clear_selection()
            self.draw_network()

    def generate_export_data(self):
        """Generate export data dictionary"""
        center_c = (self.cols - 1) / 2.0
        center_r = (self.rows - 1) / 2.0
        
        nodes_export = {}
        for node_id in self.nodes:
            r, c = self.get_node_rc(node_id)
            
            cart_x = float(c - center_c)
            cart_y = float(center_r - r)
            
            nodes_export[node_id] = (cart_x, cart_y)

        output_data = {
            "metadata": {
                "rows": self.rows,
                "cols": self.cols
            },
            "nodes": nodes_export,
            "edges": [],
            "boundary_nodes": self.get_boundary_nodes(),
            "border_points": self.get_border_points(),
            "node_connections": {
                node_id: {
                    # out_edges: list of destination nodes (where edges go TO)
                    "out_edges": [v for (u, v) in self.node_out_edges[node_id]],
                    # in_edges: list of source nodes (where edges come FROM)
                    "in_edges": [u for (u, v) in self.node_in_edges[node_id]]
                } for node_id in self.nodes
            }
        }

        # Export ALL init edges (both connected and unconnected)
        for edge_key in self.init_edges:
            u, v = edge_key
            
            # Border edges always have value 0
            if self.is_border_edge(u, v):
                value = 0
                direction = "border"
            # Check if this edge is active (connected) in either direction
            elif (u, v) in self.active_edges:
                value = self.active_edges[(u, v)]
                direction = f"{u} -> {v}"
            elif (v, u) in self.active_edges:
                value = self.active_edges[(v, u)]
                # Swap u, v to match the active edge direction
                u, v = v, u
                direction = f"{u} -> {v}"
            else:
                # Not connected - use -999
                value = -999
                direction = "none"
            
            attr_tuple, edge_type = self.calculate_triangle_attributes(u, v)
            edge_data = {
                "u": u,
                "v": v,
                "value": value,
                "attributes": attr_tuple,
                "edge_type": edge_type,
                "direction": direction
            }
            output_data["edges"].append(edge_data)

        return output_data

    def export_data_json(self):
        """Quick export with auto-generated filename to pattern directory"""
        # Get the pattern directory (relative to this script's parent)
        script_dir = os.path.dirname(os.path.abspath(__file__))
        pattern_dir = os.path.join(os.path.dirname(script_dir), "pattern")
        
        # Create pattern directory if it doesn't exist
        os.makedirs(pattern_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = os.path.join(pattern_dir, f"pattern_{timestamp}.json")
        
        output_data = self.generate_export_data()
        
        try:
            with open(filename, 'w') as f:
                json.dump(output_data, f, indent=4)
            messagebox.showinfo("Success", f"Saved to pattern/{os.path.basename(filename)}")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save JSON: {e}")

    def export_data_json_dialog(self):
        """Export with file save dialog - defaults to pattern directory"""
        # Get the pattern directory
        script_dir = os.path.dirname(os.path.abspath(__file__))
        pattern_dir = os.path.join(os.path.dirname(script_dir), "pattern")
        
        # Create pattern directory if it doesn't exist
        os.makedirs(pattern_dir, exist_ok=True)
        
        file_path = filedialog.asksaveasfilename(
            title="Export JSON File",
            initialdir=pattern_dir,
            defaultextension=".json",
            filetypes=[("JSON Files", "*.json"), ("All Files", "*.*")],
            initialfile=f"pattern_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        )
        
        if not file_path:
            return
        
        output_data = self.generate_export_data()
        
        try:
            with open(file_path, 'w') as f:
                json.dump(output_data, f, indent=4)
            messagebox.showinfo("Success", f"Saved to {os.path.basename(file_path)}")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save JSON: {e}")

    def import_data_json(self):
        file_path = filedialog.askopenfilename(
            title="Select JSON File",
            filetypes=[("JSON Files", "*.json"), ("All Files", "*.*")]
        )
        
        if not file_path:
            return

        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
            
            if "metadata" in data:
                meta = data["metadata"]
                new_rows = meta.get("rows", self.rows)
                new_cols = meta.get("cols", self.cols)
                
                if new_rows != self.rows or new_cols != self.cols:
                    self.rows = new_rows
                    self.cols = new_cols
                    self.init_edges = {}
                    self.active_edges = {}
                    self.nodes = {}
                    self.node_out_edges = {}
                    self.node_in_edges = {}
                    self.generate_grid_topology()

            # Clear existing active edges and imported boundary nodes
            self.active_edges.clear()
            self.imported_boundary_nodes.clear()
            self.imported_no_direction_edges.clear()
            for node_id in self.nodes:
                self.node_out_edges[node_id] = set()
                self.node_in_edges[node_id] = set()
            
            # Load boundary nodes from imported data
            if "boundary_nodes" in data:
                self.imported_boundary_nodes = set(data["boundary_nodes"])
            
            # Load border points from imported data
            if "border_points" in data:
                self.imported_border_points = set(data["border_points"])

            if "edges" in data:
                for edge_data in data["edges"]:
                    u = edge_data.get("u")
                    v = edge_data.get("v")
                    val = edge_data.get("value", 0.0)
                    direction = edge_data.get("direction", "none")
                    
                    if u is not None and v is not None:
                        # Import as directed edge
                        if self.can_connect(u, v):
                            self.active_edges[(u, v)] = val
                            
                            # Only track in/out edges for edges with actual direction
                            # (not "none" or "border")
                            if direction != "none" and direction != "border":
                                self.node_out_edges[u].add((u, v))
                                self.node_in_edges[v].add((u, v))
                            
                            # Track edges without direction for arrow display
                            if direction == "none" or direction == "border":
                                edge_key = tuple(sorted((u, v)))
                                self.imported_no_direction_edges.add(edge_key)
                            
            self.draw_network()
            messagebox.showinfo("Success", f"Loaded data from {os.path.basename(file_path)}")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to import JSON: {e}")

if __name__ == "__main__":
    root = tk.Tk()
    app = SegmentAssignerApp(root)
    root.mainloop()