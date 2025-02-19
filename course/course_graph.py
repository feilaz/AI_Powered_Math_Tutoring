from typing import Optional, List, Dict, Any, Tuple, Optional
from dataclasses import dataclass, field, fields
import copy
from agents import TaskOutput
import re
from langchain.tools import StructuredTool
import networkx as nx

@dataclass
class Task:
    id: int
    difficulty: str
    task: str
    step_by_step_solution: str = ""
    answer: str = ""
    solved: bool = False

@dataclass
class CourseNode:
    node_id: str  # This will serve as both identifier and topic name
    description: str = ""
    task_list: List[Task] = field(default_factory=list)
    prerequisites: List[str] = field(default_factory=list)
    children: List[str] = field(default_factory=list)

class CourseGraph:
    def __init__(self, description: str = ""):
        self.nodes: Dict[str, CourseNode] = {}
        self.description = description

    def change_description(self, description: str) -> None:
        """Changes the description of the course graph."""
        self.description = description

    def add_node(self, node: CourseNode) -> None:
        """Adds a node to the graph."""
        if node.node_id in self.nodes:
            raise ValueError(f"Node with ID '{node.node_id}' already exists.")

        self.nodes[node.node_id] = node

        for prereq_id in node.prerequisites:
            if prereq_id in self.nodes:
                self.nodes[prereq_id].children.append(node.node_id)

    def add_task_to_node(self, node_id: str, task_output: TaskOutput) -> Task:
        """Adds a new Task to the given node using TaskOutput."""
        node = self.get_node_by_id(node_id)
        if not node:
            raise ValueError(f"Node '{node_id}' doesn't exist.")
        new_task_id = len(node.task_list) + 1
        new_task = Task(
            id=new_task_id,
            task=task_output.task,
            step_by_step_solution=task_output.step_by_step_solution,
            answer=task_output.answer,
            difficulty=task_output.difficulty,
            solved=False
        )
        node.task_list.append(new_task)
        return new_task

    def remove_node(self, node_id: str) -> None:
        """Removes a node from the graph."""
        if node_id not in self.nodes:
            raise ValueError(f"Node '{node_id}' does not exist.")

        for prereq_id in self.nodes[node_id].prerequisites:
            if prereq_id in self.nodes:
                self.nodes[prereq_id].children.remove(node_id)

        for child_id in self.nodes[node_id].children:
            if child_id in self.nodes:
                self.nodes[child_id].prerequisites.remove(node_id)

        del self.nodes[node_id]

    def remove_task_from_node(self, node_id: str, task_id: int) -> None:
        """Removes a Task by ID from the given node."""
        node = self.get_node_by_id(node_id)
        if not node:
            raise ValueError(f"Node '{node_id}' does not exist.")

        original_length = len(node.task_list)
        node.task_list = [t for t in node.task_list if t.id != task_id]
        if len(node.task_list) == original_length:
            raise ValueError(f"Task with ID '{task_id}' not found in node '{node_id}'.")


    def edit_node(self, node_id: str, **kwargs) -> None:
        """Edits the attributes of a node."""
        if node_id not in self.nodes:
            raise ValueError(f"Node '{node_id}' does not exist.")

        node = self.nodes[node_id]
        valid_attrs = {f.name for f in fields(CourseNode)}

        for key, value in kwargs.items():
            if key not in valid_attrs:
                raise ValueError(f"Invalid attribute: '{key}'")

            if key == 'prerequisites':
                for prereq_id in node.prerequisites:
                    if prereq_id in self.nodes:
                        self.nodes[prereq_id].children.remove(node_id)

                if isinstance(value, str):
                    node.prerequisites = [item.strip() for item in value.split(',') if item.strip()]
                elif isinstance(value, list):
                    node.prerequisites = value
                else:
                    raise ValueError("Prerequisites must be a string or a list")

                for prereq_id in node.prerequisites:
                    if prereq_id in self.nodes:
                        self.nodes[prereq_id].children.append(node_id)
            elif key == 'task_list':
                if isinstance(value, str):
                    node.task_list = [item.strip() for item in value.split(',') if item.strip()]
                elif isinstance(value, list):
                    node.task_list = value
                else:
                    raise ValueError("Task list must be a string or a list")
            else:
                setattr(node, key, value)

    def edit_task_in_node(self, node_id: str, task_id: int, **kwargs) -> None:
        """
        Edits an existing Task in the given node.
        Example usage could be marking it as solved or updating the solution.
        """
        node = self.get_node_by_id(node_id)
        if not node:
            raise ValueError(f"Node '{node_id}' does not exist.")
        for t in node.task_list:
            if t.id == task_id:
                # Update allowed fields
                if "solved" in kwargs:
                    t.solved = bool(kwargs["solved"])
                if "task" in kwargs:
                    t.task = kwargs["task"]
                if "step_by_step_solution" in kwargs:
                    t.step_by_step_solution = kwargs["step_by_step_solution"]
                if "answer" in kwargs:
                    t.answer = kwargs["answer"]
                return
        raise ValueError(f"Task with ID '{task_id}' not found in node '{node_id}'.")


    def get_node_description(self, node_id: str) -> str:
        """Retrieves a node by its ID and returns a formatted string with node information and task list."""
        node = self.nodes.get(node_id)
        if not node:
            return f"Node '{node_id}' does not exist. Current nodes: {', '.join(self.nodes.keys())}"

        overview = f"--- Node: {node.node_id} ---\n"
        if node.description:
            overview += f"Description: {node.description}\n"
        if node.prerequisites:
            overview += f"Prerequisites: {', '.join(node.prerequisites)}\n"
        if node.children:
            overview += f"Children: {', '.join(node.children)}\n"
        if node.task_list:
            overview += "Tasks:\n"
            for task in node.task_list:
                overview += (
                    f"  - Task ID: {task.id}\n"
                    f"    Difficulty: {task.difficulty}\n"
                    f"    Task: {task.task}\n"
                    f"    Solved: {'Yes' if task.solved else 'No'}\n"
                )
        else:
            overview += "No tasks assigned to this node.\n"

        return overview
    
    def get_node_by_id(self, node_id: str) -> Optional[CourseNode]:
        """Retrieves a node by its ID."""
        return self.nodes.get(node_id)
    
    def get_task_by_id(self, node_id: str, task_id: int) -> Optional[Task]:
        """Retrieves a task by its ID."""
        node = self.get_node_by_id(node_id)
        if not node:
            return None
        for t in node.task_list:
            if t.id == task_id:
                return t
        return None

    def get_graph_overview(self) -> str:
        """Generates a high-level overview of the course graph."""
        if not self.nodes:
            return "The course graph is empty."

        overview = ""
        for node_id, node in self.nodes.items():
            overview += f"- Node: {node_id}\n"
            if node.description:
                overview += f"  Description: {node.description}\n"
            if node.prerequisites:
                overview += f"  Prerequisites: {', '.join(node.prerequisites)}\n"
            overview += "\n"
        return overview

    def _backup_state(self) -> Dict[str, CourseNode]:
        """Creates a deep copy of the current state for rollback."""
        return copy.deepcopy(self.nodes)

    def _restore_state(self, backup: Dict[str, CourseNode]) -> None:
        """Restores the state from a backup."""
        self.nodes = backup

    def apply_llm_instructions(self, instructions: str) -> None:
        backup = self._backup_state()

        try:
            for line in instructions.strip().split('\n'):
                line = line.strip()
                if not line:
                    continue

                parts = self._parse_instruction_line(line)
                command = parts[0].upper()

                # Existing commands...
                if command == 'ADD':
                    self._handle_add_command(parts[1:])
                elif command == 'REMOVE':
                    self._handle_remove_command(parts[1:])
                elif command == 'EDIT':
                    self._handle_edit_command(parts[1:])

                # New commands for Task manipulation
                elif command == 'TASK_ADD':
                    # e.g., "TASK_ADD node_id task=... step_by_step_solution=... answer=..."
                    node_id = parts[1]
                    attributes = self._parse_attributes(" ".join(parts[2:]))
                    # Convert attributes to TaskOutput
                    output_obj = TaskOutput(**attributes)
                    self.add_task_to_node(node_id, output_obj)

                elif command == 'TASK_REMOVE':
                    # e.g., "TASK_REMOVE node_id task_id"
                    node_id = parts[1]
                    task_id = int(parts[2])
                    self.remove_task_from_node(node_id, task_id)

                elif command == 'TASK_EDIT':
                    # e.g., "TASK_EDIT node_id task_id solved=True"
                    node_id = parts[1]
                    task_id = int(parts[2])
                    attributes = self._parse_attributes(" ".join(parts[3:]))
                    self.edit_task_in_node(node_id, task_id, **attributes)

                else:
                    raise ValueError(f"Unknown command: '{command}'")
        except Exception as e:
            self._restore_state(backup)
            raise ValueError(f"Failed to apply instructions: {str(e)}")


    def _parse_instruction_line(self, line: str) -> List[str]:
        """Parses a single instruction line, handling underscores as spaces within values."""
        parts = []
        current_part = []

        for char in line:
            if char == ' ':
                if current_part:
                    parts.append(''.join(current_part))
                    current_part = []
            else:
                current_part.append(char)

        if current_part:
            parts.append(''.join(current_part))

        return parts

    def _parse_attributes(self, attr_str: str) -> Dict[str, Any]:
        """Parses an attribute string into a dictionary."""
        attrs = {}
        if not attr_str:
            return attrs

        parts = [p.strip() for p in attr_str.split()]
        for part in parts:
            if '=' in part:
                key, value = part.split('=', 1)
                attrs[key.strip()] = value.strip()

        return attrs

    def _handle_add_command(self, parts: List[str]) -> None:
        """Handles the ADD command."""
        if len(parts) < 1:
            raise ValueError("ADD command requires at least: node_id [attributes]")
        
        node_id, *rest = parts  # node_id will serve as both ID and topic
        params = " ".join(rest)
        
        prerequisites = []
        description = ""
        other_attrs = {}
        
        prereq_match = re.search(r'prerequisites=([^\s]+)', params)
        if prereq_match:
            prereq_raw = prereq_match.group(1).strip()
            prerequisites = [p.strip() for p in prereq_raw.split(',') if p.strip()]
        
        desc_match = re.search(r'description="([^"]+)"', params)
        if desc_match:
            description = desc_match.group(1).strip()
            other_attrs['description'] = description
        
        try:
            node = CourseNode(
                node_id=node_id,
                prerequisites=prerequisites,
                **other_attrs
            )
        except Exception as e:
            raise ValueError(f"Failed to create node: {str(e)}")
        
        self.add_node(node)

    def _handle_remove_command(self, parts: List[str]) -> None:
        """Handles the REMOVE command."""
        if len(parts) != 1:
            raise ValueError("REMOVE command requires only node_id")

        self.remove_node(parts[0])

    def _handle_edit_command(self, parts: List[str]) -> None:
        """Handles the EDIT command."""
        if len(parts) < 1:
            raise ValueError("EDIT command requires: node_id attributes")

        node_id, *attr_parts = parts
        attributes_str = ' '.join(attr_parts)
        attrs = {}

        if attributes_str:
            for part in attributes_str.split():
                if '=' in part:
                    key, value = part.split('=', 1)
                    attrs[key.strip()] = value.strip()

        if not attrs:
            raise ValueError("EDIT command requires at least one attribute to update")

        self.edit_node(node_id, **attrs)

# ...existing code...
    def get_dag_graph(self):
        """Creates a Directed Acyclic Graph (DAG) containing all nodes of the course."""
        G = nx.DiGraph()
        for node_id, node in self.nodes.items():
            G.add_node(node_id, description=node.description)  # Remove topic attribute
            for prereq in node.prerequisites:
                if prereq in self.nodes:
                    G.add_edge(prereq, node_id)
        return G

    def draw_dag(self) -> None:
        """
        Draws the course DAG graph in a hierarchical, tree-like layout,
        with properly sized and positioned node labels.
        Repeats the overlap adjustment multiple times to reduce node collisions further.
        Uses bigger spacing for leaf nodes to avoid clustering at the bottom.
        """
        import networkx as nx
        import matplotlib.pyplot as plt
        import os

        G = self.get_dag_graph()

        # Identify roots (nodes with in-degree = 0)
        roots = [n for n, deg in G.in_degree() if deg == 0]
        if not roots:
            roots = [list(G.nodes)[0]]

        # Identify leaves (nodes with out-degree = 0)
        leaves = [n for n, deg in G.out_degree() if deg == 0]

        def hierarchy_pos(graph, root, width=1.0, vert_gap=0.3,
                        vert_loc=0.0, xcenter=0.5, pos=None):
            """
            Recursively assign positions in a top-down hierarchy (root at y=0, children below),
            with the root centered horizontally at xcenter, and children distributed
            equally to the left and right.
            """
            if pos is None:
                pos = {root: (xcenter, vert_loc)}
            else:
                pos[root] = (xcenter, vert_loc)

            successors = list(graph.successors(root))
            if successors:
                dx = width / len(successors)
                # Start child positions to the left, so children spread out evenly around xcenter
                next_x = xcenter - (width / 2) + (dx / 2)
                for child in successors:
                    pos = hierarchy_pos(
                        graph,
                        child,
                        width=dx,
                        vert_gap=vert_gap,
                        vert_loc=vert_loc - vert_gap,
                        xcenter=next_x,
                        pos=pos
                    )
                    next_x += dx

            return pos

        # 1) Compute initial positions
        pos = {}
        layer_chunk = 1.0 / max(1, len(roots))
        start_x = layer_chunk / 2
        for i, root in enumerate(roots):
            pos.update(
                hierarchy_pos(
                    G, root,
                    width=layer_chunk,
                    vert_gap=0.8,
                    vert_loc=0.0,
                    xcenter=start_x + i * layer_chunk,
                    pos=pos
                )
            )

        def adjust_positions_to_avoid_overlap(pos_dict, common_dist=0.06, leaf_dist=0.12, passes=3):
            """
            On each pass:
            - Bucket nodes by y-value (rounded).
            - Sort by x.
            - If two nodes are too close horizontally, nudge the right one further right.
            Uses a larger min_dist for leaf nodes (leaf_dist) to help spread them out.
            """
            for _ in range(passes):
                layers = {}
                # Group nodes by their y-value
                for node, (x, y) in pos_dict.items():
                    l_key = round(y, 2)
                    if l_key not in layers:
                        layers[l_key] = []
                    layers[l_key].append((node, x))

                # Sort each layer's nodes by x, then ensure spacing
                for l_key, items in layers.items():
                    items.sort(key=lambda tup: tup[1])  # sort by x ascending
                    for i in range(1, len(items)):
                        node, curr_x = items[i]
                        prev_node, prev_x = items[i - 1]
                        # Use a bigger distance if node is a leaf
                        dist = leaf_dist if node in leaves else common_dist
                        if (curr_x - prev_x) < dist:
                            new_x = prev_x + dist
                            pos_dict[node] = (new_x, l_key)
                            items[i] = (node, new_x)
            return pos_dict

        # 2) Run overlap adjustment multiple times
        #    Leaves get bigger spacing
        pos = adjust_positions_to_avoid_overlap(pos, common_dist=0.1, leaf_dist=0.15, passes=3)

        # 3) Prepare node labels
        labels = {node_id: node_id.replace('_', ' ') for node_id in G.nodes}

        # 4) Figure size
        num_nodes = len(G.nodes)
        max_label_len = max(len(text) for text in labels.values()) if labels else 1
        depths = [pos[n][1] for n in G.nodes]
        num_layers = abs(min(depths)) + 1 if depths else 1

        # Count how many nodes in each layer
        nodes_per_layer = {}
        for node, (x, y) in pos.items():
            layer = abs(int(y))
            nodes_per_layer[layer] = nodes_per_layer.get(layer, 0) + 1
        max_nodes_in_layer = max(nodes_per_layer.values())

        # Base figure size with extra space for longer labels
        label_space = max_label_len * 0.25
        width = max(12, max_nodes_in_layer * 4 + label_space)
        height = max(8, num_layers * 2)

        plt.figure(figsize=(width, height))

        # 5) Draw edges and labels
        nx.draw_networkx_edges(
            G,
            pos,
            arrows=True,
            arrowstyle="->",
            arrowsize=20,
            edge_color='gray'
        )
        nx.draw_networkx_labels(
            G,
            pos,
            labels=labels,
            font_size=9,
            bbox=dict(
                boxstyle="round4,pad=0.6,rounding_size=0.2",
                facecolor="skyblue",
                edgecolor="black",
                alpha=1
            ),
            horizontalalignment='center'
        )

        plt.title("Course Structure")
        plt.axis("off")
        plt.margins(x=0.3, y=0.2)
        plt.tight_layout(pad=2.0)

        # Save and open
        output_file = "course_dag.png"
        plt.savefig(output_file, bbox_inches='tight', dpi=300)
        plt.close()

        os.startfile(output_file)

    def create_draw_course_tool(self):
        return StructuredTool.from_function(
            func=self.draw_dag,
            name="DrawCourseDAG",
            description="Draws the course graph in a hierarchical, tree-like layout. And shows it to student in a popup window.",
            args_schema=None,
            return_direct=False)

