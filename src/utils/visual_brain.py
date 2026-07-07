import os
import sys
import math
import random
import tkinter as tk
from queue import Queue

# Add project root directory to system path
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(CURRENT_DIR))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

class NeuralBrainVisualizer:
    """
    Renders a live, glowing 3D-projected neural network node tree simulation
    with active particle data streams transferring between layers in real time.
    """
    def __init__(self, width: int = 1000, height: int = 600, active: bool = True):
        self.active = active
        if not self.active:
            return

        print("🎮 [VISUALIZER] Initializing real-time tracking graphic canvas context...")
        
        # Build Tkinter display root frame
        self.root = tk.Tk()
        self.root.title("pLLM Panini Core Neural Brain Architecture Simulation")
        self.root.geometry(f"{width}x{height}")
        self.root.configure(bg="#0A0E19")
        
        # Build vector drawing canvas layer area
        self.canvas = tk.Canvas(self.root, width=width, height=height, bg="#0A0E19", highlightthickness=0)
        self.canvas.pack()
        
        self.width = width
        self.height = height
        
        # Thread Safe Signal Queue to consume live data tokens from generate.py
        self.signal_queue = Queue()
        
        # Simulation camera angles
        self.angle_y = 0.0
        self.current_mode = "STANDBY"
        self.displayed_text = "Awaiting live pipeline prompt matrix..."
        self.pulse_intensity = 0.0
        
        # Particle tracking arrays for real-time node-to-node electrical flows
        self.active_particles = []
        
        # Generate neural architecture map grid
        self.nodes = []
        self.synapses = []
        self._build_brain_topology()
        
        # Force initial display layout refresh pass
        self.root.update()

    def _build_brain_topology(self):
        """Constructs a 3D structural lattice matching the Panini model layout layers."""
        layer_sizes = [8, 12, 16, 12, 6]  # Sized beautifully for crisp screen views
        num_layers = len(layer_sizes)
        
        for layer_idx, num_nodes in enumerate(layer_sizes):
            layer_x = (layer_idx - (num_layers - 1) / 2) * 200
            
            for node_idx in range(num_nodes):
                angle = (node_idx / num_nodes) * 2.0 * math.pi
                radius = 100 if layer_idx % 2 == 0 else 70
                
                layer_y = radius * math.cos(angle)
                layer_z = radius * math.sin(angle)
                
                # Pick beautiful glowing hex colors matching your custom nodes
                if layer_idx == 0: color = "#00FFFF"      # Cyan - Embedding
                elif layer_idx == 1: color = "#FF00FF"    # Magenta - Attention Query
                elif layer_idx == 2: color = "#00FF64"    # Green - SwiGLU Core
                elif layer_idx == 3: color = "#FFFF00"    # Yellow - KV Cached Blocks
                else: color = "#FF6400"                  # Orange - Output Head
                
                self.nodes.append({
                    "id": len(self.nodes),
                    "layer": layer_idx,
                    "x": layer_x,
                    "y": layer_y,
                    "z": layer_z,
                    "color": color,
                    "activity": 0.0
                })

        # Dynamically weave connection links between nearby node levels
        for i, n1 in enumerate(self.nodes):
            for j, n2 in enumerate(self.nodes):
                if n2["layer"] == n1["layer"] + 1:
                    if random.random() > 0.65:  # Balanced line density for routing pulses
                        self.synapses.append((i, j, random.uniform(0.3, 0.7)))

    # FIXED: Re-added the missing trigger_pulse attribute function for run_local_test.py
    def trigger_pulse(self, mode: str = "TRAIN"):
        """Alters simulation modes and injects electrical energy into node fields."""
        if not self.active: return
        self.current_mode = mode
        self.pulse_intensity = 1.0
        
        # Energize nodes during training iterations
        for node in self.nodes:
            if mode == "TRAIN":
                node["activity"] = random.uniform(0.6, 1.0)
            else:
                node["activity"] = random.uniform(0.3, 0.8)

    def inject_live_token(self, token_string: str, phase: str = "INFERENCE"):
        """Injects a real data node event from the generator codebase."""
        if not self.active: return
        self.signal_queue.put({"token": token_string, "phase": phase})
    def _process_queue_signals(self):
        """Pulls internal tokens out of pipeline memory fields and converts them to active electrical charges."""
        while not self.signal_queue.empty():
            data = self.signal_queue.get()
            self.current_mode = data["phase"]
            self.displayed_text = f"Processing Token: {data['token']}"
            
            # Find input layer nodes (Embedding Layer 0) and activate them immediately
            input_nodes = [n for n in self.nodes if n["layer"] == 0]
            for input_node in input_nodes:
                input_node["activity"] = 1.0
                
                # Spawn dynamic tracking particles moving across adjacent connection paths
                for syn in self.synapses:
                    if syn[0] == input_node["id"]:
                        self.active_particles.append({
                            "from_id": syn[0],
                            "to_id": syn[1],
                            "progress": 0.0,
                            "speed": random.uniform(0.08, 0.15),
                            "color": "#00FFFF" if self.current_mode == "INFERENCE" else "#FF3366"
                        })

    def update_and_render(self):
        """Calculates 3D rotations, updates active data particles, and paints the canvas window."""
        if not self.active: return
        
        try:
            # Check for live data signals arriving from generation threads
            self._process_queue_signals()
            
            # Clear drawing board cache from previous step frame loops
            self.canvas.delete("all")
            
            # Rotate camera viewpoint automatically
            self.angle_y += 0.02
            cos_y = math.cos(self.angle_y)
            sin_y = math.sin(self.angle_y)
            
            # Cool down node structural activations and pulse intensity
            if self.pulse_intensity > 0.0: self.pulse_intensity -= 0.05
            for node in self.nodes:
                if node["activity"] > 0.0: 
                    node["activity"] -= 0.03

            # Update and move real-time data flow particles layer-by-layer
            next_particles = []
            for p in self.active_particles:
                p["progress"] += p["speed"]
                if p["progress"] >= 1.0:
                    # Particle reached target node! Activate target layer node metrics
                    target_node = self.nodes[p["to_id"]]
                    target_node["activity"] = 1.0
                    
                    # Chain Reaction: If not at output layer, cascade down into next network fields
                    if target_node["layer"] < 4:
                        for syn in self.synapses:
                            if syn[0] == target_node["id"]:
                                next_particles.append({
                                    "from_id": syn[0],
                                    "to_id": syn[1],
                                    "progress": 0.0,
                                    "speed": random.uniform(0.08, 0.15),
                                    "color": p["color"]
                                })
                else:
                    next_particles.append(p)
            self.active_particles = next_particles

            # --- Phase 1: Render Synapse Link Wire Arrays ---
            for n1_idx, n2_idx, base_weight in self.synapses:
                n1 = self.nodes[n1_idx]
                n2 = self.nodes[n2_idx]
                
                # Project 3D vectors onto 2D viewport coordinates
                rot_x1 = n1["x"] * cos_y - n1["z"] * sin_y
                screen_x1 = int(self.width / 2 + rot_x1)
                screen_y1 = int(self.height / 2 + n1["y"])
                
                rot_x2 = n2["x"] * cos_y - n2["z"] * sin_y
                screen_x2 = int(self.width / 2 + rot_x2)
                screen_y2 = int(self.height / 2 + n2["y"])
                
                # Dynamic highlighting if connection line endpoints match active nodes or training pulses
                if (n1["activity"] > 0.3 and n2["activity"] > 0.3) or self.pulse_intensity > 0.1:
                    line_color = "#1A4B9B" if self.current_mode == "INFERENCE" else "#8B1A3E"
                    width_scale = 2
                else:
                    line_color = "#131926"  # Sleek dim standby color slate line
                    width_scale = 1
                    
                self.canvas.create_line(screen_x1, screen_y1, screen_x2, screen_y2, fill=line_color, width=width_scale)

            # --- Phase 2: Render Real-Time Data Flow Moving Particles ---
            for p in self.active_particles:
                n1 = self.nodes[p["from_id"]]
                n2 = self.nodes[p["to_id"]]
                
                rot_x1 = n1["x"] * cos_y - n1["z"] * sin_y
                sx1 = int(self.width / 2 + rot_x1)
                sy1 = int(self.height / 2 + n1["y"])
                
                rot_x2 = n2["x"] * cos_y - n2["z"] * sin_y
                sx2 = int(self.width / 2 + rot_x2)
                sy2 = int(self.height / 2 + n2["y"])
                
                px = sx1 + (sx2 - sx1) * p["progress"]
                py = sy1 + (sy2 - sy1) * p["progress"]
                
                self.canvas.create_oval(px-3, py-3, px+3, py+3, fill=p["color"], outline="#FFFFFF")

            # --- Phase 3: Render Glowing Node Dot Coordinates ---
            for node in self.nodes:
                rot_x = node["x"] * cos_y - node["z"] * sin_y
                screen_x = int(self.width / 2 + rot_x)
                screen_y = int(self.height / 2 + node["y"])
                
                radius = int(3 + (node["activity"] * 5))
                
                fill_color = node["color"] if node["activity"] > 0.1 else "#151B2D"
                self.canvas.create_oval(
                    screen_x - radius, screen_y - radius, 
                    screen_x + radius, screen_y + radius, 
                    fill=fill_color, outline=fill_color
                )

            # Paint operational dashboard tracking labels text
            self.canvas.create_text(
                140, 30, 
                text=f"🧠 PANINI ENGINE: {self.current_mode}", 
                fill="#00FFFF" if self.current_mode == "INFERENCE" else "#FF3366", font=("Helvetica", 12, "bold")
            )
            self.canvas.create_text(
                200, 55, 
                text=self.displayed_text, 
                fill="#8A9BB4", font=("Helvetica", 10, "italic")
            )
            
            # Force update window render matrix loop frames
            self.root.update_idletasks()
            self.root.update()
            
        except tk.TclError:
            self.active = False

    def close(self):
        if self.active:
            self.root.destroy()
            self.active = False
