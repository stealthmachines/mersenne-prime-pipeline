# HDGL Harmonics + Spiral10000 V3.0 Visualization
# Advanced visualization of Dₙ(r) Base(∞) Numeric Lattice mathematics

import json
import math
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle
from matplotlib.collections import LineCollection
import matplotlib.colors as mcolors

class HDGLSpiralVisualizerV30:
    """Advanced visualizer for V3.0 Dₙ(r) spiral field"""

    def __init__(self, data_file="hdgl_spiral10000_v30.json"):
        self.data_file = data_file
        self.data = None
        self.load_data()

        # V3.0 color scheme
        self.colors = {
            'spiral': '#FF6B6B',      # Coral red for spiral points
            'lattice': '#4ECDC4',     # Teal for lattice nodes
            'field': '#45B7D1',       # Blue for field overlay
            'dn_amplitude': '#96CEB4', # Mint for Dₙ amplitudes
            'energy': '#FFEAA7',      # Yellow for energy
            'base_infinity': '#DDA0DD' # Plum for Base(∞) seeds
        }

    def load_data(self):
        """Load V3.0 spiral field data"""
        try:
            with open(self.data_file, 'r') as f:
                self.data = json.load(f)
            print(f"✓ Loaded V3.0 data: {len(self.data['spiral_data'])} spiral points, {len(self.data['lattice_nodes'])} lattice nodes")
        except FileNotFoundError:
            print(f"✗ Data file {self.data_file} not found")
            return False
        return True

    def create_comprehensive_visualization(self):
        """Create comprehensive V3.0 visualization with multiple panels"""
        if not self.data:
            return

        fig = plt.figure(figsize=(20, 16))
        fig.suptitle('HDGL Harmonics + Spiral10000 (V3.0) - Dₙ(r) Base(∞) Mathematics',
                    fontsize=16, fontweight='bold', y=0.95)

        # Create 2x3 subplot grid
        gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.3)

        # 1. Main spiral field with lattice nodes
        ax1 = fig.add_subplot(gs[0, 0])
        self.plot_spiral_field(ax1)

        # 2. Dₙ(r) amplitude visualization
        ax2 = fig.add_subplot(gs[0, 1])
        self.plot_dn_amplitude(ax2)

        # 3. Energy distribution
        ax3 = fig.add_subplot(gs[0, 2])
        self.plot_energy_distribution(ax3)

        # 4. Base(∞) seed patterns
        ax4 = fig.add_subplot(gs[1, 0])
        self.plot_base_infinity_seeds(ax4)

        # 5. Lattice connectivity
        ax5 = fig.add_subplot(gs[1, 1])
        self.plot_lattice_connectivity(ax5)

        # 6. V3.0 statistics summary
        ax6 = fig.add_subplot(gs[1, 2])
        self.plot_statistics_summary(ax6)

        plt.tight_layout()
        plt.savefig('hdgl_spiral_visualization_v30.png', dpi=300, bbox_inches='tight',
                   facecolor='white', edgecolor='none')
        print("✓ Saved comprehensive V3.0 visualization to hdgl_spiral_visualization_v30.png")
        plt.show()

    def plot_spiral_field(self, ax):
        """Plot main spiral field with lattice nodes"""
        spiral_data = self.data['spiral_data']
        lattice_nodes = self.data['lattice_nodes']

        # Extract coordinates
        x_coords = [p['x'] for p in spiral_data]
        y_coords = [p['y'] for p in spiral_data]
        energies = [p['energy'] for p in spiral_data]

        # Normalize energies for coloring
        max_energy = max(energies)
        min_energy = min(energies)
        energy_colors = [(e - min_energy) / (max_energy - min_energy) for e in energies]

        # Plot spiral points with energy-based coloring
        scatter = ax.scatter(x_coords, y_coords, c=energy_colors, cmap='plasma',
                           s=1, alpha=0.7, edgecolors='none')

        # Plot lattice nodes
        if lattice_nodes:
            lattice_x = [n['x'] for n in lattice_nodes]
            lattice_y = [n['y'] for n in lattice_nodes]
            ax.scatter(lattice_x, lattice_y, c=self.colors['lattice'], s=50,
                      marker='o', edgecolors='white', linewidth=2, alpha=0.9,
                      label=f'V3.0 Lattice Nodes ({len(lattice_nodes)})')

        # Styling
        ax.set_title('V3.0 Dₙ(r) Spiral Field\nwith Base(∞) Lattice Nodes', fontsize=12, fontweight='bold')
        ax.set_xlabel('X Coordinate')
        ax.set_ylabel('Y Coordinate')
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal')

        # Colorbar for energy
        cbar = plt.colorbar(scatter, ax=ax, shrink=0.8)
        cbar.set_label('Energy Level', rotation=270, labelpad=15)

        # Legend
        ax.legend(loc='upper right', fontsize=8)

    def plot_dn_amplitude(self, ax):
        """Visualize Dₙ(r) amplitudes across the field"""
        spiral_data = self.data['spiral_data']

        x_coords = [p['x'] for p in spiral_data]
        y_coords = [p['y'] for p in spiral_data]
        dn_amplitudes = [p['dn_amplitude'] for p in spiral_data]

        # Create amplitude-based coloring
        scatter = ax.scatter(x_coords, y_coords, c=dn_amplitudes, cmap='viridis',
                           s=2, alpha=0.8, edgecolors='none')

        ax.set_title('Dₙ(r) Amplitude Distribution\nV3.0 Multi-Dimensional Coupling', fontsize=12, fontweight='bold')
        ax.set_xlabel('X Coordinate')
        ax.set_ylabel('Y Coordinate')
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal')

        # Colorbar
        cbar = plt.colorbar(scatter, ax=ax, shrink=0.8)
        cbar.set_label('Dₙ(r) Amplitude', rotation=270, labelpad=15)

    def plot_energy_distribution(self, ax):
        """Plot energy distribution histogram with V3.0 statistics"""
        energies = [p['energy'] for p in self.data['spiral_data']]
        stats = self.data['statistics']['energy_stats']

        # Create histogram
        n, bins, patches = ax.hist(energies, bins=50, alpha=0.7, color=self.colors['energy'],
                                 edgecolor='black', linewidth=0.5)

        # Add vertical lines for statistics
        ax.axvline(stats['mean'], color='red', linestyle='--', linewidth=2,
                  label=f'Mean: {stats["mean"]:.2f}')
        ax.axvline(stats['max'], color='green', linestyle='--', linewidth=2,
                  label=f'Max: {stats["max"]:.2f}')

        ax.set_title('V3.0 Energy Distribution\nBase(∞) Field Statistics', fontsize=12, fontweight='bold')
        ax.set_xlabel('Energy Level')
        ax.set_ylabel('Frequency')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Add statistics text
        stats_text = '.2f'
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
               verticalalignment='top', fontsize=8, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    def plot_base_infinity_seeds(self, ax):
        """Visualize Base(∞) seed patterns"""
        spiral_data = self.data['spiral_data']

        # Group by base seed value
        seed_groups = {}
        for point in spiral_data:
            seed = round(point['base_seed'], 6)  # Round for grouping
            if seed not in seed_groups:
                seed_groups[seed] = []
            seed_groups[seed].append(point)

        # Plot each seed group with different colors
        colors = plt.cm.rainbow(np.linspace(0, 1, len(seed_groups)))
        for i, (seed, points) in enumerate(seed_groups.items()):
            x_coords = [p['x'] for p in points]
            y_coords = [p['y'] for p in points]
            ax.scatter(x_coords, y_coords, c=[colors[i]], s=3, alpha=0.6,
                      label=f'φ≈{seed:.3f} ({len(points)} pts)', edgecolors='none')

        ax.set_title('Base(∞) Seed Distribution\nV3.0 Numeric Lattice Seeds', fontsize=12, fontweight='bold')
        ax.set_xlabel('X Coordinate')
        ax.set_ylabel('Y Coordinate')
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal')

        # Legend with fewer entries for readability
        if len(seed_groups) > 10:
            handles, labels = ax.get_legend_handles_labels()
            ax.legend(handles[:10], labels[:10], fontsize=6, loc='upper right')
        else:
            ax.legend(fontsize=6, loc='upper right')

    def plot_lattice_connectivity(self, ax):
        """Visualize lattice node connectivity"""
        lattice_nodes = self.data['lattice_nodes']

        if not lattice_nodes:
            ax.text(0.5, 0.5, 'No lattice nodes found', ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Lattice Connectivity\nV3.0 Dₙ(r) Resonance Nodes', fontsize=12, fontweight='bold')
            return

        # Create node position mapping
        node_positions = {node['id']: (node['x'], node['y']) for node in lattice_nodes}

        # Plot nodes
        for node in lattice_nodes:
            ax.scatter(node['x'], node['y'], c=self.colors['lattice'], s=100,
                      marker='o', edgecolors='white', linewidth=2, alpha=0.9)

            # Add node labels
            ax.annotate(f"D{node['dimension_n']}", (node['x'], node['y']),
                       xytext=(5, 5), textcoords='offset points', fontsize=8,
                       bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))

        # Plot connections
        lines = []
        for node in lattice_nodes:
            node_id = node['id']
            connections = node['connections']
            for conn_id in connections:
                if conn_id in node_positions:
                    lines.append([node_positions[node_id], node_positions[conn_id]])

        if lines:
            lc = LineCollection(lines, colors=self.colors['lattice'], alpha=0.3, linewidths=1)
            ax.add_collection(lc)

        ax.set_title('Lattice Connectivity\nV3.0 Dₙ(r) Resonance Network', fontsize=12, fontweight='bold')
        ax.set_xlabel('X Coordinate')
        ax.set_ylabel('Y Coordinate')
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal')

        # Add connection statistics
        total_connections = sum(len(node['connections']) for node in lattice_nodes)
        avg_connections = total_connections / len(lattice_nodes) if lattice_nodes else 0
        ax.text(0.02, 0.98, f'Nodes: {len(lattice_nodes)}\nAvg Connections: {avg_connections:.1f}',
               transform=ax.transAxes, verticalalignment='top', fontsize=8,
               bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))

    def plot_statistics_summary(self, ax):
        """Plot comprehensive V3.0 statistics summary"""
        stats = self.data['statistics']

        # Prepare statistics for display
        stat_labels = [
            f"Total Points: {stats['total_points']:,}",
            f"Lattice Nodes: {stats['lattice_stats']['total_nodes']}",
            f"Dimensions Used: {stats['lattice_stats']['dimensions_used']}",
            ".1f"            ".6f"            ".3f"            f"Base(∞) Seeds: {stats['base_infinity_stats']['unique_seeds']}",
            f"Dₙ Mean Amp: {stats['dn_stats']['mean']:.3f}",
            f"φ (Golden): {stats['golden_ratio']:.6f}",
            f"Dₙ Dimensions: {stats['num_dn_dimensions']}"
        ]

        # Create text display
        stat_text = "\n".join(stat_labels)

        ax.text(0.05, 0.95, stat_text, transform=ax.transAxes,
               fontsize=10, verticalalignment='top',
               bbox=dict(boxstyle='round,pad=1', facecolor='lightyellow', alpha=0.9))

        ax.set_title('V3.0 Field Statistics\nDₙ(r) Base(∞) Mathematics', fontsize=12, fontweight='bold')
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis('off')

        # Add V3.0 enhancement highlights
        highlights = [
            "🚀 Enhanced Dₙ(r) Mathematics",
            "🌟 Base(∞) Numeric Lattice",
            "⚡ Multi-dimensional Coupling",
            "🔮 8-Dimension Resonance",
            "♾️ Infinite Seed Foundation"
        ]

        highlight_text = "\n".join(highlights)
        ax.text(0.05, 0.3, highlight_text, transform=ax.transAxes,
               fontsize=9, verticalalignment='top',
               bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgreen', alpha=0.8))

    def create_evolution_animation(self):
        """Create evolution animation showing V3.0 field development"""
        print("Creating V3.0 evolution animation...")

        spiral_data = self.data['spiral_data']
        fig, ax = plt.subplots(figsize=(12, 10))

        # Sort by index for evolution
        sorted_data = sorted(spiral_data, key=lambda x: x['index'])

        # Animation frames
        frames = []
        for i in range(0, len(sorted_data), 100):  # Every 100 points
            frame_data = sorted_data[:i+1]
            frames.append(frame_data)

        def animate(frame_idx):
            ax.clear()
            frame_data = frames[frame_idx]

            x_coords = [p['x'] for p in frame_data]
            y_coords = [p['y'] for p in frame_data]
            energies = [p['energy'] for p in frame_data]

            # Normalize energies
            if energies:
                max_energy = max(energies)
                min_energy = min(energies)
                energy_colors = [(e - min_energy) / (max_energy - min_energy) if max_energy > min_energy else 0.5
                               for e in energies]

                scatter = ax.scatter(x_coords, y_coords, c=energy_colors, cmap='plasma',
                                   s=2, alpha=0.8, edgecolors='none')

            ax.set_title(f'V3.0 Dₙ(r) Spiral Evolution - Frame {frame_idx+1}/{len(frames)}')
            ax.set_xlabel('X Coordinate')
            ax.set_ylabel('Y Coordinate')
            ax.grid(True, alpha=0.3)
            ax.set_aspect('equal')

            # Add progress info
            progress = (frame_idx + 1) / len(frames) * 100
            ax.text(0.02, 0.98, f'Progress: {progress:.1f}%',
                   transform=ax.transAxes, fontsize=10,
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

        # Create animation (simplified - save key frames)
        import matplotlib.animation as animation

        anim = animation.FuncAnimation(fig, animate, frames=len(frames),
                                     interval=200, repeat=False)

        anim.save('hdgl_spiral_evolution_v30.gif', writer='pillow', fps=5, dpi=100)
        print("✓ Saved V3.0 evolution animation to hdgl_spiral_evolution_v30.gif")
        plt.close()

def main():
    """Main visualization function"""
    print("🌟 HDGL Harmonics + Spiral10000 V3.0 Visualization 🌟")
    print("🚀 Dₙ(r) Base(∞) Numeric Lattice Mathematics 🚀")
    print("=" * 70)

    visualizer = HDGLSpiralVisualizerV30()

    if visualizer.data:
        print("\n1. Creating comprehensive V3.0 visualization...")
        visualizer.create_comprehensive_visualization()

        print("\n2. Creating V3.0 evolution animation...")
        visualizer.create_evolution_animation()

        print("\n" + "=" * 70)
        print("🚀 HDGL V3.0 VISUALIZATION COMPLETE 🚀")
        print("=" * 70)
        print("Generated files:")
        print("  • hdgl_spiral_visualization_v30.png - Comprehensive visualization")
        print("  • hdgl_spiral_evolution_v30.gif - Evolution animation")
        print("=" * 70)
    else:
        print("✗ Failed to load V3.0 data file")

if __name__ == "__main__":
    main()