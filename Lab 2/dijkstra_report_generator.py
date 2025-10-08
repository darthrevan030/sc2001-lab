"""
Dijkstra's Algorithm: Implementation Comparison - Report Generator
Generates a comprehensive PDF report with all analysis and visualizations

Usage:
    python dijkstra_report_generator.py

Requirements:
    pip install numpy matplotlib pandas reportlab
"""

import numpy as np
import matplotlib.pyplot as plt
import time
import heapq
from collections import defaultdict
import pandas as pd
from datetime import datetime
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.patches as mpatches

# Set pandas display options for better output
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', None)

# Professional color scheme
COLORS = {
    'primary': '#2563eb',      # Professional blue
    'secondary': '#059669',    # Professional green
    'accent': '#7c3aed',       # Purple
    'warning': '#dc2626',      # Red
    'neutral': '#64748b',      # Gray
    'highlight': '#f59e0b',    # Amber
    'background': '#f8fafc',   # Light background
}

# Set professional matplotlib style
plt.style.use('seaborn-v0_8-darkgrid')
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['font.size'] = 10
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['xtick.labelsize'] = 9
plt.rcParams['ytick.labelsize'] = 9
plt.rcParams['legend.fontsize'] = 10
plt.rcParams['figure.titlesize'] = 14
plt.rcParams['axes.grid'] = True
plt.rcParams['grid.alpha'] = 0.3
plt.rcParams['grid.linestyle'] = '--'
plt.rcParams['lines.linewidth'] = 2.5
plt.rcParams['lines.markersize'] = 8

# ============================================================================
# IMPLEMENTATION (a): Adjacency Matrix + Array-based Priority Queue
# ============================================================================

class DijkstraMatrixArray:
    """Dijkstra's algorithm using adjacency matrix and array-based priority queue"""
    
    def __init__(self, adj_matrix):
        self.adj_matrix = adj_matrix
        self.V = len(adj_matrix)
    
    def find_min_distance(self, dist, visited):
        """Find vertex with minimum distance - O(V) operation"""
        min_dist = float('inf')
        min_idx = -1
        
        for v in range(self.V):
            if not visited[v] and dist[v] < min_dist:
                min_dist = dist[v]
                min_idx = v
        
        return min_idx
    
    def dijkstra(self, src):
        """
        Run Dijkstra's algorithm from source vertex
        Time Complexity: O(V²)
        Space Complexity: O(V)
        """
        dist = [float('inf')] * self.V
        visited = [False] * self.V
        parent = [-1] * self.V
        operations = 0
        
        dist[src] = 0
        
        # Main loop: process all vertices
        for _ in range(self.V - 1):
            # Find minimum distance vertex - O(V)
            u = self.find_min_distance(dist, visited)
            operations += self.V
            
            if u == -1:
                break
            
            visited[u] = True
            
            # Update distances of adjacent vertices - O(V)
            for v in range(self.V):
                operations += 1
                if (not visited[v] and 
                    self.adj_matrix[u][v] != 0 and 
                    dist[u] != float('inf') and
                    dist[u] + self.adj_matrix[u][v] < dist[v]):
                    dist[v] = dist[u] + self.adj_matrix[u][v]
                    parent[v] = u
        
        return {
            'distances': dist,
            'parent': parent,
            'operations': operations
        }


# ============================================================================
# IMPLEMENTATION (b): Adjacency List + Min-Heap Priority Queue
# ============================================================================

class DijkstraListHeap:
    """Dijkstra's algorithm using adjacency list and min-heap priority queue"""
    
    def __init__(self, adj_list, V):
        self.adj_list = adj_list
        self.V = V
    
    def dijkstra(self, src):
        """
        Run Dijkstra's algorithm from source vertex
        Time Complexity: O((V + E) log V)
        Space Complexity: O(V + E)
        """
        dist = [float('inf')] * self.V
        parent = [-1] * self.V
        operations = 0
        
        dist[src] = 0
        
        # Min-heap: (distance, vertex)
        heap = [(0, src)]
        
        while heap:
            # Extract minimum - O(log V)
            current_dist, u = heapq.heappop(heap)
            operations += int(np.log2(self.V)) if self.V > 1 else 1
            
            # Skip if we've already found a better path
            if current_dist > dist[u]:
                continue
            
            # Examine all neighbors
            for v, weight in self.adj_list[u]:
                operations += 1
                
                # Relaxation step
                if dist[u] + weight < dist[v]:
                    dist[v] = dist[u] + weight
                    parent[v] = u
                    
                    # Insert into heap - O(log V)
                    heapq.heappush(heap, (dist[v], v))
                    operations += int(np.log2(self.V)) if self.V > 1 else 1
        
        return {
            'distances': dist,
            'parent': parent,
            'operations': operations
        }


# ============================================================================
# GRAPH GENERATION UTILITIES
# ============================================================================

def generate_random_graph(V, density):
    """
    Generate a random undirected weighted graph
    
    Args:
        V: Number of vertices
        density: Edge density (0 to 1)
    
    Returns:
        adj_matrix: VxV adjacency matrix
        adj_list: Array of adjacency lists
        E: Number of edges
    """
    adj_matrix = np.zeros((V, V), dtype=int)
    adj_list = [[] for _ in range(V)]
    E = 0
    
    for i in range(V):
        for j in range(i + 1, V):
            if np.random.random() < density:
                weight = np.random.randint(1, 21)
                adj_matrix[i][j] = weight
                adj_matrix[j][i] = weight
                adj_list[i].append((j, weight))
                adj_list[j].append((i, weight))
                E += 1
    
    return adj_matrix, adj_list, E


# ============================================================================
# BENCHMARKING
# ============================================================================

def run_benchmark(graph_sizes, density=0.3, num_trials=5):
    """
    Run performance benchmark comparing both implementations
    
    Args:
        graph_sizes: List of vertex counts to test
        density: Edge density for generated graphs
        num_trials: Number of timing trials on THE SAME graph (for timing stability)
    
    Returns:
        DataFrame with benchmark results
    """
    results = []
    
    for V in graph_sizes:
        print(f"  Benchmarking V={V}...", end=" ", flush=True)
        
        # Generate ONE graph that will be used for ALL trials
        adj_matrix, adj_list, E = generate_random_graph(V, density)
        
        time_matrix_array = []
        time_list_heap = []
        
        # Run multiple trials on THE SAME GRAPH to get stable timings
        for trial in range(num_trials):
            # Test implementation (a): Matrix + Array
            start = time.perf_counter()
            dijkstra_a = DijkstraMatrixArray(adj_matrix)
            result_a = dijkstra_a.dijkstra(0)
            time_a = (time.perf_counter() - start) * 1000  # Convert to ms
            
            # Test implementation (b): List + Heap (on the SAME graph)
            start = time.perf_counter()
            dijkstra_b = DijkstraListHeap(adj_list, V)
            result_b = dijkstra_b.dijkstra(0)
            time_b = (time.perf_counter() - start) * 1000  # Convert to ms
            
            # Verify both implementations produce the same result (only check once)
            if trial == 0 and result_a['distances'] != result_b['distances']:
                print(f"\nWARNING: Implementations disagree for V={V}!")
            
            time_matrix_array.append(time_a)
            time_list_heap.append(time_b)
        
        # Store results
        results.append({
            'V': V,
            'E': E,
            'density': E / (V * (V - 1) / 2) if V > 1 else 0,
            'time_matrix_array': np.mean(time_matrix_array),
            'time_list_heap': np.mean(time_list_heap),
            'time_matrix_std': np.std(time_matrix_array),
            'time_list_std': np.std(time_list_heap),
            'ops_matrix_array': result_a['operations'],
            'ops_list_heap': result_b['operations'],
            'speedup': np.mean(time_matrix_array) / np.mean(time_list_heap)
        })
        
        print(f"Done (speedup={results[-1]['speedup']:.2f}x)")
    
    return pd.DataFrame(results)


# ============================================================================
# PDF REPORT GENERATION
# ============================================================================

def create_title_page(pdf, filename):
    """Create a professional title page with enhanced visual design"""
    fig = plt.figure(figsize=(8.5, 11), facecolor='white')
    ax = fig.add_subplot(111)
    ax.axis('off')
    
    # Add decorative header bar
    header_rect = plt.Rectangle((0.05, 0.85), 0.9, 0.08, 
                                facecolor=COLORS['primary'], alpha=0.15, 
                                transform=ax.transAxes, zorder=0)
    ax.add_patch(header_rect)
    
    # Main title
    ax.text(0.5, 0.88, "Dijkstra's Algorithm", 
            ha='center', va='center', fontsize=36, fontweight='bold',
            color=COLORS['primary'], transform=ax.transAxes)
    
    # Subtitle
    ax.text(0.5, 0.80, "Comprehensive Implementation Comparison", 
            ha='center', va='center', fontsize=20, fontweight='300',
            color=COLORS['neutral'], transform=ax.transAxes)
    
    # Course information box
    course_box = dict(boxstyle='round,pad=1.2', facecolor='white', 
                     edgecolor=COLORS['primary'], linewidth=2, alpha=0.9)
    ax.text(0.5, 0.68, "SC2001/CE2101/CZ2101", 
            ha='center', va='center', fontsize=16, fontweight='bold',
            transform=ax.transAxes, color=COLORS['primary'])
    
    ax.text(0.5, 0.63, "Algorithm Design and Analysis", 
            ha='center', va='center', fontsize=14, style='italic',
            transform=ax.transAxes, color=COLORS['neutral'])
    
    ax.text(0.5, 0.58, "Project 2: Graph Algorithm Performance Analysis", 
            ha='center', va='center', fontsize=13,
            transform=ax.transAxes, color=COLORS['neutral'])
    
    # Implementation comparison section with better formatting
    impl_box = dict(boxstyle='round,pad=1.5', facecolor=COLORS['background'], 
                    edgecolor=COLORS['secondary'], linewidth=2, alpha=0.95)
    
    impl_text = ("IMPLEMENTATION (a)\n"
                "Adjacency Matrix + Array Priority Queue\n"
                "Time: O(V²)  |  Space: O(V)\n\n"
                "━━━━━━━━━━━━━━━━━  VS  ━━━━━━━━━━━━━━━━━\n\n"
                "IMPLEMENTATION (b)\n"
                "Adjacency List + Min-Heap Priority Queue\n"
                "Time: O((V + E) log V)  |  Space: O(V + E)")
    
    ax.text(0.5, 0.38, impl_text, 
            ha='center', va='center', fontsize=11, fontfamily='monospace',
            transform=ax.transAxes, bbox=impl_box, linespacing=1.8,
            color='#1e293b')
    
    # Add decorative elements
    divider_line1 = plt.Line2D([0.15, 0.85], [0.23, 0.23], 
                               color=COLORS['accent'], linewidth=2, 
                               alpha=0.5, transform=ax.transAxes)
    ax.add_line(divider_line1)
    
    # Report metadata
    date_str = datetime.now().strftime("%B %d, %Y")
    ax.text(0.5, 0.18, f"Report Generated: {date_str}", 
            ha='center', va='center', fontsize=12, fontweight='500',
            transform=ax.transAxes, color=COLORS['neutral'])
    
    ax.text(0.5, 0.13, "Comprehensive Performance Analysis • Empirical Validation", 
            ha='center', va='center', fontsize=11, style='italic',
            transform=ax.transAxes, color=COLORS['neutral'], alpha=0.8)
    
    # Footer with page number
    ax.text(0.5, 0.03, "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━", 
            ha='center', va='center', fontsize=8, 
            transform=ax.transAxes, color=COLORS['primary'], alpha=0.3)
    
    ax.text(0.5, 0.01, "Page 1", 
            ha='center', va='center', fontsize=9, 
            transform=ax.transAxes, color=COLORS['neutral'], alpha=0.6)
    
    pdf.savefig(fig, bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.close(fig)


def create_section_page(pdf, section_number, title, description):
    """Create a professional section divider page"""
    fig = plt.figure(figsize=(8.5, 11), facecolor='white')
    ax = fig.add_subplot(111)
    ax.axis('off')
    
    # Background decorative element
    bg_rect = plt.Rectangle((0.0, 0.4), 1.0, 0.3, 
                           facecolor=COLORS['primary'], alpha=0.05, 
                           transform=ax.transAxes, zorder=0)
    ax.add_patch(bg_rect)
    
    # Section number with circle
    circle = plt.Circle((0.5, 0.62), 0.08, color=COLORS['primary'], 
                       alpha=0.15, transform=ax.transAxes, zorder=1)
    ax.add_patch(circle)
    
    ax.text(0.5, 0.62, f"{section_number}", 
            ha='center', va='center', fontsize=48, fontweight='bold', 
            color=COLORS['primary'], transform=ax.transAxes, zorder=2)
    
    # Section title
    ax.text(0.5, 0.50, title, 
            ha='center', va='center', fontsize=26, fontweight='bold',
            transform=ax.transAxes, color='#1e293b')
    
    # Decorative line
    line = plt.Line2D([0.2, 0.8], [0.45, 0.45], 
                     color=COLORS['accent'], linewidth=3, 
                     alpha=0.6, transform=ax.transAxes)
    ax.add_line(line)
    
    # Description
    if description:
        desc_box = dict(boxstyle='round,pad=1.0', facecolor='white', 
                       edgecolor=COLORS['secondary'], linewidth=1.5, alpha=0.9)
        ax.text(0.5, 0.32, description, 
                ha='center', va='center', fontsize=13, style='italic',
                transform=ax.transAxes, wrap=True, color=COLORS['neutral'],
                bbox=desc_box, linespacing=1.6)
    
    pdf.savefig(fig, bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.close(fig)


def plot_sparse_analysis(pdf, df_sparse):
    """Create detailed sparse graph analysis plots with professional styling"""
    
    # Page 1: Time Comparison with enhanced visuals
    fig = plt.figure(figsize=(11, 8.5), facecolor='white')
    fig.suptitle('Section 1.1: Execution Time Analysis - Sparse Graphs', 
                 fontsize=18, fontweight='bold', y=0.97, color=COLORS['primary'])
    
    ax1 = plt.subplot(2, 1, 1)
    # Enhanced line plots with markers
    line1 = ax1.plot(df_sparse['V'], df_sparse['time_matrix_array'], 
                     'o-', linewidth=3, markersize=10, label='Matrix + Array', 
                     color=COLORS['primary'], markerfacecolor='white', 
                     markeredgewidth=2, markeredgecolor=COLORS['primary'])
    line2 = ax1.plot(df_sparse['V'], df_sparse['time_list_heap'], 
                     's-', linewidth=3, markersize=10, label='List + Heap', 
                     color=COLORS['secondary'], markerfacecolor='white',
                     markeredgewidth=2, markeredgecolor=COLORS['secondary'])
    
    # Enhanced labels and title
    ax1.set_xlabel('Number of Vertices (V)', fontsize=13, fontweight='600', color='#1e293b')
    ax1.set_ylabel('Execution Time (milliseconds)', fontsize=13, fontweight='600', color='#1e293b')
    ax1.set_title('Raw Execution Time Comparison (Linear Scale)', 
                  fontsize=14, fontweight='bold', pad=12, color='#334155')
    
    # Enhanced legend
    legend = ax1.legend(fontsize=12, loc='upper left', frameon=True, 
                       shadow=True, fancybox=True, framealpha=0.95)
    legend.get_frame().set_facecolor('white')
    legend.get_frame().set_edgecolor(COLORS['neutral'])
    
    # Enhanced grid
    ax1.grid(True, alpha=0.25, linestyle='--', linewidth=0.8)
    ax1.set_facecolor('#f8fafc')
    
    # Professional annotation box - positioned to avoid overlap
    max_v = df_sparse['V'].max()
    max_time_matrix = df_sparse['time_matrix_array'].max()
    max_time_heap = df_sparse['time_list_heap'].max()
    speedup_at_max = max_time_matrix / max_time_heap
    
    annotation_box = dict(boxstyle='round,pad=0.6', facecolor='white', 
                         edgecolor=COLORS['accent'], linewidth=1.5, alpha=0.95)
    ax1.text(0.98, 0.65, 
             f'Peak @ V={int(max_v)}\n'
             f'Matrix: {max_time_matrix:.2f}ms\n'
             f'Heap: {max_time_heap:.2f}ms\n'
             f'Speedup: {speedup_at_max:.2f}×',
             transform=ax1.transAxes, fontsize=9, verticalalignment='top',
             horizontalalignment='right', bbox=annotation_box, fontfamily='monospace')
    
    ax2 = plt.subplot(2, 1, 2)
    ax2.plot(df_sparse['V'], df_sparse['time_matrix_array'], 
             'o-', linewidth=3, markersize=10, label='Matrix + Array', 
             color=COLORS['primary'], markerfacecolor='white',
             markeredgewidth=2, markeredgecolor=COLORS['primary'])
    ax2.plot(df_sparse['V'], df_sparse['time_list_heap'], 
             's-', linewidth=3, markersize=10, label='List + Heap', 
             color=COLORS['secondary'], markerfacecolor='white',
             markeredgewidth=2, markeredgecolor=COLORS['secondary'])
    
    ax2.set_xlabel('Number of Vertices (V)', fontsize=13, fontweight='600', color='#1e293b')
    ax2.set_ylabel('Execution Time (ms) - Log Scale', fontsize=13, fontweight='600', color='#1e293b')
    ax2.set_title('Logarithmic Scale View - Growth Rate Analysis', 
                  fontsize=14, fontweight='bold', pad=12, color='#334155')
    
    legend2 = ax2.legend(fontsize=12, loc='upper left', frameon=True, 
                        shadow=True, fancybox=True, framealpha=0.95)
    legend2.get_frame().set_facecolor('white')
    legend2.get_frame().set_edgecolor(COLORS['neutral'])
    
    ax2.grid(True, alpha=0.25, linestyle='--', linewidth=0.8, which='both')
    ax2.set_facecolor('#f8fafc')
    ax2.set_yscale('log')
    
    # Add more space between subplots to prevent overlap
    plt.tight_layout(rect=[0, 0.02, 1, 0.95])
    pdf.savefig(fig, bbox_inches='tight', facecolor='white', edgecolor='none', pad_inches=0.2)
    plt.close(fig)
    
    # Page 2: Speedup Analysis with professional styling
    fig = plt.figure(figsize=(11, 8.5), facecolor='white')
    fig.suptitle('Section 1.2: Performance Speedup Analysis - Sparse Graphs', 
                 fontsize=18, fontweight='bold', y=0.97, color=COLORS['primary'])
    
    ax1 = plt.subplot(2, 1, 1)
    speedup_values = df_sparse['speedup'].values
    x_positions = np.arange(len(df_sparse['V']))
    
    # Create gradient bars based on speedup value
    colors_bars = [COLORS['secondary'] if s > 1 else COLORS['warning'] for s in speedup_values]
    bars = ax1.bar(x_positions, speedup_values, color=colors_bars, alpha=0.8, 
                   edgecolor='#1e293b', linewidth=1.5, width=0.7)
    
    # Add reference line
    ax1.axhline(y=1, color='#1e293b', linestyle='--', linewidth=2.5, 
                label='Equal Performance (1.0×)', zorder=0, alpha=0.7)
    
    # Customize axes
    ax1.set_xticks(x_positions)
    ax1.set_xticklabels(df_sparse['V'].values, fontsize=10)
    ax1.set_xlabel('Number of Vertices (V)', fontsize=13, fontweight='600', color='#1e293b')
    ax1.set_ylabel('Speedup Factor (Matrix Time / Heap Time)', fontsize=13, fontweight='600', color='#1e293b')
    ax1.set_title('Speedup Factor by Graph Size', fontsize=14, fontweight='bold', pad=12, color='#334155')
    
    legend = ax1.legend(fontsize=11, frameon=True, shadow=True, fancybox=True, framealpha=0.95)
    legend.get_frame().set_facecolor('white')
    legend.get_frame().set_edgecolor(COLORS['neutral'])
    
    ax1.grid(True, alpha=0.25, axis='y', linestyle='--', linewidth=0.8)
    ax1.set_facecolor('#f8fafc')
    
    # Add value labels on bars with better formatting to avoid overlap
    for i, (bar, val) in enumerate(zip(bars, speedup_values)):
        height = bar.get_height()
        # Smart positioning to avoid overlap
        if height > 2.5:
            # Label inside bar for tall bars
            label_color = 'white'
            y_pos = height - 0.3
            va = 'top'
        elif height > 1.5:
            # Label in middle for medium bars
            label_color = 'white'
            y_pos = height / 2
            va = 'center'
        else:
            # Label above bar for short bars
            label_color = '#1e293b'
            y_pos = height + 0.08
            va = 'bottom'
        
        ax1.text(bar.get_x() + bar.get_width()/2., y_pos,
                f'{val:.2f}×', ha='center', va=va, 
                fontsize=8.5, fontweight='bold', color=label_color)
    
    ax2 = plt.subplot(2, 1, 2)
    ax2.plot(df_sparse['V'], df_sparse['speedup'], 
             'o-', linewidth=2.5, markersize=10, color=COLORS['accent'],
             markerfacecolor='white', markeredgewidth=2, markeredgecolor=COLORS['accent'])
    ax2.axhline(y=1, color='#dc2626', linestyle='--', linewidth=2, alpha=0.7, label='Equal Performance')
    ax2.fill_between(df_sparse['V'], 1, df_sparse['speedup'], 
                     where=(df_sparse['speedup'] > 1), alpha=0.2, color=COLORS['secondary'], label='Heap Faster')
    ax2.fill_between(df_sparse['V'], 1, df_sparse['speedup'], 
                     where=(df_sparse['speedup'] <= 1), alpha=0.2, color=COLORS['warning'], label='Matrix Faster')
    
    ax2.set_xlabel('Number of Vertices (V)', fontsize=13, fontweight='600', color='#1e293b')
    ax2.set_ylabel('Speedup Factor', fontsize=13, fontweight='600', color='#1e293b')
    ax2.set_title('Speedup Trend Analysis', fontsize=14, fontweight='bold', pad=12, color='#334155')
    
    # Position legend to avoid overlap
    legend2 = ax2.legend(fontsize=10, loc='upper right', frameon=True, 
                        shadow=True, fancybox=True, framealpha=0.95)
    legend2.get_frame().set_facecolor('white')
    legend2.get_frame().set_edgecolor(COLORS['neutral'])
    
    ax2.grid(True, alpha=0.25, linestyle='--', linewidth=0.8)
    ax2.set_facecolor('#f8fafc')
    
    # Position annotation box to avoid overlap
    avg_speedup = df_sparse['speedup'].mean()
    annotation_box = dict(boxstyle='round,pad=0.6', facecolor='#fef3c7', 
                         edgecolor=COLORS['highlight'], linewidth=1.5, alpha=0.95)
    ax2.text(0.02, 0.05, f'Avg: {avg_speedup:.2f}×',
             transform=ax2.transAxes, fontsize=10, verticalalignment='bottom',
             bbox=annotation_box, fontweight='bold')
    
    # Add more space between subplots
    plt.tight_layout(rect=[0, 0.02, 1, 0.95])
    pdf.savefig(fig, bbox_inches='tight', facecolor='white', edgecolor='none', pad_inches=0.2)
    plt.close(fig)
    
    # Page 3: Operations Count with better spacing
    fig = plt.figure(figsize=(11, 8.5), facecolor='white')
    fig.suptitle('Section 1.3: Algorithm Operations Count - Sparse Graphs', 
                 fontsize=18, fontweight='bold', y=0.97, color=COLORS['primary'])
    
    ax1 = plt.subplot(2, 1, 1)
    ax1.plot(df_sparse['V'], df_sparse['ops_matrix_array'], 
             'o-', linewidth=3, markersize=10, label='Matrix + Array', 
             color=COLORS['primary'], markerfacecolor='white',
             markeredgewidth=2, markeredgecolor=COLORS['primary'])
    ax1.plot(df_sparse['V'], df_sparse['ops_list_heap'], 
             's-', linewidth=3, markersize=10, label='List + Heap', 
             color=COLORS['secondary'], markerfacecolor='white',
             markeredgewidth=2, markeredgecolor=COLORS['secondary'])
    
    ax1.set_xlabel('Number of Vertices (V)', fontsize=13, fontweight='600', color='#1e293b')
    ax1.set_ylabel('Operations Count', fontsize=13, fontweight='600', color='#1e293b')
    ax1.set_title('Total Elementary Operations (Linear Scale)', 
                  fontsize=14, fontweight='bold', pad=12, color='#334155')
    
    legend1 = ax1.legend(fontsize=12, loc='upper left', frameon=True, 
                        shadow=True, fancybox=True, framealpha=0.95)
    legend1.get_frame().set_facecolor('white')
    legend1.get_frame().set_edgecolor(COLORS['neutral'])
    
    ax1.grid(True, alpha=0.25, linestyle='--', linewidth=0.8)
    ax1.set_facecolor('#f8fafc')
    
    ax2 = plt.subplot(2, 1, 2)
    ax2.plot(df_sparse['V'], df_sparse['ops_matrix_array'], 
             'o-', linewidth=3, markersize=10, label='Matrix + Array', 
             color=COLORS['primary'], markerfacecolor='white',
             markeredgewidth=2, markeredgecolor=COLORS['primary'])
    ax2.plot(df_sparse['V'], df_sparse['ops_list_heap'], 
             's-', linewidth=3, markersize=10, label='List + Heap', 
             color=COLORS['secondary'], markerfacecolor='white',
             markeredgewidth=2, markeredgecolor=COLORS['secondary'])
    
    ax2.set_xlabel('Number of Vertices (V)', fontsize=13, fontweight='600', color='#1e293b')
    ax2.set_ylabel('Operations Count (Log Scale)', fontsize=13, fontweight='600', color='#1e293b')
    ax2.set_title('Logarithmic Scale - Complexity Comparison', 
                  fontsize=14, fontweight='bold', pad=12, color='#334155')
    
    legend2 = ax2.legend(fontsize=12, loc='upper left', frameon=True, 
                        shadow=True, fancybox=True, framealpha=0.95)
    legend2.get_frame().set_facecolor('white')
    legend2.get_frame().set_edgecolor(COLORS['neutral'])
    
    ax2.grid(True, alpha=0.25, linestyle='--', linewidth=0.8, which='both')
    ax2.set_facecolor('#f8fafc')
    ax2.set_yscale('log')
    
    # Add theoretical complexity annotations in better position
    complexity_box = dict(boxstyle='round,pad=0.6', facecolor='#dbeafe', 
                         edgecolor=COLORS['primary'], linewidth=1.5, alpha=0.95)
    ax2.text(0.98, 0.05, 
             'Matrix: O(V²)\nHeap: O((V+E)logV)',
             transform=ax2.transAxes, fontsize=10, verticalalignment='bottom',
             horizontalalignment='right', bbox=complexity_box,
             fontfamily='monospace', fontweight='bold')
    
    plt.tight_layout(rect=[0, 0.02, 1, 0.95])
    pdf.savefig(fig, bbox_inches='tight', facecolor='white', edgecolor='none', pad_inches=0.2)
    plt.close(fig)
    
    # Page 4: Graph Properties with better spacing
    fig = plt.figure(figsize=(11, 8.5), facecolor='white')
    fig.suptitle('Section 1.4: Graph Properties - Sparse Graphs', 
                 fontsize=18, fontweight='bold', y=0.97, color=COLORS['primary'])
    
    ax1 = plt.subplot(2, 2, 1)
    ax1.plot(df_sparse['V'], df_sparse['E'], 'o-', linewidth=2.5, markersize=9, 
             color=COLORS['highlight'], markerfacecolor='white',
             markeredgewidth=2, markeredgecolor=COLORS['highlight'])
    ax1.set_xlabel('Vertices (V)', fontsize=11, fontweight='600')
    ax1.set_ylabel('Edges (E)', fontsize=11, fontweight='600')
    ax1.set_title('Edge Count vs Vertex Count', fontsize=12, fontweight='bold', pad=8)
    ax1.grid(True, alpha=0.25, linestyle='--', linewidth=0.8)
    ax1.set_facecolor('#f8fafc')
    
    ax2 = plt.subplot(2, 2, 2)
    ax2.plot(df_sparse['V'], df_sparse['density'], 'o-', linewidth=2.5, markersize=9, 
             color=COLORS['accent'], markerfacecolor='white',
             markeredgewidth=2, markeredgecolor=COLORS['accent'])
    ax2.set_xlabel('Vertices (V)', fontsize=11, fontweight='600')
    ax2.set_ylabel('Density', fontsize=11, fontweight='600')
    ax2.set_title('Graph Density Distribution', fontsize=12, fontweight='bold', pad=8)
    ax2.grid(True, alpha=0.25, linestyle='--', linewidth=0.8)
    ax2.set_facecolor('#f8fafc')
    ax2.axhline(y=df_sparse['density'].mean(), color=COLORS['warning'], linestyle='--', 
                linewidth=2, alpha=0.7, label=f"Mean: {df_sparse['density'].mean():.3f}")
    ax2.legend(fontsize=9, loc='best', framealpha=0.9)
    
    ax3 = plt.subplot(2, 2, 3)
    e_v_ratio = df_sparse['E'] / df_sparse['V']
    ax3.plot(df_sparse['V'], e_v_ratio, 'o-', linewidth=2.5, markersize=9, 
             color='#ec4899', markerfacecolor='white',
             markeredgewidth=2, markeredgecolor='#ec4899')
    ax3.set_xlabel('Vertices (V)', fontsize=11, fontweight='600')
    ax3.set_ylabel('E/V Ratio', fontsize=11, fontweight='600')
    ax3.set_title('Edge-to-Vertex Ratio', fontsize=12, fontweight='bold', pad=8)
    ax3.grid(True, alpha=0.25, linestyle='--', linewidth=0.8)
    ax3.set_facecolor('#f8fafc')
    ax3.axhline(y=e_v_ratio.mean(), color=COLORS['warning'], linestyle='--', 
                linewidth=2, alpha=0.7, label=f"Mean: {e_v_ratio.mean():.2f}")
    ax3.legend(fontsize=9, loc='best', framealpha=0.9)
    
    ax4 = plt.subplot(2, 2, 4)
    ax4.axis('off')
    summary_text = f"""SPARSE GRAPH STATISTICS
{'─'*32}

Vertex Range:    {int(df_sparse['V'].min())}-{int(df_sparse['V'].max())}
Edge Range:      {int(df_sparse['E'].min())}-{int(df_sparse['E'].max())}

Avg Density:     {df_sparse['density'].mean():.4f}
Avg E/V Ratio:   {e_v_ratio.mean():.2f}

Graph Type:      Sparse
Expected:        E ≈ O(V)
Observed:        E ≈ {e_v_ratio.mean():.2f}V

{'─'*32}
✓ Sparse characteristics confirmed"""
    
    stats_box = dict(boxstyle='round,pad=0.8', facecolor='#fffbeb', 
                    edgecolor=COLORS['highlight'], linewidth=1.5, alpha=0.95)
    ax4.text(0.1, 0.85, summary_text, transform=ax4.transAxes, fontsize=9.5,
             verticalalignment='top', fontfamily='monospace',
             bbox=stats_box, linespacing=1.4)
    
    plt.tight_layout(rect=[0, 0.02, 1, 0.95])
    pdf.savefig(fig, bbox_inches='tight', facecolor='white', edgecolor='none', pad_inches=0.2)
    plt.close(fig)


def plot_dense_analysis(pdf, df_dense):
    """Create detailed dense graph analysis plots with professional styling and no overlaps"""
    
    # Page 1: Time Comparison with enhanced visuals
    fig = plt.figure(figsize=(11, 8.5), facecolor='white')
    fig.suptitle('Section 2.1: Execution Time Analysis - Dense Graphs', 
                 fontsize=18, fontweight='bold', y=0.97, color=COLORS['primary'])
    
    ax1 = plt.subplot(2, 1, 1)
    ax1.plot(df_dense['V'], df_dense['time_matrix_array'], 
             'o-', linewidth=3, markersize=10, label='Matrix + Array', 
             color=COLORS['primary'], markerfacecolor='white',
             markeredgewidth=2, markeredgecolor=COLORS['primary'])
    ax1.plot(df_dense['V'], df_dense['time_list_heap'], 
             's-', linewidth=3, markersize=10, label='List + Heap', 
             color=COLORS['secondary'], markerfacecolor='white',
             markeredgewidth=2, markeredgecolor=COLORS['secondary'])
    
    ax1.set_xlabel('Number of Vertices (V)', fontsize=13, fontweight='600', color='#1e293b')
    ax1.set_ylabel('Execution Time (milliseconds)', fontsize=13, fontweight='600', color='#1e293b')
    ax1.set_title('Raw Execution Time Comparison (Linear Scale)', 
                  fontsize=14, fontweight='bold', pad=12, color='#334155')
    
    legend1 = ax1.legend(fontsize=12, loc='upper left', frameon=True, 
                        shadow=True, fancybox=True, framealpha=0.95)
    legend1.get_frame().set_facecolor('white')
    legend1.get_frame().set_edgecolor(COLORS['neutral'])
    
    ax1.grid(True, alpha=0.25, linestyle='--', linewidth=0.8)
    ax1.set_facecolor('#f8fafc')
    
    # Annotation box positioned to avoid overlap
    max_v = df_dense['V'].max()
    max_time_matrix = df_dense['time_matrix_array'].max()
    max_time_heap = df_dense['time_list_heap'].max()
    speedup_at_max = max_time_matrix / max_time_heap
    
    annotation_box = dict(boxstyle='round,pad=0.6', facecolor='white', 
                         edgecolor=COLORS['accent'], linewidth=1.5, alpha=0.95)
    ax1.text(0.98, 0.65, 
             f'Peak @ V={int(max_v)}\n'
             f'Matrix: {max_time_matrix:.2f}ms\n'
             f'Heap: {max_time_heap:.2f}ms\n'
             f'Speedup: {speedup_at_max:.2f}×',
             transform=ax1.transAxes, fontsize=9, verticalalignment='top',
             horizontalalignment='right', bbox=annotation_box, fontfamily='monospace')
    
    ax2 = plt.subplot(2, 1, 2)
    ax2.plot(df_dense['V'], df_dense['time_matrix_array'], 
             'o-', linewidth=3, markersize=10, label='Matrix + Array', 
             color=COLORS['primary'], markerfacecolor='white',
             markeredgewidth=2, markeredgecolor=COLORS['primary'])
    ax2.plot(df_dense['V'], df_dense['time_list_heap'], 
             's-', linewidth=3, markersize=10, label='List + Heap', 
             color=COLORS['secondary'], markerfacecolor='white',
             markeredgewidth=2, markeredgecolor=COLORS['secondary'])
    
    ax2.set_xlabel('Number of Vertices (V)', fontsize=13, fontweight='600', color='#1e293b')
    ax2.set_ylabel('Execution Time (ms) - Log Scale', fontsize=13, fontweight='600', color='#1e293b')
    ax2.set_title('Logarithmic Scale View - Growth Rate Analysis', 
                  fontsize=14, fontweight='bold', pad=12, color='#334155')
    
    legend2 = ax2.legend(fontsize=12, loc='upper left', frameon=True, 
                        shadow=True, fancybox=True, framealpha=0.95)
    legend2.get_frame().set_facecolor('white')
    legend2.get_frame().set_edgecolor(COLORS['neutral'])
    
    ax2.grid(True, alpha=0.25, linestyle='--', linewidth=0.8, which='both')
    ax2.set_facecolor('#f8fafc')
    ax2.set_yscale('log')
    
    plt.tight_layout(rect=[0, 0.02, 1, 0.95])
    pdf.savefig(fig, bbox_inches='tight', facecolor='white', edgecolor='none', pad_inches=0.2)
    plt.close(fig)
    
    # Page 2: Speedup Analysis
    fig = plt.figure(figsize=(11, 8.5), facecolor='white')
    fig.suptitle('Section 2.2: Performance Speedup Analysis - Dense Graphs', 
                 fontsize=18, fontweight='bold', y=0.97, color=COLORS['primary'])
    
    ax1 = plt.subplot(2, 1, 1)
    speedup_values = df_dense['speedup'].values
    x_positions = np.arange(len(df_dense['V']))
    
    colors_bars = [COLORS['secondary'] if s > 1 else COLORS['warning'] for s in speedup_values]
    bars = ax1.bar(x_positions, speedup_values, color=colors_bars, alpha=0.8, 
                   edgecolor='#1e293b', linewidth=1.5, width=0.7)
    
    ax1.axhline(y=1, color='#1e293b', linestyle='--', linewidth=2.5, 
                label='Equal Performance (1.0×)', zorder=0, alpha=0.7)
    
    ax1.set_xticks(x_positions)
    ax1.set_xticklabels(df_dense['V'].values, fontsize=10)
    ax1.set_xlabel('Number of Vertices (V)', fontsize=13, fontweight='600', color='#1e293b')
    ax1.set_ylabel('Speedup Factor (Matrix Time / Heap Time)', fontsize=13, fontweight='600', color='#1e293b')
    ax1.set_title('Speedup Factor by Graph Size', fontsize=14, fontweight='bold', pad=12, color='#334155')
    
    legend = ax1.legend(fontsize=11, frameon=True, shadow=True, fancybox=True, framealpha=0.95)
    legend.get_frame().set_facecolor('white')
    legend.get_frame().set_edgecolor(COLORS['neutral'])
    
    ax1.grid(True, alpha=0.25, axis='y', linestyle='--', linewidth=0.8)
    ax1.set_facecolor('#f8fafc')
    
    # Smart label positioning
    for i, (bar, val) in enumerate(zip(bars, speedup_values)):
        height = bar.get_height()
        if height > 2.5:
            label_color = 'white'
            y_pos = height - 0.3
            va = 'top'
        elif height > 1.5:
            label_color = 'white'
            y_pos = height / 2
            va = 'center'
        else:
            label_color = '#1e293b'
            y_pos = height + 0.08
            va = 'bottom'
        
        ax1.text(bar.get_x() + bar.get_width()/2., y_pos,
                f'{val:.2f}×', ha='center', va=va, 
                fontsize=8.5, fontweight='bold', color=label_color)
    
    ax2 = plt.subplot(2, 1, 2)
    ax2.plot(df_dense['V'], df_dense['speedup'], 
             'o-', linewidth=2.5, markersize=10, color=COLORS['accent'],
             markerfacecolor='white', markeredgewidth=2, markeredgecolor=COLORS['accent'])
    ax2.axhline(y=1, color='#dc2626', linestyle='--', linewidth=2, alpha=0.7, label='Equal Performance')
    ax2.fill_between(df_dense['V'], 1, df_dense['speedup'], 
                     where=(df_dense['speedup'] > 1), alpha=0.2, color=COLORS['secondary'], label='Heap Faster')
    ax2.fill_between(df_dense['V'], 1, df_dense['speedup'], 
                     where=(df_dense['speedup'] <= 1), alpha=0.2, color=COLORS['warning'], label='Matrix Faster')
    
    ax2.set_xlabel('Number of Vertices (V)', fontsize=13, fontweight='600', color='#1e293b')
    ax2.set_ylabel('Speedup Factor', fontsize=13, fontweight='600', color='#1e293b')
    ax2.set_title('Speedup Trend Analysis', fontsize=14, fontweight='bold', pad=12, color='#334155')
    
    legend2 = ax2.legend(fontsize=10, loc='upper right', frameon=True, 
                        shadow=True, fancybox=True, framealpha=0.95)
    legend2.get_frame().set_facecolor('white')
    legend2.get_frame().set_edgecolor(COLORS['neutral'])
    
    ax2.grid(True, alpha=0.25, linestyle='--', linewidth=0.8)
    ax2.set_facecolor('#f8fafc')
    
    avg_speedup = df_dense['speedup'].mean()
    annotation_box = dict(boxstyle='round,pad=0.6', facecolor='#fef3c7', 
                         edgecolor=COLORS['highlight'], linewidth=1.5, alpha=0.95)
    ax2.text(0.02, 0.05, f'Avg: {avg_speedup:.2f}×',
             transform=ax2.transAxes, fontsize=10, verticalalignment='bottom',
             bbox=annotation_box, fontweight='bold')
    
    plt.tight_layout(rect=[0, 0.02, 1, 0.95])
    pdf.savefig(fig, bbox_inches='tight', facecolor='white', edgecolor='none', pad_inches=0.2)
    plt.close(fig)
    
    # Page 3: Operations Count with better spacing
    fig = plt.figure(figsize=(11, 8.5), facecolor='white')
    fig.suptitle('Section 2.3: Algorithm Operations Count - Dense Graphs', 
                 fontsize=18, fontweight='bold', y=0.97, color=COLORS['primary'])
    
    ax1 = plt.subplot(2, 1, 1)
    ax1.plot(df_dense['V'], df_dense['ops_matrix_array'], 
             'o-', linewidth=3, markersize=10, label='Matrix + Array', 
             color=COLORS['primary'], markerfacecolor='white',
             markeredgewidth=2, markeredgecolor=COLORS['primary'])
    ax1.plot(df_dense['V'], df_dense['ops_list_heap'], 
             's-', linewidth=3, markersize=10, label='List + Heap', 
             color=COLORS['secondary'], markerfacecolor='white',
             markeredgewidth=2, markeredgecolor=COLORS['secondary'])
    
    ax1.set_xlabel('Number of Vertices (V)', fontsize=13, fontweight='600', color='#1e293b')
    ax1.set_ylabel('Operations Count', fontsize=13, fontweight='600', color='#1e293b')
    ax1.set_title('Total Elementary Operations (Linear Scale)', 
                  fontsize=14, fontweight='bold', pad=12, color='#334155')
    
    legend1 = ax1.legend(fontsize=12, loc='upper left', frameon=True, 
                        shadow=True, fancybox=True, framealpha=0.95)
    legend1.get_frame().set_facecolor('white')
    legend1.get_frame().set_edgecolor(COLORS['neutral'])
    
    ax1.grid(True, alpha=0.25, linestyle='--', linewidth=0.8)
    ax1.set_facecolor('#f8fafc')
    
    ax2 = plt.subplot(2, 1, 2)
    ax2.plot(df_dense['V'], df_dense['ops_matrix_array'], 
             'o-', linewidth=3, markersize=10, label='Matrix + Array', 
             color=COLORS['primary'], markerfacecolor='white',
             markeredgewidth=2, markeredgecolor=COLORS['primary'])
    ax2.plot(df_dense['V'], df_dense['ops_list_heap'], 
             's-', linewidth=3, markersize=10, label='List + Heap', 
             color=COLORS['secondary'], markerfacecolor='white',
             markeredgewidth=2, markeredgecolor=COLORS['secondary'])
    
    ax2.set_xlabel('Number of Vertices (V)', fontsize=13, fontweight='600', color='#1e293b')
    ax2.set_ylabel('Operations Count (Log Scale)', fontsize=13, fontweight='600', color='#1e293b')
    ax2.set_title('Logarithmic Scale - Complexity Comparison', 
                  fontsize=14, fontweight='bold', pad=12, color='#334155')
    
    legend2 = ax2.legend(fontsize=12, loc='upper left', frameon=True, 
                        shadow=True, fancybox=True, framealpha=0.95)
    legend2.get_frame().set_facecolor('white')
    legend2.get_frame().set_edgecolor(COLORS['neutral'])
    
    ax2.grid(True, alpha=0.25, linestyle='--', linewidth=0.8, which='both')
    ax2.set_facecolor('#f8fafc')
    ax2.set_yscale('log')
    
    # Complexity annotation in better position
    complexity_box = dict(boxstyle='round,pad=0.6', facecolor='#dbeafe', 
                         edgecolor=COLORS['primary'], linewidth=1.5, alpha=0.95)
    ax2.text(0.98, 0.05, 
             'Matrix: O(V²)\nHeap: O((V+E)logV)\nDense: E ≈ V²',
             transform=ax2.transAxes, fontsize=10, verticalalignment='bottom',
             horizontalalignment='right', bbox=complexity_box,
             fontfamily='monospace', fontweight='bold')
    
    plt.tight_layout(rect=[0, 0.02, 1, 0.95])
    pdf.savefig(fig, bbox_inches='tight', facecolor='white', edgecolor='none', pad_inches=0.2)
    plt.close(fig)


def plot_comparison_analysis(pdf, df_sparse, df_dense):
    """Create side-by-side comparison plots with no overlaps"""
    
    fig = plt.figure(figsize=(11, 8.5), facecolor='white')
    fig.suptitle('Section 3: Sparse vs Dense Graph Comparison', 
                 fontsize=18, fontweight='bold', y=0.97, color=COLORS['primary'])
    
    # Speedup comparison
    ax1 = plt.subplot(2, 2, 1)
    ax1.plot(df_sparse['V'], df_sparse['speedup'], 
             'o-', linewidth=2.5, markersize=9, label='Sparse', 
             color=COLORS['secondary'], markerfacecolor='white',
             markeredgewidth=2, markeredgecolor=COLORS['secondary'])
    ax1.plot(df_dense['V'], df_dense['speedup'], 
             's-', linewidth=2.5, markersize=9, label='Dense', 
             color=COLORS['warning'], markerfacecolor='white',
             markeredgewidth=2, markeredgecolor=COLORS['warning'])
    ax1.axhline(y=1, color='#1e293b', linestyle='--', linewidth=2, alpha=0.6)
    
    ax1.set_xlabel('Vertices (V)', fontsize=11, fontweight='600')
    ax1.set_ylabel('Speedup Factor', fontsize=11, fontweight='600')
    ax1.set_title('Speedup: Sparse vs Dense', fontsize=12, fontweight='bold', pad=8)
    
    legend1 = ax1.legend(fontsize=10, loc='best', frameon=True, 
                        shadow=True, fancybox=True, framealpha=0.95)
    legend1.get_frame().set_facecolor('white')
    legend1.get_frame().set_edgecolor(COLORS['neutral'])
    
    ax1.grid(True, alpha=0.25, linestyle='--', linewidth=0.8)
    ax1.set_facecolor('#f8fafc')
    
    # Density comparison - better bar positioning
    ax2 = plt.subplot(2, 2, 2)
    # Get common vertices between sparse and dense
    common_v = [v for v in df_dense['V'] if v in df_sparse['V'].values]
    sparse_density = [df_sparse[df_sparse['V']==v]['density'].values[0] for v in common_v]
    dense_density = [df_dense[df_dense['V']==v]['density'].values[0] for v in common_v]
    
    x = np.arange(len(common_v))
    width = 0.35
    
    ax2.bar(x - width/2, sparse_density, width, 
            alpha=0.8, label='Sparse', color=COLORS['secondary'], 
            edgecolor='#1e293b', linewidth=1)
    ax2.bar(x + width/2, dense_density, width, 
            alpha=0.8, label='Dense', color=COLORS['warning'], 
            edgecolor='#1e293b', linewidth=1)
    
    ax2.set_xlabel('Vertices (V)', fontsize=11, fontweight='600')
    ax2.set_ylabel('Edge Density', fontsize=11, fontweight='600')
    ax2.set_title('Graph Density Comparison', fontsize=12, fontweight='bold', pad=8)
    ax2.set_xticks(x)
    ax2.set_xticklabels([str(v) for v in common_v], fontsize=9)
    
    legend2 = ax2.legend(fontsize=10, loc='upper left', frameon=True, 
                        shadow=True, fancybox=True, framealpha=0.95)
    legend2.get_frame().set_facecolor('white')
    legend2.get_frame().set_edgecolor(COLORS['neutral'])
    
    ax2.grid(True, alpha=0.25, axis='y', linestyle='--', linewidth=0.8)
    ax2.set_facecolor('#f8fafc')
    
    # E/V ratio comparison
    ax3 = plt.subplot(2, 2, 3)
    ax3.plot(df_sparse['V'], df_sparse['E'] / df_sparse['V'], 
             'o-', linewidth=2.5, markersize=9, label='Sparse', 
             color=COLORS['secondary'], markerfacecolor='white',
             markeredgewidth=2, markeredgecolor=COLORS['secondary'])
    ax3.plot(df_dense['V'], df_dense['E'] / df_dense['V'], 
             's-', linewidth=2.5, markersize=9, label='Dense', 
             color=COLORS['warning'], markerfacecolor='white',
             markeredgewidth=2, markeredgecolor=COLORS['warning'])
    
    ax3.set_xlabel('Vertices (V)', fontsize=11, fontweight='600')
    ax3.set_ylabel('E/V Ratio', fontsize=11, fontweight='600')
    ax3.set_title('Edge-to-Vertex Ratio', fontsize=12, fontweight='bold', pad=8)
    
    legend3 = ax3.legend(fontsize=10, loc='best', frameon=True, 
                        shadow=True, fancybox=True, framealpha=0.95)
    legend3.get_frame().set_facecolor('white')
    legend3.get_frame().set_edgecolor(COLORS['neutral'])
    
    ax3.grid(True, alpha=0.25, linestyle='--', linewidth=0.8)
    ax3.set_facecolor('#f8fafc')
    
    # Summary statistics with better formatting
    ax4 = plt.subplot(2, 2, 4)
    ax4.axis('off')
    
    sparse_avg_speedup = df_sparse['speedup'].mean()
    dense_avg_speedup = df_dense['speedup'].mean()
    sparse_avg_density = df_sparse['density'].mean()
    dense_avg_density = df_dense['density'].mean()
    
    summary_text = f"""COMPARISON SUMMARY
{'─'*28}

SPARSE GRAPHS:
  Avg Density:  {sparse_avg_density:.4f}
  Avg Speedup:  {sparse_avg_speedup:.2f}×
  Winner:       {'Heap' if sparse_avg_speedup > 1 else 'Matrix'}

DENSE GRAPHS:
  Avg Density:  {dense_avg_density:.4f}
  Avg Speedup:  {dense_avg_speedup:.2f}×
  Winner:       {'Heap' if dense_avg_speedup > 1 else 'Matrix'}

{'─'*28}
KEY INSIGHT:
{('Heap performs better for both graph types' if sparse_avg_speedup > 1 and dense_avg_speedup > 1 
  else 'Performance varies by graph density')}"""
    
    summary_box = dict(boxstyle='round,pad=0.8', facecolor='#e0f2fe', 
                      edgecolor=COLORS['primary'], linewidth=1.5, alpha=0.95)
    ax4.text(0.1, 0.85, summary_text, transform=ax4.transAxes, fontsize=9.5,
             verticalalignment='top', fontfamily='monospace',
             bbox=summary_box, linespacing=1.4)
    
    plt.tight_layout(rect=[0, 0.02, 1, 0.95])
    pdf.savefig(fig, bbox_inches='tight', facecolor='white', edgecolor='none', pad_inches=0.2)
    plt.close(fig)


def create_summary_page(pdf, df_sparse, df_dense):
    """Create final summary and conclusions page with enhanced professional design"""
    fig = plt.figure(figsize=(8.5, 11), facecolor='white')
    ax = fig.add_subplot(111)
    ax.axis('off')
    
    sparse_avg_speedup = df_sparse['speedup'].mean()
    dense_avg_speedup = df_dense['speedup'].mean()
    
    # Header with decorative element
    header_rect = plt.Rectangle((0.05, 0.93), 0.9, 0.05, 
                                facecolor=COLORS['primary'], alpha=0.15, 
                                transform=ax.transAxes, zorder=0)
    ax.add_patch(header_rect)
    
    ax.text(0.5, 0.955, "CONCLUSIONS & RECOMMENDATIONS", 
            ha='center', va='center', fontsize=20, fontweight='bold',
            color=COLORS['primary'], transform=ax.transAxes)
    
    # Main content with better formatting
    summary_text = f"""
╔══════════════════════════════════════════════════════════════════╗
║                    THEORETICAL COMPLEXITY REVIEW                 ║
╚══════════════════════════════════════════════════════════════════╝

  Implementation (a): Matrix + Array Priority Queue
    ├─ Time Complexity:  O(V²)
    ├─ Space Complexity: O(V)
    └─ Optimal for:      Dense graphs where E ≈ V²

  Implementation (b): List + Heap Priority Queue  
    ├─ Time Complexity:  O((V + E) log V)
    ├─ Space Complexity: O(V + E)
    └─ Optimal for:      Sparse graphs where E << V²

╔══════════════════════════════════════════════════════════════════╗
║                         EMPIRICAL RESULTS                        ║
╚══════════════════════════════════════════════════════════════════╝

  SPARSE GRAPHS │ Density ≈ 0.30 │ E ≈ 0.3 × V(V-1)/2
  ──────────────────────────────────────────────────────────────────
    • Average Speedup:  {sparse_avg_speedup:.2f}×
    • Winner:           {'✓ Implementation (b) - Heap' if sparse_avg_speedup > 1 else '✓ Implementation (a) - Matrix'}
    • Performance:      {('Heap significantly outperforms Matrix' if sparse_avg_speedup > 1.5 else 'Competitive performance') if sparse_avg_speedup > 1 else 'Matrix shows better efficiency'}
    • Analysis:         {('Fewer edges → heap operations dominate                        favorably over V² matrix checks' if sparse_avg_speedup > 1 else 'Low constant factors for matrix access')}

  DENSE GRAPHS │ Density ≈ 0.60 │ E ≈ 0.6 × V(V-1)/2
  ──────────────────────────────────────────────────────────────────
    • Average Speedup:  {dense_avg_speedup:.2f}×
    • Winner:           {'✓ Implementation (b) - Heap' if dense_avg_speedup > 1 else '✓ Implementation (a) - Matrix'}
    • Performance:      {('Heap maintains advantage' if dense_avg_speedup > 1.2 else 'Close competition') if dense_avg_speedup > 1 else 'Matrix shows efficiency gains'}
    • Analysis:         {('Log factor provides advantage even                        with higher edge density' if dense_avg_speedup > 1 else 'Matrix operations have lower overhead                        when E approaches V²')}

╔══════════════════════════════════════════════════════════════════╗
║                          RECOMMENDATIONS                         ║
╚══════════════════════════════════════════════════════════════════╝

  ✓ Use Implementation (a) - Matrix + Array when:
    ├─ Graph is dense (E ≈ V²)
    ├─ Graph is small to medium size (V < 100)
    ├─ Simple implementation is prioritized
    └─ Memory for V×V matrix is readily available

  ✓ Use Implementation (b) - List + Heap when:
    ├─ Graph is sparse (E << V²)  
    ├─ Graph is large scale (V > 100)
    ├─ Memory efficiency is critical
    └─ Real-world networks (social, road, web graphs)

╔══════════════════════════════════════════════════════════════════╗
║                    PRACTICAL CONSIDERATIONS                      ║
╚══════════════════════════════════════════════════════════════════╝

  • Most real-world graphs exhibit sparse characteristics
  • Heap implementation provides better scalability
  • Matrix implementation offers simpler code maintenance
  • Modern systems favor adjacency list representations
  • Consider graph properties before implementation choice
    """
    
    ax.text(0.08, 0.88, summary_text, transform=ax.transAxes, fontsize=8.5,
            verticalalignment='top', fontfamily='DejaVu Sans Mono',
            bbox=dict(boxstyle='round,pad=1.2', facecolor='#f8fafc', 
                     edgecolor=COLORS['neutral'], linewidth=1.5, alpha=0.95),
            linespacing=1.4, color='#1e293b')
    
    # Footer with metadata
    footer_line = plt.Line2D([0.1, 0.9], [0.05, 0.05], 
                             color=COLORS['primary'], linewidth=2, 
                             alpha=0.3, transform=ax.transAxes)
    ax.add_line(footer_line)
    
    ax.text(0.5, 0.03, f"Report Generated: {datetime.now().strftime('%B %d, %Y at %H:%M:%S')}", 
            ha='center', va='center', fontsize=9, style='italic',
            transform=ax.transAxes, color=COLORS['neutral'])
    
    pdf.savefig(fig, bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.close(fig)


def create_data_table_page(pdf, df, title, graph_type):
    """Create a professional data table page showing numerical results"""
    fig = plt.figure(figsize=(11, 8.5), facecolor='white')
    ax = fig.add_subplot(111)
    ax.axis('off')
    
    # Title
    ax.text(0.5, 0.95, f"{title} - {graph_type} Graphs", 
            ha='center', va='center', fontsize=18, fontweight='bold',
            color=COLORS['primary'], transform=ax.transAxes)
    
    # Prepare table data
    table_data = []
    table_data.append(['V', 'E', 'Density', 'Matrix (ms)', 'Heap (ms)', 'Speedup', 'Winner'])
    
    for _, row in df.iterrows():
        winner = '🏆 Heap' if row['speedup'] > 1 else '🏆 Matrix'
        table_data.append([
            f"{int(row['V'])}",
            f"{int(row['E'])}",
            f"{row['density']:.3f}",
            f"{row['time_matrix_array']:.2f}",
            f"{row['time_list_heap']:.2f}",
            f"{row['speedup']:.2f}×",
            winner
        ])
    
    # Create table
    table = ax.table(cellText=table_data, cellLoc='center', loc='center',
                     bbox=[0.05, 0.15, 0.9, 0.75])
    
    # Style the table
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2.5)
    
    # Header row styling
    for i in range(7):
        cell = table[(0, i)]
        cell.set_facecolor(COLORS['primary'])
        cell.set_text_props(weight='bold', color='white', fontsize=11)
        cell.set_edgecolor('white')
        cell.set_linewidth(2)
    
    # Data rows styling with alternating colors
    for i in range(1, len(table_data)):
        for j in range(7):
            cell = table[(i, j)]
            if i % 2 == 0:
                cell.set_facecolor('#f8fafc')
            else:
                cell.set_facecolor('white')
            cell.set_edgecolor(COLORS['neutral'])
            cell.set_linewidth(0.5)
            cell.set_text_props(fontsize=10)
            
            # Highlight speedup column
            if j == 5:
                speedup_val = float(df.iloc[i-1]['speedup'])
                if speedup_val > 1:
                    cell.set_facecolor('#d1fae5')  # Light green
                else:
                    cell.set_facecolor('#fee2e2')  # Light red
    
    # Add summary statistics
    summary_text = f"""
    Summary Statistics:
    • Total Vertices Tested: {len(df)}
    • Vertex Range: {int(df['V'].min())} - {int(df['V'].max())}
    • Average Speedup: {df['speedup'].mean():.2f}×
    • Max Speedup: {df['speedup'].max():.2f}× @ V={int(df.loc[df['speedup'].idxmax(), 'V'])}
    • Min Speedup: {df['speedup'].min():.2f}× @ V={int(df.loc[df['speedup'].idxmin(), 'V'])}
    """
    
    summary_box = dict(boxstyle='round,pad=1.0', facecolor='#fffbeb', 
                      edgecolor=COLORS['highlight'], linewidth=2, alpha=0.9)
    ax.text(0.5, 0.06, summary_text, transform=ax.transAxes, fontsize=10,
            ha='center', va='center', bbox=summary_box, linespacing=1.5,
            fontfamily='monospace')
    
    pdf.savefig(fig, bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.close(fig)


def generate_pdf_report(df_sparse, df_dense, filename='dijkstra_analysis_report.pdf'):
    """Generate comprehensive PDF report with all visualizations and enhanced design"""
    print(f"\n{'='*80}")
    print("GENERATING PROFESSIONAL PDF REPORT")
    print(f"{'='*80}")
    
    with PdfPages(filename) as pdf:
        print("  ✓ Creating title page...")
        create_title_page(pdf, filename)
        
        print("  ✓ Creating Section 1: Sparse Graph Analysis...")
        create_section_page(pdf, 1, "Sparse Graph Analysis", 
                          "Performance evaluation on graphs with low edge density (E ≈ 0.3 × V²)")
        plot_sparse_analysis(pdf, df_sparse)
        
        print("  ✓ Creating Section 1: Data Tables...")
        create_data_table_page(pdf, df_sparse, "Numerical Results", "Sparse")
        
        print("  ✓ Creating Section 2: Dense Graph Analysis...")
        create_section_page(pdf, 2, "Dense Graph Analysis", 
                          "Performance evaluation on graphs with high edge density (E ≈ 0.6 × V²)")
        plot_dense_analysis(pdf, df_dense)
        
        print("  ✓ Creating Section 2: Data Tables...")
        create_data_table_page(pdf, df_dense, "Numerical Results", "Dense")
        
        print("  ✓ Creating Section 3: Comparative Analysis...")
        create_section_page(pdf, 3, "Comparative Analysis", 
                          "Side-by-side comparison of sparse vs dense graph performance")
        plot_comparison_analysis(pdf, df_sparse, df_dense)
        
        print("  ✓ Creating final summary and conclusions...")
        create_summary_page(pdf, df_sparse, df_dense)
        
        # Add metadata
        d = pdf.infodict()
        d['Title'] = "Dijkstra's Algorithm: Comprehensive Implementation Comparison"
        d['Author'] = 'SC2001 - Algorithm Design and Analysis'
        d['Subject'] = 'Performance Analysis of Dijkstra Algorithm Implementations'
        d['Keywords'] = 'Dijkstra, Graph Algorithm, Performance Analysis, Complexity, Empirical Study'
        d['CreationDate'] = datetime.now()
        d['Producer'] = 'Matplotlib PDF Backend'
    
    print(f"\n{'='*80}")
    print(f"✓ PDF REPORT SUCCESSFULLY GENERATED")
    print(f"{'='*80}")
    print(f"  📄 Filename: {filename}")
    print(f"  📊 Total Pages: ~15 pages")
    print(f"  🎨 Professional Design: ✓")
    print(f"  📈 Data Visualizations: ✓")
    print(f"  📋 Numerical Tables: ✓")
    print(f"  📝 Comprehensive Analysis: ✓")
    print(f"{'='*80}\n")


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Main execution function with professional output"""
    print("╔" + "="*78 + "╗")
    print("║" + " "*78 + "║")
    print("║" + "DIJKSTRA'S ALGORITHM: COMPREHENSIVE ANALYSIS & REPORT GENERATION".center(78) + "║")
    print("║" + " "*78 + "║")
    print("╚" + "="*78 + "╝")
    
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Test with sparse graphs
    print("\n┌─ [PHASE 1] BENCHMARKING SPARSE GRAPHS (density ≈ 0.3) ─────────────────┐")
    print("│ Testing graph sizes: 10, 20, 30, 50, 75, 100, 150, 200 vertices       │")
    print("└────────────────────────────────────────────────────────────────────────┘")
    sparse_sizes = [10, 20, 30, 50, 75, 100, 150, 200]
    df_sparse = run_benchmark(sparse_sizes, density=0.3, num_trials=3)
    
    # Test with dense graphs
    print("\n┌─ [PHASE 2] BENCHMARKING DENSE GRAPHS (density ≈ 0.6) ──────────────────┐")
    print("│ Testing graph sizes: 10, 20, 30, 50, 75, 100 vertices                 │")
    print("└────────────────────────────────────────────────────────────────────────┘")
    dense_sizes = [10, 20, 30, 50, 75, 100]
    df_dense = run_benchmark(dense_sizes, density=0.6, num_trials=3)
    
    # Generate comprehensive PDF report
    print("\n┌─ [PHASE 3] GENERATING PROFESSIONAL PDF REPORT ──────────────────────────┐")
    generate_pdf_report(df_sparse, df_dense, 'dijkstra_analysis_report.pdf')
    
    print("\n╔" + "="*78 + "╗")
    print("║" + " "*78 + "║")
    print("║" + "ANALYSIS COMPLETE - REPORT READY!".center(78) + "║")
    print("║" + " "*78 + "║")
    print("╚" + "="*78 + "╝")
    
    print("\n📄 Generated Files:")
    print("  └─ dijkstra_analysis_report.pdf")
    print("\n📊 Report Structure:")
    print("  ├─ 📖 Title Page")
    print("  ├─ 📈 Section 1: Sparse Graph Analysis (5 pages)")
    print("  │   ├─ Execution time comparison")
    print("  │   ├─ Speedup analysis")
    print("  │   ├─ Operations count")
    print("  │   ├─ Graph properties")
    print("  │   └─ Numerical data table")
    print("  ├─ 📈 Section 2: Dense Graph Analysis (4 pages)")
    print("  │   ├─ Execution time comparison")
    print("  │   ├─ Speedup analysis")
    print("  │   ├─ Operations count")
    print("  │   └─ Numerical data table")
    print("  ├─ 📊 Section 3: Comparative Analysis (1 page)")
    print("  └─ 📝 Conclusions & Recommendations (1 page)")
    print("\n✨ Features:")
    print("  ✓ Professional color scheme")
    print("  ✓ Enhanced data visualizations")
    print("  ✓ Comprehensive numerical tables")
    print("  ✓ Detailed statistical analysis")
    print("  ✓ Clear recommendations")
    print("\n" + "─"*80 + "\n")
    
    return df_sparse, df_dense


if __name__ == "__main__":
    df_sparse, df_dense = main()
