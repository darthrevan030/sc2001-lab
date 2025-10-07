"""
Dijkstra's Algorithm: Implementation Comparison
VSCode Python Environment Version

Usage:
    python dijkstra_comparison.py

Requirements:
    pip install numpy matplotlib pandas
"""

import numpy as np
import matplotlib.pyplot as plt
import time
import heapq
from collections import defaultdict
import pandas as pd

# Set pandas display options for better output
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', None)

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


def generate_sparse_graph(V):
    """Generate a sparse graph with E ≈ 3V"""
    return generate_random_graph(V, min(3.0 / V, 0.99))


def generate_dense_graph(V):
    """Generate a dense graph with E ≈ V²/3"""
    return generate_random_graph(V, 0.6)


# ============================================================================
# BENCHMARKING AND VISUALIZATION
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
    
    Note: Generates ONE graph per size, then runs both algorithms num_trials times
          on that same graph to get stable timing measurements.
    """
    results = []
    
    for V in graph_sizes:
        print(f"Benchmarking V={V}...", end=" ", flush=True)
        
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
        
        # Store results (operations count is deterministic, so just use last result)
        results.append({
            'V': V,
            'E': E,
            'density': E / (V * (V - 1) / 2) if V > 1 else 0,
            'time_matrix_array': np.mean(time_matrix_array),
            'time_list_heap': np.mean(time_list_heap),
            'time_matrix_std': np.std(time_matrix_array),
            'time_list_std': np.std(time_list_heap),
            'time_matrix_min': np.min(time_matrix_array),
            'time_list_min': np.min(time_list_heap),
            'ops_matrix_array': result_a['operations'],
            'ops_list_heap': result_b['operations'],
            'speedup': np.mean(time_matrix_array) / np.mean(time_list_heap)
        })
        
        print(f"Done (E={results[-1]['E']}, speedup={results[-1]['speedup']:.2f}x, "
              f"std: Matrix={results[-1]['time_matrix_std']:.3f}ms, Heap={results[-1]['time_list_std']:.3f}ms)")
    
    return pd.DataFrame(results)


def plot_comparison(df_sparse, df_dense=None, save_to_file=False):
    """Create comprehensive visualization of benchmark results"""
    
    fig = plt.figure(figsize=(16, 10))
    
    # Plot 1: Execution Time Comparison (Sparse)
    ax1 = plt.subplot(2, 3, 1)
    ax1.plot(df_sparse['V'], df_sparse['time_matrix_array'], 
             'o-', linewidth=2, markersize=8, label='Matrix + Array', color='#3b82f6')
    ax1.plot(df_sparse['V'], df_sparse['time_list_heap'], 
             's-', linewidth=2, markersize=8, label='List + Heap', color='#10b981')
    ax1.set_xlabel('Number of Vertices (V)', fontsize=11, fontweight='bold')
    ax1.set_ylabel('Time (ms)', fontsize=11, fontweight='bold')
    ax1.set_title('Execution Time - Sparse Graph', fontsize=12, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Speedup Factor (Sparse)
    ax2 = plt.subplot(2, 3, 2)
    ax2.plot(df_sparse['V'], df_sparse['speedup'], 
             'o-', linewidth=2, markersize=8, color='#8b5cf6')
    ax2.axhline(y=1, color='red', linestyle='--', alpha=0.5, label='Equal Performance')
    ax2.set_xlabel('Number of Vertices (V)', fontsize=11, fontweight='bold')
    ax2.set_ylabel('Speedup Factor (Matrix/Heap)', fontsize=11, fontweight='bold')
    ax2.set_title('Heap Speedup vs Matrix - Sparse', fontsize=12, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Operations Count (Sparse)
    ax3 = plt.subplot(2, 3, 3)
    ax3.plot(df_sparse['V'], df_sparse['ops_matrix_array'], 
             'o-', linewidth=2, markersize=8, label='Matrix + Array', color='#3b82f6')
    ax3.plot(df_sparse['V'], df_sparse['ops_list_heap'], 
             's-', linewidth=2, markersize=8, label='List + Heap', color='#10b981')
    ax3.set_xlabel('Number of Vertices (V)', fontsize=11, fontweight='bold')
    ax3.set_ylabel('Operations Count', fontsize=11, fontweight='bold')
    ax3.set_title('Algorithm Operations - Sparse', fontsize=12, fontweight='bold')
    ax3.legend(fontsize=10)
    ax3.grid(True, alpha=0.3)
    ax3.set_yscale('log')
    
    if df_dense is not None:
        # Plot 4: Execution Time (Dense)
        ax4 = plt.subplot(2, 3, 4)
        ax4.plot(df_dense['V'], df_dense['time_matrix_array'], 
                 'o-', linewidth=2, markersize=8, label='Matrix + Array', color='#3b82f6')
        ax4.plot(df_dense['V'], df_dense['time_list_heap'], 
                 's-', linewidth=2, markersize=8, label='List + Heap', color='#10b981')
        ax4.set_xlabel('Number of Vertices (V)', fontsize=11, fontweight='bold')
        ax4.set_ylabel('Time (ms)', fontsize=11, fontweight='bold')
        ax4.set_title('Execution Time - Dense Graph', fontsize=12, fontweight='bold')
        ax4.legend(fontsize=10)
        ax4.grid(True, alpha=0.3)
        
        # Plot 5: Speedup Factor (Dense)
        ax5 = plt.subplot(2, 3, 5)
        ax5.plot(df_dense['V'], df_dense['speedup'], 
                 'o-', linewidth=2, markersize=8, color='#8b5cf6')
        ax5.axhline(y=1, color='red', linestyle='--', alpha=0.5, label='Equal Performance')
        ax5.set_xlabel('Number of Vertices (V)', fontsize=11, fontweight='bold')
        ax5.set_ylabel('Speedup Factor (Matrix/Heap)', fontsize=11, fontweight='bold')
        ax5.set_title('Performance Comparison - Dense', fontsize=12, fontweight='bold')
        ax5.legend(fontsize=10)
        ax5.grid(True, alpha=0.3)
        
        # Plot 6: Edge Density Comparison
        ax6 = plt.subplot(2, 3, 6)
        ax6.bar(df_sparse['V'], df_sparse['density'], 
                alpha=0.6, label='Sparse', color='#10b981', width=8)
        ax6.bar(df_dense['V'], df_dense['density'], 
                alpha=0.6, label='Dense', color='#ef4444', width=8)
        ax6.set_xlabel('Number of Vertices (V)', fontsize=11, fontweight='bold')
        ax6.set_ylabel('Edge Density (E / max_edges)', fontsize=11, fontweight='bold')
        ax6.set_title('Graph Density Comparison', fontsize=12, fontweight='bold')
        ax6.legend(fontsize=10)
        ax6.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    if save_to_file:
        plt.savefig('dijkstra_comparison.png', dpi=300, bbox_inches='tight')
        print(f"\n[INFO] Plot saved to 'dijkstra_comparison.png'")
    
    plt.show()


def print_analysis(df):
    """Print detailed analysis of benchmark results"""
    print("\n" + "="*80)
    print("BENCHMARK RESULTS SUMMARY")
    print("="*80)
    print(df.to_string(index=False))
    print("\n" + "="*80)
    print("KEY INSIGHTS:")
    print("="*80)
    
    avg_speedup = df['speedup'].mean()
    max_speedup = df['speedup'].max()
    min_speedup = df['speedup'].min()
    
    print(f"\n1. Average Speedup (Heap vs Matrix): {avg_speedup:.2f}x")
    print(f"2. Maximum Speedup: {max_speedup:.2f}x at V={df.loc[df['speedup'].idxmax(), 'V']}")
    print(f"3. Minimum Speedup: {min_speedup:.2f}x at V={df.loc[df['speedup'].idxmin(), 'V']}")
    
    if avg_speedup > 1:
        print(f"\n→ List + Heap is faster on average for these graphs")
    else:
        print(f"\n→ Matrix + Array is faster on average for these graphs")
    
    print(f"\n4. Graph Density: {df['density'].mean():.3f} (average)")
    print(f"5. Average E/V ratio: {(df['E'] / df['V']).mean():.2f}")


def verify_correctness():
    """Verify both implementations produce the same results"""
    print("\n" + "="*80)
    print("[4] CORRECTNESS VERIFICATION - Small Example")
    print("="*80)
    
    # Create a simple 5-vertex graph
    V = 5
    adj_matrix = np.array([
        [0, 4, 0, 0, 0],
        [4, 0, 8, 0, 0],
        [0, 8, 0, 7, 9],
        [0, 0, 7, 0, 10],
        [0, 0, 9, 10, 0]
    ])
    
    adj_list = [
        [(1, 4)],
        [(0, 4), (2, 8)],
        [(1, 8), (3, 7), (4, 9)],
        [(2, 7), (4, 10)],
        [(2, 9), (3, 10)]
    ]
    
    dijkstra_a = DijkstraMatrixArray(adj_matrix)
    result_a = dijkstra_a.dijkstra(0)
    
    dijkstra_b = DijkstraListHeap(adj_list, V)
    result_b = dijkstra_b.dijkstra(0)
    
    print(f"\nShortest distances from vertex 0:")
    print(f"  Matrix + Array: {result_a['distances']}")
    print(f"  List + Heap:    {result_b['distances']}")
    print(f"  Results Match:  {result_a['distances'] == result_b['distances']}")
    
    if result_a['distances'] == result_b['distances']:
        print("\n✓ Both implementations produce identical results!")
    else:
        print("\n✗ WARNING: Results differ between implementations!")


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Main execution function"""
    print("="*80)
    print("DIJKSTRA'S ALGORITHM: IMPLEMENTATION COMPARISON")
    print("="*80)
    
    # Test with sparse graphs (typical real-world scenario)
    print("\n[1] TESTING WITH SPARSE GRAPHS (density ≈ 0.3)")
    print("-" * 80)
    sparse_sizes = [10, 20, 30, 50, 75, 100, 150, 200]
    df_sparse = run_benchmark(sparse_sizes, density=0.3, num_trials=3)
    print_analysis(df_sparse)
    
    # Test with dense graphs
    print("\n[2] TESTING WITH DENSE GRAPHS (density ≈ 0.6)")
    print("-" * 80)
    dense_sizes = [10, 20, 30, 50, 75, 100]
    df_dense = run_benchmark(dense_sizes, density=0.6, num_trials=3)
    print_analysis(df_dense)
    
    # Verify correctness
    verify_correctness()
    
    # Create visualizations
    print("\n[3] GENERATING VISUALIZATIONS...")
    print("-" * 80)
    print("Close the plot window to continue...")
    plot_comparison(df_sparse, df_dense, save_to_file=True)
    
    print("\n" + "="*80)
    print("ANALYSIS COMPLETE!")
    print("="*80)
    print("\nResults saved to:")
    print("  - dijkstra_comparison.png (visualization)")
    print("\nDataFrames available as: df_sparse, df_dense")
    
    return df_sparse, df_dense


if __name__ == "__main__":
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Run main analysis
    df_sparse, df_dense = main()
    
    # Keep the script running to display plots
    # (plots will show until you close them)
    print("\n[INFO] Close all plot windows to exit.")
