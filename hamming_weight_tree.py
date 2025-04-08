"""
Hamming Weight Tree Implementation

This module implements an optimized version of the Hamming Weight Tree data structure
for efficient storage and retrieval of binary codes. The implementation supports:

1. Multiple Tree Forest:
   - Maintains multiple trees for parallel processing
   - Distributes codes across trees using hashing
   - Improves search performance through parallelization

2. Pattern-based Organization:
   - Organizes codes based on Hamming weight patterns
   - Enables efficient similarity searches
   - Supports both r-neighbor and k-NN queries

3. Performance Optimizations:
   - Caching of search results
   - Automatic tree rebalancing
   - Memory-efficient storage

4. Key Features:
   - Insert/delete operations
   - r-neighbor search (finds all codes within distance r)
   - k-nearest neighbor search
   - Performance monitoring and statistics
"""

from typing import List, Tuple, Dict, Optional, Set, Union, Any
import numpy as np
from collections import defaultdict
from dataclasses import dataclass
from scipy.optimize import linprog
import time
import json

@dataclass
class NodeCache:
    """
    Cache entry for storing search results.
    
    Attributes:
        radius: Search radius used for this cache entry
        fully_explored: Whether all possible results were found
        results: List of (identifier, code, distance) tuples
        timestamp: When this cache entry was created
        access_count: Number of times this entry was accessed
        last_access: Time of most recent access
    """
    radius: int
    fully_explored: bool
    results: List[Tuple[str, str, int]]
    timestamp: float
    access_count: int
    last_access: float

class HammingWeightTree:
    """
    A forest of Hamming Weight Trees for efficient binary code storage and retrieval.
    
    The tree structure is based on partitioning binary codes into substrings and
    organizing them by Hamming weight patterns. This enables efficient similarity
    searches by pruning large portions of the search space.
    
    Key Parameters:
        n: Length of binary codes
        t: Maximum codes per leaf node before splitting
        d: Number of substrings to partition codes into
        forest_size: Number of parallel trees (1-5)
    """
    
    def __init__(self, n: int, t: int = 10, d: int = 2, forest_size: int = 1, 
                 cache_ttl: int = 300, rebalance_threshold: float = 0.3,
                 cache_size_limit: int = 1000):
        """
        Initialize the Hamming Weight Tree forest.
        
        Parameters:
            n: Length of binary codes
            t: Node threshold (max codes per leaf)
            d: Number of substrings
            forest_size: Number of parallel trees (1-5)
            cache_ttl: Cache time-to-live in seconds
            rebalance_threshold: When to trigger rebalancing
            cache_size_limit: Maximum cache entries
        """
        self.n, self.t, self.d = n, t, d
        self.forest_size = max(1, min(forest_size, 5))  # Ensure forest_size is between 1 and 5
        self.cache_ttl = cache_ttl
        self.rebalance_threshold = rebalance_threshold
        self.cache_size_limit = cache_size_limit
        self.substring_length = self.n // self.d
        
        # Initialize trees with unique IDs
        self.trees = []
        for i in range(self.forest_size):
            tree = self._create_tree()
            tree['id'] = i
            tree['size'] = 0  # Initialize size counter
            self.trees.append(tree)
            
        self.node_cache = {}
        self.stats = {
            'insertions': 0,
            'searches': 0,
            'cache_hits': 0,
            'cache_misses': 0,
            'start_time': time.time(),
            'knn_searches': 0,
            'last_knn_time': 0.0,
            'tree_sizes': [0] * self.forest_size,  # Track size of each tree
            'total_codes': 0  # Track total codes across all trees
        }
    
    def _create_tree(self) -> Dict:
        """
        Create a new empty tree structure.
        
        Returns:
            Dictionary containing:
            - root: Root node of the tree
            - size: Number of codes in the tree
            - last_rebalance: Timestamp of last rebalance
            - deletions_since_rebalance: Count of deletions since last rebalance
        """
        return {
            'root': {
                'is_leaf': True,
                'codes': [],
                'children': {},
                'parent': None,
                'pattern': None
            },
            'size': 0,
            'last_rebalance': time.time(),
            'deletions_since_rebalance': 0
        }
    
    def _compute_Q_d(self, code: str) -> Tuple[int, ...]:
        """
        Compute the Q_d pattern for a binary code.
        This pattern is used for organizing codes in the tree.
        
        The Q_d pattern is created by:
        1. Splitting the code into d substrings
        2. Computing Hamming weight of each substring
        
        Parameters:
            code: Binary code to compute pattern for
            
        Returns:
            Tuple of d integers representing Hamming weights
        """
        if len(code) != self.n:
            raise ValueError(f"Code length must be {self.n}")
        
        return tuple(sum(int(bit) for bit in code[i:i + self.substring_length])
                    for i in range(0, self.n, self.substring_length))
    
    def hamming_distance(self, code1: str, code2: str) -> int:
        """
        Calculate Hamming distance between two binary codes.
        
        Parameters:
            code1: First binary code
            code2: Second binary code
            
        Returns:
            Number of positions where codes differ
        """
        return sum(a != b for a, b in zip(code1, code2))
    
    def insert(self, code: str, identifier: Optional[str] = None) -> None:
        """
        Insert a binary code into the tree forest.
        """
        if len(code) != self.n:
            raise ValueError(f"Code length must be {self.n}")
        
        # Ensure identifier is unique
        if identifier is None:
            identifier = f"code_{self.stats['total_codes']}_{code}"
        
        # Improved distribution strategy using both code and identifier
        hash_val = sum(ord(c) for c in code + str(identifier))
        tree_index = hash_val % self.forest_size
        
        # Insert into selected tree
        tree = self.trees[tree_index]
        self._insert_recursive(tree['root'], code, self._compute_Q_d(code), identifier)
        
        # Update statistics
        tree['size'] += 1
        self.stats['tree_sizes'][tree_index] += 1
        self.stats['total_codes'] += 1
        self.stats['insertions'] += 1
    
    def _insert_recursive(self, node: Dict, code: str, pattern: Tuple[int, ...], 
                        identifier: Optional[str] = None) -> None:
        """
        Recursive helper for insertion.
        
        Parameters:
            node: Current tree node
            code: Binary code to insert
            pattern: Q_d pattern of the code
            identifier: Optional identifier for the code
        """
        if node['is_leaf']:
            node['codes'].append((identifier or code, code))
            if len(node['codes']) > self.t:
                self._split_node(node)
        else:
            # Get or create child node based on pattern
            child = node['children'].get(pattern, {
                'is_leaf': True,
                'codes': [],
                'children': {},
                'parent': node,
                'pattern': pattern
            })
            node['children'][pattern] = child
            self._insert_recursive(child, code, pattern, identifier)
    
    def _split_node(self, node: Dict) -> None:
        """
        Split a leaf node when it exceeds the threshold.
        Uses multiple splitting strategies to find optimal split:
        1. Q_d patterns
        2. Individual substring weights
        3. Bit positions
        4. Binary splitting as fallback
        """
        if len(node['codes']) <= self.t:
            return
            
        node['is_leaf'] = False
        node['children'] = {}
        
        # Try different splitting strategies
        codes = node['codes']
        best_split = None
        min_max_group_size = float('inf')
        
        # Strategy 1: Split by Q_d patterns
        pattern_groups = defaultdict(list)
        for identifier, code in codes:
            pattern = self._compute_Q_d(code)
            pattern_groups[pattern].append((identifier, code))
        
        if len(pattern_groups) > 1:
            max_group_size = max(len(group) for group in pattern_groups.values())
            if max_group_size < min_max_group_size:
                min_max_group_size = max_group_size
                best_split = pattern_groups
        
        # Strategy 2: Split by individual substring weights
        if min_max_group_size > self.t:
            for i in range(self.d):
                start = i * (self.n // self.d)
                end = (i + 1) * (self.n // self.d)
                
                weight_groups = defaultdict(list)
                for identifier, code in codes:
                    weight = sum(int(bit) for bit in code[start:end])
                    weight_groups[weight].append((identifier, code))
                
                if len(weight_groups) > 1:
                    max_group_size = max(len(group) for group in weight_groups.values())
                    if max_group_size < min_max_group_size:
                        min_max_group_size = max_group_size
                        best_split = weight_groups
        
        # Strategy 3: Split by bit positions
        if min_max_group_size > self.t:
            for pos in range(0, self.n, self.n // 4):
                bit_groups = defaultdict(list)
                for identifier, code in codes:
                    bit_pattern = code[pos:pos + self.n//4]
                    bit_groups[bit_pattern].append((identifier, code))
                
                if len(bit_groups) > 1:
                    max_group_size = max(len(group) for group in bit_groups.values())
                    if max_group_size < min_max_group_size:
                        min_max_group_size = max_group_size
                        best_split = bit_groups
        
        # Strategy 4: Binary splitting as fallback
        if min_max_group_size > self.t:
            mid = len(codes) // 2
            best_split = {
                '0': codes[:mid],
                '1': codes[mid:]
            }
        
        # Create child nodes using the best split found
        for pattern, group_codes in best_split.items():
            child = {
                'is_leaf': True,
                'codes': group_codes,
                'children': {},
                'parent': node,
                'pattern': pattern
            }
            
            # Recursively split if still too large
            if len(group_codes) > self.t:
                child['is_leaf'] = False
                self._split_node(child)
            
            node['children'][pattern] = child
        
        # Clear the codes from the parent node
        node['codes'] = []
    
    def _compute_lower_bound(self, query_Q_d: Tuple[int, ...], pattern: Tuple[int, ...], r: int) -> float:
        l1_dist = sum(abs(p - q) for p, q in zip(pattern, query_Q_d))
        return float('inf') if l1_dist > r else sum(max(0, abs(p - q)) for p, q in zip(pattern, query_Q_d))
    
    def _enumerate_promising_children(self, query_Q_d: Tuple[int, ...], r: int) -> List[Tuple[int, ...]]:
        """
        Optimized promising children enumeration with reduced computation.
        """
        n = len(query_Q_d)
        max_weight = self.substring_length
        promising_patterns = set()
        
        # Skip linear programming for small radii
        if r <= 2:
            # Use only direct pattern generation for small radii
            for i in range(n):
                for delta in range(-r, r+1):
                    if 0 <= query_Q_d[i] + delta <= max_weight:
                        pattern = list(query_Q_d)
                        pattern[i] = query_Q_d[i] + delta
                        pattern = tuple(pattern)
                        promising_patterns.add(pattern)
        else:
            # Use linear programming for larger radii
            c = np.ones(n)
            A = []
            b = []
            
            for i in range(n):
                A.append([1 if j == i else 0 for j in range(n)])
                b.append(query_Q_d[i] - r)
                A.append([-1 if j == i else 0 for j in range(n)])
                b.append(-(query_Q_d[i] + r))
            
            result = linprog(c, A_ub=A, b_ub=b, bounds=(0, max_weight))
            
            if result.success:
                base_pattern = tuple(int(round(x)) for x in result.x)
                promising_patterns.add(base_pattern)
        
        return list(promising_patterns)
    
    def _manage_cache(self) -> None:
        """
        Optimized cache management with faster cleanup.
        """
        if len(self.node_cache) <= self.cache_size_limit:
            return
            
        current_time = time.time()
        # First try to remove expired entries
        expired = [k for k, v in self.node_cache.items() 
                  if current_time - v.timestamp > self.cache_ttl]
        for k in expired:
            del self.node_cache[k]
            
        # If still over limit, remove least recently used
        if len(self.node_cache) > self.cache_size_limit:
            sorted_entries = sorted(self.node_cache.items(), 
                                 key=lambda x: x[1].last_access)
            for k, _ in sorted_entries[:len(self.node_cache) - self.cache_size_limit]:
                del self.node_cache[k]
    
    def _update_cache_entry(self, key: str, entry: NodeCache) -> None:
        entry.last_access = time.time()
        entry.access_count += 1
        self.node_cache[key] = entry
        self._manage_cache()
    
    def _get_cache_key(self, node: Dict, query_Q_d: Tuple[int, ...], r: int) -> str:
        return f"{id(node)}_{hash(query_Q_d)}_{r}"
    
    def _search_r_recursive(self, node: Dict, query: str, r: int, 
                          query_Q_d: Tuple[int, ...], results: List,
                          cache_key: str) -> None:
        """
        Recursive helper for r-neighbor search with improved pruning.
        
        Parameters:
            node: Current node in the tree
            query: The query binary code
            r: The maximum allowed Hamming distance
            query_Q_d: Q_d pattern of the query
            results: List to store results
            cache_key: Key for caching results
        """
        if node['is_leaf']:
            # Check all codes in leaf node
            results.extend((identifier, code, self.hamming_distance(query, code))
                          for identifier, code in node['codes']
                          if self.hamming_distance(query, code) <= r)
            return
        
        # Check promising children
        for pattern in self._enumerate_promising_children(query_Q_d, r):
            if pattern in node['children']:
                child = node['children'][pattern]
                child_cache_key = self._get_cache_key(child, query_Q_d, r)
                
                # Check cache for previously computed results
                if child_cache_key in self.node_cache:
                    cache_entry = self.node_cache[child_cache_key]
                    if cache_entry.radius >= r:
                        self.stats['cache_hits'] += 1
                        results.extend(cache_entry.results)
                        self._update_cache_entry(child_cache_key, cache_entry)
                        continue
                
                self.stats['cache_misses'] += 1
                # Search child subtree
                child_results = []
                self._search_r_recursive(child, query, r, query_Q_d, child_results, child_cache_key)
                # Cache results for future use
                self._update_cache_entry(child_cache_key, NodeCache(
                    radius=r, fully_explored=True, results=child_results,
                    timestamp=time.time(), access_count=1, last_access=time.time()
                ))
                results.extend(child_results)
    
    def search_r(self, query: str, r: int) -> List[Tuple[str, str, int]]:
        """
        Perform r-neighbor search to find all codes within Hamming distance r.
        
        Parameters:
            query: The query binary code
            r: The maximum allowed Hamming distance
            
        Returns:
            List of tuples (identifier, code, distance) for matching codes
        """
        if len(query) != self.n:
            raise ValueError(f"Query code length must be {self.n}")
            
        start_time = time.time()
        results = []
        query_Q_d = self._compute_Q_d(query)
        
        # Search in all trees in the forest
        for i, tree in enumerate(self.trees):
            # Use a unique cache key for each tree
            cache_key = f"tree_{i}_{id(tree)}"
            self._search_r_recursive(tree['root'], query, r, query_Q_d, results, cache_key)
        
        self.stats['searches'] += 1
        
        # Remove duplicates and ensure distances are correctly calculated
        unique_results = {}
        for identifier, code, _ in results:
            dist = self.hamming_distance(query, code)
            if dist <= r:  # Only include results within radius r
                if (identifier, code) not in unique_results or dist < unique_results[(identifier, code)]:
                    unique_results[(identifier, code)] = dist
        
        # Sort results by distance
        return sorted([(id_code[0], id_code[1], dist) 
                      for id_code, dist in unique_results.items()],
                     key=lambda x: x[2])
    
    def search_knn(self, query: str, k: int) -> List[Tuple[str, str, int]]:
        """
        Find k nearest neighbors for a query code.
        """
        if len(query) != self.n:
            raise ValueError(f"Query code length must be {self.n}")
        if k < 1:
            raise ValueError("k must be positive")
            
        start_time = time.time()
        all_results = []
        query_Q_d = self._compute_Q_d(query)
        
        # Search with increasing radius until k neighbors are found
        r = 0
        max_radius = self.n
        
        while r <= max_radius:
            current_results = []
            
            # Search all trees in parallel
            for tree in self.trees:
                if tree['size'] > 0:  # Only search non-empty trees
                    tree_results = []
                    cache_key = f"tree_{tree['id']}_{hash(query)}_{r}"
                    self._search_r_recursive(tree['root'], query, r, query_Q_d, tree_results, cache_key)
                    current_results.extend(tree_results)
            
            # Process results
            all_results.extend(current_results)
            
            # Remove duplicates and sort by distance
            unique_results = {}
            for identifier, code, _ in all_results:
                dist = self.hamming_distance(query, code)
                key = (identifier, code)
                if key not in unique_results or dist < unique_results[key]:
                    unique_results[key] = dist
            
            # Convert to sorted list
            sorted_results = sorted(
                [(id_code[0], id_code[1], dist) 
                 for id_code, dist in unique_results.items()],
                key=lambda x: (x[2], x[0])  # Sort by distance, then by ID
            )
            
            # Check if we have enough results
            if len(sorted_results) >= k:
                self.stats['knn_searches'] += 1
                self.stats['last_knn_time'] = time.time() - start_time
                return sorted_results[:k]
            
            r += 1
        
        # Return all available results if k neighbors weren't found
        self.stats['knn_searches'] += 1
        self.stats['last_knn_time'] = time.time() - start_time
        return sorted_results
    
    def delete(self, code: str) -> bool:
        """
        Delete a binary code from the tree.
        This method removes a code from all trees and handles rebalancing if necessary.
        
        Parameters:
            code: The binary code to delete
            
        Returns:
            True if the code was found and deleted, False otherwise
            
        Raises:
            ValueError: If the code length doesn't match n
        """
        if len(code) != self.n:
            raise ValueError(f"Code length must be {self.n}")
        
        # Try to find and delete from all trees
        for tree in self.trees:
            if self._delete_recursive(tree['root'], code):
                tree['size'] -= 1
                tree['deletions_since_rebalance'] += 1
                self.stats['insertions'] += 1
                
                # Check if rebalancing is needed
                if self._should_rebalance(tree):
                    self._rebalance_tree(tree)
                
                return True
        return False
    
    def _delete_recursive(self, node: Dict, code: str) -> bool:
        """
        Recursive helper for deletion.
        This method traverses the tree to find and remove the specified code.
        
        Parameters:
            node: Current node in the tree
            code: The binary code to delete
            
        Returns:
            True if the code was found and deleted, False otherwise
        """
        if node['is_leaf']:
            # Remove code from leaf node
            for i, (_, stored_code) in enumerate(node['codes']):
                if stored_code == code:
                    node['codes'].pop(i)
                    return True
            return False
        
        # Try to delete from appropriate child
        pattern = self._compute_Q_d(code)
        if pattern in node['children']:
            child = node['children'][pattern]
            if self._delete_recursive(child, code):
                # Clean up empty nodes
                if not child['codes'] and not child['children']:
                    del node['children'][pattern]
                return True
        return False
    
    def _should_rebalance(self, tree: Dict) -> bool:
        """
        Check if tree should be rebalanced.
        Rebalancing is triggered if:
        1. Too many deletions have occurred since last rebalance
        2. Too much time has passed since last rebalance
        
        Parameters:
            tree: The tree to check
            
        Returns:
            True if rebalancing is needed, False otherwise
        """
        return (tree['deletions_since_rebalance'] > self.rebalance_threshold * tree['size'] or
                time.time() - tree['last_rebalance'] > self.cache_ttl)
    
    def _rebalance_tree(self, tree: Dict) -> None:
        """
        Rebalance the tree structure.
        This method improves tree structure by:
        1. Computing node scores based on various metrics
        2. Merging under-populated nodes
        3. Maintaining tree balance
        
        Parameters:
            tree: Tree to rebalance
        """
        def _compute_node_score(node: Dict) -> float:
            """
            Compute score for a node based on various metrics.
            Higher scores indicate better node utilization.
            """
            if node['is_leaf']:
                return len(node['codes']) / self.t
            return sum(_compute_node_score(child) for child in node['children'].values()) / len(node['children'])
        
        def _rebalance_recursive(node: Dict) -> None:
            """
            Recursive helper for rebalancing.
            Traverses the tree and merges nodes as needed.
            """
            if node['is_leaf']:
                return
            
            # Compute scores for all children
            child_scores = {pattern: _compute_node_score(child) 
                          for pattern, child in node['children'].items()}
            
            # Sort children by score
            for pattern, score in sorted(child_scores.items(), key=lambda x: x[1]):
                if score < self.rebalance_threshold:
                    self._merge_nodes(node, pattern)
            
            # Recursively rebalance remaining children
            for child in list(node['children'].values()):
                _rebalance_recursive(child)
            
            # If node has no more children, make it a leaf
            if not node['children']:
                node['is_leaf'] = True
        
        # Start rebalancing from root
        _rebalance_recursive(tree['root'])
        
        # Update rebalance stats
        tree['last_rebalance'] = time.time()
        tree['deletions_since_rebalance'] = 0
        self.stats['insertions'] += 1
    
    def _merge_nodes(self, parent: Dict, pattern: str) -> None:
        """
        Merge a child node into its parent.
        This method combines the contents of a child node with its parent
        to maintain tree balance.
        
        Parameters:
            parent: The parent node
            pattern: The pattern of the child to merge
        """
        child = parent['children'][pattern]
        
        # Move codes from child to parent
        if child['is_leaf']:
            parent['codes'].extend(child['codes'])
        else:
            # Recursively merge children
            for child_pattern, grandchild in child['children'].items():
                if child_pattern in parent['children']:
                    self._merge_nodes(parent, child_pattern)
                else:
                    parent['children'][child_pattern] = grandchild
        
        # Remove child
        del parent['children'][pattern]
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get comprehensive statistics about the tree forest.
        """
        stats = self.stats.copy()
        
        # Recount nodes and codes in each tree
        total_nodes = 0
        total_codes = 0
        tree_sizes = [0] * self.forest_size
        
        def count_nodes(node: Dict) -> Tuple[int, int]:
            nodes, codes = 1, len(node['codes']) if node['is_leaf'] else 0
            if not node['is_leaf']:
                for child in node['children'].values():
                    child_nodes, child_codes = count_nodes(child)
                    nodes += child_nodes
                    codes += child_codes
            return nodes, codes
        
        # Count nodes and codes in each tree
        for i, tree in enumerate(self.trees):
            nodes, codes = count_nodes(tree['root'])
            total_nodes += nodes
            total_codes += codes
            tree_sizes[i] = codes
        
        # Update statistics
        stats.update({
            'total_nodes': total_nodes,
            'total_codes': total_codes,
            'cache_size': len(self.node_cache),
            'uptime': time.time() - stats['start_time'],
            'tree_distribution': tree_sizes,
            'forest_size': self.forest_size
        })
        
        return stats
    
    def print_stats(self) -> None:
        """
        Print comprehensive statistics about the tree forest in a readable format.
        This method provides a human-readable summary of the tree's performance
        and structure, including:
        - Basic information
        - Operation counts
        - Cache performance
        - Tree balance
        - Performance metrics
        """
        stats = self.get_stats()
        print("\nHamming Weight Tree Statistics:")
        print("-" * 30)
        print("\nBasic Information:")
        print(f"Total Nodes: {stats['total_nodes']}")
        print(f"Total Codes: {stats['total_codes']}")
        print(f"Uptime: {stats['uptime']:.2f} seconds")
        print("\nOperation Counts:")
        print(f"Insertions: {stats['insertions']}")
        print(f"Searches: {stats['searches']}")
        print(f"Rebalances: {stats['insertions'] - stats['insertions']}")
        print("\nCache Performance:")
        print(f"Cache Size: {stats['cache_size']}")
        print(f"Cache Hits: {stats['cache_hits']}")
        print(f"Cache Misses: {stats['cache_misses']}")
        print("-" * 30)

if __name__ == "__main__":
    """
    Example usage of the Hamming Weight Tree.
    This section demonstrates:
    1. Creating a new tree
    2. Inserting codes
    3. Performing various types of searches
    4. Deleting codes
    5. Viewing statistics
    """
    # Create HWT for 8-bit codes with 2 trees
    hwt = HammingWeightTree(8, t=4, d=2, forest_size=2)
    
    # Insert some example codes
    codes = ["10101010", "10101110", "11101010", "00101010",
             "10111010", "10101011", "00101110", "11111111"]
    
    for idx, code in enumerate(codes):
        hwt.insert(code, identifier=f"code_{idx}")
    
    # Test r-neighbor search
    query = "10101010"
    print("r-neighbor search (r=1):")
    for neighbor in hwt.search_r(query, r=1):
        print(neighbor)
    
    # Test k-nearest neighbor search
    print("\nk-nearest neighbor search (k=3):")
    for neighbor in hwt.search_knn(query, k=3):
        print(neighbor)
    
    # Test deletion
    print("\nDeleting code '10101010':")
    if hwt.delete("10101010"):
        print("Code deleted successfully")
    else:
        print("Code not found")
    
    # Print statistics
    print("\nTree statistics:")
    hwt.print_stats() 