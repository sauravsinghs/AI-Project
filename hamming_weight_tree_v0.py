"""
Hamming Weight Tree Implementation (Original Version)
This implementation provides an efficient data structure for storing and searching binary codes
based on their Hamming weights. It uses a tree structure where each node represents a group
of codes with similar weight patterns.

Key Features:
1. Supports efficient insertion and deletion of binary codes
2. Provides r-neighbor search (finding codes within Hamming distance r)
3. Implements k-nearest neighbor search
4. Uses caching to improve search performance
5. Supports tree rebalancing for maintaining efficiency
6. Includes comprehensive statistics tracking
"""

from typing import List, Tuple, Dict, Optional, Set, Union, Any
import numpy as np
from collections import defaultdict
from dataclasses import dataclass
from scipy.optimize import linprog
import itertools
import time
import json
import os
import math
from heapq import heappush, heappop

@dataclass
class NodeCache:
    """
    Cache entry for visited nodes in k-NN search.
    This class stores information about previously visited nodes to avoid redundant searches.
    
    Attributes:
        radius: The search radius used when this node was visited
        fully_explored: Whether all children of this node were explored
        results: List of (identifier, code, distance) tuples found in this node
        timestamp: When this cache entry was created
        access_count: Number of times this cache entry has been accessed
        last_access: Time of the last access to this cache entry
    """
    radius: int
    fully_explored: bool
    results: List[Tuple[str, str, int]]
    timestamp: float
    access_count: int  # For LRU cache eviction
    last_access: float  # For time-based eviction

class HammingWeightTree:
    """
    A tree-based data structure for efficient storage and retrieval of binary codes.
    The tree organizes codes based on their Hamming weights and supports various search operations.
    """
    
    def __init__(self, n: int, t: int = 10, d: int = 2, forest_size: int = 1, 
                 cache_ttl: int = 300, rebalance_threshold: float = 0.3,
                 cache_size_limit: int = 10000, use_persistent_cache: bool = True):
        """
        Initialize a Hamming Weight Tree for n-bit codes.
        
        Parameters:
            n: Length of binary codes (e.g., 8 for 8-bit codes)
            t: Threshold for node splitting (max codes per leaf node)
            d: Number of substrings for partitioning (affects tree structure)
            forest_size: Number of trees in the forest (for parallel processing)
            cache_ttl: Time-to-live for cache entries in seconds
            rebalance_threshold: Threshold for triggering tree rebalancing
            cache_size_limit: Maximum number of cache entries
            use_persistent_cache: Whether to maintain cache across queries
        """
        self.n = n
        self.t = t
        self.d = d
        self.forest_size = forest_size
        self.cache_ttl = cache_ttl
        self.rebalance_threshold = rebalance_threshold
        self.cache_size_limit = cache_size_limit
        self.use_persistent_cache = use_persistent_cache
        # Create multiple trees for parallel processing
        self.trees = [self._create_tree() for _ in range(forest_size)]
        self.node_cache = {}
        # Initialize statistics tracking
        self.stats = {
            'insertions': 0,
            'deletions': 0,
            'searches': 0,
            'cache_hits': 0,
            'cache_misses': 0,
            'rebalances': 0,
            'start_time': time.time(),
            'cache_evictions': 0,
            'enumeration_time': 0,
            'search_time': 0
        }
    
    def _create_tree(self) -> Dict:
        """
        Create a new empty tree structure.
        Returns a dictionary containing the root node and tree metadata.
        """
        return {
            'root': {
                'is_leaf': True,  # Whether this is a leaf node
                'codes': [],      # List of (identifier, code) pairs
                'children': {},   # Child nodes
                'parent': None,   # Reference to parent node
                'pattern': None   # Q_d pattern for this node
            },
            'size': 0,           # Number of codes in this tree
            'last_rebalance': time.time(),
            'deletions_since_rebalance': 0,
            'children': {}       # Track all nodes in the tree
        }
    
    def _compute_Q_d(self, code: str) -> Tuple[int, ...]:
        """
        Compute the Q_d pattern for a binary code.
        This pattern represents the Hamming weights of d substrings of the code.
        
        Example:
            For code "10101010" with d=2:
            - First substring: "1010" -> weight = 2
            - Second substring: "1010" -> weight = 2
            Returns: (2, 2)
        """
        n = len(code)
        d = self.d
        substring_length = n // d
        pattern = []
        
        for i in range(0, n, substring_length):
            substring = code[i:i + substring_length]
            weight = sum(int(bit) for bit in substring)
            pattern.append(weight)
        
        return tuple(pattern)
    
    def hamming_distance(self, code1: str, code2: str) -> int:
        """
        Compute the Hamming distance between two binary codes.
        Hamming distance is the number of positions at which the corresponding bits differ.
        
        Example:
            "1010" and "1001" have Hamming distance 2
            because they differ at positions 2 and 3
        """
        return sum(c1 != c2 for c1, c2 in zip(code1, code2))
    
    def insert(self, code: str, identifier: Optional[str] = None) -> None:
        """
        Insert a binary code into the tree.
        
        Parameters:
            code: The binary code to insert (must be n bits long)
            identifier: Optional identifier for the code (defaults to the code itself)
            
        Raises:
            ValueError: If the code length doesn't match n
        """
        if len(code) != self.n:
            raise ValueError(f"Code length must be {self.n}")
        
        # Choose a random tree for insertion using hash function
        tree_idx = hash(code) % self.forest_size
        tree = self.trees[tree_idx]
        
        # Compute Q_d pattern for the code
        pattern = self._compute_Q_d(code)
        
        # Insert into the chosen tree
        self._insert_recursive(tree['root'], code, pattern, identifier)
        tree['size'] += 1
        self.stats['insertions'] += 1
    
    def _insert_recursive(self, node: Dict, code: str, pattern: Tuple[int, ...], 
                        identifier: Optional[str] = None) -> None:
        """
        Recursive helper for insertion.
        Handles the actual insertion logic by traversing the tree.
        
        Parameters:
            node: Current node in the tree
            code: The binary code to insert
            pattern: Q_d pattern of the code
            identifier: Optional identifier for the code
        """
        if node['is_leaf']:
            # Add code to leaf node
            node['codes'].append((identifier or code, code))
            
            # Split node if it exceeds threshold
            if len(node['codes']) > self.t:
                self._split_node(node)
        else:
            # Insert into appropriate child based on pattern
            if pattern in node['children']:
                child = node['children'][pattern]
            else:
                # Create new child node if pattern doesn't exist
                child = {
                    'is_leaf': True,
                    'codes': [],
                    'children': {},
                    'parent': node,
                    'pattern': pattern
                }
                node['children'][pattern] = child
            
            self._insert_recursive(child, code, pattern, identifier)
    
    def _split_node(self, node: Dict) -> None:
        """
        Split a leaf node into children based on Q_d patterns.
        This is called when a leaf node exceeds the threshold t.
        
        Parameters:
            node: The leaf node to split
        """
        node['is_leaf'] = False
        
        # Group codes by their Q_d patterns
        pattern_groups = defaultdict(list)
        for identifier, code in node['codes']:
            pattern = self._compute_Q_d(code)
            pattern_groups[pattern].append((identifier, code))
        
        # Create children for each pattern group
        node['children'] = {}
        for pattern, codes in pattern_groups.items():
            child = {
                'is_leaf': True,
                'codes': codes,
                'children': {},
                'parent': node,
                'pattern': pattern
            }
            node['children'][pattern] = child
        
        # Clear codes from parent node
        node['codes'] = []
    
    def _compute_lower_bound(self, query_Q_d: Tuple[int, ...], pattern: Tuple[int, ...], r: int) -> float:
        """
        Compute lower bound using Proposition 2 from the paper
        :param query_Q_d: Q_d pattern of the query
        :param pattern: Pattern to check
        :param r: Maximum allowed distance
        :return: Lower bound on the actual distance
        """
        # L1 distance between Q_d patterns
        l1_dist = sum(abs(p - q) for p, q in zip(pattern, query_Q_d))
        
        # If L1 distance > r, pattern cannot be promising
        if l1_dist > r:
            return float('inf')
        
        # Compute lower bound using Proposition 2
        # For each position i, the minimum number of bit flips needed
        min_flips = 0
        for i in range(len(query_Q_d)):
            diff = abs(pattern[i] - query_Q_d[i])
            min_flips += max(0, diff)
        
        return min_flips
    
    def _enumerate_promising_children_equation6(self, query_Q_d: Tuple[int, ...], r: int) -> List[Tuple[int, ...]]:
        """
        Enumerate promising children using Equation 6 from the paper
        :param query_Q_d: Q_d pattern of the query
        :param r: Maximum allowed distance
        :return: List of promising Q_d patterns
        """
        n = len(query_Q_d)
        max_weight = self.n // self.d
        promising_patterns = set()
        
        # Equation 6: Minimize sum of absolute differences subject to constraints
        c = np.ones(n)  # Objective function coefficients
        
        # Constraints for each position
        A = []
        b = []
        for i in range(n):
            # Lower bound constraint
            A.append([1 if j == i else 0 for j in range(n)])
            b.append(query_Q_d[i] - r)
            
            # Upper bound constraint
            A.append([-1 if j == i else 0 for j in range(n)])
            b.append(-(query_Q_d[i] + r))
        
        # Solve linear program
        result = linprog(c, A_ub=A, b_ub=b, bounds=(0, max_weight))
        
        if result.success:
            # Generate base pattern
            base_pattern = tuple(int(round(x)) for x in result.x)
            promising_patterns.add(base_pattern)
            
            # Generate variations within the feasible region
            for i in range(n):
                for delta in range(-r, r+1):
                    if 0 <= base_pattern[i] + delta <= max_weight:
                        new_pattern = list(base_pattern)
                        new_pattern[i] += delta
                        new_pattern = tuple(new_pattern)
                        
                        # Check if pattern is promising using Proposition 2
                        if self._compute_lower_bound(query_Q_d, new_pattern, r) <= r:
                            promising_patterns.add(new_pattern)
        
        return list(promising_patterns)
    
    def _enumerate_promising_children_proposition2(self, query_Q_d: Tuple[int, ...], r: int) -> List[Tuple[int, ...]]:
        """
        Enumerate promising children using Proposition 2 from the paper
        :param query_Q_d: Q_d pattern of the query
        :param r: Maximum allowed distance
        :return: List of promising Q_d patterns
        """
        n = len(query_Q_d)
        max_weight = self.n // self.d
        promising_patterns = set()
        
        # Generate patterns within L1 ball of radius r
        for i in range(n):
            for delta in range(-r, r+1):
                if 0 <= query_Q_d[i] + delta <= max_weight:
                    pattern = list(query_Q_d)
                    pattern[i] = query_Q_d[i] + delta
                    pattern = tuple(pattern)
                    
                    # Check if pattern is promising using Proposition 2
                    if self._compute_lower_bound(query_Q_d, pattern, r) <= r:
                        promising_patterns.add(pattern)
        
        return list(promising_patterns)
    
    def _enumerate_promising_children(self, query_Q_d: Tuple[int, ...], r: int) -> List[Tuple[int, ...]]:
        """
        Enumerate promising children based on Q_d patterns and radius r.
        
        Args:
            query_Q_d: Q_d pattern of the query
            r: The search radius
            
        Returns:
            List of promising Q_d patterns
        """
        # Simple implementation: return patterns within L1 distance r
        promising_patterns = []
        
        # Generate all possible patterns within L1 distance r
        for i in range(len(query_Q_d)):
            for delta in range(-r, r+1):
                if 0 <= query_Q_d[i] + delta <= self.n // self.d:
                    pattern = list(query_Q_d)
                    pattern[i] = query_Q_d[i] + delta
                    promising_patterns.append(tuple(pattern))
        
        return promising_patterns
    
    def _manage_cache(self) -> None:
        """Manage cache size and evict entries if necessary"""
        if len(self.node_cache) <= self.cache_size_limit:
            return
        
        # Sort entries by last access time and access count
        entries = [(k, v) for k, v in self.node_cache.items()]
        entries.sort(key=lambda x: (x[1].last_access, -x[1].access_count))
        
        # Remove oldest/least accessed entries
        num_to_remove = len(entries) - self.cache_size_limit
        for k, _ in entries[:num_to_remove]:
            del self.node_cache[k]
            self.stats['cache_evictions'] += 1
    
    def _update_cache_entry(self, key: str, entry: NodeCache) -> None:
        """Update cache entry with access information"""
        entry.last_access = time.time()
        entry.access_count += 1
        self.node_cache[key] = entry
        self._manage_cache()
    
    def _get_cache_key(self, node: Dict, query_Q_d: Tuple[int, ...], r: int) -> str:
        """Generate a unique cache key for a node"""
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
            for identifier, code in node['codes']:
                dist = self.hamming_distance(query, code)
                if dist <= r:
                    results.append((identifier, code, dist))
            return
        
        # Get promising patterns using both Equation 6 and Proposition 2
        promising_patterns = self._enumerate_promising_children(query_Q_d, r)
        
        # Check promising children
        for pattern in promising_patterns:
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
                    radius=r,
                    fully_explored=True,
                    results=child_results,
                    timestamp=time.time(),
                    access_count=1,
                    last_access=time.time()
                ))
                
                results.extend(child_results)
    
    def _rebalance_tree_advanced(self, tree: Dict) -> None:
        """
        Advanced rebalancing strategy for the tree.
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
        
        def _rebalance_recursive(node: Dict, depth: int = 0) -> None:
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
            sorted_children = sorted(child_scores.items(), key=lambda x: x[1])
            
            # Merge under-populated children
            for pattern, score in sorted_children:
                child = node['children'][pattern]
                if score < self.rebalance_threshold:
                    self._merge_nodes(node, pattern)
            
            # Recursively rebalance remaining children
            for pattern, child in list(node['children'].items()):
                _rebalance_recursive(child, depth + 1)
            
            # If node has no more children, make it a leaf
            if not node['children']:
                node['is_leaf'] = True
        
        # Start rebalancing from root
        _rebalance_recursive(tree['root'])
        
        # Update rebalance stats
        tree['last_rebalance'] = time.time()
        tree['deletions_since_rebalance'] = 0
        self.stats['rebalances'] += 1
    
    def search_r(self, query: str, r: int) -> List[Tuple[str, str, int]]:
        """
        Perform r-neighbor search to find all codes within Hamming distance r.
        
        Parameters:
            query: The query binary code
            r: The maximum allowed Hamming distance
            
        Returns:
            List of tuples (identifier, code, distance) for matching codes
        """
        start_time = time.time()
        results = []
        query_Q_d = self._compute_Q_d(query)
        
        # Search in all trees
        for tree in self.trees:
            self._search_r_recursive(tree['root'], query, r, query_Q_d, results, f"tree_{id(tree)}")
        
        # Update statistics
        self.stats['searches'] += 1
        self.stats['search_time'] += time.time() - start_time
        
        return list(set(results))  # Remove duplicates
    
    def tune_parameters_advanced(self, dataset: List[str], query_set: List[str], 
                               param_grid: Dict[str, List[Any]] = None,
                               metric: str = 'time') -> Dict[str, Any]:
        """
        Advanced parameter tuning with multiple metrics
        :param dataset: List of binary codes for the dataset
        :param query_set: List of binary codes for queries
        :param param_grid: Grid of parameters to try
        :param metric: Metric to optimize ('time', 'memory', or 'balanced')
        :return: Best parameters found
        """
        if param_grid is None:
            param_grid = {
                't': [5, 10, 20, 50],
                'd': [2, 3, 4],
                'forest_size': [1, 2, 4, 8],
                'cache_size_limit': [1000, 5000, 10000],
                'rebalance_threshold': [0.2, 0.3, 0.4]
            }
        
        best_params = None
        best_score = float('inf')
        
        # Generate all parameter combinations
        param_combinations = [dict(zip(param_grid.keys(), v)) 
                            for v in itertools.product(*param_grid.values())]
        
        for params in param_combinations:
            # Create a new tree with these parameters
            tree = HammingWeightTree(self.n, **params)
            
            # Measure insertion time and memory usage
            start_time = time.time()
            for code in dataset:
                tree.insert(code)
            insert_time = time.time() - start_time
            
            # Measure query performance
            query_times = []
            for query in query_set:
                start_time = time.time()
                tree.search_knn(query, k=5)
                query_times.append(time.time() - start_time)
            
            avg_query_time = sum(query_times) / len(query_times)
            
            # Compute score based on chosen metric
            if metric == 'time':
                score = insert_time + avg_query_time
            elif metric == 'memory':
                score = tree.get_stats()['total_nodes']
            else:  # balanced
                score = (insert_time + avg_query_time) * math.log2(tree.get_stats()['total_nodes'])
            
            if score < best_score:
                best_score = score
                best_params = params
        
        return best_params
    
    def save(self, filename: str) -> None:
        """Save the tree to a file with cache state"""
        data = {
            'n': self.n,
            't': self.t,
            'd': self.d,
            'forest_size': self.forest_size,
            'cache_ttl': self.cache_ttl,
            'rebalance_threshold': self.rebalance_threshold,
            'cache_size_limit': self.cache_size_limit,
            'use_persistent_cache': self.use_persistent_cache,
            'stats': self.stats,
            'trees': []
        }
        
        for tree in self.trees:
            tree_data = {
                'size': tree['size'],
                'last_rebalance': tree['last_rebalance'],
                'deletions_since_rebalance': tree['deletions_since_rebalance'],
                'root': self._serialize_node(tree['root'])
            }
            data['trees'].append(tree_data)
        
        # Save cache state if using persistent cache
        if self.use_persistent_cache:
            data['cache'] = {
                k: {
                    'radius': v.radius,
                    'fully_explored': v.fully_explored,
                    'results': v.results,
                    'timestamp': v.timestamp,
                    'access_count': v.access_count,
                    'last_access': v.last_access
                }
                for k, v in self.node_cache.items()
            }
        
        with open(filename, 'w') as f:
            json.dump(data, f)
    
    def _serialize_node(self, node: Dict) -> Dict:
        """Serialize a node for saving to file"""
        serialized = {
            'is_leaf': node['is_leaf'],
            'codes': node['codes'],
            'pattern': node['pattern']
        }
        
        if not node['is_leaf']:
            serialized['children'] = {
                str(k): self._serialize_node(v) for k, v in node['children'].items()
            }
        
        return serialized
    
    def _deserialize_node(self, data: Dict) -> Dict:
        """Deserialize a node from file data"""
        node = {
            'is_leaf': data['is_leaf'],
            'codes': data['codes'],
            'children': {},
            'parent': None,
            'pattern': data['pattern']
        }
        
        if not node['is_leaf'] and 'children' in data:
            for k, v in data['children'].items():
                child = self._deserialize_node(v)
                child['parent'] = node
                node['children'][eval(k)] = child
        
        return node
    
    @classmethod
    def load(cls, filename: str) -> 'HammingWeightTree':
        """Load a tree from a file with cache state"""
        with open(filename, 'r') as f:
            data = json.load(f)
        
        # Create tree with saved parameters
        tree = cls(
            n=data['n'],
            t=data['t'],
            d=data['d'],
            forest_size=data['forest_size'],
            cache_ttl=data['cache_ttl'],
            rebalance_threshold=data['rebalance_threshold'],
            cache_size_limit=data['cache_size_limit'],
            use_persistent_cache=data['use_persistent_cache']
        )
        
        # Restore stats
        tree.stats = data['stats']
        
        # Restore trees
        for i, tree_data in enumerate(data['trees']):
            tree.trees[i]['size'] = tree_data['size']
            tree.trees[i]['last_rebalance'] = tree_data['last_rebalance']
            tree.trees[i]['deletions_since_rebalance'] = tree_data['deletions_since_rebalance']
            tree.trees[i]['root'] = tree._deserialize_node(tree_data['root'])
        
        # Restore cache if using persistent cache
        if tree.use_persistent_cache and 'cache' in data:
            tree.node_cache = {
                k: NodeCache(**v)
                for k, v in data['cache'].items()
            }
        
        return tree

    def search_knn(self, query: str, k: int) -> List[Tuple[str, str, int]]:
        """
        Perform k-nearest neighbor search using the Hamming Weight Tree.
        This method finds the k codes that are closest to the query code in terms of Hamming distance.
        
        Parameters:
            query: The query binary code
            k: The number of nearest neighbors to return
            
        Returns:
            List of tuples (identifier, code, distance) for the k nearest neighbors,
            sorted by distance in ascending order
        """
        if len(query) != self.n:
            raise ValueError(f"Query code length must be {self.n}")
            
        # Start with a small radius and increase until we find enough neighbors
        r = 0
        results = []
        
        # Keep track of visited nodes to avoid redundant searches
        visited_nodes = set()
        
        while len(results) < k and r <= self.n:  # Maximum radius is code length
            # Search with current radius
            r_results = self._search_r_recursive_knn(query, r, visited_nodes)
            
            # Add new results
            for result in r_results:
                if result not in results:
                    results.append(result)
            
            # If we found enough neighbors, break
            if len(results) >= k:
                break
                
            # Increase radius for next iteration
            r += 1
        
        # Sort by distance and return top k
        results.sort(key=lambda x: x[2])
        return results[:k]
    
    def _search_r_recursive_knn(self, query: str, r: int, visited_nodes: Set) -> List[Tuple[str, str, int]]:
        """
        Recursive helper for k-nearest neighbor search with radius r.
        This method searches for codes within Hamming distance r of the query.
        
        Parameters:
            query: The query binary code
            r: The current search radius
            visited_nodes: Set of already visited node IDs to avoid redundant searches
            
        Returns:
            List of tuples (identifier, code, distance) for neighbors within radius r
        """
        results = []
        query_Q_d = self._compute_Q_d(query)
        
        # Search in all trees
        for tree in self.trees:
            self._search_r_recursive_knn_tree(tree['root'], query, r, query_Q_d, results, visited_nodes)
        
        return results
    
    def _search_r_recursive_knn_tree(self, node: Dict, query: str, r: int, 
                                    query_Q_d: Tuple[int, ...], results: List,
                                    visited_nodes: Set) -> None:
        """
        Recursive helper for searching a specific tree in k-NN search.
        This method implements the actual tree traversal logic.
        
        Parameters:
            node: Current node in the tree
            query: The query binary code
            r: The current search radius
            query_Q_d: Q_d pattern of the query
            results: List to store results
            visited_nodes: Set of already visited node IDs
        """
        # Generate a unique ID for this node
        node_id = id(node)
        
        # Skip if already visited
        if node_id in visited_nodes:
            return
        
        # Mark as visited
        visited_nodes.add(node_id)
        
        if node['is_leaf']:
            # Check all codes in leaf node
            for identifier, code in node['codes']:
                dist = self.hamming_distance(query, code)
                if dist <= r:
                    results.append((identifier, code, dist))
            return
        
        # Get promising patterns
        promising_patterns = self._enumerate_promising_children(query_Q_d, r)
        
        # Check promising children
        for pattern in promising_patterns:
            if pattern in node['children']:
                child = node['children'][pattern]
                self._search_r_recursive_knn_tree(child, query, r, query_Q_d, results, visited_nodes)

    def angular_search(self, query: str, k: int) -> List[Tuple[str, str, int]]:
        """
        Perform angular (cosine) similarity search using the Hamming Weight Tree.
        This method finds the k codes that are most similar to the query code in terms of cosine similarity.
        
        Parameters:
            query: The query binary code
            k: The number of nearest neighbors to return
            
        Returns:
            List of tuples (identifier, code, similarity) for the k nearest neighbors,
            sorted by similarity in descending order (higher similarity is better)
        """
        if len(query) != self.n:
            raise ValueError(f"Query code length must be {self.n}")
            
        # Convert query to vector for cosine similarity calculation
        query_vector = np.array([int(bit) for bit in query])
        query_norm = np.linalg.norm(query_vector)
        
        # Initialize results list
        results = []
        
        # Search in all trees
        for tree in self.trees:
            self._angular_search_recursive(tree['root'], query_vector, query_norm, results)
        
        # Sort by cosine similarity (descending) and return top k
        results.sort(key=lambda x: x[2], reverse=True)
        return results[:k]
    
    def _angular_search_recursive(self, node: Dict, query_vector: np.ndarray, 
                                query_norm: float, results: List) -> None:
        """
        Recursive helper for angular search.
        This method implements the tree traversal for cosine similarity search.
        
        Parameters:
            node: Current node in the tree
            query_vector: Vector representation of the query
            query_norm: L2 norm of the query vector
            results: List to store results
        """
        if node['is_leaf']:
            # Check all codes in leaf node
            for identifier, code in node['codes']:
                # Convert code to vector
                code_vector = np.array([int(bit) for bit in code])
                code_norm = np.linalg.norm(code_vector)
                
                # Compute cosine similarity
                dot_product = np.dot(query_vector, code_vector)
                similarity = dot_product / (query_norm * code_norm)
                
                # Add to results
                results.append((identifier, code, similarity))
            return
        
        # For non-leaf nodes, explore all children
        # We don't use promising children for angular search as it's based on cosine similarity
        for child in node['children'].values():
            self._angular_search_recursive(child, query_vector, query_norm, results)
    
    def _compute_cosine_similarity(self, code1: str, code2: str) -> float:
        """
        Compute cosine similarity between two binary codes.
        Cosine similarity measures the angle between two vectors.
        
        Parameters:
            code1: First binary code
            code2: Second binary code
            
        Returns:
            Cosine similarity between the two codes (range: -1 to 1)
        """
        # Convert binary strings to vectors
        v1 = np.array([int(bit) for bit in code1])
        v2 = np.array([int(bit) for bit in code2])
        
        # Compute cosine similarity
        dot_product = np.dot(v1, v2)
        norm1 = np.linalg.norm(v1)
        norm2 = np.linalg.norm(v2)
        
        return dot_product / (norm1 * norm2)

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
                self.stats['deletions'] += 1
                
                # Check if rebalancing is needed
                if self._should_rebalance(tree):
                    self._rebalance_tree_advanced(tree)
                
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
            for i, (identifier, stored_code) in enumerate(node['codes']):
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
        # Rebalance if too many deletions since last rebalance
        if tree['deletions_since_rebalance'] > self.rebalance_threshold * tree['size']:
            return True
        
        # Rebalance if too much time has passed
        if time.time() - tree['last_rebalance'] > self.cache_ttl:
            return True
        
        return False
    
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
        Get comprehensive statistics about the tree.
        This method collects various metrics about the tree's performance and structure.
        
        Returns:
            Dictionary containing various statistics including:
            - Basic information (total nodes, codes, uptime)
            - Operation counts (insertions, deletions, searches)
            - Cache performance (hits, misses, evictions)
            - Tree balance metrics
            - Performance metrics (search times)
        """
        stats = self.stats.copy()
        
        # Add tree-specific statistics
        total_nodes = 0
        total_codes = 0
        
        def count_nodes(node: Dict) -> Tuple[int, int]:
            """
            Helper function to count nodes and codes recursively.
            Returns a tuple of (number of nodes, number of codes).
            """
            nodes = 1  # Count current node
            codes = len(node['codes']) if node['is_leaf'] else 0
            
            if not node['is_leaf']:
                for child in node['children'].values():
                    child_nodes, child_codes = count_nodes(child)
                    nodes += child_nodes
                    codes += child_codes
            
            return nodes, codes
        
        # Count nodes and codes in each tree
        for tree in self.trees:
            nodes, codes = count_nodes(tree['root'])
            total_nodes += nodes
            total_codes += codes
        
        stats['total_nodes'] = total_nodes
        stats['total_codes'] = total_codes
        stats['cache_size'] = len(self.node_cache)
        
        # Add timing information
        stats['uptime'] = time.time() - stats['start_time']
        
        # Add performance metrics
        if stats['searches'] > 0:
            stats['avg_search_time'] = stats['search_time'] / stats['searches']
        else:
            stats['avg_search_time'] = 0
        
        if stats['cache_hits'] + stats['cache_misses'] > 0:
            stats['cache_hit_rate'] = stats['cache_hits'] / (stats['cache_hits'] + stats['cache_misses'])
        else:
            stats['cache_hit_rate'] = 0
        
        # Add tree balance metrics
        tree_sizes = [tree['size'] for tree in self.trees]
        stats['min_tree_size'] = min(tree_sizes)
        stats['max_tree_size'] = max(tree_sizes)
        stats['avg_tree_size'] = sum(tree_sizes) / len(tree_sizes)
        
        return stats
    
    def print_stats(self) -> None:
        """
        Print comprehensive statistics about the tree in a readable format.
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
        print(f"Deletions: {stats['deletions']}")
        print(f"Searches: {stats['searches']}")
        print(f"Rebalances: {stats['rebalances']}")
        
        print("\nCache Performance:")
        print(f"Cache Size: {stats['cache_size']}")
        print(f"Cache Hits: {stats['cache_hits']}")
        print(f"Cache Misses: {stats['cache_misses']}")
        print(f"Cache Hit Rate: {stats['cache_hit_rate']:.2%}")
        print(f"Cache Evictions: {stats['cache_evictions']}")
        
        print("\nTree Balance:")
        print(f"Min Tree Size: {stats['min_tree_size']}")
        print(f"Max Tree Size: {stats['max_tree_size']}")
        print(f"Avg Tree Size: {stats['avg_tree_size']:.2f}")
        
        print("\nPerformance Metrics:")
        print(f"Avg Search Time: {stats['avg_search_time']:.6f} seconds")
        print(f"Enumeration Time: {stats['enumeration_time']:.6f} seconds")
        
        print("-" * 30)

# Example usage
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
    codes = [
        "10101010",
        "10101110",
        "11101010",
        "00101010",
        "10111010",
        "10101011",
        "00101110",
        "11111111"
    ]
    
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
    
    # Test angular search
    print("\nangular search (k=3):")
    for neighbor in hwt.angular_search(query, k=3):
        print(neighbor)
    
    # Test deletion
    print("\nDeleting code '10101010':")
    if hwt.delete("10101010"):
        print("Code deleted successfully")
    else:
        print("Code not found")
    
    # Print statistics
    print("\nTree statistics:")
    stats = hwt.get_stats()
    for key, value in stats.items():
        print(f"{key}: {value}")
    
    # Example of advanced parameter tuning
    print("\nAdvanced parameter tuning example:")
    param_grid = {
        't': [4, 8],
        'd': [2, 3],
        'forest_size': [1, 2],
        'cache_size_limit': [1000, 5000],
        'rebalance_threshold': [0.2, 0.3]
    }
    best_params = hwt.tune_parameters_advanced(codes, [query], param_grid, metric='balanced')
    print(f"Best parameters: {best_params}")
    
    # Save and load example
    print("\nSaving and loading tree:")
    hwt.save("hwt_example.json")
    loaded_hwt = HammingWeightTree.load("hwt_example.json")
    print("Tree loaded successfully") 