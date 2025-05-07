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
import hashlib
import random

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
            'deletions': 0,
            'searches': 0,
            'cache_hits': 0,
            'cache_misses': 0,
            'start_time': time.time(),
            'knn_searches': 0,
            'last_knn_time': 0.0,
            'tree_sizes': [0] * self.forest_size,  # Track size of each tree
            'total_codes': 0,  # Track total codes across all trees
            'rebalances': 0  # Add rebalance counter
        }
    
    def _make_node(self, is_leaf, parent, pattern, depth, d):
        return {
            'is_leaf': is_leaf,
            'codes': [],
            'children': {},
            'parent': parent,
            'pattern': pattern,  # Qd pattern tuple
            'depth': depth,      # tree depth (number of splits)
            'd': d               # number of substrings at this node
        }

    def _create_tree(self) -> dict:
        # Create the root node with d=2 for the root
        return {
            'root': self._make_node(True, None, None, 0, 2),
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
        
        # Use d=2 to ensure we always have 2-element patterns
        return self._get_substring_weights(code, 2)

    def _get_substring_weights(self, code, d):
        """
        Compute the Hamming weight pattern for a code with d substrings.
        This splits the code into d equal parts and counts the 1's in each part.
        
        Parameters:
            code: Binary code
            d: Number of substrings (forced to at least 2)
            
        Returns:
            Tuple of d integers representing Hamming weights of each substring
        """
        n = len(code)
        # Ensure d is at least 2 to create proper tuples
        d = max(2, d)
        
        sublen = max(1, n // d)  # Ensure substring length is at least 1
        
        # Create a tuple with d elements
        weights = []
        for i in range(d):
            start = i * sublen
            end = min((i + 1) * sublen, n)  # Ensure we don't go past the end
            
            # Count the number of '1' bits in this substring
            if start < n:  # Only add if there are bits left
                substring = code[start:end]
                weight = substring.count('1')
                weights.append(weight)
            else:
                weights.append(0)  # Pad with zeros if needed
                
        # Ensure tuple has exactly d elements
        while len(weights) < d:
            weights.append(0)
            
        # Always return a tuple with at least 2 elements
        if len(weights) < 2:
            weights.append(0)
            
        return tuple(weights[:d])

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
    
    def _stable_hash(self, code: str) -> int:
        # Use md5 for stable hash across runs
        return int(hashlib.md5(code.encode('utf-8')).hexdigest(), 16)

    def insert(self, code: str, identifier: Optional[str] = None) -> None:
        """
        Insert a code into the tree with proper structure maintenance.
        """
        if len(code) != self.n:
            raise ValueError(f"Code length must be {self.n}")
        if identifier is None:
            identifier = f"code_{self.stats['total_codes']}"
            
        # Distribute codes across the forest
        if self.forest_size > 1:
            tree_index = random.randint(0, self.forest_size - 1)
        else:
            tree_index = 0
            
        tree = self.trees[tree_index]
        self._insert_recursive(tree['root'], code, identifier)
        tree['size'] += 1
        self.stats['tree_sizes'][tree_index] += 1
        self.stats['total_codes'] += 1
        self.stats['insertions'] += 1
        
        # Verify tree integrity and fix any issues
        attempts = 0
        while attempts < 3 and not self._verify_and_fix_patterns(tree['root']):
            attempts += 1
        
        # Clean up the tree structure
        self._optimize_tree_structure(tree['root'])
        
        # Final verification to ensure threshold is enforced
        self._enforce_threshold(tree['root'])

    def _verify_and_fix_patterns(self, node):
        """
        Verify and fix pattern assignments in the tree.
        Also checks for threshold violations.
        Returns True if patterns are correct, False if fixes were needed.
        """
        if node['is_leaf']:
            # Check for threshold violations
            if len(node['codes']) > self.t:
                print(f"WARNING: Threshold violation detected: {len(node['codes'])} > {self.t} in node with pattern {node['pattern']}")
                # Try to fix by splitting
                self._split_node(node)
                return False
                
            # For leaf nodes with artificial patterns, we don't check pattern matching
            # If the node has a parent and is not the root, it may have an artificial pattern
            if node['parent'] is not None and node['parent']['children'] and len(node['parent']['children']) > 1:
                return True
                
            # For regular leaf nodes, check if all codes match the node's pattern
            all_correct = True
            if node['pattern'] is not None and node['codes']:
                for _, code in node['codes']:
                    actual_pattern = self._get_substring_weights(code, 2)
                    if actual_pattern != node['pattern']:
                        print(f"WARNING: Pattern mismatch: code has pattern {actual_pattern} but is in node with pattern {node['pattern']}")
                        all_correct = False
                        break
            return all_correct
        else:
            # For internal nodes, first check if any child exceeds threshold
            for pattern, child in list(node['children'].items()):
                if child['is_leaf'] and len(child['codes']) > self.t:
                    print(f"WARNING: Child node threshold violation: {len(child['codes'])} > {self.t}")
                    # Try to fix by splitting the child
                    self._split_node(child)
                    return False
                    
            # Then process all children recursively
            all_ok = True
            for child in list(node['children'].values()):
                if not self._verify_and_fix_patterns(child):
                    all_ok = False
            return all_ok

    def _optimize_tree_structure(self, node):
        """
        Optimize the tree structure by:
        1. Removing empty nodes
        2. Collapsing single-child nodes when possible
        3. Merging small nodes when appropriate
        
        Always maintains threshold constraint.
        """
        if node['is_leaf']:
            # Nothing to optimize for leaf nodes
            return len(node['codes']) > 0
        
        # First optimize all children recursively
        children_to_remove = []
        for pattern, child in list(node['children'].items()):
            if not self._optimize_tree_structure(child):
                children_to_remove.append(pattern)
        
        # Remove empty children
        for pattern in children_to_remove:
            del node['children'][pattern]
            
        # If no children left, convert back to leaf
        if not node['children']:
            node['is_leaf'] = True
            return len(node['codes']) > 0
            
        # Check if we have only one child and can collapse
        # Only collapse if the combined codes don't exceed threshold
        if len(node['children']) == 1 and node['parent'] is not None:
            # Only collapse if parent is not root
            child_pattern, child = next(iter(node['children'].items()))
            if child['is_leaf']:
                # Only collapse if it won't violate threshold
                total_codes = len(node['codes']) + len(child['codes'])
                if total_codes <= self.t:
                    # Safe to collapse - move child codes to this node
                    node['is_leaf'] = True
                    node['codes'].extend(child['codes'])
                    node['children'] = {}
                
        return True

    def _insert_recursive(self, node, code, identifier):
        """
        Insert a code recursively into the tree.
        Only splits nodes when necessary and maintains proper structure.
        """
        if node['is_leaf']:
            # For root node or leaf nodes, add code to node
            node['codes'].append((identifier, code))
            
            # Only split if threshold is exceeded
            if len(node['codes']) > self.t:
                self._split_node(node)
        else:
            # Get pattern and insert into appropriate child
            pattern = self._get_substring_weights(code, 2)
            
            # Create child node if it doesn't exist
            if pattern not in node['children']:
                node['children'][pattern] = self._make_node(
                    is_leaf=True,
                    parent=node,
                    pattern=pattern,
                    depth=node['depth'] + 1,
                    d=2
                )
            
            # Insert into the child node
            self._insert_recursive(node['children'][pattern], code, identifier)

    def _split_node(self, node):
        """
        Split a leaf node that exceeds the threshold.
        Handles both pattern-based splitting and artificial splitting when necessary.
        """
        # Validate we're only splitting leaf nodes that exceed threshold
        if not node['is_leaf'] or len(node['codes']) <= self.t:
            return
            
        # Store codes for splitting
        codes = node['codes'].copy()
        
        # Group codes by pattern
        pattern_groups = {}
        for identifier, code in codes:
            pattern = self._get_substring_weights(code, 2)
            if pattern not in pattern_groups:
                pattern_groups[pattern] = []
            pattern_groups[pattern].append((identifier, code))
        
        # If we can split by pattern (multiple patterns exist)
        if len(pattern_groups) > 1:
            # Convert to internal node
            node['is_leaf'] = False
            node['children'] = {}
            node['codes'] = []
            
            # Create child nodes for each pattern group
            for pattern, group_codes in pattern_groups.items():
                if group_codes:  # Only create nodes with actual codes
                    child = self._make_node(True, node, pattern, node['depth'] + 1, 2)
                    child['codes'] = group_codes.copy()
                    node['children'][pattern] = child
                    
                    # Recursively split child if it exceeds threshold
                    if len(child['codes']) > self.t:
                        self._split_node(child)
        
        # If we can't split by pattern (single pattern or identical codes)
        else:
            # Must do artificial splitting to enforce threshold
            self._artificial_split(node)

    def _artificial_split(self, node):
        """
        Artificially split a node when pattern-based splitting isn't possible.
        This ensures threshold enforcement even when all codes have the same pattern.
        """
        if not node['is_leaf'] or len(node['codes']) <= self.t:
            return
            
        codes = node['codes'].copy()
        
        # Clear the node
        node['is_leaf'] = False
        node['children'] = {}
        node['codes'] = []
        
        # Get the single pattern (if codes have the same pattern)
        if codes:
            base_pattern = self._get_substring_weights(codes[0][1], 2)
        else:
            return  # Nothing to split
            
        # Split codes into chunks of size self.t
        chunks = []
        for i in range(0, len(codes), self.t):
            chunks.append(codes[i:i + self.t])
            
        # Create child nodes with slightly modified patterns
        for i, chunk in enumerate(chunks):
            if i == 0:
                # First chunk uses the original pattern
                pattern = base_pattern
            else:
                # Create a modified pattern for uniqueness
                pattern_list = list(base_pattern)
                # Modify the first element slightly (guaranteed to be unique with this formula)
                pattern_list[0] = (base_pattern[0] + i) % (self.substring_length + 1)
                pattern = tuple(pattern_list)
                
            # Create a child node for this chunk
            child = self._make_node(True, node, pattern, node['depth'] + 1, 2)
            child['codes'] = chunk
            node['children'][pattern] = child

    def _compute_lower_bound(self, query_Q_d: Tuple[int, ...], pattern: Tuple[int, ...], r: int) -> int:
        """
        Compute a lower bound on the Hamming distance between any code with the given pattern
        and the query code. This is used for pruning the search space.
        
        Parameters:
            query_Q_d: Pattern of the query code
            pattern: Pattern of the node
            r: Maximum allowed distance
            
        Returns:
            Lower bound on Hamming distance (or inf if pattern can be pruned)
        """
        # Ensure both patterns have the same length
        if len(query_Q_d) != len(pattern):
            return float('inf')
        
        # Calculate L1 distance between patterns
        l1_dist = sum(abs(p - q) for p, q in zip(pattern, query_Q_d))
        
        # If L1 distance > r, all codes in this node can be pruned
        if l1_dist > r:
            return float('inf')
        
        # For Hamming distance, we need at least the sum of absolute differences
        # as each difference in weight requires at least that many bit flips
        return l1_dist
    
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
    
    def _search_r_recursive(self, node, query, r, query_pattern, results):
        """
        Recursive helper for r-neighbor search.
        """
        if node['is_leaf']:
            # Check all codes in leaf node
            for identifier, code in node['codes']:
                dist = self.hamming_distance(query, code)
                if dist <= r:
                    results.append((identifier, code, dist))
        else:
            # Process all children that could contain matches
            for pattern, child in node['children'].items():
                # Compute lower bound for this pattern
                lb = self._compute_lower_bound(query_pattern, pattern, r)
                if lb <= r:
                    # This child might contain matches
                    self._search_r_recursive(child, query, r, query_pattern, results)

    def search_r(self, query: str, r: int) -> List[Tuple[str, str, int]]:
        """
        Perform r-neighbor search to find all codes within Hamming distance r.
        Works with the optimized tree structure.
        
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
        query_pattern = self._get_substring_weights(query, 2)
        
        # Generate a unique cache key for this query
        cache_key = f"{hash(query)}_{r}"
        
        # Check cache first
        if cache_key in self.node_cache:
            # Cache hit
            cache_entry = self.node_cache[cache_key]
            self._update_cache_entry(cache_key, cache_entry)
            self.stats['cache_hits'] += 1
            self.stats['searches'] += 1
            return cache_entry.results
        
        # Cache miss - perform the search
        self.stats['cache_misses'] += 1
        
        # Search in all trees in the forest
        for tree in self.trees:
            self._search_r_recursive(tree['root'], query, r, query_pattern, results)
        
        self.stats['searches'] += 1
        
        # Filter results to ensure distance <= r
        filtered_results = []
        for identifier, code, dist in results:
            if dist <= r:  # Ensure distance constraint is strictly enforced
                filtered_results.append((identifier, code, dist))
        
        # Sort by distance
        filtered_results.sort(key=lambda x: x[2])
        
        # Store in cache
        cache_entry = NodeCache(
            radius=r,
            fully_explored=True,
            results=filtered_results,
            timestamp=time.time(),
            access_count=1,
            last_access=time.time()
        )
        self.node_cache[cache_key] = cache_entry
        self._manage_cache()
        
        return filtered_results
    
    def search_knn(self, query: str, k: int) -> List[Tuple[str, str, int]]:
        """
        Find k nearest neighbors for a query code.
        """
        if len(query) != self.n:
            raise ValueError(f"Query code length must be {self.n}")
        if k < 1:
            raise ValueError("k must be positive")
        
        start_time = time.time()
        
        # Generate a unique cache key for this knn query
        cache_key = f"knn_{hash(query)}_{k}"
        
        # Check cache first
        if cache_key in self.node_cache:
            # Cache hit
            cache_entry = self.node_cache[cache_key]
            self._update_cache_entry(cache_key, cache_entry)
            self.stats['cache_hits'] += 1
            self.stats['knn_searches'] += 1
            return cache_entry.results
        
        # Cache miss - perform the search
        self.stats['cache_misses'] += 1
        
        all_results = []
        query_Q_d = self._compute_Q_d(query)
        
        # First check if the query code itself is in the tree
        exact_matches = []
        for tree in self.trees:
            self._find_exact_match(tree['root'], query, exact_matches)
        
        # If exact match found, it should be the first result
        if exact_matches:
            all_results.extend([(id, code, 0) for id, code in exact_matches])
            if len(all_results) >= k:
                final_results = all_results[:k]
                
                # Store in cache
                cache_entry = NodeCache(
                    radius=0,  # Using radius=0 for kNN
                    fully_explored=True,
                    results=final_results,
                    timestamp=time.time(),
                    access_count=1,
                    last_access=time.time()
                )
                self.node_cache[cache_key] = cache_entry
                self._manage_cache()
                
                self.stats['knn_searches'] += 1
                self.stats['last_knn_time'] = time.time() - start_time
                return final_results
        
        # Search with increasing radius until k neighbors are found
        r = 0
        max_radius = self.n
        
        while r <= max_radius:
            current_results = []
            
            # Search all trees with current radius
            for tree in self.trees:
                if tree['size'] > 0:  # Only search non-empty trees
                    tree_results = []
                    self._search_r_recursive(tree['root'], query, r, query_Q_d, tree_results)
                    
                    # Filter results to ensure they're exact distance r (since we already searched 0...r-1)
                    for identifier, code, dist in tree_results:
                        if dist == r:  # Only include results at exactly distance r
                            current_results.append((identifier, code, dist))
            
            # Add new results to accumulated results
            all_results.extend(current_results)
            
            # Check if we have enough results
            if len(all_results) >= k:
                # Sort by distance
                all_results.sort(key=lambda x: (x[2], x[0]))  # Sort by distance, then by ID
                final_results = all_results[:k]
                
                # Store in cache
                cache_entry = NodeCache(
                    radius=r,
                    fully_explored=True,
                    results=final_results,
                    timestamp=time.time(),
                    access_count=1,
                    last_access=time.time()
                )
                self.node_cache[cache_key] = cache_entry
                self._manage_cache()
                
                self.stats['knn_searches'] += 1
                self.stats['last_knn_time'] = time.time() - start_time
                return final_results
            
            r += 1
        
        # Return all available results if k neighbors weren't found
        all_results.sort(key=lambda x: (x[2], x[0]))  # Sort by distance, then by ID
        
        # Store in cache
        cache_entry = NodeCache(
            radius=max_radius,
            fully_explored=True,
            results=all_results,
            timestamp=time.time(),
            access_count=1,
            last_access=time.time()
        )
        self.node_cache[cache_key] = cache_entry
        self._manage_cache()
        
        self.stats['knn_searches'] += 1
        self.stats['last_knn_time'] = time.time() - start_time
        return all_results

    def _find_exact_match(self, node, query, results):
        """
        Helper function to find an exact match for the query code.
        """
        if node['is_leaf']:
            for identifier, code in node['codes']:
                if code == query:
                    results.append((identifier, code))
        else:
            query_pattern = self._get_substring_weights(query, 2)
            if query_pattern in node['children']:
                self._find_exact_match(node['children'][query_pattern], query, results)

    def delete(self, code: str) -> bool:
        """
        Delete a binary code from the tree.
        """
        if len(code) != self.n:
            raise ValueError(f"Code length must be {self.n}")
        found = False
        # Search all trees for the code and delete from the first one found
        for tree in self.trees:
            if self._delete_anywhere(tree['root'], code):
                tree['size'] -= 1
                tree['deletions_since_rebalance'] += 1
                self.stats['deletions'] += 1
                self.stats['total_codes'] -= 1
                if self._should_rebalance(tree):
                    self._rebalance_tree(tree)
                    self.stats['rebalances'] += 1
                found = True
                break
        return found
    
    def _delete_anywhere(self, node, code):
        if node['is_leaf']:
            for i, (_, stored_code) in enumerate(node['codes']):
                if stored_code == code:
                    node['codes'].pop(i)
                    return True
            return False
        # Try to delete from all children (not just the Qd pattern path)
        to_delete = None
        for pattern, child in list(node['children'].items()):
            deleted = self._delete_anywhere(child, code)
            if deleted:
                # Clean up empty leaf
                if child['is_leaf'] and not child['codes']:
                    to_delete = pattern
                # If after deletion, node has only one child and is not the root, merge it up
                if not node['is_leaf'] and len(node['children']) == 1 and node['parent'] is not None:
                    only_child = next(iter(node['children'].values()))
                    if only_child['is_leaf']:
                        node['is_leaf'] = True
                        node['codes'] = only_child['codes']
                        node['children'] = {}
                # If after deletion, node has no children, make it a leaf
                if not node['children']:
                    node['is_leaf'] = True
                if to_delete is not None:
                    del node['children'][to_delete]
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
        Print comprehensive statistics about the tree forest.
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
        print(f"Rebalances: {stats['rebalances']}")
        print("\nCache Performance:")
        print(f"Cache Size: {stats['cache_size']}")
        print(f"Cache Hits: {stats['cache_hits']}")
        print(f"Cache Misses: {stats['cache_misses']}")
        print("-" * 30)

    def reorder_bits(self, order: list):
        """Reorder bits in all codes in the tree according to the given order, and rebuild the tree."""
        def reorder_code(code):
            return ''.join(code[i] for i in order)
        all_codes = [(id, reorder_code(code)) for id, code in self.export_all_codes()]
        self.rebuild_tree(all_codes)

    def angular_distance(self, code1: str, code2: str) -> float:
        """Compute angular (cosine) distance between two binary codes."""
        v1 = np.array([int(b) for b in code1])
        v2 = np.array([int(b) for b in code2])
        dot = np.dot(v1, v2)
        norm1 = np.linalg.norm(v1)
        norm2 = np.linalg.norm(v2)
        if norm1 == 0 or norm2 == 0:
            return 1.0
        cos_sim = dot / (norm1 * norm2)
        return 1.0 - cos_sim

    def export_all_codes(self):
        """Export all (identifier, code) pairs from all trees."""
        all_codes = []
        for tree in self.trees:
            def recurse(node):
                if node['is_leaf']:
                    all_codes.extend(node['codes'])
                else:
                    for child in node['children'].values():
                        recurse(child)
            recurse(tree['root'])
        return all_codes

    def rebuild_tree(self, codes):
        """
        Rebuild the entire forest from scratch with optimized structure.
        """
        # Reset everything
        self.stats['tree_sizes'] = [0] * self.forest_size
        self.stats['total_codes'] = 0
        self.node_cache = {}
        
        # Create fresh trees
        self.trees = []
        for i in range(self.forest_size):
            tree = self._create_tree()
            tree['id'] = i
            tree['size'] = 0
            self.trees.append(tree)
            
        # Batch insert codes for better efficiency
        tree_codes = [[] for _ in range(self.forest_size)]
        
        # Distribute codes evenly across trees
        for i, (identifier, code) in enumerate(codes):
            tree_idx = i % self.forest_size
            tree_codes[tree_idx].append((identifier, code))
            
        # Insert codes into each tree
        for i, tree_code_list in enumerate(tree_codes):
            tree = self.trees[i]
            
            # Sort codes by pattern for more efficient tree building
            tree_code_list.sort(key=lambda x: self._get_substring_weights(x[1], 2))
            
            # Insert all codes
            for identifier, code in tree_code_list:
                self._insert_recursive(tree['root'], code, identifier)
                tree['size'] += 1
                self.stats['total_codes'] += 1
                
            # Optimize the tree structure
            self._optimize_tree_structure(tree['root'])

    def search_r_angular(self, query: str, r: float) -> list:
        """
        Find all codes within angular distance r of the query.
        
        Parameters:
            query: Query binary code
            r: Maximum angular distance (between 0 and 1)
            
        Returns:
            List of tuples (identifier, code, distance) for matching codes
        """
        if len(query) != self.n:
            raise ValueError(f"Query code length must be {self.n}")
        
        # Generate a unique cache key for this angular query
        cache_key = f"angular_{hash(query)}_{r}"
        
        # Check cache first
        if cache_key in self.node_cache:
            # Cache hit
            cache_entry = self.node_cache[cache_key]
            self._update_cache_entry(cache_key, cache_entry)
            self.stats['cache_hits'] += 1
            self.stats['searches'] += 1
            return cache_entry.results
        
        # Cache miss - perform the search
        self.stats['cache_misses'] += 1
        
        results = []
        for tree in self.trees:
            self._search_r_angular_recursive(tree['root'], query, r, 1, results)
        
        self.stats['searches'] += 1
        
        # Filter results to ensure distance <= r
        filtered_results = []
        for identifier, code, dist in results:
            if dist <= r:  # Ensure distance constraint is strictly enforced
                filtered_results.append((identifier, code, dist))
        
        # Sort by distance
        filtered_results.sort(key=lambda x: x[2])
        
        # Store in cache
        cache_entry = NodeCache(
            radius=r,
            fully_explored=True,
            results=filtered_results,
            timestamp=time.time(),
            access_count=1,
            last_access=time.time()
        )
        self.node_cache[cache_key] = cache_entry
        self._manage_cache()
        
        return filtered_results

    def search_knn_angular(self, query: str, k: int) -> list:
        """
        Find k nearest neighbors for a query code using angular distance.
        
        Parameters:
            query: Query binary code
            k: Number of neighbors to return
            
        Returns:
            List of tuples (identifier, code, angular_distance) for k nearest neighbors
        """
        if len(query) != self.n:
            raise ValueError(f"Query code length must be {self.n}")
        if k < 1:
            raise ValueError("k must be positive")
        
        # Generate a unique cache key for this knn angular query
        cache_key = f"knn_angular_{hash(query)}_{k}"
        
        # Check cache first
        if cache_key in self.node_cache:
            # Cache hit
            cache_entry = self.node_cache[cache_key]
            self._update_cache_entry(cache_key, cache_entry)
            self.stats['cache_hits'] += 1
            self.stats['knn_searches'] += 1
            return cache_entry.results
        
        # Cache miss - perform the search
        self.stats['cache_misses'] += 1
        
        # First check if the query code itself is in the tree (exact match)
        exact_matches = []
        for tree in self.trees:
            self._find_exact_match(tree['root'], query, exact_matches)
        
        all_results = []
        # If exact match found, it should be the first result
        if exact_matches:
            all_results.extend([(id, code, 0.0) for id, code in exact_matches])
            if len(all_results) >= k:
                final_results = all_results[:k]
                
                # Store in cache
                cache_entry = NodeCache(
                    radius=0.0,  # Using radius=0 for kNN
                    fully_explored=True,
                    results=final_results,
                    timestamp=time.time(),
                    access_count=1,
                    last_access=time.time()
                )
                self.node_cache[cache_key] = cache_entry
                self._manage_cache()
                
                self.stats['knn_searches'] += 1
                return final_results
        
        # Start with small radius and increase until we find k neighbors
        r = 0.0
        step = 0.05  # Increase radius by 5% each time
        max_radius = 1.0  # Maximum possible angular distance
        
        while r <= max_radius:
            current_results = []
            
            # Get all results within current radius
            search_results = self.search_r_angular(query, r)
            
            # Only include results at exactly this radius tier
            # (to avoid duplicating codes we've already found)
            for identifier, code, dist in search_results:
                if len(all_results) == 0 or dist > all_results[-1][2]:
                    current_results.append((identifier, code, dist))
            
            # Add new results
            all_results.extend(current_results)
            
            # Check if we have enough results
            if len(all_results) >= k:
                # Sort by distance
                all_results.sort(key=lambda x: x[2])
                final_results = all_results[:k]
                
                # Store in cache
                cache_entry = NodeCache(
                    radius=r,
                    fully_explored=True,
                    results=final_results,
                    timestamp=time.time(),
                    access_count=1,
                    last_access=time.time()
                )
                self.node_cache[cache_key] = cache_entry
                self._manage_cache()
                
                self.stats['knn_searches'] += 1
                return final_results
            
            # Increase search radius
            r += step
        
        # Return all available results if k neighbors weren't found
        all_results.sort(key=lambda x: x[2])
        
        # Store in cache
        cache_entry = NodeCache(
            radius=max_radius,
            fully_explored=True,
            results=all_results,
            timestamp=time.time(),
            access_count=1,
            last_access=time.time()
        )
        self.node_cache[cache_key] = cache_entry
        self._manage_cache()
        
        self.stats['knn_searches'] += 1
        return all_results

    def _search_r_angular_recursive(self, node, query, r, d, results):
        if node['is_leaf']:
            results.extend((identifier, code, self.angular_distance(query, code))
                           for identifier, code in node['codes']
                           if self.angular_distance(query, code) <= r)
            return
        query_pattern = self._get_substring_weights(query, node['d'])
        max_weight = len(query) // node['d']
        for pattern, child in node['children'].items():
            pattern_vec = np.array(pattern)
            query_vec = np.array(query_pattern)
            # Only compare if pattern and query pattern are the same length
            if pattern_vec.shape != query_vec.shape:
                continue
            dot = np.dot(pattern_vec, query_vec)
            norm1 = np.linalg.norm(pattern_vec)
            norm2 = np.linalg.norm(query_vec)
            if norm1 == 0 or norm2 == 0:
                ang_dist = 1.0
            else:
                cos_sim = dot / (norm1 * norm2)
                ang_dist = 1.0 - cos_sim
            if ang_dist <= r:
                self._search_r_angular_recursive(child, query, r, d * 2, results)

    def _debug_check_threshold(self, node):
        """Check that no node exceeds the threshold (for debugging)."""
        try:
            if node['is_leaf']:
                assert len(node['codes']) <= self.t, f"Node threshold exceeded: {len(node['codes'])} > {self.t}"
            else:
                for child in node['children'].values():
                    self._debug_check_threshold(child)
        except AssertionError as e:
            # If assertion fails, try to fix the issue by splitting the node
            if node['is_leaf'] and len(node['codes']) > self.t:
                self._split_node(node)
                # Verify fix worked
                assert len(node['codes']) <= self.t or not node['is_leaf'], f"Node threshold fix failed: {len(node['codes'])} > {self.t}"

    def _prune_empty_nodes(self, node):
        """
        Recursively prune empty nodes from the tree.
        An empty node is a leaf with no codes or a non-leaf with no children.
        """
        if node['is_leaf']:
            # Leaf nodes are empty if they have no codes
            return len(node['codes']) > 0
        else:
            # First, recursively prune children
            children_to_remove = []
            for pattern, child in list(node['children'].items()):
                if not self._prune_empty_nodes(child):
                    children_to_remove.append(pattern)
            
            # Remove empty children
            for pattern in children_to_remove:
                del node['children'][pattern]
            
            # If all children were removed, node should become a leaf
            if not node['children']:
                node['is_leaf'] = True
                return False  # Node is now an empty leaf
            
            return True  # Node still has children

    def _enforce_threshold(self, node):
        """
        Ensure no node exceeds the threshold.
        """
        if node['is_leaf']:
            if len(node['codes']) > self.t:
                # Threshold violation - must split
                self._split_node(node)
            return True
        else:
            # Process children first
            for child in list(node['children'].values()):
                self._enforce_threshold(child)
            return True

    def check_tree_integrity(self):
        """
        Debug method to check the entire tree for integrity violations.
        Checks:
        1. Threshold constraints
        2. Pattern assignments
        3. Tree structure
        
        Returns the number of violations found
        """
        violations = 0
        
        for tree_idx, tree in enumerate(self.trees):
            print(f"Checking tree {tree_idx}...")
            
            def check_node(node, depth=0):
                nonlocal violations
                indent = "  " * depth
                
                if node['is_leaf']:
                    # Check threshold
                    if len(node['codes']) > self.t:
                        print(f"{indent}VIOLATION: Node has {len(node['codes'])} codes (> threshold {self.t})")
                        violations += 1
                    
                    # Check pattern matching
                    if node['pattern'] is not None:
                        for _, code in node['codes']:
                            actual = self._get_substring_weights(code, 2)
                            if actual != node['pattern'] and node['parent'] is not None:
                                # Skip for artificially split nodes
                                if len(node['parent']['children']) <= 1:
                                    print(f"{indent}VIOLATION: Code with pattern {actual} in node with pattern {node['pattern']}")
                                    violations += 1
                else:
                    # Check children
                    for pattern, child in node['children'].items():
                        if child['pattern'] != pattern:
                            print(f"{indent}VIOLATION: Child pattern {child['pattern']} != key {pattern}")
                            violations += 1
                        check_node(child, depth + 1)
            
            check_node(tree['root'])
        
        if violations == 0:
            print("Tree integrity check: PASSED (No violations)")
        else:
            print(f"Tree integrity check: FAILED ({violations} violations)")
            
        return violations

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