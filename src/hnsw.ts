import type { DBInterface } from "./db/interfaces";
import type { KNNResult, LayerNode, Node, Point } from "./types";
import { NodeHeap, compareNode, cosine_distance } from "./utils";

/**
 * Hierarchical Navigable Small Worlds (HNSW) Implementation
 *
 * HNSW is a graph-based algorithm for approximate nearest neighbor search in high-dimensional spaces.
 * It builds a multi-layer graph structure where:
 * - Layer 0 contains all points and forms the base layer
 * - Higher layers contain progressively fewer points (sampled probabilistically)
 * - Each layer maintains connections between nearby points
 *
 * Key Concepts:
 * - **Multi-layer structure**: Higher layers enable long-range navigation, lower layers provide precision
 * - **Entry point**: A single node in the top layer that serves as the starting point for all searches
 * - **Greedy search**: Navigate by always moving to the closest unvisited neighbor
 * - **Layer selection**: New points are assigned to layers using an exponential decay probability
 *
 * Algorithm Benefits:
 * - Logarithmic search complexity: O(log N) for both search and insertion
 * - High recall: Can find very good approximate nearest neighbors
 * - Scalable: Works well with millions of high-dimensional vectors
 *
 * This implementation works over a key-value database interface, allowing different storage backends.
 *
 * @template M Type of the metadata attached to each point (e.g., document IDs, labels, etc.)
 *
 * @see https://arxiv.org/pdf/1603.09320.pdf Original HNSW paper by Malkov & Yashunin
 */
export class HNSW<M = unknown> {
	/** Database interface for storing points, graph connections, and metadata */
	db: DBInterface<M>;

	/**
	 * Maximum number of bi-directional links for each node during construction.
	 * This parameter controls the connectivity of the graph:
	 * - Higher values = better search quality but slower construction and more memory
	 * - Lower values = faster construction but potentially worse search quality
	 * - Paper suggests m ∈ [5, 48], with 16 being a good default
	 * - Weaviate uses 64 for high-dimensional data
	 */
	m: number;

	/**
	 * Maximum number of connections for layer 0 (base layer).
	 * Set to 2 * m as recommended in the paper.
	 * Layer 0 can have more connections since it contains all points.
	 */
	m_max0: number;

	/**
	 * Normalization factor for level generation probability.
	 * Used in the exponential decay formula: level = floor(-ln(uniform(0,1)) * ml)
	 * Set to 1/ln(m) as per the paper's heuristic.
	 */
	ml: number;

	/**
	 * Size of the dynamic candidate list during construction.
	 * Controls the search scope when finding neighbors for new points:
	 * - Higher values = better graph quality but slower construction
	 * - Lower values = faster construction but potentially worse search quality
	 * - Common values: 40 (fast), 100 (balanced), 400 (high quality)
	 */
	ef_construction: number;

	/**
	 * Size of the dynamic candidate list during search.
	 * Controls the search scope when performing kNN queries:
	 * - Higher values = better recall but slower search
	 * - Lower values = faster search but potentially lower recall
	 * - Should be >= k (number of neighbors to return)
	 * - Can be adjusted at search time for different speed/quality tradeoffs
	 */
	ef: number;

	/**
	 * Constructs a new HNSW index with the specified parameters.
	 *
	 * @param db Database interface for persistence (must implement DBInterface)
	 * @param M Maximum number of connections per node (recommended: 16, range: [5-48])
	 * @param ef_construction Size of candidate list during construction (recommended: 200)
	 * @param ef_search Size of candidate list during search (recommended: 50, must be >= k)
	 *
	 * @example
	 * ```typescript
	 * const hnsw = new HNSW(
	 *   database,    // Your database implementation
	 *   16,          // M: good balance of speed/quality
	 *   200,         // ef_construction: high quality graph
	 *   50           // ef_search: good search performance
	 * );
	 * ```
	 */
	constructor(
		db: DBInterface<M>,
		M: number,
		ef_construction: number,
		ef_search: number,
	) {
		this.db = db;

		// Connection parameters (from paper recommendations)
		this.m = M;
		this.m_max0 = M * 2; // Layer 0 gets double the connections
		this.ml = 1 / Math.log(M); // Level generation probability factor

		// Search quality parameters
		this.ef_construction = ef_construction;
		this.ef = ef_search;
	}

	/**
	 * Retrieves a vector and its associated metadata by index.
	 *
	 * This is a convenience method that fetches both the vector data and any
	 * metadata stored with it in a single operation.
	 *
	 * @param idx The index of the vector to retrieve
	 * @returns Object containing the vector data and metadata (null if no metadata exists)
	 *
	 * @example
	 * ```typescript
	 * const result = await hnsw.get_vector(42);
	 * console.log('Vector:', result.point);      // [0.1, 0.2, 0.3, ...]
	 * console.log('Metadata:', result.metadata); // { filename: 'doc.pdf', category: 'research' }
	 * ```
	 */
	async get_vector(idx: number): Promise<{
		point: Point;
		metadata: M | null;
	}> {
		const point = await this.db.get_point(idx);
		const metadata = await this.db.get_metadata(idx);
		return { point, metadata };
	}

	/**
	 * Selects which layer a new point should be inserted into.
	 *
	 * Uses an exponential decay probability distribution as recommended in the paper.
	 * Most points will be inserted into layer 0, with progressively fewer points
	 * in higher layers. This creates the hierarchical structure that makes HNSW efficient.
	 *
	 * The probability of a point being in layer L or higher is: (1/2)^L
	 * This means:
	 * - ~50% of points are only in layer 0
	 * - ~25% of points reach layer 1 or higher
	 * - ~12.5% of points reach layer 2 or higher
	 * - etc.
	 *
	 * @returns The layer number (0-based) where the new point should be inserted
	 *
	 * @example
	 * ```typescript
	 * const layer = hnsw.select_layer();  // Returns 0, 1, 2, 3, ... with decreasing probability
	 * ```
	 */
	select_layer() {
		return Math.floor(-Math.log(Math.random()) * this.ml);
	}

	/**
	 * Inserts a new point into the HNSW index.
	 *
	 * This is the core method that implements Algorithm 1 from the HNSW paper.
	 * The insertion process works in several phases:
	 *
	 * 1. **Layer Selection**: Randomly determine which layer the new point belongs to
	 * 2. **Entry Point Search**: Navigate from top layer down to find the best entry point
	 * 3. **Layer-by-layer Insertion**: Insert the point into each layer from selected layer down to 0
	 * 4. **Neighbor Selection**: For each layer, find the best neighbors and create bidirectional links
	 * 5. **Pruning**: Ensure no node has too many connections by removing the worst ones
	 *
	 * Time Complexity: O(log N) expected, where N is the number of points
	 * Space Complexity: O(M * N) where M is the average number of connections per point
	 *
	 * @param q The vector to insert (array of numbers representing the point in space)
	 * @param metadata Optional metadata to associate with this point (e.g., document ID, labels)
	 *
	 * @example
	 * ```typescript
	 * // Insert a simple vector
	 * await hnsw.insert([0.1, 0.2, 0.3, 0.4]);
	 *
	 * // Insert a vector with metadata
	 * await hnsw.insert(
	 *   [0.1, 0.2, 0.3, 0.4],
	 *   { filename: 'document.pdf', category: 'research' }
	 * );
	 * ```
	 *
	 * @see https://arxiv.org/pdf/1603.09320.pdf Algorithm 1 (page 7)
	 */
	async insert(q: Point, metadata?: M) {
		// Get current state of the index
		const ep_index = await this.db.get_ep();
		const L = (await this.db.get_num_layers()) - 1; // Current top layer (0-indexed)
		const l = this.select_layer(); // Layer where new point will be inserted

		// Add the point to database and get its assigned index
		const idx = await this.db.new_point(q);
		if (metadata) {
			await this.db.set_metadata(idx, metadata);
		}

		// CASE 1: Index is not empty - need to find entry point and insert into layers
		if (ep_index !== null) {
			const dist = cosine_distance(q, await this.db.get_point(ep_index));

			// PHASE 1: Search from top layer (L) down to layer (l+1) to find entry point
			// This is a greedy search that finds the closest point in each layer
			let ep = [[dist, ep_index] as Node];
			for (let i = L; i > l; i--) {
				const ep_copy: Node[] = ep.map((e) => [e[0], e[1]]);

				// Search with ef=1 (only find the single closest neighbor)
				const W = await this.search_layer(q, ep_copy, 1, i);

				// Update entry point if we found something closer
				if (W.length > 0 && ep[0][0] > W[0][0]) {
					ep = W;
				}
			}

			// PHASE 2: Insert the point into each layer from min(L, l) down to 0
			for (let l_c = Math.min(L, l); l_c >= 0; l_c--) {
				// Search for neighbors in current layer with larger candidate list
				const W = await this.search_layer(q, ep, this.ef_construction, l_c);
				const newNode: LayerNode = {}; // Connections for the new point

				// Update entry point for next layer
				ep = W.map((e) => [e[0], e[1]] as Node);

				// Select the best neighbors using Algorithm 4 (neighbor selection heuristic)
				const neighbors = this.select_neighbors(q, W, l_c);
				const indices = neighbors.map(([, idx]) => idx);
				const nodes = await this.db.get_neighbors(l_c, indices);

				// Determine maximum connections allowed for this layer
				const M = l_c === 0 ? this.m_max0 : this.m;

				// PHASE 3: Create bidirectional connections between new point and selected neighbors
				for (const e of neighbors) {
					newNode[e[1]] = e[0]; // Connect new point to neighbor
					nodes[e[1]][idx] = e[0]; // Connect neighbor back to new point
				}

				// PHASE 4: Prune connections if any neighbor now has too many connections
				for (const e of neighbors) {
					// Get all connections for this neighbor
					const eConn = Object.entries(nodes[e[1]]).map(
						([k, v]) => [v, Number.parseInt(k)] as Node,
					);

					// If neighbor has too many connections, prune to the best ones
					if (eConn.length > M) {
						// Use neighbor selection heuristic to keep only the best connections
						const eNewConn = this.select_neighbors(
							await this.db.get_point(e[1]),
							eConn,
							l_c,
						);

						// Convert selected connections back to dictionary format
						const dict: Record<number, number> = {};
						for (const eNew of eNewConn) {
							dict[eNew[1]] = eNew[0];
						}

						nodes[e[1]] = dict; // Update neighbor's connection list
					}
				}

				// Save the new connections to the database
				await this.db.upsert_neighbor(l_c, idx, newNode);
				await this.db.upsert_neighbors(l_c, nodes);
			}
		}

		// PHASE 5: Update global state if new point created new layers
		const LL = await this.db.get_num_layers();
		if (LL < l + 1) {
			// New point is now the entry point (highest layer)
			await this.db.set_ep(idx);
		}

		// Create empty neighbor lists for new layers
		// TODO: This could be optimized to create all new layers in parallel
		for (let i = LL; i < l + 1; i++) {
			await this.db.new_neighbor(idx);
		}
	}

	/**
	 * Performs a greedy search within a single layer of the HNSW graph.
	 *
	 * This implements Algorithm 2 from the HNSW paper and is the core search primitive
	 * used by both insertion and query operations. The algorithm uses a best-first search
	 * strategy with two priority queues:
	 *
	 * - **Candidates (C)**: Min-heap of points to explore next (closest first)
	 * - **Dynamic list (W)**: Max-heap of found neighbors (furthest first, for easy removal)
	 *
	 * The search expands outward from the entry points, visiting the closest unvisited
	 * neighbors first, until either:
	 * - No more promising candidates remain (all remaining candidates are further than current furthest result)
	 * - The desired number of neighbors (ef) has been found
	 *
	 * @param q The query point to search for
	 * @param ep Array of entry points to start the search from (typically 1 point, but can be multiple)
	 * @param ef Maximum number of neighbors to return (controls search scope vs speed)
	 * @param l_c The layer to search in (0 = base layer with all points, higher = sparser layers)
	 *
	 * @returns Array of [distance, point_id] pairs representing the closest neighbors found
	 *
	 * @example
	 * ```typescript
	 * // Search for 5 closest points starting from entry point 42 in layer 0
	 * const entryPoints = [[0.5, 42]];  // [distance_to_query, point_id]
	 * const neighbors = await hnsw.search_layer(queryVector, entryPoints, 5, 0);
	 * // Returns: [[0.1, 15], [0.2, 23], [0.3, 8], [0.4, 31], [0.5, 42]]
	 * ```
	 *
	 * @see https://arxiv.org/pdf/1603.09320.pdf Algorithm 2 (page 8)
	 */
	async search_layer(q: Point, ep: Node[], ef: number, l_c: number) {
		// Initialize visited set with entry points to avoid revisiting them
		const V = new Set<number>(ep.map(([, id]) => id));

		// Candidates queue (min-heap): points to explore next, ordered by distance to query
		const C = new NodeHeap(ep);

		// Dynamic list (max-heap): current best neighbors found
		// We negate distances to turn min-heap into max-heap (furthest neighbor at top)
		const W = new NodeHeap(ep.map(([mdist, p]) => [-mdist, p]));

		// Main search loop: expand outward from entry points
		while (!C.isEmpty()) {
			const c = C.pop(); // Get closest unexplored candidate
			if (!c) break; // Safety check for empty heap

			const c_v: number = c[0]; // Distance from query to candidate
			const topW = W.top(1)[0];
			if (!topW) break; // Safety check for empty heap

			const f_dist = -topW[0]; // Distance to furthest point in results (undo negation)

			// Stopping condition: if closest candidate is further than furthest result,
			// we won't find any better neighbors, so stop searching
			if (c_v > f_dist) {
				break;
			}

			// Explore neighbors of current candidate
			// Get all unvisited neighbors of the current candidate in this layer
			const neighbors = Object.keys(await this.db.get_neighbor(l_c, c[1]))
				.map((k) => Number.parseInt(k))
				.filter((k) => !V.has(k));

			// Calculate distances from query to all neighbors
			const points = await this.db.get_points(neighbors);
			const dists = points.map((p) => cosine_distance(p, q));

			// Process each neighbor
			dists.forEach((dist, i) => {
				const e = neighbors[i];
				V.add(e); // Mark neighbor as visited

				// Add neighbor to results if:
				// 1. It's closer than our furthest current result, OR
				// 2. We don't have enough results yet (W.length < ef)
				if (dist < f_dist || W.length < ef) {
					C.push([dist, e]); // Add to candidates for future exploration
					W.push([-dist, e]); // Add to results (with negated distance)

					// If we have too many results, remove the furthest one
					if (W.length > ef) {
						W.pop();
					}
				}
			});
		}

		// Handle special case for ef=1 (single neighbor search)
		if (ef === 1) {
			if (W.length !== 0) {
				// For single neighbor search, extract the closest one
				// TODO: This could be optimized - we're creating a new heap just to get the minimum
				const dd = new NodeHeap(W.heapArray.map((W_i) => [-W_i[0], W_i[1]]));
				const result = dd.pop();
				return result ? [result] : [];
			}
			return [];
		}

		// Convert max-heap back to normal distances and return all neighbors
		return W.heapArray.map((W_i) => [-W_i[0], W_i[1]]) as Node[];
	}

	/**
	 * Selects the best neighbors from a candidate set using a simple heuristic.
	 *
	 * This implements Algorithm 4 from the HNSW paper (Simple heuristic for selecting neighbors).
	 * The goal is to select diverse, high-quality connections that:
	 * 1. Are close to the query point
	 * 2. Provide good graph connectivity
	 * 3. Don't create redundant paths
	 *
	 * The algorithm works by:
	 * 1. Always preferring closer neighbors first (greedy selection)
	 * 2. Optionally keeping some pruned connections to maintain graph connectivity
	 *
	 * This is a simplified version of the neighbor selection heuristic. The paper also
	 * describes a more complex "extended heuristic" (Algorithm 4*) that considers
	 * the distance between candidates to avoid clustering, but this implementation
	 * uses the simpler approach for performance.
	 *
	 * @param q The query point (either a new point being inserted or existing point being pruned)
	 * @param C Candidate neighbors with their distances: [distance, point_id]
	 * @param l_c Current layer (affects maximum number of connections allowed)
	 * @param keepPrunedConnections Whether to fill remaining slots with pruned candidates (recommended: true)
	 *
	 * @returns Array of selected neighbors, up to M (or M_max0 for layer 0) neighbors
	 *
	 * @example
	 * ```typescript
	 * // Select best neighbors for a point in layer 1
	 * const candidates = [[0.1, 5], [0.2, 10], [0.15, 8], [0.3, 15]];
	 * const selected = hnsw.select_neighbors(queryPoint, candidates, 1, true);
	 * // Returns: [[0.1, 5], [0.15, 8]] (assuming M=2)
	 * ```
	 *
	 * @see https://arxiv.org/pdf/1603.09320.pdf Algorithm 4 (page 9)
	 */
	select_neighbors(
		q: Point,
		C: Node[],
		l_c: number,
		keepPrunedConnections = true,
	) {
		// Priority queue for selected neighbors (min-heap by distance)
		const R = new NodeHeap();
		// Working copy of candidates (min-heap by distance)
		const W = new NodeHeap(C);
		// Maximum connections allowed for this layer
		const M = l_c > 0 ? this.m : this.m_max0;

		// Queue for discarded candidates (for potential reuse if keepPrunedConnections=true)
		const W_d = new NodeHeap();

		// PHASE 1: Greedy selection of closest neighbors
		while (W.length > 0 && R.length < M) {
			const e = W.pop(); // Get closest remaining candidate
			if (!e) break; // Safety check for empty heap

			const r_top = R.top(1)[0] as Node | undefined; // Current closest selected neighbor

			// Select this candidate if:
			// 1. No neighbors selected yet (R is empty), OR
			// 2. This candidate is closer to query than our current closest selected neighbor
			//
			// Note: This is the "simple heuristic" - it just picks closest neighbors greedily
			// The "extended heuristic" would also consider distances between candidates
			if (R.length === 0 || (r_top && e[0] < r_top[0])) {
				R.push([e[0], e[1]]);
			} else {
				// Candidate was not selected, save it for potential later use
				W_d.push([e[0], e[1]]);
			}
		}

		// PHASE 2: Fill remaining slots with pruned connections (if enabled)
		// This helps maintain graph connectivity even when the greedy selection
		// doesn't fill all available connection slots
		if (keepPrunedConnections) {
			while (W_d.length > 0 && R.length < M) {
				const discarded = W_d.pop();
				if (discarded) {
					R.push(discarded);
				}
			}
		}

		return R.heapArray;
	}

	/**
	 * Performs k-nearest neighbor search to find the closest points to a query.
	 *
	 * This implements Algorithm 5 from the HNSW paper and is the main query interface.
	 * The search works in two phases:
	 *
	 * 1. **Routing Phase**: Navigate from top layer down to layer 1 using greedy search
	 *    with ef=1 to quickly find a good entry point in the base layer
	 *
	 * 2. **Search Phase**: Perform a more thorough search in layer 0 (base layer)
	 *    using the configured ef parameter to find the k best neighbors
	 *
	 * The multi-layer approach provides logarithmic search complexity because:
	 * - Higher layers have fewer points but longer connections (for fast routing)
	 * - Lower layers have more points but shorter connections (for precise search)
	 *
	 * Time Complexity: O(log N) expected, where N is the number of points
	 *
	 * @param q The query vector to search for
	 * @param K Number of nearest neighbors to return
	 *
	 * @returns Array of KNNResult objects containing id, distance, and metadata for each neighbor,
	 *          sorted by distance (closest first). Returns empty array if no points in index.
	 *
	 * @example
	 * ```typescript
	 * // Find 5 most similar vectors to query
	 * const results = await hnsw.knn_search([0.1, 0.2, 0.3, 0.4], 5);
	 *
	 * // Results format:
	 * // [
	 * //   { id: 42, distance: 0.1, metadata: { filename: 'doc1.pdf' } },
	 * //   { id: 15, distance: 0.2, metadata: { filename: 'doc2.pdf' } },
	 * //   { id: 8,  distance: 0.25, metadata: null },
	 * //   ...
	 * // ]
	 *
	 * for (const result of results) {
	 *   console.log(`Point ${result.id} with distance ${result.distance}`);
	 *   if (result.metadata) {
	 *     console.log(`  Metadata:`, result.metadata);
	 *   }
	 * }
	 * ```
	 *
	 * @see https://arxiv.org/pdf/1603.09320.pdf Algorithm 5 (page 10)
	 */
	async knn_search(q: Point, K: number): Promise<KNNResult<M>[]> {
		// Get the entry point (starting point for search)
		const ep_index = await this.db.get_ep();

		// Handle edge case: empty index
		if (ep_index === null) return [];

		// Get the current top layer and calculate initial distance
		const L = (await this.db.get_num_layers()) - 1;
		const dist = cosine_distance(q, await this.db.get_point(ep_index));

		// PHASE 1: Routing through upper layers (L down to 1)
		// Use ef=1 for fast navigation - we only need to find a good entry point for layer 0
		let ep: Node[] = [[dist, ep_index]];
		for (let l_c = L; l_c > 0; l_c--) {
			ep = await this.search_layer(q, ep, 1, l_c);
		}

		// PHASE 2: Comprehensive search in base layer (layer 0)
		// Use the configured ef parameter for quality search
		ep = await this.search_layer(q, ep, this.ef, 0);

		// Sort results by distance and take top K
		ep.sort(compareNode);
		const ep_topk = ep.slice(0, K);

		// Fetch metadata for all results in batch for efficiency
		const metadatas = await this.db.get_metadatas(ep_topk.map((ep) => ep[1]));

		// Combine results with metadata and return in the expected format
		return ep_topk.map((ep, i) => ({
			id: ep[1],
			distance: ep[0],
			metadata: metadatas[i],
		}));
	}
}

/**
 * HNSW Usage Guide and Performance Tips
 * ===================================
 *
 * ## Basic Usage Pattern
 *
 * ```typescript
 * // 1. Initialize with your database and parameters
 * const hnsw = new HNSW(database, 16, 200, 50);
 *
 * // 2. Insert vectors with optional metadata
 * await hnsw.insert([0.1, 0.2, 0.3], { type: 'document', id: 'doc1' });
 * await hnsw.insert([0.4, 0.5, 0.6], { type: 'image', id: 'img1' });
 *
 * // 3. Search for similar vectors
 * const results = await hnsw.knn_search([0.15, 0.25, 0.35], 5);
 * ```
 *
 * ## Parameter Tuning Guide
 *
 * ### Construction Parameters (set once, affect build quality):
 *
 * **M (connections per node)**:
 * - Low values (4-8): Fast insertion, uses less memory, lower search quality
 * - Medium values (12-16): Good balance for most use cases
 * - High values (32-64): Slower insertion, more memory, better search quality
 * - Rule of thumb: Increase M for higher-dimensional data
 *
 * **ef_construction (candidate list size during building)**:
 * - Low values (40-100): Fast building, lower graph quality
 * - Medium values (100-400): Good balance for most use cases
 * - High values (400+): Slow building, high graph quality
 * - Should be >= M and preferably >= 2*M
 *
 * ### Search Parameters (can be adjusted per query):
 *
 * **ef (candidate list size during search)**:
 * - Must be >= K (number of results requested)
 * - Higher values: Better recall (finds more true neighbors) but slower search
 * - Lower values: Faster search but potentially misses some true neighbors
 * - Typical range: K to 10*K depending on quality requirements
 *
 * ## Performance Characteristics
 *
 * **Time Complexity**:
 * - Insertion: O(log N) expected
 * - Search: O(log N) expected
 * - Memory: O(M * N) where M is avg connections per point
 *
 * **Scalability**:
 * - Works efficiently with millions of vectors
 * - Performance degrades gracefully with dimension (unlike tree-based methods)
 * - Parallelizable: Multiple insertions/searches can run concurrently
 *
 * ## Best Practices
 *
 * 1. **Normalize your vectors** if using cosine distance
 * 2. **Insert in batches** for better database performance
 * 3. **Start with conservative parameters** and tune based on your data
 * 4. **Monitor recall vs latency** tradeoffs during parameter tuning
 * 5. **Use metadata** to filter results post-search if needed
 *
 * ## Common Issues & Solutions
 *
 * **Poor search quality**:
 * - Increase M (more connections per node)
 * - Increase ef_construction (better graph during building)
 * - Increase ef during search (larger candidate list)
 *
 * **Slow insertions**:
 * - Decrease ef_construction
 * - Decrease M
 * - Use batch insertions if supported by your database
 *
 * **High memory usage**:
 * - Decrease M (fewer connections per node)
 * - Consider dimensionality reduction techniques
 * - Use quantization if precision allows
 *
 * **Slow searches**:
 * - Decrease ef (smaller candidate list)
 * - Ensure your database is optimized for batch reads
 * - Consider caching frequently accessed vectors
 */
