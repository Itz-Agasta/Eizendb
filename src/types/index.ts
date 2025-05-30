/**
 * Type definitions for HNSW (Hierarchical Navigable Small World) implementation
 *
 * These types define the core data structures used throughout the HNSW algorithm.
 */

/**
 * A point in high-dimensional space, represented as an array of numbers.
 *
 * Each number represents the coordinate value in one dimension.
 * All points in an HNSW index should have the same dimensionality.
 *
 * @example
 * ```typescript
 * // 3-dimensional point
 * const point: Point = [0.1, 0.5, -0.3];
 *
 * // High-dimensional embedding (e.g., from text or image)
 * const embedding: Point = [0.12, -0.34, 0.56, 0.78, ...]; // 512 dimensions
 * ```
 */
export type Point = number[];

/**
 * Represents the graph structure for a single layer in the HNSW index.
 *
 * Maps point indices to their neighbor information (LayerNode).
 * Each key is a point index, and each value contains that point's connections.
 *
 * @example
 * ```typescript
 * // Layer 1 graph with 3 points
 * const layer1: Graph = {
 *   5: { 10: 0.2, 15: 0.3 },    // Point 5 connects to points 10 and 15
 *   10: { 5: 0.2, 15: 0.1 },    // Point 10 connects to points 5 and 15
 *   15: { 5: 0.3, 10: 0.1 }     // Point 15 connects to points 5 and 10
 * };
 * ```
 */
export type Graph = Record<number, LayerNode>;

/**
 * Represents all neighbors of a single point in a layer.
 *
 * Maps neighbor point indices to their distances from this point.
 * The distances are used for efficient neighbor traversal during search.
 *
 * @example
 * ```typescript
 * // Point connects to three neighbors with their respective distances
 * const neighbors: LayerNode = {
 *   42: 0.15,   // Point 42 is distance 0.15 away
 *   18: 0.23,   // Point 18 is distance 0.23 away
 *   7: 0.31     // Point 7 is distance 0.31 away
 * };
 * ```
 */
export type LayerNode = Record<number, number>;

/**
 * A tuple representing a point with its distance from a query.
 *
 * Used throughout the search algorithms to track candidates and results.
 * The first element is the distance, the second is the point's index.
 * This format allows efficient sorting by distance.
 *
 * @example
 * ```typescript
 * // Point 25 is distance 0.42 from the query
 * const node: Node = [0.42, 25];
 *
 * // Array of nodes sorted by distance (closest first)
 * const candidates: Node[] = [
 *   [0.12, 5],   // Point 5 is closest
 *   [0.18, 12],  // Point 12 is second closest
 *   [0.25, 8]    // Point 8 is third closest
 * ];
 * ```
 */
export type Node = [distance: number, id: number];

/**
 * Result object returned by k-nearest neighbor search.
 *
 * Contains the point index, its distance from the query, and any associated metadata.
 * The metadata can be any type (specified via the generic parameter M).
 *
 * @template M The type of metadata associated with points (e.g., string, object, etc.)
 *
 * @example
 * ```typescript
 * // Results from searching for documents
 * type DocMetadata = { filename: string; category: string };
 * const results: KNNResult<DocMetadata>[] = [
 *   {
 *     id: 42,
 *     distance: 0.15,
 *     metadata: { filename: 'research.pdf', category: 'science' }
 *   },
 *   {
 *     id: 18,
 *     distance: 0.23,
 *     metadata: null // No metadata for this point
 *   }
 * ];
 * ```
 */
export type KNNResult<M = unknown> = {
	/** The unique index/ID of the point in the HNSW index */
	id: number;
	/** The distance from the query point (lower = more similar) */
	distance: number;
	/** Optional metadata associated with this point */
	metadata: M | null;
};
