import type { Graph, LayerNode, Point } from "../../types";

/**
 * Database interface for HNSW (Hierarchical Navigable Small World) implementation.
 *
 * This interface abstracts the storage layer for the HNSW algorithm, allowing
 * different backends (in-memory, file-based, database, etc.) to be used.
 *
 * @template M - Type for point metadata (optional, defaults to unknown)
 */
export interface DBInterface<M = unknown> {
	/////////////// GRAPH STRUCTURE OPERATIONS ///////////////

	/**
	 * Initializes a new layer in the HNSW graph structure.
	 * Creates an empty neighbor map for a point at the specified index.
	 */
	new_neighbor(idx: number): Promise<void>;

	/**
	 * Retrieves all neighbors of a specific node in a given layer.
	 */
	get_neighbor(layer: number, idx: number): Promise<LayerNode>;

	/**
	 * Batch retrieval of neighbors for multiple nodes in a layer.
	 * More efficient than multiple individual get_neighbor calls.
	 */
	get_neighbors(layer: number, idxs: number[]): Promise<Graph>;

	/**
	 * Updates or inserts neighbor connections for a node.
	 * Creates the connection if it doesn't exist, updates if it does.
	 */
	upsert_neighbor(layer: number, idx: number, node: LayerNode): Promise<void>;

	/**
	 * Batch update/insert of neighbor connections.
	 * More efficient than multiple individual upsert_neighbor calls.
	 */
	upsert_neighbors(layer: number, nodes: Graph): Promise<void>;

	/**
	 * Returns the total number of layers in the HNSW structure.
	 * Each layer represents a different level of the hierarchical graph.
	 */
	get_num_layers(): Promise<number>;

	/////////////// POINT STORAGE OPERATIONS ///////////////

	/**
	 * Adds a new vector point to the database.
	 * Points are assigned sequential indices starting from 0.
	 *
	 * @returns The assigned index for the new point
	 */
	new_point(q: Point): Promise<number>;

	/**
	 * Retrieves a single point by its index.
	 */
	get_point(idx: number): Promise<Point>;

	/**
	 * Batch retrieval of multiple points.
	 *
	 * @throws Error if any point doesn't exist at the given indices
	 */
	get_points(idxs: number[]): Promise<Point[]>;

	/**
	 * Returns the total number of points stored in the database.
	 * Equivalent to the next index that would be assigned to a new point.
	 */
	get_datasize(): Promise<number>;

	/////////////// ENTRY POINT MANAGEMENT ///////////////

	/**
	 * Gets the index of the current entry point for HNSW search.
	 * The entry point is typically the node in the highest layer.
	 *
	 * @returns Entry point index, or null if no points have been added
	 */
	get_ep(): Promise<number | null>;

	/**
	 * Sets the entry point for HNSW search operations.
	 * This should be a node that exists in the highest layer.
	 */
	set_ep(ep: number): Promise<void>;

	/////////////// METADATA OPERATIONS ///////////////

	/**
	 * Retrieves application-specific metadata for a point.
	 * Metadata can be any additional information associated with a vector.
	 *
	 * @returns Metadata object or null if no metadata exists
	 */
	get_metadata(idx: number): Promise<M | null>;

	/**
	 * Batch retrieval of metadata for multiple points.
	 * Returns array with same length as input, with null for missing metadata.
	 */
	get_metadatas(idxs: number[]): Promise<(M | null)[]>;

	/**
	 * Associates metadata with a point.
	 * Overwrites existing metadata if present.
	 */
	set_metadata(idx: number, data: M): Promise<void>;
}
