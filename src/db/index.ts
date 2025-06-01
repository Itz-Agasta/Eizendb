import type { SetSDK } from "hollowdb";
import {
	decodeLayerNode,
	decodePoint,
	encodeLayerNode,
	encodePoint,
} from "../codec";
import type { Graph, LayerNode, Point } from "../types";
import { keys, safeParse } from "./common";
import type { DBInterface } from "./interfaces";

/**
 * EizenMemory - A distributed memory implementation for vector similarity search
 *
 * This class provides a hierarchical navigation structure for approximate nearest neighbor search.
 * It uses layered graphs where higher layers contain fewer, more connected nodes for efficient search.
 *
 * Key concepts:
 * - Points: Vector data stored with unique indices
 * - Layers: Hierarchical levels of the graph structure
 * - Neighbors: Connected nodes in each layer forming the searchable graph
 * - Entry Point (EP): Starting node for search operations
 *
 * @template M - Type for optional metadata associated with points
 */
export class EizenMemory<M = unknown> implements DBInterface<M> {
	client: SetSDK<string>;

	/**
	 * Deploy a new contract for this database instance
	 *
	 * @param initialState - Initial state configuration for the contract
	 * @param source - ( Optional) source transaction ID for contract deployment
	 * @returns The deployed contract transaction ID
	 */
	async deploy(
		initialState: Awaited<ReturnType<typeof this.client.getState>>,
		source = "",
	) {
		const { contractTxId } = await this.client.warp.deployFromSourceTx({
			wallet: this.client.signer,
			srcTxId: source,
			initState: JSON.stringify(initialState),
		});

		return contractTxId;
	}

	constructor(client: SetSDK<string>) {
		this.client = client;
	}

	// === Entry Point Management ===

	/**
	 * Get the current entry point index for search operations
	 * The entry point is the starting node for navigating the graph structure
	 */
	async get_ep(): Promise<number | null> {
		const ep = await this.client.get(keys.ep);
		return ep === null ? null : Number.parseInt(ep);
	}

	/**
	 * Set the entry point for search operations
	 * Should typically be a well-connected node in the highest layer
	 */
	async set_ep(ep: number): Promise<void> {
		await this.client.set(keys.ep, ep.toString());
	}

	// === Point (Vector) Operations ===

	/**
	 * Retrieve a single point (vector) by its index
	 *
	 * @param idx - Unique identifier for the point
	 * @returns The point data
	 * @throws Error if point doesn't exist or has no value
	 */
	async get_point(idx: number): Promise<Point> {
		const data = await this.client.get(keys.point(idx));
		if (!data) {
			throw new Error(`No point with index ${idx}`);
		}
		const point = decodePoint(data);
		if (!point.v) {
			throw new Error(`Point at index ${idx} has no value`);
		}
		return point.v;
	}

	/**
	 * Retrieve multiple points in a single operation for better performance
	 *
	 * @param idxs - Array of point indices to retrieve
	 * @returns Array of points in the same order as input indices
	 * @throws Error if any point is missing or invalid
	 */
	async get_points(idxs: number[]): Promise<Point[]> {
		if (idxs.length === 0) return [];
		const datas = await this.safe_get_many(idxs.map((idx) => keys.point(idx)));

		// Validate all points exist before processing
		const nullPos = datas.indexOf(null);
		if (nullPos !== -1) {
			throw new Error(`No point with index ${idxs[nullPos]}`);
		}

		// Decode all points and validate they have values
		const points = datas.map((data, i) => {
			if (data === null) {
				throw new Error(`No data for point at index ${idxs[i]}`);
			}
			return decodePoint(data);
		});

		return points.map((point, i) => {
			if (!point.v) {
				throw new Error(`Point at index ${idxs[i]} has no value`);
			}
			return point.v;
		});
	}

	/**
	 * Add a new point to the database and assign it the next available index
	 *
	 * @param q - Point data to store
	 * @returns The assigned index for this point
	 */
	async new_point(q: Point): Promise<number> {
		const idx = await this.get_datasize();

		const point = encodePoint({ v: q, idx });
		await this.client.set(keys.point(idx), point);
		await this.client.set(keys.points, (idx + 1).toString());

		return idx;
	}

	// === Graph Structure Queries ===

	/**
	 * Get the total number of layers in the hierarchical structure
	 * Higher layers have fewer, more connected nodes for efficient search
	 */
	async get_num_layers(): Promise<number> {
		const numLayers = await this.client.get(keys.layers);
		return numLayers ? Number.parseInt(numLayers) : 0;
	}

	/**
	 * Get the total number of points stored in the database
	 * This represents the next available index for new points
	 */
	async get_datasize(): Promise<number> {
		const datasize = await this.client.get(keys.points);
		return datasize ? Number.parseInt(datasize) : 0;
	}

	// === Neighbor (Graph Edge) Operations ===

	/**
	 * Get the neighbor connections for a specific node in a specific layer
	 *
	 * @param layer - Layer level (0 = base layer, higher = more sparse)
	 * @param idx - Node index within that layer
	 * @returns The neighbor connections for this node
	 * @throws Error if node doesn't exist or has no neighbors
	 */
	async get_neighbor(layer: number, idx: number): Promise<LayerNode> {
		const data = await this.client.get(keys.neighbor(layer, idx));
		if (!data) {
			throw new Error(`No neighbors at layer ${layer}, index ${idx}"`);
		}
		const node = decodeLayerNode(data);
		if (!node.neighbors) {
			throw new Error(`Node at layer ${layer}, index ${idx} has no neighbors`);
		}
		return node.neighbors;
	}

	/**
	 * Get neighbor connections for multiple nodes in a layer (batch operation)
	 *
	 * @param layer - Layer level to query
	 * @param idxs - Array of node indices to retrieve
	 * @returns Graph object mapping node indices to their neighbor lists
	 * @throws Error if any node is missing or invalid
	 */
	async get_neighbors(layer: number, idxs: number[]): Promise<Graph> {
		const datas = await this.safe_get_many(
			idxs.map((idx) => keys.neighbor(layer, idx)),
		);

		// Validate all nodes exist
		const nullPos = datas.indexOf(null);
		if (nullPos !== -1) {
			throw new Error(
				`No neighbors at layer ${layer}, index ${idxs[nullPos]}"`,
			);
		}

		// Decode and validate all nodes
		const nodes = datas.map((data, i) => {
			if (data === null) {
				throw new Error(
					`No data for neighbor at layer ${layer}, index ${idxs[i]}`,
				);
			}
			return decodeLayerNode(data);
		});

		const neighbors = nodes.map((node, i) => {
			if (!node.neighbors) {
				throw new Error(
					`Node at layer ${layer}, index ${idxs[i]} has no neighbors`,
				);
			}
			return node.neighbors;
		});

		return Object.fromEntries(idxs.map((idx, i) => [idx, neighbors[i]]));
	}

	/**
	 * Create or update neighbor connections for a single node
	 *
	 * @param layer - Layer level where the node exists
	 * @param idx - Node index to update
	 * @param node - New neighbor connections for this node
	 */
	async upsert_neighbor(
		layer: number,
		idx: number,
		node: LayerNode,
	): Promise<void> {
		const data = encodeLayerNode({
			idx,
			level: layer,
			neighbors: node,
		});
		await this.client.set(keys.neighbor(layer, idx), data);
	}

	/**
	 * Batch update neighbor connections for multiple nodes in a layer
	 * More efficient than individual upsert operations for bulk updates
	 *
	 * @param layer - Layer level to update
	 * @param nodes - Graph mapping node indices to their new neighbor connections
	 */
	async upsert_neighbors(layer: number, nodes: Graph): Promise<void> {
		await this.safe_set_many(
			Object.keys(nodes).map((idx) => {
				const i = Number.parseInt(idx);
				const key = keys.neighbor(layer, i);
				const value = encodeLayerNode({
					idx: i,
					level: layer,
					neighbors: nodes[i],
				});

				return [key, value];
			}),
		);
	}

	/**
	 * Initialize a new layer and add a node with empty neighbors
	 * This creates a new level in the hierarchical structure
	 *
	 * WARNING: Concurrent calls may cause race conditions in layer counting
	 *
	 * @param idx - Index of the node to add to the new layer
	 */
	async new_neighbor(idx: number): Promise<void> {
		const l = await this.get_num_layers();
		await this.upsert_neighbor(l, idx, {});

		// NOTE: if `new_neighbor` is run in parallel,
		// this might cause a race-condition
		await this.client.set(keys.layers, (l + 1).toString());
	}

	// === Metadata Operations ===

	/**
	 * Retrieve optional metadata associated with a point
	 *
	 * @param idx - Point index
	 * @returns Metadata object or null if none exists
	 */
	async get_metadata(idx: number): Promise<M | null> {
		const data = await this.client.get(keys.metadata(idx));
		return safeParse<M>(data);
	}

	/**
	 * Retrieve metadata for multiple points
	 * Returns null for points that have no metadata
	 *
	 * @param idxs - Array of point indices
	 * @returns Array of metadata objects (or null) in same order as input
	 */
	async get_metadatas(idxs: number[]): Promise<(M | null)[]> {
		// const datas =
		return Promise.all(idxs.map((idx) => this.get_metadata(idx)));
	}

	/**
	 * Store metadata for a specific point
	 *
	 * @param idx - Point index
	 * @param data - Metadata to associate with this point
	 */
	async set_metadata(idx: number, data: M): Promise<void> {
		await this.client.set(keys.metadata(idx), JSON.stringify(data));
	}

	// === Private Utility Methods ===

	/**
	 * Safely retrieve multiple keys with automatic request splitting
	 *
	 * Base-DB has transaction size limits. This method automatically splits
	 * large requests into smaller chunks when the limit is exceeded.
	 * Uses recursive binary splitting on errors.
	 */
	private async safe_get_many(keys: string[]): Promise<(string | null)[]> {
		try {
			return await this.client.getMany(keys);
		} catch (err) {
			// TODO: check error type
			const half = Math.floor(keys.length >> 1);

			// prettier-ignore
			return await Promise.all([
				this.safe_get_many(keys.slice(0, half)),
				this.safe_get_many(keys.slice(half)),
			]).then((results) => results.flat());
		}
	}

	/**
	 * Safely set multiple key-value pairs with automatic request splitting
	 *
	 * Similar to safe_get_many, this handles transaction size limits by
	 * recursively splitting large batches into smaller ones.
	 */
	private async safe_set_many(
		entries: [key: string, value: string][],
	): Promise<void> {
		try {
			await this.client.setMany(
				entries.map((e) => e[0]),
				entries.map((e) => e[1]),
			);
		} catch (err) {
			//TODO: check error type
			const half = Math.floor(entries.length >> 1);

			// prettier-ignore
			await Promise.all([
				this.safe_set_many(entries.slice(0, half)),
				this.safe_set_many(entries.slice(half)),
			]).then((results) => results.flat());
		}
	}

	toString() {
		return "yay, EizenDB Set with Protobufs";
	}
}
