import type { Redis } from "ioredis";
import {
	decodeLayerNode,
	decodePoint,
	encodeLayerNode,
	encodePoint,
} from "../../src/codec";
import { keys, safeParse } from "../../src/db/common";
import type { DBInterface } from "../../src/db/interfaces";
import type { Graph, LayerNode, Point } from "../../src/types";

export class RedisMemory<M = unknown> implements DBInterface<M> {
	client: Redis;

	constructor(client: Redis) {
		this.client = client;
	}

	async get_ep(): Promise<number | null> {
		const ep = await this.client.get(keys.ep);
		return ep === null ? null : Number.parseInt(ep);
	}

	async set_ep(ep: number): Promise<void> {
		await this.client.set(keys.ep, ep.toString());
	}

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

	async get_points(idxs: number[]): Promise<Point[]> {
		if (idxs.length === 0) return [];
		const datas = await this.client.mget(idxs.map((idx) => keys.point(idx)));

		// see if there is a null value in there
		const nullPos = datas.indexOf(null);
		if (nullPos !== -1) {
			throw new Error(`No point with index ${idxs[nullPos]}`);
		}

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

	async new_point(q: Point): Promise<number> {
		const idx = await this.get_datasize();
		const point = encodePoint({ v: q, idx });
		await this.client.set(keys.point(idx), point);
		await this.client.set(keys.points, (idx + 1).toString());
		return idx;
	}

	async get_num_layers(): Promise<number> {
		const numLayers = await this.client.get(keys.layers);
		return numLayers ? Number.parseInt(numLayers) : 0;
	}

	async get_datasize(): Promise<number> {
		const datasize = await this.client.get(keys.points);
		return datasize ? Number.parseInt(datasize) : 0;
	}

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

	async get_neighbors(layer: number, idxs: number[]): Promise<Graph> {
		const datas = await this.client.mget(
			idxs.map((idx) => keys.neighbor(layer, idx)),
		);

		// see if there is a null value in there
		const nullPos = datas.indexOf(null);
		if (nullPos !== -1) {
			throw new Error(
				`No neighbors at layer ${layer}, index ${idxs[nullPos]}"`,
			);
		}

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

	async upsert_neighbors(layer: number, nodes: Graph): Promise<void> {
		await this.client.mset(
			Object.keys(nodes).flatMap((idx) => {
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

		await Promise.all(
			Object.keys(nodes).map((idx) => {
				const i = Number.parseInt(idx);
				return this.upsert_neighbor(layer, i, nodes[i]);
			}),
		);
	}

	async new_neighbor(idx: number): Promise<void> {
		const l = await this.get_num_layers();
		await this.upsert_neighbor(l, idx, {});

		// NOTE: if `new_neighbor` is run in parallel,
		// this might cause a race-condition
		await this.client.set(keys.layers, (l + 1).toString());
	}

	async get_metadata(idx: number): Promise<M | null> {
		const data = await this.client.get(keys.metadata(idx));
		return safeParse<M>(data);
	}

	async get_metadatas(idxs: number[]): Promise<(M | null)[]> {
		const datas = await this.client.mget(idxs.map((idx) => keys.metadata(idx)));
		const parsed = datas.map((data) => safeParse<M>(data));
		return parsed;
	}

	async set_metadata(idx: number, data: M): Promise<void> {
		await this.client.set(keys.metadata(idx), JSON.stringify(data));
	}

	toString() {
		return "Redis with Protobufs";
	}
}
