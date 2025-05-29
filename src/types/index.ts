/**  types used in (Hierarchical Navigable Small World) graph  */
export type Point = number[];

export type Graph = Record<number, LayerNode>;

export type LayerNode = Record<number, number>;

export type Node = [distance: number, id: number];

export type KNNResult<M = unknown> = {
	id: number;
	distance: number;
	metadata: M | null;
};
