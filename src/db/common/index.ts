/** KVdb key generators for different data types */
export const keys = {
	layers: "layers",
	ep: "ep",
	points: "points",
	metadata: (idx: number) => `m:${idx}` as const,
	point: (idx: number) => `${idx}` as const,
	neighbor: (layer: number, idx: number) => `${layer}__${idx}` as const,
} as const;

/** Parse JSON string, returns null for falsy values */
export function safeParse<V = unknown>(
	data: string | null | undefined,
): V | null {
	return data ? JSON.parse(data) : null;
}
