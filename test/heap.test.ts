import Heap from "heap-js";

describe("heap", () => {
	const arr = [1, 3, 4, 2, 6, 5, 7, 9, 8];
	let heap: Heap<number>;

	test("should construct heap", () => {
		heap = new Heap();
		heap.addAll(arr);
		expect(heap.length).toBe(arr.length);
	});

	test("should give correct top (min for min-heap)", () => {
		expect(heap.top(1)[0]).toBe(Math.min(...arr));
	});

	test("should give correct bottom (max for max-heap)", () => {
		expect(heap.bottom(1)[0]).toBe(Math.max(...arr));
	});
});
