import { defineConfig } from "vitest/config";

export default defineConfig({
	test: {
		testTimeout: 999_000,
		environment: "node",
		globals: true, // enables Jest-like globals (describe, it, expect)
		include: ["test/**/*.test.ts"],
		coverage: {
			provider: "v8",
			reporter: ["text", "html"],
			exclude: [
				"proto/**", // Exclude generated protobuf files
				"lib/**", // Exclude compiled output
				"test/**", // Exclude test files themselves
			],
		},
	},
});
