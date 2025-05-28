import { defineConfig } from "tsdown";

export default defineConfig({
	entry: ["src/index.ts"],
	format: ["cjs", "esm"],
	dts: true,
	clean: true,
	sourcemap: true,
	external: ["hollowdb", "ioredis", "warp-contracts", "warp-contracts-redis"], // Bundle external dependencies (set to false for libraries)
	platform: "node",
	outDir: "lib",
});
