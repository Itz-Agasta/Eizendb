/**
 * Python HNSW Reference Implementation Test
 *
 * This test validates the Python reference implementation that serves as the ground truth
 * for our TypeScript HNSW implementation. The Python version generates the expected
 * results that are used in other test files (see test/data/index.ts).
 *
 * CI/CD Considerations:
 * - This test requires Python + uv to be installed
 * - In CI environments, this test is automatically skipped
 * - Use `npm run test:ci` for CI pipelines (excludes Python tests)
 * - Use `npm run test:local` for local development (includes Python tests)
 * - Use `npm run test:python` to run only the Python script
 *
 * Purpose:
 * - Validates that the Python reference implementation runs correctly
 * - Ensures consistent results between development environments
 * - Provides baseline for comparing TypeScript implementation accuracy
 */

import { execSync } from "node:child_process";
import { join } from "node:path";

describe("Python HNSW Reference Implementation", () => {
	test("should run successfully and produce expected output", () => {
		// Skip in CI environments where Python/uv might not be available
		const isCI =
			process.env.CI || process.env.GITHUB_ACTIONS || process.env.JENKINS_URL;
		if (isCI) {
			console.log("⏭️  Skipping Python test in CI environment");
			return;
		}

		// Check if uv is available before running
		try {
			execSync("uv --version", { stdio: "pipe" });
		} catch {
			console.log("⏭️  Skipping Python test - uv not available");
			return;
		}

		const pythonDir = join(__dirname, "python");

		// Run the Python script
		const output = execSync("uv run main.py", {
			cwd: pythonDir,
			encoding: "utf-8",
		});

		// Parse the output lines
		const lines = output.trim().split("\n");

		// Validate that we get 10 results (K=10)
		expect(lines).toHaveLength(10);

		// Parse each line as [similarity, index]
		const results = lines.map((line) => {
			const [similarity, index] = line.split(" ");
			return [Number.parseFloat(similarity), Number.parseInt(index, 10)];
		});

		// Validate the first result is perfect match with itself
		expect(results[0][0]).toBe(1.0);
		expect(results[0][1]).toBe(0);

		// Validate that similarities are in descending order
		for (let i = 1; i < results.length; i++) {
			expect(results[i][0]).toBeLessThanOrEqual(results[i - 1][0]);
		}

		// Validate similarity scores are between 0 and 1
		for (const [similarity] of results) {
			expect(similarity).toBeGreaterThanOrEqual(0);
			expect(similarity).toBeLessThanOrEqual(1);
		}

		// Validate indices are valid
		for (const [, index] of results) {
			expect(index).toBeGreaterThanOrEqual(0);
			expect(index).toBeLessThan(100); // Since we only insert 100 vectors
		}

		console.log("Python HNSW test results:");
		results.forEach(([similarity, index], i) => {
			console.log(
				`  ${i + 1}. Similarity: ${similarity.toFixed(4)}, Index: ${index}`,
			);
		});
	}, 30000); // 30 second timeout for Python execution
});
