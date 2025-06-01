import { readFileSync } from "node:fs";
import ArLocal from "arlocal";
import { SetSDK } from "hollowdb";
import { Redis } from "ioredis";
import {
	LoggerFactory,
	WarpFactory,
	defaultCacheOptions,
} from "warp-contracts";
import { DeployPlugin } from "warp-contracts-plugin-deploy";
import { RedisCache } from "warp-contracts-redis";
import { EizenDbVector } from "../src";
import { testcases } from "./data";

type Metadata = { id: number };

/** describe.skip() --> If you want to skip this long test */
describe("Eizen Vector", () => {
	let train: number[][];
	let metadatas: Metadata[];

	let arlocal: ArLocal;
	let vectordb: EizenDbVector<Metadata>;
	const ARWEAVE_PORT = 1984;

	// testing just for 100 because that takes long enough (approx 200 seconds insert)
	const N = 100;
	const K = 10;

	beforeAll(async () => {
		try {
			console.log("Starting test setup...");

			const trainRaw = readFileSync("./test/data/data.json", "utf-8");
			train = JSON.parse(trainRaw);

			expect(Array.isArray(train)).toBe(true);
			expect(train.length).toBe(2000);

			// metadata is simply the id of each data (i.e. their index)
			// for testing purposes of course
			metadatas = train.map((_, i) => ({ id: i }));
			console.log("Data loaded successfully");

			// start arlocal
			console.log("Starting ArLocal...");
			arlocal = new ArLocal(ARWEAVE_PORT, false);
			await arlocal.start();
			console.log("ArLocal started");

			// connect to redis
			console.log("Connecting to Redis...");
			const redis = new Redis();
			expect(await redis.ping()).toBe("PONG");
			console.log("Redis connected");

			// setup warp
			console.log("Setting up Warp...");
			const warp = WarpFactory.forLocal(ARWEAVE_PORT)
				.use(new DeployPlugin())
				.useKVStorageFactory(
					(contractTxId: string) =>
						new RedisCache(
							{ ...defaultCacheOptions, dbLocation: `${contractTxId}` },
							{ client: redis },
						),
				);

			// setup wallet
			console.log("Generating wallet...");
			const wallet = await warp.generateWallet();
			const owner = wallet.jwk;

			// deploy contract
			console.log("Deploying contract...");
			const { contractTxId } = await warp.deploy(
				{
					wallet: owner,
					initState: readFileSync("./test/data/state.json", "utf-8"),
					src: readFileSync("./test/data/contract.js", "utf-8"),
					evaluationManifest: {
						evaluationOptions: {
							allowBigInt: true,
							useKVStorage: true,
						},
					},
				},
				true, // bundling disabled for local network
			);
			console.log("Contract deployed with ID:", contractTxId);

			// create sdk
			console.log("Creating SDK...");
			const sdk = new SetSDK<string>(owner, contractTxId, warp);
			LoggerFactory.INST.logLevel("none");

			// create eizen vector database
			console.log("Creating EizenDbVector...");
			vectordb = new EizenDbVector(sdk);

			LoggerFactory.INST.logLevel("none");
			console.log("Test setup completed successfully");
		} catch (error) {
			console.error("Test setup failed:", error);
			throw error;
		}
	}, 60000); // 60 second timeout

	test("deployment", async () => {
		if (!vectordb) {
			throw new Error("vectordb not initialized - beforeAll setup failed");
		}
		const contractTxId = await vectordb.sdk.warp.arweave.transactions.get(
			vectordb.sdk.contract.txId(),
		);
		expect(contractTxId).not.toBeNull();
	});

	test("insert", async () => {
		if (!vectordb) {
			throw new Error("vectordb not initialized - beforeAll setup failed");
		}
		for (let i = 0; i < N; i++) {
			const msg = `inserting point [${i + 1}/${N}]`;
			console.time(msg);
			await vectordb.insert(train[i], metadatas[i]);
			console.timeEnd(msg);
		}
		expect(await vectordb.db.get_datasize()).toBe(N);
	});

	test("KNN search", async () => {
		if (!vectordb) {
			throw new Error("vectordb not initialized - beforeAll setup failed");
		}
		const res = await vectordb.knn_search(train[0], K);
		expect(res.length).toBe(K);

		const testcase = testcases[`${N}.${K}`];
		expect(testcase.length).toBe(K);
		for (let i = 0; i < K; i++) {
			expect(testcase[i][0]).toBeCloseTo(1 - res[i].distance);
			expect(testcase[i][1]).toBe(res[i].id);
			expect(testcase[i][1]).toBe(res[i].metadata?.id);
		}
	});

	test("get vector", async () => {});

	afterAll(async () => {
		if (vectordb?.sdk) {
			const redis = vectordb.sdk.warp
				.kvStorageFactory(vectordb.sdk.contractTxId)
				.storage<Redis>();
			await redis.flushdb();
			await redis.quit();
		}

		if (arlocal) {
			await arlocal.stop().then(() => console.log("arlocal stopped"));
		}
	});
});
