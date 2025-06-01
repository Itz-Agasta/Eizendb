# Eizen - HNSW Vector Database on Arweave

## Overview

Eizen is a high-performance vector database backend built on Arweave that implements the Hierarchical Navigable Small Worlds (HNSW) algorithm for approximate nearest neighbor search. It provides efficient vector storage, similarity search, and metadata management with blockchain-based persistence.

## Key Features

- **HNSW Algorithm**: State-of-the-art approximate nearest neighbor search with O(log N) complexity
- **Blockchain Storage**: Persistent vector storage on Arweave with HollowDB integration
- **Protobuf Encoding**: Efficient serialization for optimal storage and network transfer
- **Metadata Support**: Rich metadata attachment to vectors for enhanced search capabilities
- **Flexible Interface**: Database-agnostic interface supporting multiple storage backends
- **Scalable**: Handles millions of high-dimensional vectors efficiently

## HNSW Implementation

### Algorithm Overview

The Hierarchical Navigable Small Worlds (HNSW) algorithm creates a multi-layer graph structure:

- **Layer 0**: Contains all vectors with dense local connections
- **Higher Layers**: Contain progressively fewer vectors with long-range connections
- **Search Process**: Navigate from top to bottom for logarithmic search complexity

### Core Components

#### 1. HNSW Class (`src/hnsw.ts`)

The main implementation containing:

- **`insert()`**: Add new vectors with metadata (Algorithm 1)
- **`knn_search()`**: Find k nearest neighbors (Algorithm 5)
- **`search_layer()`**: Core search primitive (Algorithm 2)
- **`select_neighbors()`**: Neighbor selection heuristic (Algorithm 4)

#### 2. Database Interface (`src/db/interfaces/`)

Abstraction layer supporting different storage backends:

- Point storage and retrieval
- Graph structure management
- Metadata operations
- Entry point tracking

#### 3. Utility Functions (`src/utils/`)

Mathematical operations and data structures:

- Distance functions (cosine, euclidean)
- Priority queues for search algorithms
- Vector operations (dot product, norm)

### Usage Example

```typescript
import { HNSW } from "./src/hnsw";
import { EizenMemory } from "./src/db";

// Initialize database and HNSW index
const database = new EizenMemory();
const hnsw = new HNSW(
  database,
  16, // M: connections per node
  200, // ef_construction: build quality
  50 // ef_search: search quality
);

// Insert vectors with metadata
await hnsw.insert([0.1, 0.2, 0.3, 0.4], {
  filename: "document.pdf",
  category: "research",
});

// Search for similar vectors
const results = await hnsw.knn_search([0.15, 0.25, 0.35, 0.45], 5);
console.log(results);
// Output: [{ id: 0, distance: 0.1, metadata: { ... } }, ...]
```

### Parameter Tuning Guide

| Parameter           | Purpose                    | Recommended Range      | Impact                                |
| ------------------- | -------------------------- | ---------------------- | ------------------------------------- |
| **M**               | Connections per node       | 5-48 (default: 16)     | Higher = better quality, more memory  |
| **ef_construction** | Build candidate list size  | 100-400 (default: 200) | Higher = better graph, slower build   |
| **ef_search**       | Search candidate list size | >= K (default: 50)     | Higher = better recall, slower search |

### Performance Characteristics

- **Time Complexity**: O(log N) for both insertion and search
- **Space Complexity**: O(M × N) where M is average connections per node
- **Scalability**: Efficiently handles millions of high-dimensional vectors
- **Distance Function**: Currently uses cosine distance (configurable)

## Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/Itz-Agasta/Eizendb.git
   ```

2. Navigate to the project directory:

   ```bash
   cd Eizendb
   ```

3. Install dependencies:

   ```bash
   pnpm install
   ```

4. Build the project:
   ```bash
   pnpm build
   ```

## Development

### Testing

The project includes comprehensive tests covering different aspects:

```bash
# Run all tests (local development)
npm run test:local

# Run tests for CI (excludes Python tests)
npm run test:ci

# Run only TypeScript tests
npm test

# Run specific test suites
npm run test:heap     # Heap data structure tests
npm run test:hnsw     # Core HNSW algorithm tests
npm run test:proto    # Protocol buffer serialization tests
npm run test:eizen    # Full blockchain integration tests

# Run Python reference implementation
npm run test:python # After setting up the python env.
```

#### Test Structure

- **`test/heap.test.ts`**: Tests the heap data structure used in HNSW
- **`test/proto.test.ts`**: Tests Protocol Buffer serialization
- **`test/hnsw.test.ts`**: Tests the core HNSW algorithm with different storage backends
- **`test/eizen.test.ts`**: Integration tests with Arweave blockchain and Redis
- **`test/python.test.ts`**: Python reference implementation validation (skipped in CI)

#### Python Reference Implementation

The `test/python/` directory contains a Python implementation of the HNSW algorithm that serves as the ground truth for validation. This requires Python and `uv` to be installed:

```bash
# Setup Python environment (one-time)
cd test/python
uv sync

# Run Python reference implementation
uv run main.py
```

**Note**: For some reason `Arlocal` doest work with `pnpm`, so the `eizen.test.ts` will fail. Thats why for testing `npm` is recommended. If you are using pnpm run:

```bash
pnpm run use-npm
```

## Architecture

```
src/
├── hnsw.ts              # Main HNSW implementation
├── types/               # TypeScript type definitions
├── utils/               # Utility functions and data structures
├── db/                  # Database abstraction layer
│   ├── interfaces/      # Database interface definitions
│   ├── index.ts         # EizenMemory implementation
│   └── operations/      # Database operation modules
├── codec.ts             # Protobuf encoding/decoding
└── index.ts             # Main exports
```

## References

- [HNSW Paper](https://arxiv.org/pdf/1603.09320.pdf): "Efficient and robust approximate nearest neighbor search using Hierarchical Navigable Small World graphs" by Malkov & Yashunin
- [Arweave](https://www.arweave.org/): Permanent data storage blockchain

## Contributing

We welcome contributions! Please see our contributing guidelines:

1. Fork the repository
2. Create a feature branch
3. Add comprehensive tests
4. Update documentation
5. Submit a pull request

### Code Style

- Use TypeScript for type safety
- Follow existing code formatting (Biome)
- Add JSDoc comments for public APIs
- Include performance considerations in code reviews

## License

This project is licensed under the MIT License. See the LICENSE file for details.
