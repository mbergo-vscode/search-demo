import { Smartphone, Globe, Zap, Search, BrainCircuit, Layers, Database, Server, Filter, BarChart, HardDrive, Cpu, FileText, ArrowLeftRight } from 'lucide-react';
import { PipelineNodeDef, PipelineEdgeDef, NodeDetail } from './types';

// Coordinate System:
// Left: User/Query | Center: Serving/Logic | Right: Data/Indexing

export const NODES: PipelineNodeDef[] = [
  // --- Frontend / Entry ---
  { id: 'user_device', label: 'User Client', x: 50, y: 100, icon: Smartphone, description: 'Multi-tenant client app sending queries.', category: 'frontend' },
  { id: 'api_gateway', label: 'API Gateway', x: 250, y: 100, icon: Globe, description: 'Auth, Rate Limiting, & Routing.', category: 'frontend' },

  // --- Online Serving (The Query Pipeline) ---
  { id: 'redis_cache', label: 'Redis Cache', x: 250, y: 250, icon: Zap, description: 'Token Bucket & Result Caching.', category: 'serving' },
  { id: 'query_understanding', label: 'Query Processor', x: 450, y: 100, icon: Filter, description: 'Rewrite, Expand, Spellcheck.', category: 'serving' },
  { id: 'hybrid_retriever', label: 'Hybrid Retrieval', x: 650, y: 100, icon: Search, description: 'Merge BM25 + Vector Scores.', category: 'serving' },
  { id: 'ranking_engine', label: 'LTR Ranker', x: 850, y: 100, icon: BarChart, description: 'Learning-to-Rank Re-scoring.', category: 'serving' },
  { id: 'rag_orchestrator', label: 'RAG GenAI', x: 1050, y: 100, icon: BrainCircuit, description: 'LLM Synthesis & Grounding.', category: 'serving' },

  // --- Data Engineering (The Indexing Pipeline) ---
  { id: 'kafka_backbone', label: 'Apache Kafka', x: 450, y: 400, icon: Layers, description: 'Event Streaming Backbone.', category: 'data_engineering' },
  { id: 'flink_processor', label: 'Apache Flink', x: 650, y: 400, icon: ArrowLeftRight, description: 'Real-time Enrichment & Windowing.', category: 'data_engineering' },
  
  // --- Storage / Indices ---
  { id: 'data_sources', label: 'Data Sources', x: 250, y: 550, icon: Database, description: 'DBs, PDFs, Crawlers.', category: 'storage' },
  { id: 'search_index', label: 'Lexical Index', x: 850, y: 300, icon: FileText, description: 'Inverted Index (Lucene/Solr).', category: 'storage' },
  { id: 'vector_db', label: 'Vector Memory', x: 850, y: 500, icon: Server, description: 'ANN Index (HNSW).', category: 'storage' },
  { id: 'analytics_dw', label: 'Analytics DW', x: 1050, y: 400, icon: HardDrive, description: 'Offline Eval & Metrics.', category: 'storage' },
];

export const EDGES: PipelineEdgeDef[] = [
  // Query Flow
  { from: 'user_device', to: 'api_gateway', activeInFlow: true, payloadInfo: 'HTTP GET /search' },
  { from: 'api_gateway', to: 'redis_cache', activeInFlow: true, payloadInfo: 'Check Cache / Rate Limit' },
  { from: 'redis_cache', to: 'query_understanding', activeInFlow: true, payloadInfo: 'Cache Miss -> Process' },
  
  // Retrieval Flow (Fork)
  { from: 'query_understanding', to: 'hybrid_retriever', activeInFlow: true, payloadInfo: 'Expanded Query' },
  { from: 'hybrid_retriever', to: 'search_index', activeInFlow: true, payloadInfo: 'Lexical Query (BM25)' },
  { from: 'hybrid_retriever', to: 'vector_db', activeInFlow: true, payloadInfo: 'Dense Vector (KNN)' },
  
  // Ranking & RAG
  { from: 'hybrid_retriever', to: 'ranking_engine', activeInFlow: true, payloadInfo: 'Top-N Candidates' },
  { from: 'ranking_engine', to: 'rag_orchestrator', activeInFlow: true, payloadInfo: 'Top-K Ranked Docs' },
  { from: 'rag_orchestrator', to: 'api_gateway', activeInFlow: true, payloadInfo: 'Generated Answer' },

  // Indexing Flow (Backbone)
  { from: 'data_sources', to: 'kafka_backbone', activeInFlow: true, payloadInfo: 'CDC / Ingest' },
  { from: 'kafka_backbone', to: 'flink_processor', activeInFlow: true, payloadInfo: 'Raw Events' },
  { from: 'flink_processor', to: 'search_index', activeInFlow: true, payloadInfo: 'Tokenized Docs' },
  { from: 'flink_processor', to: 'vector_db', activeInFlow: true, payloadInfo: 'Embeddings' },
  
  // Feedback Loop
  { from: 'api_gateway', to: 'kafka_backbone', activeInFlow: false, payloadInfo: 'Clickstream Logs' },
  { from: 'kafka_backbone', to: 'analytics_dw', activeInFlow: false, payloadInfo: 'Training Data' },
];

export const NODE_DETAILS: Record<string, NodeDetail> = {
  user_device: {
    title: 'Client / User Interface',
    subtitle: 'Entry Point & Telemetry',
    content: `## Search Relevance Signals
The client is not just for display; it is the primary source of **Implicit Feedback** signals used for Ranking Tuning.

### Tracking Signals
*   **CTR (Click-Through Rate)**: Ratio of clicks to impressions.
*   **Dwell Time**: Time spent on a result. Long dwell time = Good result.
*   **Pogo-sticking**: Clicking a result, then quickly navigating back. Strong negative signal (Bad result).
*   **Zero-Result Rate**: Queries that returned no items. Used to identify missing content.

### Autosuggest Strategy (Performance)
*   **< 3 Chars**: **Prefix Indexing** (Trie/FST) from Redis. No network trip to search engine. Latency < 10ms.
*   **> 3 Chars**: Triggers full pipeline (Infix matching, Fuzzy search).`,
    algorithms: ['Debounce', 'Telemetry Tracking', 'Prefix Trie'],
    techStack: ['React', 'WebSockets', 'OpenTelemetry'],
    kpis: ['P99 Latency', 'Search Abandonment Rate'],
    crossDomainImpact: {
        inputs: [{ source: "API Gateway", benefit: "Receives processed results" }],
        outputs: [{ target: "Kafka", improvement: "Sends interaction logs for LTR" }]
    }
  },
  api_gateway: {
    title: 'API Gateway',
    subtitle: 'Scalability & API Integration',
    content: `## Distributed Systems Patterns
The gateway ensures reliability and scalability of the search platform.

### Reliability Patterns
*   **Rate Limiting**: Uses **Token Bucket** or **Leaky Bucket** algorithms to smooth out traffic spikes. Essential for multi-tenant isolation.
*   **Circuit Breakers**: If the Ranking Engine starts timing out (e.g., >500ms), the breaker opens and returns raw results from the Retriever immediately, degrading quality but preserving uptime.
*   **Bulkhead Pattern**: Isolates resources (threads) for different tenants so one heavy tenant doesn't crash the system.

### API Integration
*   **Contract**: Usually REST or GraphQL.
*   **Auth**: Validates JWT and extracts \`tenant_id\` to enforce data silo policies at the query level (Filter: \`tenant_id: "acme"\`).`,
    algorithms: ['Token Bucket', 'Circuit Breaker', 'Consistent Hashing'],
    techStack: ['Kong', 'Nginx', 'Lua'],
    kpis: ['Availability (99.99%)', 'Error Rate (5xx)'],
    crossDomainImpact: {
        inputs: [{ source: "User Device", benefit: "Raw Requests" }],
        outputs: [{ target: "Redis", improvement: "Reduces backend load via cache" }]
    }
  },
  redis_cache: {
    title: 'Redis Cache',
    subtitle: 'Scalability & Performance',
    content: `## Caching Strategies for Search
Search workloads follow a **Zipfian Distribution** (Power Law) - 20% of queries account for 80% of traffic.

### Caching Layers
1.  **Result Cache**: Stores full JSON response. TTL: 5-10 mins. Key: \`hash(query + filters + tenant)\`.
2.  **Filter Cache (Bitsets)**: Caches the list of docIDs for common filters (e.g., \`category:shoes\`). Used to quickly intersect with query results.
3.  **Vector Cache**: Caches nearest neighbors for common vector embeddings to avoid expensive HNSW traversals.

### Eviction Policies
*   **LRU (Least Recently Used)**: Standard for search.
*   **LFU (Least Frequently Used)**: Better for "Top of Funnel" persistent queries.`,
    algorithms: ['LRU Eviction', 'Consistent Hashing', 'Bloom Filters'],
    techStack: ['Redis Cluster', 'Resp Protocol'],
    kpis: ['Cache Hit Ratio (>30%)', 'Latency (<2ms)'],
    crossDomainImpact: {
        inputs: [{ source: "API Gateway", benefit: "High throughput requests" }],
        outputs: [{ target: "Query Processor", improvement: "Only forwards unique tail queries" }]
    }
  },
  query_understanding: {
    title: 'Query Processor',
    subtitle: 'Query Understanding & Rewriting',
    content: `## From "User Speak" to "Engine Speak"
Raw user queries are often ambiguous or sparse. This stage fixes them.

### Query Rewriting Techniques
*   **Normalization**: Lowercasing, removing accents, stemming ("running" -> "run").
*   **Stop Word Removal**: Removing "the", "and", "a" (though modern transformers often keep them for context).
*   **Spell Check**: Uses **Levenshtein Edit Distance** or SymSpell to fix typos.
*   **Entity Recognition (NER)**: Identifies parts of speech. 
    *   *Input*: "cheap nike shoes red"
    *   *Output*: \`brand:nike\`, \`category:shoes\`, \`color:red\`, \`sort:price_asc\`.

### Query Expansion
*   **Synonyms**: Maps "laptop" -> "notebook", "computer".
*   **Vector Expansion**: Adds related terms found in the vector space to the boolean query to increase **Recall**.`,
    algorithms: ['Levenshtein Distance', 'NER (Named Entity Recognition)', 'Stemming (Porter/Snowball)'],
    techStack: ['Python', 'Spacy', 'HuggingFace'],
    kpis: ['Zero-Result Reduction', 'Query Parsing Latency'],
    crossDomainImpact: {
        inputs: [{ source: "Redis", benefit: "Uncached queries" }],
        outputs: [{ target: "Hybrid Retriever", improvement: "Structured Boolean + Vector query" }]
    }
  },
  search_index: {
    title: 'Lexical Index (Lucene)',
    subtitle: 'Data Indexing & Search Clusters',
    content: `## Deep Dive: Apache Lucene Internals
The industry standard for Lexical Search (Elasticsearch/Solr/OpenSearch).

### The Inverted Index
Data structure mapping **Terms** to **Postings Lists** (lists of Document IDs).
*   *Term*: "Search" -> *Docs*: [1, 5, 88, 92]
*   Allows O(1) or O(log N) lookups for exact keywords.

### Distributed Indexing
*   **Sharding**: Splitting the index into N pieces (Shards) distributed across nodes. allows horizontal scaling of volume.
*   **Replication**: Copying shards to M nodes for High Availability and throughput (scaling reads).
*   **Segments**: Lucene indices are composed of immutable segments. When a doc is updated, it's marked "deleted" in a bitmap and a new version is written to a new segment. Background **Merging** compacts these segments.

### Scoring: BM25 (Best Matching 25)
An evolution of TF-IDF (Term Frequency - Inverse Document Frequency).
*   **TF**: How often term appears in doc. (Saturates unlike raw TF).
*   **IDF**: How rare is the term in the corpus? (Penalizes common words).
*   **Field Length Norm**: Penalizes long documents (a match in a short title is worth more than in a long body).`,
    algorithms: ['BM25', 'TF-IDF', 'Inverted Index', 'Segment Merging (LSM)'],
    techStack: ['Apache Solr', 'Elasticsearch', 'Apache Lucene'],
    kpis: ['Indexing Rate (docs/sec)', 'Segment Merge Time'],
    crossDomainImpact: {
        inputs: [{ source: "Flink", benefit: "Tokenized Documents" }],
        outputs: [{ target: "Hybrid Retriever", improvement: "High Precision Candidates" }]
    }
  },
  vector_db: {
    title: 'Vector Memory',
    subtitle: 'Vector Space Models & ANN',
    content: `## Deep Dive: Semantic Search
Solves the "Vocabulary Mismatch Problem".

### Vector Space Model
*   **Embedding Model**: Transformer (BERT) converting text to high-dimensional vectors (e.g. 768d).
*   **Similarity**: Cosine Similarity (angle) or Dot Product (magnitude).

### Indexing: HNSW
*   **Hierarchical Navigable Small World**: A multi-layer graph for Approximate Nearest Neighbor (ANN) search.
*   Trades 100% accuracy for O(log N) speed.

### Vector DB Sharding & Scalability
When the index doesn't fit in RAM (e.g., 1B vectors = ~4TB), we must shard.
*   **Horizontal Sharding**: Split vectors across N nodes.
*   **Routing Strategies**:
    *   **Scatter-Gather**: Query is sent to ALL shards. Coordinator merges Top-K. (High Recall, Higher CPU/Latency).
    *   **Segment/Tenant Routing**: Data partitioned by \`tenant_id\`. Query routed only to relevant shard. (Low Latency).
*   **IVF (Inverted File Index)**: A form of local partitioning where vectors are clustered (Voronoi cells). Search only visits nearest clusters.`,
    algorithms: ['HNSW', 'IVF-PQ', 'Cosine Similarity', 'Consistent Hashing'],
    techStack: ['Milvus', 'Qdrant', 'Faiss'],
    kpis: ['Recall@K', 'QPS (Queries Per Second)'],
    crossDomainImpact: {
        inputs: [{ source: "Flink", benefit: "Document Embeddings" }],
        outputs: [{ target: "Hybrid Retriever", improvement: "High Recall Candidates" }]
    }
  },
  hybrid_retriever: {
    title: 'Hybrid Retriever',
    subtitle: 'Search Relevance Strategy',
    content: `## Why Hybrid?
Combines **Precision** (Lexical/BM25) with **Recall** (Semantic/Vector).

### Fusion Algorithms
**RRF (Reciprocal Rank Fusion)**:
*   \`Score = 1 / (k + Rank_BM25) + 1 / (k + Rank_Vector)\`
*   Robust, zero-calibration needed.

### Building Hybrid Search (Implementation)
1.  **Parallel Execution**: Fire BM25 and Vector Search simultaneously (Scatter-Gather).
2.  **Score Normalization**:
    *   BM25 scores are unbounded (0 to infinity).
    *   Vector scores are normalized (0.0 to 1.0).
    *   *Fix*: Apply **Min-Max Scaling** or **Sigmoid** to BM25 scores before linear combination.
3.  **Windowing**:
    *   Fetch Top-100 from Vector.
    *   Fetch Top-100 from BM25.
    *   Pool is 200 docs (max).
4.  **Dedup**: Remove duplicates by DocID before passing to Ranking Engine.`,
    algorithms: ['Reciprocal Rank Fusion (RRF)', 'Min-Max Normalization', 'WAND'],
    techStack: ['Java', 'Elasticsearch Plugin'],
    kpis: ['Mean Reciprocal Rank (MRR)', 'Recall@100'],
    crossDomainImpact: {
        inputs: [{ source: "Query Processor", benefit: "Dual Query Intent" }],
        outputs: [{ target: "Ranking Engine", improvement: "Broad candidate set (Top-1000)" }]
    }
  },
  ranking_engine: {
    title: 'LTR (Learning to Rank)',
    subtitle: 'Ranking & Relevance Tuning',
    content: `## Deep Dive: LTR Pipeline
Retrieval gets the "Possible" documents. Ranking finds the "Best".

### The Model: LambdaMART
*   A gradient boosted tree approach (like XGBoost) adapted for ranking.
*   Optimizes **NDCG** directly using pairwise gradients.

### ML Model Deployment & Inference
*   **Training-Serving Skew**: Logic for features (e.g., "Click Rate last 7 days") must be identical in Spark (Training) and Java/C++ (Inference).
*   **Model Stores**: Models are versioned (v1.2.3) in S3. The engine hot-reloads new models without downtime (Canary Rollout).
*   **Inference Engines**:
    *   **ONNX Runtime**: Standard format for model portability.
    *   **Triton Inference Server**: For high-throughput GPU scoring.
*   **Feature Store**: Low-latency KV store (Redis) serving user/item features to the model in <2ms.`,
    algorithms: ['LambdaMART', 'XGBoost', 'ListNet', 'ONNX Runtime'],
    techStack: ['XGBoost', 'LightGBM', 'TensorFlow Ranking'],
    kpis: ['NDCG@10', 'P99 Inference Latency (<10ms)'],
    crossDomainImpact: {
        inputs: [{ source: "Hybrid Retriever", benefit: "Candidate Set" }],
        outputs: [{ target: "RAG Orchestrator", improvement: "Top-5 most relevant context chunks" }]
    }
  },
  kafka_backbone: {
    title: 'Apache Kafka',
    subtitle: 'Data Pipeline Integration',
    content: `## Data Consistency & Distributed Logs
Decouples the "Write" path (Indexing) from the "Read" path (Serving).

### CDC (Change Data Capture)
*   Instead of polling the DB ("SELECT * WHERE updated_at > now()"), we tail the database **WAL (Write Ahead Log)**.
*   Tools: Debezium, Maxwell.
*   Ensures we capture **Hard Deletes**, not just updates.

### Partitioning Strategy
*   Kafka topics are partitioned (sharded).
*   **Keying**: We must key messages by \`DocumentID\`. This ensures all updates for Doc #123 go to the same partition.
*   **Ordering**: Kafka guarantees ordering *within* a partition. This is critical so we don't overwrite a "New Version" with an "Old Version".`,
    algorithms: ['Log-Structured Storage', 'Consistent Hashing', 'ISR (In-Sync Replicas)'],
    techStack: ['Apache Kafka', 'Confluent Schema Registry'],
    kpis: ['Replication Lag', 'Throughput (MB/s)'],
    crossDomainImpact: {
        inputs: [{ source: "Data Sources", benefit: "Raw Change Stream" }],
        outputs: [{ target: "Flink", improvement: "Ordered, durable event stream" }]
    }
  },
  flink_processor: {
    title: 'Apache Flink',
    subtitle: 'Real-time Processing & Indexing',
    content: `## Stream Processing
Flink provides the computational power to transform raw DB rows into Searchable Documents.

### Indexing Pipeline Steps
1.  **De-normalization**: Joining related tables. (e.g., Join "Product" with "Brand" and "Inventory").
2.  **Enrichment**: Calling external APIs (e.g., Vision API for image tags).
3.  **Vectorization**: Calling Embedding Model to generate vectors.

### Exactly-Once Semantics
*   Ensures documents are not indexed twice (duplicates) or lost.
*   Uses **Chandy-Lamport** algorithm for distributed state snapshots.

### Backpressure
*   If the Vector DB is slow, Flink manages backpressure to slow down consumption from Kafka, preventing OOM (Out of Memory) crashes.`,
    algorithms: ['Watermarking', 'Tumbling Windows', 'Chandy-Lamport Snapshots'],
    techStack: ['Apache Flink', 'RocksDB', 'Java'],
    kpis: ['Processing Latency', 'Checkpoint Duration'],
    crossDomainImpact: {
        inputs: [{ source: "Kafka", benefit: "Raw Events" }],
        outputs: [{ target: "Search Indices", improvement: "Ready-to-index batches" }]
    }
  },
  analytics_dw: {
    title: 'Analytics DW',
    subtitle: 'Search Quality Metrics',
    content: `## Search Relevance Metrics
How do we know if search is "Good"?

### Offline Metrics (Golden Sets)
*   **Precision@K**: % of top-K results that are relevant.
*   **Recall**: % of *all* relevant items that were found.
*   **NDCG (Normalized Discounted Cumulative Gain)**:
    *   Measures ranking quality.
    *   Rewards relevant items at the top.
    *   Penalizes relevant items at the bottom.
*   **MRR (Mean Reciprocal Rank)**: 1 / Rank of the first relevant item. Good for "Known Item Search".

### Online Metrics (A/B Testing)
*   **CTR**: Click-through rate.
*   **Conversion Rate**: Purchases / Searches.
*   **SAT (Satisfied) Clicks**: Clicks with dwell time > 30s.`,
    algorithms: ['Counterfactual Estimation', 'Wilson Score Interval'],
    techStack: ['Snowflake', 'BigQuery', 'dbt'],
    kpis: ['NDCG', 'MAP (Mean Average Precision)'],
    crossDomainImpact: {
        inputs: [{ source: "Kafka", benefit: "Interaction Logs" }],
        outputs: [{ target: "Ranking Engine", improvement: "Training data for LTR models" }]
    }
  },
  data_sources: {
    title: 'Data Sources',
    subtitle: 'System of Record',
    content: `## Data Ingestion Challenges
The source of truth.

### Challenges
*   **Schema Drift**: Upstream team changes a column name. Pipeline breaks. Solution: Schema Registry.
*   **Throughput**: Initial "Bootstrap" (re-indexing 100M items) vs Incremental updates.
*   **Polyglot Persistence**: Handling data from SQL (Relational), NoSQL (Document), and Object Stores (Images/PDFs).`,
    algorithms: ['CDC (Change Data Capture)', 'Snapshotting'],
    techStack: ['PostgreSQL', 'Debezium', 'MongoDB'],
    kpis: ['Data Freshness', 'Source Availability'],
    crossDomainImpact: {
        inputs: [],
        outputs: [{ target: "Kafka", improvement: "Source of Truth" }]
    }
  },
  rag_orchestrator: {
    title: 'RAG Orchestrator',
    subtitle: 'Generative AI Integration',
    content: `## Retrieval-Augmented Generation
Combines Parametric Memory (LLM Weights) with Non-Parametric Memory (Search Index).

### The "RAG" Stack
1.  **Retrieve**: Get top-K chunks from Search.
2.  **Augment**: Insert chunks into the Prompt Context.
3.  **Generate**: LLM synthesizes answer.

### Challenges
*   **Context Window Limits**: We can't feed 100 documents. Ranking must be excellent to find the top 5.
*   **Hallucination**: Model making things up. Fixed by "Grounding" (citing sources).
*   **Latency**: LLMs are slow (Tokens/sec). We stream the response to the user (TTFT - Time To First Token).`,
    algorithms: ['Prompt Chaining', 'Self-Consistency', 'Token Generation'],
    techStack: ['LangChain', 'Google Gemini', 'LlamaIndex'],
    kpis: ['TTFT (Time To First Token)', 'Faithfulness Score'],
    crossDomainImpact: {
        inputs: [{ source: "Ranking Engine", benefit: "High quality context" }],
        outputs: [{ target: "API Gateway", improvement: "Natural language answer" }]
    }
  }
};