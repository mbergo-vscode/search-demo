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
    subtitle: 'Entry Point',
    content: `**Multi-Tenant Access**: The application handles strict data isolation. TenantID is injected into every request header.
    
**Autosuggest Strategy**:
*   **< 3 Chars**: Uses simple Prefix Indexing (Trie) from Redis for ultra-low latency (~10ms).
*   **> 3 Chars**: Triggers the full Search Pipeline.
    
**Telemetry**: Captures "Signals" (Clicks, Dwell Time, Good Abandon vs. Bad Abandon) to feed the Feedback Loop.`,
    algorithms: ['Debounce', 'Telemetry Tracking'],
    techStack: ['React', 'WebSockets', 'OpenTelemetry']
  },
  api_gateway: {
    title: 'API Gateway',
    subtitle: 'Security & Traffic Control',
    content: `**Role**: The bouncer of the system.
    
**Key Functions**:
*   **Authentication**: Validates JWTs to determine Tenant Context (Context filtering).
*   **Rate Limiting**: Implementation of the **Token Bucket Algorithm** (backed by Redis) to prevent noisy neighbor issues in a multi-tenant environment.
*   **Circuit Breaking**: Fails fast if the downstream Ranking Engine is overloaded.`,
    algorithms: ['Token Bucket', 'Leaky Bucket', 'Round Robin'],
    techStack: ['Kong', 'Nginx', 'Lua']
  },
  redis_cache: {
    title: 'Redis Cache',
    subtitle: 'Hot Memory & Session Store',
    content: `**Performance Layer**: Serves approx 30% of traffic without hitting the heavy search engines.

**Strategies**:
*   **Result Caching**: Stores the final JSON response for identical queries (TTL: 5 mins).
*   **Filter Caching**: Caches "Bitsets" for common filters (e.g., "Category:Electronics") to speed up Lucene filtering.
*   **Token Store**: Manages user session tokens and API quotas.

**Eviction**: Uses **LRU (Least Recently Used)** to keep only hot data in memory.`,
    algorithms: ['LRU Eviction', 'Consistent Hashing'],
    techStack: ['Redis Cluster', 'Resp Protocol'],
    kpis: ['< 2ms Latency', '35% Cache Hit Ratio']
  },
  query_understanding: {
    title: 'Query Understanding',
    subtitle: 'NLP Pre-processing',
    content: `**Goal**: Translate "user intent" into "machine query".
    
**Steps**:
1.  **Normalization**: Lowercase, accent removal.
2.  **Entity Extraction (NER)**: Detects "Nike" (Brand) vs "Shoes" (Category).
3.  **Expansion**: Uses a Synonym Graph to map "sneakers" -> "trainers" OR "athletic shoes".
4.  **Vectorization**: Converts text to dense vectors (768d) using a Transformer model (e.g., BERT) for the Semantic Retrieval leg.`,
    algorithms: ['Named Entity Recognition (NER)', 'Query Expansion', 'Spell Check (SymSpell)'],
    techStack: ['Python', 'Spacy', 'HuggingFace Transformers']
  },
  hybrid_retriever: {
    title: 'Hybrid Retrieval Engine',
    subtitle: 'Lexical + Semantic Fusion',
    content: `**The "Secret Sauce" of Modern Search**:
It combines the precision of keywords with the understanding of concepts.

**Formula**:
\`Final_Score = (α * Normalized_BM25) + (β * Normalized_Cosine)\`

*   **Lexical (BM25)**: Best for exact matches (Part Numbers, Specific Names).
*   **Semantic (Vector)**: Best for descriptions ("shoe that is good for running").
*   **Reciprocal Rank Fusion (RRF)**: A robust method to merge the two ranked lists without worrying about scale calibration.`,
    algorithms: ['Reciprocal Rank Fusion (RRF)', 'WAND (Weak AND)', 'Query Coordination'],
    techStack: ['Java', 'Elasticsearch Plugin']
  },
  search_index: {
    title: 'Lexical Index (Lucene)',
    subtitle: 'Inverted Index Storage',
    content: `**Structure**: The "Book Index" of the internet. Maps *Terms* -> *Document IDs*.

**Key Algorithm: BM25 (Best Matching 25)**:
A probabilistic retrieval model that ranks documents based on the query terms appearing in each document, heavily weighing rare terms (IDF - Inverse Document Frequency).

**Optimization**:
*   **Segments**: Immutable data files merged in background (Log-structured merge tree).
*   **Postings List**: Compressed lists of doc IDs using Frame of Reference (PFOR) encoding.`,
    algorithms: ['BM25', 'TF-IDF', 'Boolean Logic', 'Fuzzy Search (Levenshtein)'],
    techStack: ['Apache Solr', 'Elasticsearch', 'Apache Lucene']
  },
  vector_db: {
    title: 'Vector Memory',
    subtitle: 'Semantic Index',
    content: `**Structure**: Stores high-dimensional embeddings (e.g., 1536 dimensions).

**Key Algorithm: HNSW (Hierarchical Navigable Small World)**:
A graph-based algorithm. Imagine a "multi-layer highway" where you jump long distances on the top layer and refine your search on lower layers. It trades 100% accuracy for massive speed (Approximate Nearest Neighbor).

**Operations**:
*   **ANN Search**: Find top-k neighbors in < 10ms.
*   **Filtering**: Pre-filtering (using metadata) vs Post-filtering (simpler but wasteful).`,
    algorithms: ['HNSW', 'IVF-PQ (Inverted File + Product Quantization)', 'Cosine Similarity'],
    techStack: ['Milvus', 'Qdrant', 'Faiss']
  },
  ranking_engine: {
    title: 'LTR (Learning to Rank)',
    subtitle: 'Precision Re-ranking',
    content: `**The "Last Mile" of Relevance**:
Retrieval gets the top 1000 docs. Ranking sorts the top 50 perfectly.

**Model**: LambdaMART (Gradient Boosted Decision Trees).
**Features**:
*   **Query-Dependent**: BM25 score, Vector Score.
*   **Doc-Dependent**: PageRank, Recency, Popularity.
*   **Business Logic**: "Boost sponsored items", "Demote out-of-stock".

**Cross-Encoders**: Sometimes used for the very top 10 items. Slow, but extremely accurate (reads query and doc together).`,
    algorithms: ['LambdaMART', 'XGBoost', 'Cross-Encoders (BERT)'],
    techStack: ['XGBoost', 'LightGBM', 'TensorFlow Ranking'],
    kpis: ['NDCG@10 (Normalized Discounted Cumulative Gain)', 'MRR (Mean Reciprocal Rank)']
  },
  rag_orchestrator: {
    title: 'RAG Orchestrator',
    subtitle: 'Generation & Synthesis',
    content: `**Role**: The "Consumer" of Search.
    
**Why RAG?**
Search retrieves *facts*. LLMs provide *synthesis*.

**Process**:
1.  **Context Stuffing**: Takes the top 5 chunks from the Ranking Engine.
2.  **Prompt Engineering**: "Answer the user query using ONLY the following context..."
3.  **Citation**: Maps generated sentences back to source document IDs for auditability.

**Safety**: Hallucination checks using "NLI" (Natural Language Inference) models.`,
    algorithms: ['Prompt Chaining', 'Self-Consistency', 'Token Generation'],
    techStack: ['LangChain', 'Google Gemini', 'LlamaIndex']
  },
  kafka_backbone: {
    title: 'Apache Kafka',
    subtitle: 'Event Streaming Backbone',
    content: `**Role**: The central nervous system. Decouples Ingestion from Indexing.

**Topics**:
*   \`db.changes\`: CDC stream from databases.
*   \`user.clicks\`: Raw interaction logs for training data.
*   \`index.updates\`: Processed documents ready for indexing.

**Durability**: Allows replaying the log to rebuild indices from scratch without hitting the primary database.`,
    algorithms: ['Log-Structured Storage', 'Partitioning', 'Replication'],
    techStack: ['Apache Kafka', 'Confluent Schema Registry']
  },
  flink_processor: {
    title: 'Apache Flink',
    subtitle: 'Stateful Stream Processing',
    content: `**Role**: Real-time ETL.

**Tasks**:
1.  **Enrichment**: Joins a "ProductID" from the stream with "ProductDetails" from a side-input (cached in RocksDB).
2.  **Windowing**: Aggregates click signals over 1-hour tumbling windows ("Most Popular Items").
3.  **Embedding Generation**: Calls an external embedding model API for every new document *before* sending to Vector DB.

**Why Flink?**: Guarantees "Exactly-Once" processing semantics, critical for index consistency.`,
    algorithms: ['Watermarking', 'Tumbling Windows', 'State Snapshots (Chandy-Lamport)'],
    techStack: ['Apache Flink', 'RocksDB', 'Java']
  },
  data_sources: {
    title: 'Data Sources',
    subtitle: 'System of Record',
    content: `**Origins**:
*   **Relational (Postgres)**: Customer data, Inventory.
*   **NoSQL (Mongo)**: Product catalogs, unstructured metadata.
*   **Web Crawlers**: External content.

**CDC (Change Data Capture)**: We don't run SQL queries for indexing. We read the *Write Ahead Log (WAL)* to detect changes immediately.`,
    algorithms: ['CDC (Change Data Capture)', 'WAL Tailing'],
    techStack: ['PostgreSQL', 'Debezium', 'MongoDB']
  },
  analytics_dw: {
    title: 'Analytics DW',
    subtitle: 'Offline Evaluation',
    content: `**Role**: The "Scorecard".

**Evaluation**:
*   **Offline**: Uses historical click logs to test if a new ranking model *would have* performed better (Counterfactual Evaluation).
*   **Online**: Analyzes A/B test results.

**Golden Sets**: Stores manually labeled "Query, Document, Relevance" triplets to calculate NDCG.`,
    algorithms: ['Counterfactual Estimation', 'Confidence Intervals'],
    techStack: ['Snowflake', 'BigQuery', 'dbt']
  }
};