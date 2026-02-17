# Enterprise Search & RAG Architecture Visualization

An interactive, production-grade simulation of a modern **Enterprise Search** and **Retrieval-Augmented Generation (RAG)** platform. This application visualizes the complex data flows, distributed systems patterns, and ML pipelines required to build a system like Google Search or Amazon Product Search within an enterprise context.

It integrates with **Google Gemini API** (`@google/genai`) to simulate RAG synthesis and generate dynamic cinematic visualizations using the **Veo** video model.

![Architecture Visualization](https://img.shields.io/badge/Architecture-Event%20Driven-orange)
![Search](https://img.shields.io/badge/Search-Hybrid%20(BM25%20%2B%20Vector)-blue)
![AI](https://img.shields.io/badge/AI-Gemini%20Flash%20%26%20Veo-purple)

## 🧠 Architectural Deep Dive

This project visualizes two distinct but interconnected pipelines:

### 1. The Online Serving Pipeline (Query Time)
This is the synchronous path that executes when a user types a query. Latency budget: <500ms.

*   **API Gateway**: Implements the **Token Bucket Algorithm** for rate limiting and handles tenant isolation via JWT inspection.
*   **Redis Cache**: Implements a **Look-Aside Cache** strategy. It utilizes the **Zipfian Distribution** of search queries (20% of queries = 80% of volume) to serve cached results in <2ms.
*   **Query Understanding (NLP)**:
    *   **NER (Named Entity Recognition)**: Extracts filters (e.g., "cheap" -> `price_range: low`).
    *   **Vectorization**: Converts text to 768-dimensional dense vectors using BERT-based models.
*   **Hybrid Retrieval (The "Fork")**:
    *   **Lexical Path (Apache Lucene)**: Queries an Inverted Index using **BM25** scoring for exact keyword matching (Precision).
    *   **Semantic Path (Vector DB)**: Queries an **HNSW (Hierarchical Navigable Small World)** graph for Approximate Nearest Neighbor (ANN) search (Recall).
    *   **Fusion**: Merges results using **RRF (Reciprocal Rank Fusion)** to balance precision and recall without manual weight tuning.
*   **LTR (Learning to Rank)**: A re-ranking stage using **LambdaMART** (Gradient Boosted Decision Trees). It re-orders the Top-50 candidates based on features like Click-Through Rate (CTR) and Freshness.
*   **RAG Orchestrator**: The final stage where the Top-5 documents are stuffed into the **Gemini 1.5 Pro** context window to generate a grounded, natural language answer.

### 2. The Offline/Nearline Data Pipeline (Indexing Time)
This is the asynchronous backbone ensuring data freshness.

*   **CDC (Change Data Capture)**: Uses Debezium patterns to tail the **Write-Ahead Log (WAL)** of the source databases, ensuring hard deletes are captured.
*   **Apache Kafka**: Provides the persistent, ordered log of all domain events. Partitioned by `DocumentID` to ensure sequential processing.
*   **Apache Flink**: Performs stateful stream processing:
    *   **Windowing**: Aggregates clickstreams for analytics.
    *   **Enrichment**: Joins raw IDs with metadata before indexing.
*   **Vector Indexing**: Computes embeddings on the fly before upserting into the Vector DB.

## 🛠️ Technologies Used

*   **Frontend**: React 19, Vite, TailwindCSS.
*   **Visualization**: SVG-based interactive graph with pan/zoom (d3-like behavior).
*   **AI Integration**: 
    *   `@google/genai` SDK for interaction with Gemini Models.
    *   **Gemini 3 Flash**: For RAG text generation.
    *   **Veo 3.1**: For generative video visualization of infrastructure.

## 🚀 Local Development Guide

Follow these instructions to run the visualization locally.

### Prerequisites
*   **Node.js**: v18.0.0 or higher.
*   **Google AI Studio API Key**: You need an API key to run the RAG and Video generation features. 
    *   *Note: Video generation (Veo) requires a paid tier project.*

### Step 1: Clone and Install
Clone the repository and install the dependencies. **Do not skip the install step.**

```bash
git clone <repository-url>
cd enterprise-search-viz

# CRITICAL STEP: Install Node Modules
npm install
```

### Step 2: Configure Environment
You can provide your API key via an environment variable or select it via the UI prompt at runtime.

**Option A: Create a .env file (Recommended)**
Create a file named `.env` in the root directory:

```env
API_KEY=your_google_api_key_here
```

**Option B: Runtime Selection**
If no key is found, the app will prompt you to select a Google Cloud Project / API Key via the Google AI SDK overlay when you try to generate a video.

### Step 3: Run the Application
Start the Vite development server.

```bash
npm run dev
```

Open your browser to: `http://localhost:10000` (or the port shown in your terminal).

## 🐳 Docker Deployment

To run this as a containerized application:

```bash
# Build the image
docker build -t search-rag-viz .

# Run the container (Map port 10000)
docker run -p 10000:10000 -e API_KEY=your_key search-rag-viz
```

## 📂 Project Structure

```
├── src/
│   ├── App.tsx                 # Main Simulation Controller & Layout
│   ├── constants.tsx           # Graph Definition (Nodes, Edges, Technical Content)
│   ├── types.ts                # TypeScript Interfaces
│   ├── services/
│   │   └── geminiService.ts    # Google GenAI SDK Implementation (RAG + Veo)
│   └── components/
│       ├── PipelineVisualizer.tsx # SVG Graph Engine
│       ├── DetailPanel.tsx        # Technical Deep Dive Side Panel
│       └── DataInspector.tsx      # JSON Payload & Video Viewer
├── index.html                  # Entry Point
├── package.json                # Dependencies
└── vite.config.ts              # Vite Build Configuration
```

## 🤝 Contributing

1.  Fork the repository.
2.  Create a feature branch (`git checkout -b feature/amazing-feature`).
3.  Commit your changes.
4.  Push to the branch.
5.  Open a Pull Request.

---
*Built with ❤️ for System Design Interviews & Architecture Reviews.*
