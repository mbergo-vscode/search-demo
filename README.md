# iFood AI Platform Strategy - Interactive Map

An interactive, educational visualization of a production-grade AI/RAG pipeline (simulating iFood's architecture). This application visualizes how data flows from Consumer Apps through Kafka, Feature Stores, and AI Models (RecSys, RAG, Risk) to deliver real-time results.

It integrates with **Google Gemini API** to simulate RAG responses and generate UI assets (images/videos) dynamically.

## Features

*   **Interactive Architecture Graph**: Zoom, pan, and explore nodes (Kafka, Vertex AI, Redis, etc.).
*   **Step-by-Step Simulation**: Visualizes the lifecycle of an order (Discovery -> Transaction -> Fulfillment -> Support).
*   **RAG & GenAI Integration**: Uses Gemini to simulate support agent responses and generate videos.
*   **Real-time ROI Inspector**: View simulated metrics (latency, conversion lift, GMV impact) for every step.

## Prerequisites

*   **Node.js** (v18 or higher)
*   **npm** (or yarn/pnpm)
*   **Google Gemini API Key**: You need a paid API key from Google AI Studio (for Veo video generation) or a standard key for text generation.

## 🚀 Quick Start (Local)

1.  **Install Dependencies**
    ```bash
    npm install
    ```

2.  **Configure API Key**
    You can set the API key in a `.env` file or pass it inline.
    Create a `.env` file in the root directory:
    ```env
    API_KEY=your_google_api_key_here
    ```

3.  **Run Development Server**
    ```bash
    npm run dev
    ```
    Open `http://localhost:10000` in your browser.

## 🛠 Building for Production

To create a static build optimized for deployment:

1.  **Build**
    ```bash
    npm run build
    ```
    This generates a `dist/` folder containing static assets.

2.  **Preview Production Build**
    ```bash
    npm run preview
    ```

## 🐳 Running with Docker

This project includes a `Dockerfile` for easy containerization.

### Option 1: Using Make (Easiest)

If you have `make` installed:

*   **Build & Run**:
    ```bash
    API_KEY=your_key make docker-run
    ```

### Option 2: Manual Docker Commands

1.  **Build Image**:
    ```bash
    docker build -t ifood-ai-viz .
    ```

2.  **Run Container**:
    ```bash
    docker run -p 10000:10000 -e API_KEY=your_actual_api_key ifood-ai-viz
    ```

## 📂 Project Structure

*   `src/components/PipelineVisualizer.tsx`: The main interactive graph engine (SVG + React).
*   `src/constants.tsx`: Definitions of nodes, edges, and static content.
*   `src/services/geminiService.ts`: Integration with Google GenAI SDK.
*   `src/App.tsx`: Main application logic and simulation state machine.

## Technologies

*   **Frontend**: React 19, Vite, TailwindCSS (CDN).
*   **AI**: Google GenAI SDK (Gemini 2.5/3.0 Models).
*   **Icons**: Lucide React.

## License

MIT
