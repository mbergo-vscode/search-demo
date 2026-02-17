import React, { useState, useEffect, useCallback } from 'react';
import PipelineVisualizer from './components/PipelineVisualizer';
import DetailPanel from './components/DetailPanel';
import DataInspector from './components/DataInspector';
import { generateRAGResponse, generateVeoVideo } from './services/geminiService';
import { NodeType, SimulationStepDef } from './types';
import { Play, RotateCcw, ChevronRight, Terminal, Search, Server, Activity, Info, GripVertical } from 'lucide-react';

// --- Enterprise Search Simulation Definitions ---

const getSearchPipelineSteps = (query: string): SimulationStepDef[] => [
  // --- PHASE 1: QUERY INGESTION ---
  { 
    stepId: 0, node: 'user_device', log: `User types: "${query}"`,
    inspectorData: { 
      title: 'User Input', 
      description: 'User interacts with the search bar. Keystrokes are debounced (300ms) to prevent flooding the API.', 
      data: { query: query, tenant_id: "org_acme_corp", session_id: "sess_8829", timestamp: Date.now() } 
    }
  },
  { 
    stepId: 1, node: 'api_gateway', edge: {from: 'user_device', to: 'api_gateway'}, log: 'Gateway: Auth & Rate Limit',
    inspectorData: { 
      title: 'Gateway Verification', 
      description: 'Checking JWT signature for tenant isolation. Token Bucket algorithm checks if user has quota remaining.', 
      impact: 'Prevents DDoS and enforces billing tiers.',
      roiMetric: { label: 'Quota Remaining', value: '98%', trend: 'down' },
      data: { auth: "JWT_VALID", rate_limit: "OK", bucket_tokens: 495 } 
    }
  },
  { 
    stepId: 2, node: 'redis_cache', edge: {from: 'api_gateway', to: 'redis_cache'}, log: 'Redis: Cache Miss',
    inspectorData: { 
      title: 'Cache Lookup', 
      description: 'Checking Redis Cluster for key "hash(query + tenant)". Result is NULL (Cache Miss), proceeding to full pipeline.', 
      impact: 'Cache Hit would save ~200ms of backend processing.',
      roiMetric: { label: 'Cache Hit Rate', value: '32%', trend: 'up' },
      data: { key: "sha256:enterprise_search", result: null, latency: "2ms" } 
    }
  },
  
  // --- PHASE 2: QUERY UNDERSTANDING ---
  { 
    stepId: 3, node: 'query_understanding', edge: {from: 'redis_cache', to: 'query_understanding'}, log: 'NLP: Expansion & Vectorization',
    inspectorData: { 
      title: 'Query Processing', 
      description: '1. NER identifies "Enterprise" as a context.\n2. Synonym Expansion: "Strategy" -> "Plan", "Architecture".\n3. BERT Model generates 768d vector.', 
      data: { 
        tokens: ["enterprise", "search", "strategy"], 
        expansion: ["corporate search", "retrieval system"],
        vector_sample: [0.021, -0.45, 0.11, "...", 0.98] 
      } 
    }
  },

  // --- PHASE 3: PARALLEL RETRIEVAL ---
  { 
    stepId: 4, node: 'hybrid_retriever', edge: {from: 'query_understanding', to: 'hybrid_retriever'}, log: 'Retriever: Forking Request',
    inspectorData: { 
      title: 'Hybrid Dispatch', 
      description: 'Sending parallel requests to Lucene (Lexical) and Vector DB (Semantic).', 
      data: { strategy: "RRF (Reciprocal Rank Fusion)", alpha: 0.4, beta: 0.6 } 
    }
  },
  { 
    stepId: 5, node: 'search_index', edge: {from: 'hybrid_retriever', to: 'search_index'}, log: 'Lucene: BM25 Search',
    inspectorData: { 
      title: 'Lexical Retrieval', 
      description: 'Inverted Index lookup. Matching exact terms "enterprise" AND "search". Scoring via BM25 algorithm.', 
      impact: 'Ensures exact keyword matches are found (Precision).',
      data: { hits: 450, top_doc: "doc_123", bm25_score: 12.4 } 
    }
  },
  { 
    stepId: 6, node: 'vector_db', edge: {from: 'hybrid_retriever', to: 'vector_db'}, log: 'Vector DB: ANN Search',
    inspectorData: { 
      title: 'Semantic Retrieval', 
      description: 'HNSW Graph Traversal. Finding nearest neighbors in embedding space. Retrieves conceptually related docs even without keyword overlap.', 
      impact: 'Finds "System Design" even if user typed "Architecture" (Recall).',
      data: { algorithm: "HNSW", distance_metric: "Cosine", candidates: 100 } 
    }
  },

  // --- PHASE 4: RANKING & GENAI ---
  { 
    stepId: 7, node: 'ranking_engine', edge: {from: 'hybrid_retriever', to: 'ranking_engine'}, log: 'LTR: Re-ranking Candidates',
    inspectorData: { 
      title: 'Learning to Rank', 
      description: 'XGBoost model re-scores the merged list (Lexical + Semantic). Features: Click-through-rate, Recency, and Exact Match Boost.', 
      impact: 'Moves the most relevant document to Position #1.',
      roiMetric: { label: 'NDCG@10', value: '0.92', trend: 'up' },
      data: { input_count: 50, output_count: 10, model: "LambdaMART_v4" } 
    }
  },
  { 
    stepId: 8, node: 'rag_orchestrator', edge: {from: 'ranking_engine', to: 'rag_orchestrator'}, log: 'GenAI: Synthesizing Answer',
    inspectorData: { 
      title: 'RAG Synthesis', 
      description: 'LLM receives the Top 5 Ranked Documents as context window. Generates a grounded response with citations.', 
      data: { model: "Gemini 1.5 Pro", prompt_tokens: 4096, context_window: "5 Docs" } 
    }
  },
  { 
    stepId: 9, node: 'api_gateway', edge: {from: 'rag_orchestrator', to: 'api_gateway'}, log: 'Gateway: Streaming Response',
    inspectorData: { 
      title: 'Final Response', 
      description: 'Returning JSON with: 1. Direct Answer (GenAI) 2. Source Links 3. Search Results.', 
      roiMetric: { label: 'Total Latency', value: '450ms', trend: 'down' },
      data: { status: 200, size: "14KB" } 
    }
  },

  // --- PHASE 5: ASYNC FEEDBACK LOOP ---
  { 
    stepId: 10, node: 'kafka_backbone', edge: {from: 'api_gateway', to: 'kafka_backbone'}, log: 'Async: Logging Event',
    inspectorData: { 
      title: 'Event Logging', 
      description: 'User interaction (and the query itself) is pushed to Kafka topic `search.logs` for offline analysis and model training.', 
      data: { topic: "search.logs", partition: 2, offset: 99120 } 
    }
  },
  { 
    stepId: 11, node: 'flink_processor', edge: {from: 'kafka_backbone', to: 'flink_processor'}, log: 'Flink: Updating Metrics',
    inspectorData: { 
      title: 'Stream Processing', 
      description: 'Flink window aggregates "Top Queries" for the last hour and updates the Analytics Dashboard.', 
      impact: 'Real-time visibility into system performance.',
      data: { window: "1 hour", metric: "zero_result_rate", value: 0.05 } 
    }
  }
];

const App: React.FC = () => {
  const [activeNode, setActiveNode] = useState<NodeType | null>(null);
  const [consoleLogs, setConsoleLogs] = useState<string[]>([]);
  const [searchQuery, setSearchQuery] = useState('Enterprise Search Architecture');
  const [isSimulating, setIsSimulating] = useState(false);
  const [currentStepIndex, setCurrentStepIndex] = useState(-1);
  const [animatingEdge, setAnimatingEdge] = useState<{from: string, to: string} | null>(null);
  const [inspectorData, setInspectorData] = useState<any>(null);
  const [latencyBudget, setLatencyBudget] = useState(500);

  // Veo Video State
  const [rerankVideoUrl, setRerankVideoUrl] = useState<string | null>(null);
  const [isGeneratingRerankVideo, setIsGeneratingRerankVideo] = useState(false);

  // Resize State
  const [sidebarWidth, setSidebarWidth] = useState(480);
  const [isResizing, setIsResizing] = useState(false);

  const startResizing = useCallback(() => setIsResizing(true), []);
  const stopResizing = useCallback(() => setIsResizing(false), []);

  const resize = useCallback((mouseMoveEvent: MouseEvent) => {
      if (isResizing) {
          const newWidth = window.innerWidth - mouseMoveEvent.clientX;
          // Clamp width between 300px and 800px
          if (newWidth > 300 && newWidth < 900) {
              setSidebarWidth(newWidth);
          }
      }
  }, [isResizing]);

  useEffect(() => {
      if (isResizing) {
          window.addEventListener('mousemove', resize);
          window.addEventListener('mouseup', stopResizing);
      }
      return () => {
          window.removeEventListener('mousemove', resize);
          window.removeEventListener('mouseup', stopResizing);
      };
  }, [isResizing, resize, stopResizing]);

  const log = (msg: string) => {
    setConsoleLogs(prev => [`[${new Date().toLocaleTimeString()}] ${msg}`, ...prev.slice(0, 8)]); 
  };

  const startSimulation = () => {
    setIsSimulating(true);
    setCurrentStepIndex(-1);
    setConsoleLogs([]);
    log(`Initializing Search Pipeline for query: "${searchQuery}"...`);
    setInspectorData(null);
    setRerankVideoUrl(null); 
    advanceStep(0); 
  };

  const advanceStep = (stepIdx: number) => {
    const steps = getSearchPipelineSteps(searchQuery);
    
    if (stepIdx >= steps.length) {
      log('Request Lifecycle Complete.');
      setIsSimulating(false);
      setAnimatingEdge(null);
      return;
    }

    const step = steps[stepIdx];
    setCurrentStepIndex(stepIdx);
    
    // Update UI
    setActiveNode(step.node);
    if (step.edge) {
      setAnimatingEdge({ from: step.edge.from, to: step.edge.to });
    } else {
      setAnimatingEdge(null);
    }
    
    log(step.log);
    setInspectorData(step.inspectorData);
  };

  const handleNextStep = () => {
    if (!isSimulating) return;
    advanceStep(currentStepIndex + 1);
  };

  const resetSimulation = () => {
    setIsSimulating(false);
    setActiveNode(null);
    setAnimatingEdge(null);
    setCurrentStepIndex(-1);
    setInspectorData(null);
    setRerankVideoUrl(null);
    log('System Ready.');
  };

  // Video Generation for "Server Room / Data Center" visualization
  const handleGenerateDataCenterVideo = async () => {
      const aistudio = (window as any).aistudio;
      if (aistudio) {
          const hasKey = await aistudio.hasSelectedApiKey();
          if (!hasKey) {
              await aistudio.openSelectKey();
          }
      }

      setIsGeneratingRerankVideo(true);
      setRerankVideoUrl(null);
      log('Generating Data Center Visualization...');
      
      const prompt = "Cinematic shot of a futuristic server room. Rows of servers with blinking blue and green lights. Fast moving streams of light representing data flowing through fiber optic cables. High tech, cyberpunk aesthetic, clean and professional.";

      try {
          const videoUrl = await generateVeoVideo(prompt, undefined, '16:9');
          if (videoUrl) {
              setRerankVideoUrl(videoUrl);
              log('Visualization generated.');
          } else {
              log('Failed to generate video.');
          }
      } catch(e: any) {
          console.error("Veo Error:", e);
          const errStr = JSON.stringify(e) + (e.message || "") + (e.toString() || "");

          if (aistudio) {
            // Check for 403 Permission Denied
            if (errStr.includes("403") || errStr.includes("PERMISSION_DENIED")) {
                log('Error: Permission Denied. Please select a PAID API Key.');
                await aistudio.openSelectKey();
            } 
            // Check for 429 Resource Exhausted (Quota)
            else if (errStr.includes("429") || errStr.includes("RESOURCE_EXHAUSTED")) {
                log('Error: Quota Exceeded. Please check billing/quota.');
                await aistudio.openSelectKey();
            }
            // Check for Not Found (often invalid key context)
            else if (errStr.includes("Requested entity was not found")) {
                log('Key context invalid. Re-selecting key...');
                await aistudio.openSelectKey();
            } else {
                log('Error generating video. Check console.');
            }
          } else {
              log(`Error generating video: ${e.message || "Unknown"}`);
          }
      } finally {
          setIsGeneratingRerankVideo(false);
      }
  };

  return (
    <div className={`h-screen flex flex-col font-sans text-slate-200 selection:bg-blue-500 selection:text-white ${isResizing ? 'cursor-col-resize select-none' : ''}`}>
      {/* Header */}
      <header className="border-b border-tech-800 bg-tech-900 px-6 py-4 flex items-center justify-between shrink-0 shadow-lg relative z-20">
        <div className="flex items-center gap-4">
          <div className="bg-blue-600 p-2 rounded-lg shadow-[0_0_15px_rgba(37,99,235,0.4)]">
             <Search className="w-6 h-6 text-white" />
          </div>
          <div>
            <h1 className="text-xl font-bold tracking-tight text-white flex items-center gap-2">
              Modern Enterprise <span className="text-blue-500">Search Platform</span>
            </h1>
            <p className="text-xs text-slate-400 font-medium tracking-wide uppercase">RAG • Hybrid Retrieval • Stream Processing</p>
          </div>
        </div>
        <div className="flex items-center gap-6">
            <div className="flex items-center gap-3 bg-tech-800 px-4 py-2 rounded-full border border-tech-700">
                <Activity className="w-4 h-4 text-emerald-400" />
                <span className="text-xs text-slate-400 font-bold uppercase">Latency Budget:</span>
                <select 
                  value={latencyBudget} 
                  onChange={(e) => setLatencyBudget(Number(e.target.value))}
                  className="bg-transparent text-sm font-bold text-white outline-none cursor-pointer"
                >
                    <option value="100" className="text-black">100ms (Strict)</option>
                    <option value="500" className="text-black">500ms (Standard)</option>
                    <option value="2000" className="text-black">2000ms (RAG Heavy)</option>
                </select>
            </div>
        </div>
      </header>

      {/* Main Content (Flex Layout for Resizable Sidebar) */}
      <main className="flex-1 flex flex-col lg:flex-row overflow-hidden bg-[#0a0a0c]">
        
        {/* Left: Visualization & Controls (Flexible Width) */}
        <div className="flex-1 min-w-0 p-6 flex flex-col gap-6 overflow-y-auto">
          
          {/* Controls Bar */}
          <div className="bg-tech-800/60 backdrop-blur-md border border-tech-700 rounded-2xl p-4 flex items-center justify-between shadow-xl">
             <div className="flex items-center gap-4 flex-1 mr-4">
                <div className="bg-tech-900 p-2 rounded-lg border border-tech-700">
                    <Server className="w-5 h-5 text-blue-500" />
                </div>
                <div className="flex-1">
                    <h3 className="text-[10px] font-bold text-slate-400 uppercase tracking-wider mb-1">Search Query Simulation</h3>
                    <input 
                      type="text" 
                      value={searchQuery}
                      onChange={(e) => !isSimulating && setSearchQuery(e.target.value)}
                      disabled={isSimulating}
                      className="w-full bg-tech-900 border border-tech-700 rounded px-3 py-1 text-sm text-white focus:border-blue-500 outline-none transition"
                    />
                </div>
             </div>

             <div className="flex gap-2">
                {!isSimulating ? (
                    <button 
                        onClick={startSimulation}
                        className="flex items-center gap-2 px-6 py-2 bg-blue-600 hover:bg-blue-500 text-sm font-bold rounded-lg text-white transition shadow-[0_4px_14px_rgba(37,99,235,0.4)]"
                    >
                        <Play className="w-4 h-4 fill-current" />
                        Run Pipeline
                    </button>
                ) : (
                    <>
                    <button 
                        onClick={handleNextStep}
                        className="flex items-center gap-2 px-6 py-2 bg-white text-blue-600 hover:bg-slate-100 text-sm font-bold rounded-lg transition shadow-lg animate-pulse-fast"
                    >
                        Next Step <ChevronRight className="w-4 h-4" />
                    </button>
                    <button 
                        onClick={resetSimulation}
                        className="p-2 text-slate-400 hover:text-white hover:bg-tech-700 rounded-lg transition"
                        title="Reset"
                    >
                        <RotateCcw className="w-5 h-5" />
                    </button>
                    </>
                )}
             </div>
          </div>

          {/* Visualization Stage */}
          <div className="flex-1 min-h-[500px] flex flex-col relative">
             <div className="flex items-center justify-between mb-4 px-1">
               <h2 className="text-xs font-bold text-slate-500 uppercase tracking-widest flex items-center gap-2">
                 <Info className="w-3 h-3" /> Architecture Diagram
               </h2>
               {isSimulating && (
                 <span className="text-xs bg-blue-500/10 text-blue-400 px-3 py-1 rounded-full border border-blue-500/20 font-mono font-bold">
                    STEP {currentStepIndex + 1} / {getSearchPipelineSteps(searchQuery).length}
                 </span>
               )}
             </div>
             <PipelineVisualizer 
                activeNode={activeNode} 
                onNodeClick={setActiveNode} 
                activeFlow={isSimulating ? 'both' : null} 
                animatingEdge={animatingEdge}
             />
          </div>

          {/* Bottom Panels */}
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4 h-60">
             {/* Data Inspector */}
             <DataInspector 
                data={inspectorData} 
                onGenerateVideo={handleGenerateDataCenterVideo}
                videoUrl={rerankVideoUrl}
                isGeneratingVideo={isGeneratingRerankVideo}
             />

             {/* Console Log */}
             <div className="bg-tech-900 border border-tech-700 rounded-xl p-0 font-mono text-xs overflow-hidden flex flex-col shadow-inner">
              <div className="bg-tech-800 px-4 py-2 border-b border-tech-700 flex items-center justify-between">
                <div className="flex items-center gap-2">
                    <Terminal className="w-3.5 h-3.5 text-slate-400" />
                    <span className="font-bold text-slate-300">System Logs</span>
                </div>
                <div className="flex gap-1.5">
                    <div className="w-2.5 h-2.5 rounded-full bg-red-500/20 border border-red-500"></div>
                    <div className="w-2.5 h-2.5 rounded-full bg-yellow-500/20 border border-yellow-500"></div>
                    <div className="w-2.5 h-2.5 rounded-full bg-green-500/20 border border-green-500"></div>
                </div>
              </div>
              <div className="flex-1 overflow-y-auto p-4 space-y-2 scrollbar-thin scrollbar-thumb-tech-700">
                {consoleLogs.map((msg, i) => (
                  <div key={i} className={`truncate font-medium ${i === 0 ? 'text-blue-400' : 'text-slate-500'}`}>
                    <span className="opacity-30 mr-2">{'>'}</span>{msg}
                  </div>
                ))}
                {consoleLogs.length === 0 && <span className="text-slate-700 italic opacity-50">Cluster online. Waiting for input.</span>}
              </div>
            </div>
          </div>
        </div>

        {/* Resizer Handle */}
        <div 
           className="hidden lg:flex w-2 bg-tech-900 border-l border-r border-tech-800 hover:bg-blue-600/50 hover:border-blue-500/50 cursor-col-resize items-center justify-center transition-colors z-30 flex-none"
           onMouseDown={startResizing}
        >
            <GripVertical className="w-3 h-3 text-slate-600" />
        </div>

        {/* Right: Info Panel (Dynamic Width) */}
        <div 
          className="flex-none h-full min-h-0 bg-tech-800/30 backdrop-blur-sm relative z-10 flex flex-col overflow-hidden border-t lg:border-t-0 border-tech-700"
          style={{ width: window.innerWidth >= 1024 ? sidebarWidth : '100%' }}
        >
          <DetailPanel nodeId={activeNode} onClose={() => setActiveNode(null)} />
        </div>

      </main>
    </div>
  );
};

export default App;