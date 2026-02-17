import React from 'react';

export type NodeType = 
  // User & Entry
  | 'user_device'
  | 'api_gateway'
  
  // Fast Data / Caching
  | 'redis_cache'
  
  // Query Pipeline (Online)
  | 'query_understanding'
  | 'hybrid_retriever' // The "Search" Node
  | 'ranking_engine'   // LTR
  | 'rag_orchestrator' // LLM
  
  // Data / Indexing Pipeline (Offline/Nearline)
  | 'kafka_backbone'
  | 'flink_processor'
  | 'vector_db'        // Semantic Index
  | 'search_index'     // Lexical Index (Elastic/Solr)
  | 'data_sources'     // DBs/Connectors
  | 'analytics_dw';    // Data Warehouse

export interface PipelineNodeDef {
  id: NodeType;
  label: string;
  x: number;
  y: number;
  icon: React.ComponentType<any>;
  description: string;
  category: 'frontend' | 'serving' | 'data_engineering' | 'storage';
}

export interface PipelineEdgeDef {
  from: NodeType;
  to: NodeType;
  label?: string;
  activeInFlow: boolean;
  payloadInfo?: string; 
}

export interface NodeDetail {
  title: string;
  subtitle: string;
  content: string; 
  algorithms: string[];
  techStack: string[];
  kpis?: string[]; 
  crossDomainImpact?: {
    inputs: { source: string; benefit: string }[];
    outputs: { target: string; improvement: string }[];
  };
}

export interface SimulationStepData {
  title: string;
  data: Record<string, any> | string;
  description: string;
  visualType?: 'json' | 'ranking';
  impact?: string; 
  roiMetric?: { label: string; value: string; trend: 'up' | 'down' };
}

export interface SimulationStepDef {
  stepId: number;
  node: NodeType;
  edge?: { from: string; to: string };
  log: string;
  inspectorData?: SimulationStepData;
}