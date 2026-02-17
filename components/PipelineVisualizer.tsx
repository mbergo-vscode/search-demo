import React, { useState, useRef, useEffect } from 'react';
import { NODES, EDGES } from '../constants';
import { NodeType, PipelineNodeDef } from '../types';
import { Eye, EyeOff, Maximize2, X, ZoomIn, ZoomOut } from 'lucide-react';

interface PipelineVisualizerProps {
  activeNode: NodeType | null;
  onNodeClick: (node: NodeType) => void;
  activeFlow: any; 
  animatingEdge: { from: string; to: string } | null;
}

// Enterprise Search Theme
const CATEGORY_STYLES = {
  frontend: {
    bg: 'bg-slate-800',
    border: 'border-slate-600',
    activeBorder: 'border-white',
    icon: 'text-white',
    glow: 'shadow-white/20',
    label: 'text-slate-200'
  },
  serving: {
    bg: 'bg-blue-900/80', // Query Pipeline -> Blue
    border: 'border-blue-500/50',
    activeBorder: 'border-blue-500',
    icon: 'text-blue-400',
    glow: 'shadow-blue-500/40',
    label: 'text-blue-100'
  },
  data_engineering: {
    bg: 'bg-orange-900/80', // Kafka/Flink -> Orange/Amber
    border: 'border-orange-500/50',
    activeBorder: 'border-orange-400',
    icon: 'text-orange-400', 
    glow: 'shadow-orange-500/20',
    label: 'text-orange-200'
  },
  storage: {
    bg: 'bg-emerald-900/80', // Data/Indices -> Green
    border: 'border-emerald-500/50',
    activeBorder: 'border-emerald-400',
    icon: 'text-emerald-400',
    glow: 'shadow-emerald-500/40',
    label: 'text-emerald-200'
  }
};

const PipelineVisualizer: React.FC<PipelineVisualizerProps> = ({ 
  activeNode, 
  onNodeClick, 
  animatingEdge 
}) => {
  const [showPayloads, setShowPayloads] = useState(true);
  const [nodes, setNodes] = useState<PipelineNodeDef[]>(NODES);
  
  // Viewport State
  const [view, setView] = useState({ x: 0, y: 0, zoom: 0.9 });

  // Interaction State
  const [expandedNodeId, setExpandedNodeId] = useState<NodeType | null>(null);
  const [draggingNode, setDraggingNode] = useState<{ id: NodeType; startX: number; startY: number; initialNodeX: number; initialNodeY: number } | null>(null);
  const [isPanning, setIsPanning] = useState(false);
  const [panStart, setPanStart] = useState({ x: 0, y: 0 });

  const containerRef = useRef<HTMLDivElement>(null);

  // --- Zoom Logic ---
  const handleWheel = (e: React.WheelEvent) => {
    e.stopPropagation();
    if (!containerRef.current) return;
    
    const rect = containerRef.current.getBoundingClientRect();
    const mouseX = e.clientX - rect.left;
    const mouseY = e.clientY - rect.top;

    const delta = -e.deltaY * 0.001;
    const newZoom = Math.min(Math.max(0.4, view.zoom + delta), 3);
    
    const worldX = (mouseX - view.x) / view.zoom;
    const worldY = (mouseY - view.y) / view.zoom;
    
    const newPanX = mouseX - worldX * newZoom;
    const newPanY = mouseY - worldY * newZoom;

    setView({ x: newPanX, y: newPanY, zoom: newZoom });
  };

  const handleManualZoom = (delta: number) => {
      const newZoom = Math.min(Math.max(0.4, view.zoom + delta), 3);
      setView(v => ({ ...v, zoom: newZoom }));
  };

  const handleZoomReset = () => setView({ x: 0, y: 0, zoom: 0.9 });

  // --- Interaction Handlers ---

  const handleMouseDownBg = (e: React.MouseEvent) => {
      setIsPanning(true);
      setPanStart({ x: e.clientX - view.x, y: e.clientY - view.y });
      setExpandedNodeId(null);
  };

  const handleMouseDownNode = (e: React.MouseEvent, node: PipelineNodeDef) => {
      e.stopPropagation();
      setDraggingNode({ 
          id: node.id, 
          startX: e.clientX, 
          startY: e.clientY, 
          initialNodeX: node.x, 
          initialNodeY: node.y 
      });
  };

  const handleNodeClick = (e: React.MouseEvent, node: PipelineNodeDef) => {
      e.stopPropagation();
      if (draggingNode) {
          const dist = Math.abs(e.clientX - draggingNode.startX) + Math.abs(e.clientY - draggingNode.startY);
          if (dist > 5) return;
      }

      if (expandedNodeId === node.id) {
          onNodeClick(node.id);
      } else {
          setExpandedNodeId(node.id);
          onNodeClick(node.id);
      }
  };

  useEffect(() => {
      const handleGlobalMouseMove = (e: MouseEvent) => {
          if (isPanning) {
              setView(v => ({ ...v, x: e.clientX - panStart.x, y: e.clientY - panStart.y }));
          }
          
          if (draggingNode) {
              const dx = (e.clientX - draggingNode.startX) / view.zoom;
              const dy = (e.clientY - draggingNode.startY) / view.zoom;
              
              setNodes(prev => prev.map(n => {
                  if (n.id === draggingNode.id) {
                      return { ...n, x: draggingNode.initialNodeX + dx, y: draggingNode.initialNodeY + dy };
                  }
                  return n;
              }));
          }
      };

      const handleGlobalMouseUp = () => {
          setIsPanning(false);
          setDraggingNode(null);
      };

      window.addEventListener('mousemove', handleGlobalMouseMove);
      window.addEventListener('mouseup', handleGlobalMouseUp);
      return () => {
          window.removeEventListener('mousemove', handleGlobalMouseMove);
          window.removeEventListener('mouseup', handleGlobalMouseUp);
      };
  }, [isPanning, draggingNode, view.zoom, panStart]);

  return (
    <div 
      ref={containerRef}
      className="relative w-full h-full bg-[#0d0d0f] rounded-xl border border-tech-700 overflow-hidden shadow-2xl group select-none cursor-move"
      onWheel={handleWheel}
      onMouseDown={handleMouseDownBg}
    >
      <div className="absolute inset-0 opacity-10 pointer-events-none" 
           style={{ 
             backgroundImage: 'radial-gradient(#ffffff 1px, transparent 1px)', 
             backgroundSize: '24px 24px',
             backgroundPosition: `${view.x}px ${view.y}px`,
             transform: `scale(${view.zoom})`,
             transformOrigin: '0 0'
           }}>
      </div>

      <div className="absolute top-4 left-4 z-50 flex flex-col gap-1 bg-tech-800 rounded-lg border border-tech-600 p-1 shadow-lg cursor-default" onMouseDown={e => e.stopPropagation()}>
          <button onClick={() => handleManualZoom(0.2)} className="p-1.5 hover:bg-tech-700 text-slate-300 rounded"><ZoomIn className="w-4 h-4"/></button>
          <button onClick={handleZoomReset} className="p-1.5 hover:bg-tech-700 text-slate-300 rounded text-[10px] font-bold font-mono text-center w-8">{Math.round(view.zoom * 100)}%</button>
          <button onClick={() => handleManualZoom(-0.2)} className="p-1.5 hover:bg-tech-700 text-slate-300 rounded"><ZoomOut className="w-4 h-4"/></button>
      </div>

      <button 
        onClick={(e) => { e.stopPropagation(); setShowPayloads(!showPayloads); }}
        className="absolute top-4 right-4 z-50 p-2 bg-tech-800 hover:bg-tech-700 text-slate-400 rounded-lg border border-tech-600 transition-colors cursor-pointer"
        onMouseDown={e => e.stopPropagation()}
      >
        {showPayloads ? <EyeOff className="w-4 h-4" /> : <Eye className="w-4 h-4" />}
      </button>

      {/* World Container */}
      <div 
        style={{ 
            transform: `translate(${view.x}px, ${view.y}px) scale(${view.zoom})`, 
            transformOrigin: '0 0',
            width: '100%', 
            height: '100%',
            willChange: 'transform'
        }}
      >
        <svg className="absolute top-0 left-0 w-[4000px] h-[4000px] pointer-events-none z-0 overflow-visible">
            <defs>
            <marker id="arrow-blue" markerWidth="10" markerHeight="7" refX="24" refY="3.5" orient="auto">
                <polygon points="0 0, 10 3.5, 0 7" fill="#3b82f6" />
            </marker>
            <marker id="arrow-slate" markerWidth="10" markerHeight="7" refX="24" refY="3.5" orient="auto">
                <polygon points="0 0, 10 3.5, 0 7" fill="#475569" />
            </marker>
            <marker id="arrow-orange" markerWidth="10" markerHeight="7" refX="24" refY="3.5" orient="auto">
                <polygon points="0 0, 10 3.5, 0 7" fill="#f97316" />
            </marker>
            <filter id="glow-blue">
                <feGaussianBlur stdDeviation="2.5" result="coloredBlur"/>
                <feFlood floodColor="#3b82f6" result="glowColor" />
                <feComposite in="glowColor" in2="coloredBlur" operator="in" result="softGlow_colored" />
                <feMerge>
                <feMergeNode in="softGlow_colored"/>
                <feMergeNode in="SourceGraphic"/>
                </feMerge>
            </filter>
            </defs>

            {EDGES.map((edge, idx) => {
            const fromNode = nodes.find(n => n.id === edge.from)!;
            const toNode = nodes.find(n => n.id === edge.to)!;
            const isAnimating = animatingEdge?.from === edge.from && animatingEdge?.to === edge.to;
            
            // Check category for coloring edge
            const isDataEdge = fromNode.category === 'data_engineering' || fromNode.category === 'storage';

            const getCenter = (n: PipelineNodeDef) => {
                const isExp = expandedNodeId === n.id;
                return {
                    x: n.x + (isExp ? 128 : 40),
                    y: n.y + (isExp ? 60 : 40)
                };
            };

            const p1 = getCenter(fromNode);
            const p2 = getCenter(toNode);

            return (
                <g key={`${edge.from}-${edge.to}-${idx}`}>
                <line
                    x1={p1.x} y1={p1.y} x2={p2.x} y2={p2.y}
                    stroke={isAnimating ? "#3b82f6" : (isDataEdge ? "#f97316" : "#475569")}
                    strokeWidth={isAnimating ? 3 : 2}
                    strokeDasharray={!edge.activeInFlow ? "5,5" : "0"}
                    markerEnd={isAnimating ? "url(#arrow-blue)" : (isDataEdge ? "url(#arrow-orange)" : "url(#arrow-slate)")}
                    className="transition-colors duration-300"
                    opacity={isAnimating ? 1 : (isDataEdge ? 0.4 : 0.3)}
                />
                {showPayloads && edge.payloadInfo && (
                    <foreignObject x={(p1.x+p2.x)/2 - 60} y={(p1.y+p2.y)/2 - 10} width="120" height="24" className="overflow-visible pointer-events-none">
                        <div className={`text-[9px] text-center px-1.5 py-0.5 rounded border backdrop-blur-md transform transition-all duration-300 font-mono font-medium truncate
                            ${isAnimating 
                                ? 'bg-blue-900/80 border-blue-500 text-white scale-110 shadow-lg' 
                                : 'bg-tech-900/80 border-tech-600 text-slate-500'}
                        `}>
                            {edge.payloadInfo}
                        </div>
                    </foreignObject>
                )}
                {isAnimating && (
                    <circle r="4" fill="#fff" filter="url(#glow-blue)">
                    <animateMotion dur="1s" repeatCount="1" path={`M${p1.x},${p1.y} L${p2.x},${p2.y}`} />
                    </circle>
                )}
                </g>
            );
            })}
        </svg>

        {/* Nodes */}
        {nodes.map((node) => {
            const Icon = node.icon;
            const isActive = activeNode === node.id;
            const isExpanded = expandedNodeId === node.id;
            const style = CATEGORY_STYLES[node.category];

            return (
            <div
                key={node.id}
                onMouseDown={(e) => handleMouseDownNode(e, node)}
                onClick={(e) => handleNodeClick(e, node)}
                className={`absolute flex flex-col rounded-xl cursor-pointer transition-all duration-300 ease-spring z-10 border-2 shadow-xl backdrop-blur-md overflow-hidden
                    ${isActive ? style.activeBorder : style.border}
                    ${isExpanded ? `w-64 h-auto z-50 ${style.bg.replace('/40','/95')}` : `w-20 h-20 items-center justify-center ${style.bg}`}
                    ${isActive && !isExpanded ? `${style.glow} scale-110` : ''}
                    ${isExpanded ? 'shadow-2xl shadow-black/50' : ''}
                `}
                style={{ left: node.x, top: node.y }}
            >
                {isExpanded ? (
                    // Expanded Content
                    <div className="p-4 flex flex-col h-full relative cursor-default">
                        <button 
                            onClick={(e) => { e.stopPropagation(); setExpandedNodeId(null); }} 
                            className="absolute top-2 right-2 text-white/50 hover:text-white"
                        >
                            <X className="w-3 h-3"/>
                        </button>
                        <div className="flex items-center gap-2 mb-3">
                            <Icon className={`w-5 h-5 ${style.icon}`} />
                            <span className={`text-sm font-bold text-white`}>{node.label}</span>
                        </div>
                        <p className="text-xs text-slate-300 leading-snug mb-3 select-text">
                            {node.description}
                        </p>
                        <div className="mt-auto pt-2 border-t border-white/10 flex items-center gap-2 text-[10px] font-bold text-blue-400 uppercase tracking-wider">
                            <Maximize2 className="w-3 h-3" /> View Details
                        </div>
                    </div>
                ) : (
                    // Collapsed Content
                    <>
                        <Icon className={`w-8 h-8 mb-2 ${isActive ? 'text-white' : style.icon}`} />
                        <span className={`text-[9px] font-bold text-center leading-tight px-1 select-none ${isActive ? 'text-white' : style.label}`}>
                        {node.label}
                        </span>
                    </>
                )}
            </div>
            );
        })}
      </div>

      {/* Legend */}
      <div className="absolute bottom-4 left-4 flex gap-4 text-[10px] font-bold uppercase tracking-wider bg-tech-900/90 border border-tech-700 p-3 rounded-lg z-0 pointer-events-none">
          <div className="flex items-center gap-2 text-white"><div className="w-2 h-2 rounded-full bg-slate-500"></div> User/Edge</div>
          <div className="flex items-center gap-2 text-blue-400"><div className="w-2 h-2 rounded-full bg-blue-500"></div> Query/Serving</div>
          <div className="flex items-center gap-2 text-orange-400"><div className="w-2 h-2 rounded-full bg-orange-500"></div> Streaming/Ingest</div>
          <div className="flex items-center gap-2 text-emerald-400"><div className="w-2 h-2 rounded-full bg-emerald-500"></div> Storage/Index</div>
      </div>
    </div>
  );
};

export default PipelineVisualizer;