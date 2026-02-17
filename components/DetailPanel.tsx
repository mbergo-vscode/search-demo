import React from 'react';
import { NodeType, NodeDetail, PipelineNodeDef } from '../types';
import { NODE_DETAILS, NODES } from '../constants';
import { X, Code, Layers, TrendingUp, GitMerge, ArrowRightFromLine, ArrowLeftToLine, BookOpen } from 'lucide-react';

interface DetailPanelProps {
  nodeId: NodeType | null;
  onClose: () => void;
}

const DetailPanel: React.FC<DetailPanelProps> = ({ nodeId, onClose }) => {
  if (!nodeId) {
    return (
      <div className="h-full flex items-center justify-center text-slate-500 p-8 text-center bg-tech-900">
        <div>
          <div className="w-20 h-20 bg-tech-800 rounded-full flex items-center justify-center mx-auto mb-6 border border-tech-700">
             <Layers className="w-8 h-8 opacity-30" />
          </div>
          <p className="text-lg font-bold text-slate-300">Architecture Details</p>
          <p className="text-sm mt-2 text-slate-500">Select a component to view the Interview Deep Dive.</p>
        </div>
      </div>
    );
  }

  const details: NodeDetail = NODE_DETAILS[nodeId];
  const nodeDef = NODES.find(n => n.id === nodeId);
  
  // Theme logic
  let accentColor = "text-slate-200";
  let borderColor = "border-slate-500";
  let bgGradient = "from-slate-800 to-slate-900";

  if (nodeDef?.category === 'serving') {
      accentColor = "text-blue-400";
      borderColor = "border-blue-500";
      bgGradient = "from-blue-950 to-tech-900";
  } else if (nodeDef?.category === 'data_engineering') {
      accentColor = "text-orange-400";
      borderColor = "border-orange-500";
      bgGradient = "from-orange-950 to-tech-900";
  } else if (nodeDef?.category === 'storage') {
      accentColor = "text-emerald-400";
      borderColor = "border-emerald-500";
      bgGradient = "from-emerald-950 to-tech-900";
  }

  // Simple Markdown Parser
  const renderContent = (text: string) => {
      return text.split('\n').map((line, i) => {
          if (!line.trim()) return <div key={i} className="h-2" />;

          // Headers
          if (line.startsWith('###')) {
              return <h3 key={i} className={`text-sm font-bold uppercase tracking-wider mt-4 mb-2 ${accentColor}`}>{line.replace('###', '').trim()}</h3>
          }
          if (line.startsWith('##')) {
              return <h2 key={i} className="text-lg font-bold text-white mt-6 mb-3 border-b border-white/10 pb-1">{line.replace('##', '').trim()}</h2>
          }

          // List Items
          if (line.trim().startsWith('* ')) {
              const content = line.trim().substring(2);
              return (
                  <div key={i} className="flex items-start gap-3 mb-2 pl-2">
                      <div className={`mt-2 w-1.5 h-1.5 rounded-full shrink-0 ${accentColor.replace('text-', 'bg-')}`}></div>
                      <p className="leading-6 m-0 text-slate-300 text-xs">
                          {renderBold(content)}
                      </p>
                  </div>
              );
          }

          // Paragraphs
          return <p key={i} className="mb-2 leading-6 text-slate-300 text-xs">{renderBold(line)}</p>;
      });
  };

  const renderBold = (text: string) => {
      const parts = text.split(/(\*\*.*?\*\*)/g);
      return parts.map((part, index) => {
          if (part.startsWith('**') && part.endsWith('**')) {
              return <strong key={index} className="text-white font-bold">{part.slice(2, -2)}</strong>;
          }
          return part;
      });
  };

  return (
    <div className={`h-full overflow-y-auto relative bg-gradient-to-b ${bgGradient} scrollbar-thin scrollbar-thumb-tech-600`}>
      <div className="p-6 pb-24">
          <button 
            onClick={onClose}
            className="absolute top-4 right-4 p-2 text-slate-400 hover:text-white rounded-full hover:bg-white/10 transition-colors z-20"
          >
            <X className="w-5 h-5" />
          </button>

          <div className="mb-8 pr-8">
            <div className="inline-flex items-center gap-2 mb-2">
                <span className={`h-2 w-2 rounded-full bg-current ${accentColor}`}></span>
                <span className="text-[10px] font-bold uppercase tracking-widest text-slate-400">
                    {nodeDef?.category.replace('_', ' ')}
                </span>
            </div>
            <h2 className="text-3xl font-bold text-white mb-2 tracking-tight">{details.title}</h2>
            <p className="text-sm text-slate-300 font-medium leading-relaxed">{details.subtitle}</p>
          </div>

          <div className="space-y-6">
            
            {/* Main Content Area */}
            <div className="prose prose-invert prose-sm max-w-none">
               {renderContent(details.content)}
            </div>

            {/* KPI / Metrics Section */}
            {details.kpis && details.kpis.length > 0 && (
                <div className="bg-tech-900/50 p-4 rounded-xl border border-tech-700">
                    <h3 className={`text-xs font-bold uppercase tracking-wider mb-3 flex items-center gap-2 ${accentColor}`}>
                        <TrendingUp className="w-4 h-4" /> Interview Metrics (KPIs)
                    </h3>
                    <div className="grid grid-cols-1 gap-2">
                        {details.kpis.map((kpi, idx) => (
                            <div key={idx} className="flex items-center justify-between p-2 bg-tech-800 rounded border border-tech-700/50">
                                <span className="text-xs font-medium text-slate-200">{kpi}</span>
                            </div>
                        ))}
                    </div>
                </div>
            )}

            {/* Cross-Domain Synergy */}
            {details.crossDomainImpact && (
                <div className="bg-tech-900/50 p-4 rounded-xl border border-tech-700">
                     <h3 className={`text-xs font-bold uppercase tracking-wider mb-4 flex items-center gap-2 ${accentColor}`}>
                        <GitMerge className="w-4 h-4" /> Pipeline Integration
                    </h3>
                    
                    <div className="space-y-4">
                        <div>
                            <div className="flex items-center gap-2 text-[10px] font-bold text-emerald-400 uppercase tracking-wide mb-2">
                                <ArrowLeftToLine className="w-3 h-3" /> Upstream Dependencies
                            </div>
                            <div className="space-y-2">
                                {details.crossDomainImpact.inputs.map((item, idx) => (
                                    <div key={idx} className="bg-tech-800/80 p-2.5 rounded border-l-2 border-emerald-500 text-xs">
                                        <span className="font-bold text-slate-200 block mb-0.5">{item.source}</span>
                                        <span className="text-slate-400 leading-snug">{item.benefit}</span>
                                    </div>
                                ))}
                            </div>
                        </div>

                        <div>
                            <div className="flex items-center gap-2 text-[10px] font-bold text-blue-400 uppercase tracking-wide mb-2">
                                <ArrowRightFromLine className="w-3 h-3" /> Downstream Impact
                            </div>
                            <div className="space-y-2">
                                {details.crossDomainImpact.outputs.map((item, idx) => (
                                    <div key={idx} className="bg-tech-800/80 p-2.5 rounded border-l-2 border-blue-500 text-xs">
                                        <span className="font-bold text-slate-200 block mb-0.5">{item.target}</span>
                                        <span className="text-slate-400 leading-snug">{item.improvement}</span>
                                    </div>
                                ))}
                            </div>
                        </div>
                    </div>
                </div>
            )}

            {/* Tech Stack */}
            <div className="bg-tech-900/50 p-4 rounded-xl border border-tech-700">
               <h3 className="text-xs font-bold text-slate-400 uppercase tracking-wider mb-3 flex items-center gap-2">
                 <Layers className="w-4 h-4" /> Tech Stack
               </h3>
               <div className="flex flex-wrap gap-2">
                 {details.techStack.map((tech, idx) => (
                   <span key={idx} className="px-2.5 py-1 bg-white/5 border border-white/10 text-slate-200 text-[11px] font-bold rounded hover:bg-white/10 transition-colors">
                     {tech}
                   </span>
                 ))}
               </div>
            </div>

            {/* Algorithms */}
            <div className="bg-tech-900/50 p-4 rounded-xl border border-tech-700">
               <h3 className="text-xs font-bold text-slate-400 uppercase tracking-wider mb-3 flex items-center gap-2">
                 <Code className="w-4 h-4" /> Key Algorithms
               </h3>
               <ul className="space-y-2">
                 {details.algorithms.map((algo, idx) => (
                   <li key={idx} className="flex items-start gap-2 text-xs text-slate-400">
                     <span className={`mt-1 w-1.5 h-1.5 rounded-full flex-shrink-0 bg-current ${accentColor}`}></span>
                     {algo}
                   </li>
                 ))}
               </ul>
            </div>

            {/* Interview Prep Badge */}
            <div className="flex items-center justify-center gap-2 p-3 rounded-lg border border-white/5 bg-white/5 text-slate-400 text-[10px] font-mono">
                <BookOpen className="w-3 h-3" />
                <span>Interview Ready Content</span>
            </div>

          </div>
      </div>
    </div>
  );
};

export default DetailPanel;