import React from 'react';
import { SimulationStepData } from '../types';
import { Code, Database, Video, Loader2, Play, TrendingUp, TrendingDown, CheckCircle2 } from 'lucide-react';

interface DataInspectorProps {
  data: SimulationStepData | null;
  onGenerateVideo?: () => void;
  videoUrl?: string | null;
  isGeneratingVideo?: boolean;
}

const DataInspector: React.FC<DataInspectorProps> = ({ 
  data, 
  onGenerateVideo, 
  videoUrl, 
  isGeneratingVideo 
}) => {
  if (!data) {
    return (
      <div className="h-full flex flex-col items-center justify-center text-slate-600 bg-tech-900 rounded-xl border border-tech-700 p-4">
        <Database className="w-8 h-8 mb-2 opacity-20" />
        <span className="text-[10px] uppercase tracking-widest font-bold opacity-50">Aguardando Sinais</span>
      </div>
    );
  }

  // Helper to render video section
  const renderVideoSection = () => (
     <div className="mb-4 bg-tech-800 rounded-lg border border-tech-600 p-3 flex flex-col items-center shadow-lg">
        <div className="flex items-center justify-between w-full mb-2">
            <div className="flex items-center gap-2">
                <div className="p-1 bg-red-500/10 rounded text-red-500">
                    <Video className="w-3.5 h-3.5" />
                </div>
                <span className="text-xs font-bold text-slate-200">Visualizador IA</span>
            </div>
            {onGenerateVideo && !videoUrl && !isGeneratingVideo && (
                <button 
                    onClick={onGenerateVideo}
                    className="flex items-center gap-1.5 px-3 py-1 bg-red-600 hover:bg-red-500 text-white text-[10px] font-bold rounded transition shadow-lg"
                >
                    <Play className="w-2.5 h-2.5 fill-current" /> Gerar
                </button>
            )}
        </div>
        
        {isGeneratingVideo && (
            <div className="w-full h-32 bg-black/50 rounded flex flex-col items-center justify-center text-red-500 gap-2 border border-white/5">
                <Loader2 className="w-6 h-6 animate-spin" />
                <span className="text-[10px] animate-pulse font-mono">Renderizando Vídeo Veo...</span>
            </div>
        )}

        {videoUrl && (
            <div className="w-full h-48 bg-black rounded overflow-hidden border border-slate-700 relative group">
                <video src={videoUrl} controls autoPlay loop className="w-full h-full object-contain" />
            </div>
        )}
    </div>
  );

  return (
    <div className="h-full flex flex-col bg-tech-900 rounded-xl border border-tech-700 shadow-xl overflow-hidden">
      <div className="bg-tech-800 px-4 py-2 border-b border-tech-700 flex items-center justify-between">
        <div className="flex items-center gap-2">
           <Code className="w-3.5 h-3.5 text-ifood-red" />
           <span className="text-xs font-bold text-slate-200">{data.title}</span>
        </div>
        <span className="text-[9px] text-emerald-400 font-mono font-bold">
            AO VIVO
        </span>
      </div>
      
      <div className="p-4 overflow-y-auto flex-1 font-mono text-xs scrollbar-thin scrollbar-thumb-tech-600 scrollbar-track-transparent">
         {/* Show Video Generator if available */}
         {onGenerateVideo && renderVideoSection()}

         {/* Impact / ROI Section (New) */}
         {(data.impact || data.roiMetric) && (
             <div className="mb-4 bg-[#0d1f12] border border-emerald-900/50 rounded-lg p-3">
                 {data.impact && (
                     <div className="flex items-start gap-2 mb-3">
                        <CheckCircle2 className="w-3.5 h-3.5 text-emerald-500 shrink-0 mt-0.5" />
                        <span className="text-slate-300 leading-snug">{data.impact}</span>
                     </div>
                 )}
                 {data.roiMetric && (
                     <div className="flex items-center justify-between bg-black/20 rounded p-2 border border-white/5">
                         <span className="text-emerald-500/70 font-bold uppercase tracking-wider text-[10px]">{data.roiMetric.label}</span>
                         <div className="flex items-center gap-1.5">
                             <span className="text-white font-bold text-sm">{data.roiMetric.value}</span>
                             {data.roiMetric.trend === 'up' ? (
                                 <TrendingUp className="w-3 h-3 text-emerald-500" />
                             ) : (
                                 <TrendingDown className="w-3 h-3 text-emerald-500" />
                             )}
                         </div>
                     </div>
                 )}
             </div>
         )}

         <div className="mb-3 text-slate-400 border-b border-tech-700 pb-2 italic leading-relaxed">
            {data.description}
         </div>
         
         <div className="bg-[#0b0b0d] p-3 rounded border border-tech-700 text-blue-300 shadow-inner">
            <pre className="whitespace-pre-wrap">{JSON.stringify(data.data, null, 2)}</pre>
         </div>
      </div>
    </div>
  );
};

export default DataInspector;