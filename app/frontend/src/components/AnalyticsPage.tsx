import { BarChart3, Activity, AlertCircle } from 'lucide-react';
import { motion } from 'motion/react';

export const AnalyticsPage = () => {
  return (
    <div className="max-w-6xl mx-auto px-8 pb-32">
      <div className="mb-12">
        <h2 className="text-4xl font-display font-black tracking-tighter mb-2 text-white">Model Insights</h2>
        <p className="text-slate-400">Live performance indicators and interpretability visualizations for the current model snapshot.</p>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6 mb-12">
        {[
          { label: 'Recall @1', val: '86.4%', icon: Activity },
          { label: 'Recall @5', val: '92.1%', icon: Activity },
          { label: 'Avg Pipeline Accuracy', val: '89.2%', icon: BarChart3 },
          { label: 'NEI Predicted Rate', val: '12.4%', icon: AlertCircle }
        ].map((m, i) => (
          <div key={i} className="metric-card bg-slate-900/50 border-slate-800">
            <div className="flex justify-between items-start mb-6">
              <span className="text-[10px] font-mono font-bold text-slate-500 uppercase tracking-widest">{m.label}</span>
              <m.icon className="w-4 h-4 text-slate-600" />
            </div>
            <div className="text-4xl font-display font-black tracking-tighter text-white">{m.val}</div>
          </div>
        ))}
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-12 mt-12">
        <div className="lg:col-span-2 p-10 bg-slate-900/50 rounded-3xl border border-slate-800 shadow-sm">
          <h3 className="text-xl font-display font-bold text-white mb-6">Claim–Evidence Token Heatmap</h3>
          <p className="text-sm text-slate-400 mb-8 max-w-xl">
            This visualization displays mock alignment scores between individual claim tokens and evidence tokens, helping identify which parts of the evidence triggered the verifier.
          </p>
          <div className="grid gap-1 mt-6" style={{ gridTemplateColumns: 'repeat(10, 1fr)' }}>
            {Array.from({ length: 40 }).map((_, i) => {
              const opacity = Math.random() * 0.8 + 0.1;
              return (
                <div 
                  key={i} 
                  className="w-12 h-12 rounded-sm flex items-center justify-center text-[10px] font-mono transition-opacity bg-sky-500" 
                  style={{ opacity }}
                  title={`Score: ${opacity.toFixed(2)}`}
                >
                  <span className="text-slate-900 font-bold">{(opacity * 100).toFixed(0)}</span>
                </div>
              );
            })}
          </div>
          <div className="flex justify-between mt-4 font-mono text-[10px] text-slate-500 uppercase">
            <span>Low Alignment</span>
            <span>High Alignment</span>
          </div>
        </div>

        <div className="p-10 bg-slate-900/50 rounded-3xl border border-slate-800 shadow-sm space-y-8">
          <h3 className="text-xl font-display font-bold text-white">Label Confidence Breakdown</h3>
          <div className="space-y-6">
            {[
              { label: 'Entailment', score: 0.86, color: 'bg-emerald-500' },
              { label: 'Neutral', score: 0.09, color: 'bg-slate-500' },
              { label: 'Contradiction', score: 0.05, color: 'bg-red-500' }
            ].map(l => (
              <div key={l.label} className="space-y-2">
                <div className="flex justify-between text-xs font-mono font-bold uppercase">
                  <span className="text-slate-400">{l.label}</span>
                  <span className="text-white">{(l.score * 100).toFixed(1)}%</span>
                </div>
                <div className="h-2 bg-slate-800 rounded-full overflow-hidden">
                  <motion.div initial={{ width: 0 }} animate={{ width: `${l.score * 100}%` }} className={`h-full ${l.color} shadow-[0_0_8px_rgba(0,0,0,0.5)]`}></motion.div>
                </div>
              </div>
            ))}
          </div>
        </div>
      </div>
    </div>
  );
};
