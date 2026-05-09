import { Cpu, ShieldCheck } from 'lucide-react';
import { PIPELINE_STEPS } from '../constants';

export const TutorialPage = () => {
  return (
    <div className="max-w-4xl mx-auto px-8 pb-32">
      <div className="mb-20 text-center">
        <h2 className="text-5xl font-display font-black tracking-tighter mb-4 text-white">Watch the Pipeline Work</h2>
        <p className="text-slate-400 text-lg">A scroll-based demonstration of how our system processes factual claims.</p>
      </div>

      <div className="tutorial-section">
        {PIPELINE_STEPS.map((step, i) => (
          <div key={i} className="pipeline-step pb-20">
            <div className="md:w-1/3 space-y-4">
              <div className="w-12 h-12 bg-sky-500 text-slate-950 rounded-2xl flex items-center justify-center text-xl font-bold font-display shadow-lg shadow-sky-500/20">
                0{step.id}
              </div>
              <h3 className="text-3xl font-display font-bold tracking-tight text-white">{step.title}</h3>
              <p className="text-slate-400 leading-relaxed">{step.description}</p>
            </div>
            <div className="md:w-2/3">
              <div className="pipeline-card bg-slate-900 border-slate-800 shadow-2xl">
                {step.id === 1 && (
                  <div className="p-4 bg-slate-950 rounded-xl border border-dashed border-slate-800 font-medium italic text-sky-400">
                    "Roman Atwood is a content creator."
                  </div>
                )}
                {step.id === 2 && (
                  <div className="space-y-4">
                    <div className="flex items-center gap-4 p-3 bg-sky-500/10 rounded-lg text-sky-400 text-sm">
                      <Cpu className="w-5 h-5" /> Embedding Claim @ 768 Dimensions
                    </div>
                    <div className="grid grid-cols-8 gap-1 opacity-20">
                      {Array.from({ length: 16 }).map((_, i) => (
                        <div key={i} className="h-4 bg-sky-500 rounded"></div>
                      ))}
                    </div>
                  </div>
                )}
                {step.id === 3 && (
                  <div className="space-y-2">
                    {[1, 2, 3].map(i => (
                      <div key={i} className={`p-3 rounded-lg border text-xs ${i === 1 ? 'border-sky-500 bg-sky-500/5 text-sky-400' : 'border-slate-800 text-slate-500'}`}>
                        Evidence candidate sequence #{i} score update...
                      </div>
                    ))}
                  </div>
                )}
                {step.id === 4 && (
                  <div className="flex items-center justify-center p-8 bg-emerald-500/5 rounded-3xl">
                    <ShieldCheck className="w-16 h-16 text-emerald-500 animate-pulse" />
                  </div>
                )}
              </div>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
};
