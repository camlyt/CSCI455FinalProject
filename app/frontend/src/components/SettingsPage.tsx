import { RETRIEVER_MODELS } from '../constants';
import { AppSettings } from '../types';

interface SettingsPageProps {
  settings: AppSettings;
  updateSettings: (newSettings: Partial<AppSettings>) => void;
}

export const SettingsPage = ({ settings, updateSettings }: SettingsPageProps) => {
  return (
    <div className="max-w-4xl mx-auto px-8 pb-32">
      <div className="mb-12">
        <h2 className="text-4xl font-display font-black tracking-tighter mb-2 text-white">Advanced Settings</h2>
        <p className="text-slate-400">Configure retrieval parameters and model selection for the verification pipeline.</p>
      </div>

      <div className="advanced-settings bg-slate-900/50 border-slate-800">
        <div className="grid grid-cols-1 md:grid-cols-2 gap-12 p-12">
          {/* Retrieval Settings */}
          <div className="space-y-8">
            <div className="setting-card">
              <label className="text-xs font-mono font-bold uppercase tracking-widest text-slate-400">Retrieval Volume (K)</label>
              <div className="flex gap-2">
                {[5, 10, 20, 50].map(k => (
                  <button 
                    key={k} 
                    onClick={() => updateSettings({ topK: k })}
                    className={`flex-1 py-3 rounded-xl border text-sm font-bold transition-all ${settings.topK === k ? 'bg-sky-500 text-slate-900 border-sky-500 shadow-lg shadow-sky-500/20' : 'bg-slate-900 text-slate-400 border-slate-800 hover:border-slate-600'}`}
                  >
                    {k}
                  </button>
                ))}
              </div>
            </div>

            <div className="setting-card">

                <label className="text-xs font-mono font-bold uppercase tracking-widest text-slate-400">
                    CrossEncoder Reranking
                </label>

                <div className="flex gap-3">

                    <button
                    onClick={() =>
                        updateSettings({ useReranker: true })
                    }
                    className={`flex-1 py-3 rounded-xl border text-sm font-bold transition-all ${
                        settings.useReranker
                        ? 'bg-sky-500 text-slate-900 border-sky-500 shadow-lg shadow-sky-500/20'
                        : 'bg-slate-900 text-slate-400 border-slate-800 hover:border-slate-600'
                    }`}
                    >
                    ENABLED
                    </button>

                    <button
                    onClick={() =>
                        updateSettings({ useReranker: false })
                    }
                    className={`flex-1 py-3 rounded-xl border text-sm font-bold transition-all ${
                        !settings.useReranker
                        ? 'bg-red-500 text-white border-red-500 shadow-lg shadow-red-500/20'
                        : 'bg-slate-900 text-slate-400 border-slate-800 hover:border-slate-600'
                    }`}
                    >
                    DISABLED
                    </button>

                </div>

                <p className="text-xs text-slate-500 leading-relaxed italic mt-2">
                    Enables CrossEncoder reranking after dense retrieval. Improves evidence quality and ranking precision at the cost of additional latency.
                </p>

                </div>
          </div>

          {/* Model Selection */}
          <div className="space-y-8">
            <div className="setting-card">
              <label className="text-xs font-mono font-bold uppercase tracking-widest text-slate-400">Active Retriever</label>
              <div className="space-y-2">
                {RETRIEVER_MODELS.map(m => (
                  <button 
                    key={m.id} 
                    onClick={() => updateSettings({ retriever: m.id })}
                    className={`w-full p-4 rounded-xl border-2 text-left transition-all ${settings.retriever === m.id ? 'border-sky-500 bg-sky-500/10' : 'border-slate-800 bg-slate-900/50 hover:border-slate-600'}`}
                  >
                    <div className={`font-bold text-sm ${settings.retriever === m.id ? 'text-sky-400' : 'text-slate-300'}`}>{m.name}</div>
                    <div className="text-[10px] text-slate-500 uppercase tracking-tighter">{m.desc}</div>
                  </button>
                ))}
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};
