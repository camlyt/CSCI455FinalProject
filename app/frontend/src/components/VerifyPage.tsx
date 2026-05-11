import { useState } from 'react';
import { motion, AnimatePresence } from 'motion/react';
import { History, Zap, CheckCircle2, CheckCircle, ExternalLink, BrainCircuit } from 'lucide-react';
import { MOCK_PREVIOUS_CLAIMS } from '../constants';
import { AppSettings } from '../types';

interface VerifyPageProps {
  isVerifying: boolean;
  handleVerify: (claim: string) => void;
  showResult: boolean;
  settings: AppSettings;
  result: any;
  error: string | null;
}

export const VerifyPage = ({  isVerifying, handleVerify, showResult, settings, result, error}: VerifyPageProps) => {
  const [claimText, setClaimText] = useState("Roman Atwood is a content creator.");
  
  const normalizedLabel = result?.label?.toUpperCase().replaceAll("_", " ");
  const isSupports = normalizedLabel === "SUPPORTS";
  const isRefutes = normalizedLabel === "REFUTES";
  const isNEI =
    normalizedLabel === "NOT ENOUGH INFO" ||
    normalizedLabel === "NEI";
  
  return (
    <div className="grid grid-cols-1 lg:grid-cols-12 gap-12 max-w-7xl mx-auto px-8 pb-32">
      {/* Sidebar: History */}
      <aside className="lg:col-span-3 space-y-8 hidden lg:block">
        <div className="space-y-4">
          <h3 className="text-xs font-mono font-bold text-slate-500 uppercase tracking-widest flex items-center gap-2">
            <History className="w-4 h-4" /> Claim History
          </h3>
          <div className="space-y-2">
            {MOCK_PREVIOUS_CLAIMS.map((c, i) => (
              <div key={i} className="p-3 bg-slate-900/50 border border-slate-800 rounded-xl text-xs text-slate-400 hover:border-slate-600 cursor-pointer transition-all truncate">
                {c}
              </div>
            ))}
          </div>
        </div>

        <div className="p-6 bg-sky-500/10 border border-sky-500/20 rounded-2xl text-slate-50 space-y-4 shadow-xl shadow-sky-500/5">
          <div className="w-8 h-8 bg-sky-500 rounded-lg flex items-center justify-center text-slate-950">
            <Zap className="w-5 h-5" />
          </div>
          <h4 className="font-display font-medium leading-tight text-white">Fast, Reliable, Interpretable.</h4>
          <p className="text-slate-400 text-xs text-balance">Exposing the full pipeline of automated fact-checking.</p>
          <div className="text-[10px] font-mono text-sky-500/60 mt-4 border-t border-sky-500/10 pt-4">
            SETTING: K={settings.topK} | T={settings.threshold}
          </div>
        </div>
      </aside>

      {/* Main Content */}
      <main className="lg:col-span-9 space-y-12">
        <section className="space-y-6">

            {error && (
            <div className="p-4 rounded-xl border border-red-500/30 bg-red-500/10 text-red-300 text-sm">
                {error}
            </div>
            )}

          <div className="relative">
            <textarea 
              className="claim-input"
              value={claimText}
              onChange={(e) => setClaimText(e.target.value)}
              placeholder="Enter your claim here..."
            />
            <div className="absolute top-4 right-4 bg-slate-800 text-slate-400 font-mono text-[10px] px-2 py-1 rounded">
              L: {claimText.length}
            </div>
          </div>
          <div className="flex flex-col md:flex-row justify-between items-center gap-6">
            <p className="text-sm text-slate-500 italic max-w-xl">
              Verification uses FEVER-trained DeBERTa models against 2018 Wikipedia snapshots.
            </p>
            <button 
              onClick={() => handleVerify(claimText)}
              disabled={isVerifying}
              className="verify-button w-full md:w-auto"
            >
              {isVerifying ? (
                <div className="flex items-center gap-2">
                  <motion.div animate={{ rotate: 360 }} transition={{ duration: 1, repeat: Infinity, ease: "linear" }}>
                    <BrainCircuit className="w-5 h-5" />
                  </motion.div>
                  Verifying...
                </div>
              ) : 'Run Verification'}
            </button>
          </div>
        </section>

        {/* Results */}
        <AnimatePresence>
          {showResult && !isVerifying && (
            <motion.div
              initial={{ opacity: 0, y: 30 }}
              animate={{ opacity: 1, y: 0 }}
              className="space-y-16"
            >
              {/* Score Card */}
              <div className="result-card">
                <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-12">
                  <div className="lg:col-span-2">
                    <span className="text-xs font-mono font-bold text-slate-500 uppercase tracking-widest block mb-4">
                      Predicted Verdict
                    </span>
                    <div className="flex items-center gap-6">
                        <div
                            className={`result-label text-xl shadow-2xl ${
                                isSupports
                                ? "label-supports shadow-emerald-500/20"
                                : isRefutes
                                ? "label-refutes shadow-red-500/20"
                                : "label-nei shadow-slate-500/20"
                            }`}
                            >
                            <CheckCircle2 className="w-6 h-6 mr-3" />
                            {normalizedLabel}
                        </div>
                    </div>
                    <div className="mt-8 p-4 bg-emerald-500/5 rounded-xl border border-emerald-500/20 text-emerald-100 text-sm">
                      <span className="font-bold block mb-1">Reasoning Explanation</span>
                        The model predicted {result?.label} using the retrieved evidence and the current verifier scores.
                           
                    </div>
                  </div>

                  <div>
                    <span className="text-xs font-mono font-bold text-slate-500 uppercase tracking-widest block mb-4">
                      Confidence
                    </span>
                    <div className="confidence-score italic mb-2 font-black">{result?.confidence?.toFixed(2)}</div>
                    <div className="score-bar-container">
                      <motion.div initial={{ width: 0 }} animate={{ width: `${(result?.confidence ?? 0) * 100}%` }} transition={{ duration: 1.5, ease: "easeOut" }} className="score-bar-fill"></motion.div>
                    </div>
                    <div className="text-xs font-mono font-bold text-white tracking-wide block mb-4 pt-5 flex flex-col gap-1">
                        <span>
                            Entailment: {result?.scores?.entailment?.toFixed(2)}
                        </span>

                        <span>
                            Neutral: {result?.scores?.neutral?.toFixed(2)}
                        </span>

                        <span>
                            Contradiction: {result?.scores?.contradiction?.toFixed(2)}
                        </span>
                    </div>
                  </div>

                  <div className="space-y-6">
                    <div>
                      <span className="text-xs font-mono font-bold text-slate-500 uppercase mb-2 block">Evidence Count</span>
                      <span className="text-xl font-bold text-white">{settings.topK} sentences</span>
                    </div>
                    <div>
                      <span className="text-xs font-mono font-bold text-slate-500 uppercase mb-2 block">Reranker</span>
                      <span className="text-xs font-mono bg-slate-800 text-slate-300 px-2 py-1 rounded inline-block">Cross-Encoder-v3</span>
                    </div>
                  </div>
                </div>
              </div>

              {/* Evidence Trail */}
              <section className="evidence-section space-y-8">
                <div className="flex items-center justify-between border-b border-slate-800 pb-4">
                  <h2 className="text-2xl font-display font-black tracking-tighter text-white">Evidence Trail</h2>
                  <div className="flex items-center gap-4 text-[10px] font-mono text-slate-500 uppercase">
                    <span>Ranked Output</span>
                    <span className="w-px h-3 bg-slate-800"></span>
                    <span>Wikipedia Retrieval</span>
                  </div>
                </div>

                <div className="space-y-6">
                  {result?.evidence?.map((ev: any, i: number) => (
                    <motion.div
                      key={i}
                      initial={{ opacity: 0, x: -20 }}
                      animate={{ opacity: 1, x: 0 }}
                      transition={{ delay: i * 0.1 }}
                      className="evidence-card"
                    >
                      <div className="flex justify-between items-start mb-6">
                        <div className="space-y-1">
                          <span className="text-[10px] font-mono text-slate-500 uppercase font-bold tracking-widest block">
                            Evidence Rank 0{i + 1}
                          </span>
                          <h4 className="text-lg font-bold text-slate-100 flex items-center gap-2">
                            {ev.page.replace(/_/g, ' ')}
                            <ExternalLink className="w-4 h-4 text-slate-600 hover:text-sky-500 cursor-pointer transition-colors" />
                          </h4>
                        </div>
                        <div className="bg-slate-900 px-2 py-1 rounded font-mono text-[10px] text-slate-500 border border-slate-800">
                          S_ID: {ev.sentence_id}
                        </div>
                      </div>

                      <p className="text-slate-300 leading-relaxed text-lg">
                        {ev.text}
                      </p>

                      <div className="grid grid-cols-2 lg:grid-cols-4 gap-8 mt-8 pt-8 border-t border-slate-800/50">
                        <div className="space-y-2">
                          <span className="text-[10px] font-mono text-slate-500 uppercase">Similarity</span>
                          <div className="font-black text-sm tracking-tighter font-mono text-slate-100">{ev.score.toFixed(3)}</div>
                          <div className="h-1 bg-slate-800 rounded-full overflow-hidden">
                            <div className="h-full bg-slate-500" style={{ width: `${ev.score * 100}%` }}></div>
                          </div>
                        </div>
                        <div className="space-y-2">
                          <span className="text-[10px] font-mono text-slate-500 uppercase">Rerank Score</span>
                          <div className="font-black text-sm tracking-tighter font-mono text-sky-400">{ev.rerank_score.toFixed(2)}</div>
                          <div className="h-1 bg-slate-800 rounded-full overflow-hidden">
                            <div className="h-full bg-sky-500" style={{ width: `${(ev.rerank_score / 6) * 100}%` }}></div>
                          </div>
                        </div>
                      </div>
                    </motion.div>
                  ))}
                </div>
              </section>
            </motion.div>
          )}
        </AnimatePresence>
      </main>
    </div>
  );
};
