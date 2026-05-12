import {
  Cpu,
  ShieldCheck,
  Database,
  Layers3,
  BrainCircuit,
  Globe,
  Search,
  Binary,
  ArrowRight,
} from 'lucide-react';

import { motion } from 'framer-motion';

export const TutorialPage = () => {
  return (
    <div className="max-w-7xl mx-auto px-8 pb-40">
      {/* Hero */}
      <section className="mb-28 text-center space-y-6">
        <div className="inline-flex items-center gap-3 px-5 py-2 rounded-full border border-sky-500/20 bg-sky-500/10 text-sky-300 text-xs font-mono uppercase tracking-[0.25em]">
          <BrainCircuit className="w-4 h-4" />
          Retrieval-Augmented Fact Verification
        </div>

        <h1 className="text-6xl md:text-7xl font-display font-black tracking-tighter text-white leading-none">
          Inside the
          <br />
          Verification Pipeline
        </h1>

        <p className="max-w-3xl mx-auto text-slate-400 text-xl leading-relaxed">
          A deep walkthrough of the full FEVER-based
          retrieval and verification system, from
          semantic embedding generation to final
          natural language inference prediction.
        </p>
      </section>

      {/* Architecture Overview */}
      <section className="mb-32">
        <div className="result-card">
          <div className="flex items-center justify-between flex-wrap gap-6 mb-12">
            <div>
              <h2 className="text-4xl font-display font-black tracking-tight text-white mb-3">
                System Architecture
              </h2>

              <p className="text-slate-400 max-w-3xl leading-relaxed">
                The pipeline combines dense semantic
                retrieval, cross-encoder reranking,
                and DeBERTa-based natural language
                inference to verify factual claims
                against Wikipedia evidence.
              </p>
            </div>

            <div className="px-5 py-3 rounded-2xl border border-slate-800 bg-slate-900">
              <div className="text-[10px] font-mono uppercase tracking-widest text-slate-500 mb-1">
                Current Stack
              </div>

              <div className="text-white font-bold">
                MiniLM + FAISS + DeBERTa-v3
              </div>
            </div>
          </div>

          <div className="grid grid-cols-1 lg:grid-cols-5 gap-6">
            {[
              {
                icon: Search,
                title: 'Claim Input',
                desc: 'User claim normalization and preprocessing.',
              },
              {
                icon: Cpu,
                title: 'Dense Retrieval',
                desc: 'Semantic vector retrieval over Wikipedia evidence.',
              },
              {
                icon: Layers3,
                title: 'Reranking',
                desc: 'CrossEncoder relevance scoring and filtering.',
              },
              {
                icon: ShieldCheck,
                title: 'Verification',
                desc: 'Natural language inference prediction.',
              },
              {
                icon: BrainCircuit,
                title: 'Explainability',
                desc: 'Confidence scores and evidence visualization.',
              },
            ].map((item, i) => (
              <motion.div
                key={item.title}
                initial={{ opacity: 0, y: 10 }}
                whileInView={{ opacity: 1, y: 0 }}
                transition={{ delay: i * 0.05 }}
                viewport={{ once: true }}
                className="p-6 rounded-3xl border border-slate-800 bg-slate-900/40 relative overflow-hidden"
              >
                <div className="absolute top-0 right-0 w-24 h-24 bg-sky-500/5 blur-3xl rounded-full" />

                <div className="w-12 h-12 rounded-2xl bg-sky-500/10 border border-sky-500/20 flex items-center justify-center mb-6">
                  <item.icon className="w-6 h-6 text-sky-400" />
                </div>

                <h3 className="text-white font-bold text-lg mb-3">
                  {item.title}
                </h3>

                <p className="text-sm text-slate-400 leading-relaxed">
                  {item.desc}
                </p>

                {i < 4 && (
                  <ArrowRight className="hidden lg:block absolute -right-4 top-1/2 -translate-y-1/2 text-slate-700 w-8 h-8" />
                )}
              </motion.div>
            ))}
          </div>
        </div>
      </section>

      {/* Pipeline Stages */}
      <section className="space-y-32">
        {/* STEP 1 */}
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-16 items-center">
          <div className="space-y-8">
            <div className="w-16 h-16 rounded-3xl bg-sky-500 text-slate-950 flex items-center justify-center text-2xl font-black shadow-2xl shadow-sky-500/20">
              01
            </div>

            <div>
              <div className="text-xs uppercase tracking-[0.25em] font-mono text-sky-500 mb-3">
                Claim Processing
              </div>

              <h2 className="text-5xl font-display font-black tracking-tight text-white mb-6">
                Semantic Claim Encoding
              </h2>

              <p className="text-slate-400 leading-relaxed text-lg">
                Incoming claims are transformed into
                dense semantic vector representations
                using SentenceTransformer embeddings.
                These embeddings allow the system to
                search Wikipedia semantically rather
                than relying on exact keyword overlap.
              </p>
            </div>

            <div className="grid grid-cols-2 gap-4">
              <div className="p-5 rounded-2xl border border-slate-800 bg-slate-900/40">
                <div className="text-[10px] uppercase tracking-widest text-slate-500 font-mono mb-2">
                  Embedding Model
                </div>

                <div className="text-white font-bold">
                  all-MiniLM-L6-v2
                </div>
              </div>

              <div className="p-5 rounded-2xl border border-slate-800 bg-slate-900/40">
                <div className="text-[10px] uppercase tracking-widest text-slate-500 font-mono mb-2">
                  Vector Size
                </div>

                <div className="text-white font-bold">
                  384 Dimensions
                </div>
              </div>
            </div>
          </div>

          <div className="pipeline-card space-y-8">
            <div className="p-5 rounded-2xl bg-slate-950 border border-slate-800 font-medium italic text-sky-400 text-lg">
              "Roman Atwood is a content creator."
            </div>

            <div className="space-y-3">
              <div className="flex items-center justify-between text-xs font-mono uppercase tracking-widest text-slate-500">
                <span>Embedding Representation</span>
                <span>384D</span>
              </div>

              <div className="grid grid-cols-12 gap-1">
                {Array.from({ length: 72 }).map(
                  (_, i) => (
                    <motion.div
                      key={i}
                      initial={{ opacity: 0.2 }}
                      animate={{
                        opacity:
                          Math.random() * 0.8 + 0.2,
                      }}
                      transition={{
                        repeat: Infinity,
                        repeatType: 'reverse',
                        duration:
                          Math.random() * 2 + 1,
                      }}
                      className="h-4 rounded bg-sky-500"
                    />
                  )
                )}
              </div>
            </div>
          </div>
        </div>

        {/* STEP 2 */}
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-16 items-center">
          <div className="pipeline-card space-y-8 order-2 lg:order-1">
            <div className="flex items-center gap-3 text-sky-400 font-mono text-sm uppercase tracking-widest">
              <Database className="w-5 h-5" />
              FAISS Semantic Search
            </div>

            {[
              {
                page: 'Roman_Atwood',
                score: 0.92,
              },
              {
                page: 'YouTube_Personalities',
                score: 0.81,
              },
              {
                page: 'Digital_Influencers',
                score: 0.77,
              },
            ].map((item, i) => (
              <motion.div
                key={item.page}
                initial={{ opacity: 0, x: -20 }}
                whileInView={{
                  opacity: 1,
                  x: 0,
                }}
                transition={{ delay: i * 0.1 }}
                viewport={{ once: true }}
                className="p-5 rounded-2xl border border-slate-800 bg-slate-900/40 flex justify-between items-center"
              >
                <div>
                  <div className="text-white font-bold">
                    {item.page}
                  </div>

                  <div className="text-xs text-slate-500 font-mono mt-1">
                    Wikipedia Candidate
                  </div>
                </div>

                <div className="text-sky-400 font-black text-xl font-mono">
                  {item.score}
                </div>
              </motion.div>
            ))}
          </div>

          <div className="space-y-8 order-1 lg:order-2">
            <div className="w-16 h-16 rounded-3xl bg-sky-500 text-slate-950 flex items-center justify-center text-2xl font-black shadow-2xl shadow-sky-500/20">
              02
            </div>

            <div>
              <div className="text-xs uppercase tracking-[0.25em] font-mono text-sky-500 mb-3">
                Retrieval Layer
              </div>

              <h2 className="text-5xl font-display font-black tracking-tight text-white mb-6">
                Dense Wikipedia Retrieval
              </h2>

              <p className="text-slate-400 leading-relaxed text-lg">
                The claim embedding is queried against
                a FAISS vector index constructed from
                Wikipedia evidence sentences. Dense
                retrieval enables semantic matching
                between claims and evidence even when
                wording differs significantly.
              </p>
            </div>

            <div className="flex items-center gap-6 flex-wrap">
              <div className="px-5 py-3 rounded-2xl bg-slate-900 border border-slate-800">
                <div className="text-[10px] uppercase tracking-widest text-slate-500 font-mono mb-1">
                  Retrieval Method
                </div>

                <div className="text-white font-bold">
                  FAISS Similarity Search
                </div>
              </div>

              <div className="px-5 py-3 rounded-2xl bg-slate-900 border border-slate-800">
                <div className="text-[10px] uppercase tracking-widest text-slate-500 font-mono mb-1">
                  Corpus
                </div>

                <div className="text-white font-bold">
                  Wikipedia Evidence
                </div>
              </div>
            </div>
          </div>
        </div>

        {/* STEP 3 */}
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-16 items-center">
          <div className="space-y-8">
            <div className="w-16 h-16 rounded-3xl bg-sky-500 text-slate-950 flex items-center justify-center text-2xl font-black shadow-2xl shadow-sky-500/20">
              03
            </div>

            <div>
              <div className="text-xs uppercase tracking-[0.25em] font-mono text-sky-500 mb-3">
                CrossEncoder
              </div>

              <h2 className="text-5xl font-display font-black tracking-tight text-white mb-6">
                Evidence Reranking
              </h2>

              <p className="text-slate-400 leading-relaxed text-lg">
                Retrieved evidence candidates are
                rescored using a CrossEncoder reranker,
                allowing the model to jointly analyze
                the claim and evidence sentence for
                deeper semantic alignment.
              </p>
            </div>

            <div className="p-6 rounded-3xl border border-sky-500/20 bg-sky-500/5">
              <div className="text-xs uppercase tracking-widest text-sky-400 font-mono mb-3">
                Performance Gain
              </div>

              <div className="text-5xl font-black tracking-tighter text-white mb-2">
                +14.7%
              </div>

              <div className="text-slate-400">
                Recall@1 improvement from reranking
              </div>
            </div>
          </div>

          <div className="pipeline-card space-y-4">
            {[
              {
                rank: 1,
                score: 5.42,
                active: true,
              },
              {
                rank: 2,
                score: 4.11,
                active: false,
              },
              {
                rank: 3,
                score: 3.62,
                active: false,
              },
            ].map((item) => (
              <div
                key={item.rank}
                className={`p-5 rounded-2xl border transition-all ${
                  item.active
                    ? 'border-sky-500 bg-sky-500/10'
                    : 'border-slate-800 bg-slate-900/30'
                }`}
              >
                <div className="flex justify-between items-center">
                  <div>
                    <div className="text-white font-bold">
                      Evidence Candidate #{item.rank}
                    </div>

                    <div className="text-xs text-slate-500 font-mono mt-1">
                      CrossEncoder Relevance
                    </div>
                  </div>

                  <div
                    className={`text-2xl font-black font-mono ${
                      item.active
                        ? 'text-sky-400'
                        : 'text-slate-500'
                    }`}
                  >
                    {item.score}
                  </div>
                </div>
              </div>
            ))}
          </div>
        </div>

        {/* STEP 4 */}
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-16 items-center">
          <div className="pipeline-card space-y-10 order-2 lg:order-1">
            <div className="flex items-center justify-between">
              <div>
                <div className="text-xs uppercase tracking-widest text-slate-500 font-mono mb-2">
                  DeBERTa-v3 NLI
                </div>

                <div className="text-3xl font-black text-white">
                  SUPPORTS
                </div>
              </div>

              <ShieldCheck className="w-16 h-16 text-emerald-500" />
            </div>

            <div className="space-y-5">
              {[
                {
                  label: 'Entailment',
                  score: 0.91,
                  color: 'bg-emerald-500',
                },
                {
                  label: 'Neutral',
                  score: 0.06,
                  color: 'bg-slate-500',
                },
                {
                  label: 'Contradiction',
                  score: 0.03,
                  color: 'bg-red-500',
                },
              ].map((item) => (
                <div
                  key={item.label}
                  className="space-y-2"
                >
                  <div className="flex justify-between text-sm">
                    <span className="text-slate-300">
                      {item.label}
                    </span>

                    <span className="font-mono text-white">
                      {(item.score * 100).toFixed(1)}%
                    </span>
                  </div>

                  <div className="h-3 bg-slate-800 rounded-full overflow-hidden">
                    <motion.div
                      initial={{ width: 0 }}
                      whileInView={{
                        width: `${item.score * 100}%`,
                      }}
                      transition={{
                        duration: 1,
                      }}
                      viewport={{ once: true }}
                      className={`h-full ${item.color}`}
                    />
                  </div>
                </div>
              ))}
            </div>
          </div>

          <div className="space-y-8 order-1 lg:order-2">
            <div className="w-16 h-16 rounded-3xl bg-sky-500 text-slate-950 flex items-center justify-center text-2xl font-black shadow-2xl shadow-sky-500/20">
              04
            </div>

            <div>
              <div className="text-xs uppercase tracking-[0.25em] font-mono text-sky-500 mb-3">
                Verification Layer
              </div>

              <h2 className="text-5xl font-display font-black tracking-tight text-white mb-6">
                Natural Language Inference
              </h2>

              <p className="text-slate-400 leading-relaxed text-lg">
                The verifier jointly evaluates the
                claim and reranked evidence using a
                DeBERTa-v3 NLI model. Final predictions
                are generated across SUPPORTS,
                REFUTES, and NOT ENOUGH INFO classes.
              </p>
            </div>

            <div className="grid grid-cols-2 gap-4">
              <div className="p-5 rounded-2xl border border-slate-800 bg-slate-900/40">
                <div className="text-[10px] uppercase tracking-widest text-slate-500 font-mono mb-2">
                  Verifier
                </div>

                <div className="text-white font-bold">
                  DeBERTa-v3
                </div>
              </div>

              <div className="p-5 rounded-2xl border border-slate-800 bg-slate-900/40">
                <div className="text-[10px] uppercase tracking-widest text-slate-500 font-mono mb-2">
                  Task
                </div>

                <div className="text-white font-bold">
                  NLI Classification
                </div>
              </div>
            </div>
          </div>
        </div>

        {/* FINAL SUMMARY */}
        <section className="result-card relative overflow-hidden">
          <div className="absolute inset-0 bg-gradient-to-br from-sky-500/5 via-transparent to-purple-500/5" />

          <div className="relative z-10">
            <div className="flex items-center gap-4 mb-8">
              <Globe className="w-10 h-10 text-sky-400" />

              <div>
                <h2 className="text-4xl font-display font-black tracking-tight text-white">
                  Beyond Static Fact Checking
                </h2>

                <p className="text-slate-400 mt-2">
                  Real-world misinformation detection
                  requires both retrieval and reasoning.
                </p>
              </div>
            </div>

            <div className="grid grid-cols-1 md:grid-cols-3 gap-8 mt-12">
              {[
                {
                  title: 'Semantic Retrieval',
                  desc:
                    'Finds evidence even when wording differs from the claim.',
                },
                {
                  title: 'Evidence Ranking',
                  desc:
                    'Improves retrieval precision through CrossEncoder scoring.',
                },
                {
                  title: 'Interpretable Verification',
                  desc:
                    'Produces confidence scores and evidence trails.',
                },
              ].map((item) => (
                <div
                  key={item.title}
                  className="p-6 rounded-2xl border border-slate-800 bg-slate-900/40"
                >
                  <h3 className="text-white font-bold text-lg mb-3">
                    {item.title}
                  </h3>

                  <p className="text-slate-400 text-sm leading-relaxed">
                    {item.desc}
                  </p>
                </div>
              ))}
            </div>
          </div>
        
        </section>
        {/* INTERACTIVE SETTINGS */}
        <section className="mt-32">
        <div className="result-card relative overflow-hidden">
            <div className="absolute inset-0 bg-gradient-to-br from-purple-500/5 via-transparent to-sky-500/5" />

            <div className="relative z-10">
            <div className="flex items-center gap-4 mb-8">
                <Binary className="w-10 h-10 text-purple-400" />

                <div>
                <h2 className="text-4xl font-display font-black tracking-tight text-white">
                    Interactive Pipeline Configuration
                </h2>

                <p className="text-slate-400 mt-2 max-w-3xl leading-relaxed">
                    The system exposes configurable retrieval
                    and verification settings so users can
                    experimentally compare how different
                    retrieval strategies affect evidence quality,
                    ranking precision, and final verification
                    accuracy.
                </p>
                </div>
            </div>

            {/* DEFAULT CONFIG */}
            <div className="grid grid-cols-1 lg:grid-cols-3 gap-6 mb-14">
                {[
                {
                    title: 'Top-K Retrieval',
                    value: 'K = 5',
                    desc:
                    'Retrieves the top 5 semantic evidence candidates from the vector index.',
                    color:
                    'border-sky-500/20 bg-sky-500/5',
                },
                {
                    title: 'CrossEncoder',
                    value: 'Enabled',
                    desc:
                    'Reranking is enabled by default to improve evidence relevance.',
                    color:
                    'border-emerald-500/20 bg-emerald-500/5',
                },
                {
                    title: 'Retriever Model',
                    value: 'MiniLM',
                    desc:
                    'Uses SentenceTransformer embeddings for dense semantic retrieval.',
                    color:
                    'border-purple-500/20 bg-purple-500/5',
                },
                ].map((item) => (
                <div
                    key={item.title}
                    className={`p-7 rounded-3xl border ${item.color}`}
                >
                    <div className="text-[10px] uppercase tracking-widest font-mono text-slate-500 mb-3">
                    Default Setting
                    </div>

                    <div className="text-white text-2xl font-black tracking-tight mb-2">
                    {item.value}
                    </div>

                    <div className="text-sm font-semibold text-slate-300 mb-3">
                    {item.title}
                    </div>

                    <p className="text-sm text-slate-400 leading-relaxed">
                    {item.desc}
                    </p>
                </div>
                ))}
            </div>

            {/* SETTING VISUALIZERS */}
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-10">
                {/* K VISUALIZER */}
                <div className="p-8 rounded-3xl border border-slate-800 bg-slate-900/40">
                <div className="flex items-center justify-between mb-8">
                    <div>
                    <div className="text-xs uppercase tracking-widest font-mono text-slate-500 mb-2">
                        Retrieval Volume
                    </div>

                    <h3 className="text-2xl font-black text-white">
                        Adjustable Top-K
                    </h3>
                    </div>

                    <div className="text-sky-400 font-mono font-black">
                    5 / 10 / 20 / 50
                    </div>
                </div>

                <div className="space-y-5">
                    {[5, 10, 20, 50].map((k, i) => (
                    <motion.div
                        key={k}
                        initial={{ opacity: 0, x: -20 }}
                        whileInView={{
                        opacity: 1,
                        x: 0,
                        }}
                        transition={{
                        delay: i * 0.08,
                        }}
                        viewport={{ once: true }}
                        className={`flex items-center justify-between p-4 rounded-2xl border ${
                        k === 5
                            ? 'border-sky-500 bg-sky-500/10'
                            : 'border-slate-800 bg-slate-900/30'
                        }`}
                    >
                        <div>
                        <div className="text-white font-bold">
                            Top-{k}
                        </div>

                        <div className="text-xs text-slate-500 font-mono mt-1">
                            Retrieved Evidence Candidates
                        </div>
                        </div>

                        {k === 5 && (
                        <div className="px-3 py-1 rounded-full bg-sky-500 text-slate-950 text-xs font-black uppercase tracking-wider">
                            Default
                        </div>
                        )}
                    </motion.div>
                    ))}
                </div>
                </div>

                {/* RERANKER VISUALIZER */}
                <div className="p-8 rounded-3xl border border-slate-800 bg-slate-900/40">
                <div className="flex items-center justify-between mb-8">
                    <div>
                    <div className="text-xs uppercase tracking-widest font-mono text-slate-500 mb-2">
                        Retrieval Precision
                    </div>

                    <h3 className="text-2xl font-black text-white">
                        CrossEncoder Reranking
                    </h3>
                    </div>

                    <div className="px-4 py-2 rounded-full bg-emerald-500/10 border border-emerald-500/20 text-emerald-400 text-sm font-black uppercase tracking-wide">
                    Enabled
                    </div>
                </div>

                <div className="space-y-6">
                    <div>
                    <div className="flex justify-between text-sm mb-2">
                        <span className="text-slate-300">
                        Dense Retrieval
                        </span>

                        <span className="font-mono text-slate-500">
                        Recall@1 = 66.7%
                        </span>
                    </div>

                    <div className="h-3 bg-slate-800 rounded-full overflow-hidden">
                        <motion.div
                        initial={{ width: 0 }}
                        whileInView={{
                            width: '66.7%',
                        }}
                        transition={{
                            duration: 1,
                        }}
                        viewport={{ once: true }}
                        className="h-full bg-slate-500"
                        />
                    </div>
                    </div>

                    <div>
                    <div className="flex justify-between text-sm mb-2">
                        <span className="text-slate-300">
                        Dense + Reranker
                        </span>

                        <span className="font-mono text-sky-400">
                        Recall@1 = 81.3%
                        </span>
                    </div>

                    <div className="h-3 bg-slate-800 rounded-full overflow-hidden">
                        <motion.div
                        initial={{ width: 0 }}
                        whileInView={{
                            width: '81.3%',
                        }}
                        transition={{
                            duration: 1,
                            delay: 0.2,
                        }}
                        viewport={{ once: true }}
                        className="h-full bg-sky-500"
                        />
                    </div>
                    </div>
                </div>

                <div className="mt-8 p-5 rounded-2xl border border-sky-500/20 bg-sky-500/5 text-sm text-slate-300 leading-relaxed">
                    Enabling reranking significantly improves
                    top-ranked evidence quality by allowing
                    the CrossEncoder to jointly analyze both
                    the claim and candidate evidence sentence.
                </div>
                </div>
            </div>

            {/* FOOTNOTE */}
            <div className="mt-14 p-6 rounded-3xl border border-slate-800 bg-slate-900/30">
                <div className="text-xs uppercase tracking-widest font-mono text-slate-500 mb-3">
                Experimental Workflow
                </div>

                <p className="text-slate-400 leading-relaxed">
                Users can toggle retriever models, retrieval
                volume, and reranking behavior in real time
                to experimentally compare retrieval dynamics,
                evidence quality, and final verifier behavior.
                This transforms the interface from a static
                demo into an interactive research sandbox for
                studying retrieval-augmented fact verification
                systems.
                </p>
            </div>
            </div>
        </div>
        </section>
        
      </section>
    </div>
  );
};