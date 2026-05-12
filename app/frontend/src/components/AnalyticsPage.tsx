import {
  BarChart3,
  Activity,
  AlertTriangle,
  CheckCircle2,
} from 'lucide-react';

import { motion } from 'framer-motion';

const retrievalMetrics = [
  {
    pipeline: 'Dense Retrieval',
    r1: 0.6667,
    r5: 0.9333,
    r10: 0.9733,
  },
  {
    pipeline: 'Dense + Reranker',
    r1: 0.8133,
    r5: 0.9467,
    r10: 0.9733,
  },
];

const errorBreakdown = [
  {
    label: 'Retrieval Miss',
    count: 1,
    color: 'bg-red-500',
  },
  {
    label: 'Verifier Wrong Despite Gold Evidence',
    count: 9,
    color: 'bg-yellow-500',
  },
  {
    label: 'Verifier Too Conservative',
    count: 7,
    color: 'bg-slate-500',
  },
];

export const AnalyticsPage = () => {
  const totalErrors = 17;

  return (
    <div className="max-w-7xl mx-auto px-8 pb-32 space-y-16">
      {/* Header */}
      <section className="space-y-3">
        <h1 className="text-5xl font-display font-black tracking-tighter text-white">
          FEVER Pipeline Analytics
        </h1>

        <p className="text-slate-400 max-w-3xl text-lg leading-relaxed">
          Evaluation metrics from the FEVER claim verification
          pipeline, including dense retrieval, reranking,
          and DeBERTa-based verification performance.
        </p>
      </section>

      {/* Top Metrics */}
      <section className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
        {[
          {
            label: 'Recall@1',
            value: '81.3%',
            icon: Activity,
            desc: 'Dense + reranker',
          },
          {
            label: 'Recall@5',
            value: '94.7%',
            icon: CheckCircle2,
            desc: 'Top-5 evidence recall',
          },
          {
            label: 'Recall@10',
            value: '97.3%',
            icon: BarChart3,
            desc: 'Top-10 evidence recall',
          },
          {
            label: 'Pipeline Accuracy',
            value: '77.3%',
            icon: AlertTriangle,
            desc: 'End-to-end FEVER accuracy',
          },
        ].map((metric, i) => (
          <motion.div
            key={metric.label}
            initial={{ opacity: 0, y: 10 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: i * 0.05 }}
            className="metric-card"
          >
            <div className="flex items-start justify-between mb-6">
              <span className="text-[10px] font-mono uppercase tracking-widest text-slate-500 font-bold">
                {metric.label}
              </span>

              <metric.icon className="w-5 h-5 text-slate-600" />
            </div>

            <div className="text-5xl font-black tracking-tighter text-white mb-3">
              {metric.value}
            </div>

            <div className="text-sm text-slate-500">
              {metric.desc}
            </div>
          </motion.div>
        ))}
      </section>

      {/* Retrieval Table */}
      <section className="result-card space-y-8">
        <div>
          <h2 className="text-3xl font-display font-black tracking-tight text-white mb-2">
            Retrieval Evaluation
          </h2>

          <p className="text-slate-400 max-w-3xl">
            Retrieval Recall@K evaluation on FEVER claims.
            Metrics measure whether at least one gold
            evidence sentence appeared in the retrieved set.
          </p>
        </div>

        <div className="overflow-x-auto">
          <table className="w-full border-collapse">
            <thead>
              <tr className="border-b border-slate-800">
                <th className="text-left py-4 text-xs uppercase tracking-widest text-slate-500 font-mono">
                  Pipeline
                </th>

                <th className="text-left py-4 text-xs uppercase tracking-widest text-slate-500 font-mono">
                  Recall@1
                </th>

                <th className="text-left py-4 text-xs uppercase tracking-widest text-slate-500 font-mono">
                  Recall@5
                </th>

                <th className="text-left py-4 text-xs uppercase tracking-widest text-slate-500 font-mono">
                  Recall@10
                </th>
              </tr>
            </thead>

            <tbody>
              {retrievalMetrics.map((row) => (
                <tr
                  key={row.pipeline}
                  className="border-b border-slate-900"
                >
                  <td className="py-6 text-white font-semibold">
                    {row.pipeline}
                  </td>

                  <td className="py-6 font-mono text-slate-300">
                    {(row.r1 * 100).toFixed(2)}%
                  </td>

                  <td className="py-6 font-mono text-slate-300">
                    {(row.r5 * 100).toFixed(2)}%
                  </td>

                  <td className="py-6 font-mono text-slate-300">
                    {(row.r10 * 100).toFixed(2)}%
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </section>

      {/* Final Pipeline */}
      <section className="grid grid-cols-1 lg:grid-cols-2 gap-12">
        <div className="result-card space-y-8">
          <div>
            <h2 className="text-3xl font-display font-black tracking-tight text-white mb-2">
              Final Pipeline Result
            </h2>

            <p className="text-slate-400">
              End-to-end verification accuracy using
              dense retrieval, reranking, and
              DeBERTa-v3 NLI verification.
            </p>
          </div>

          <div className="p-8 rounded-2xl border border-slate-800 bg-slate-900/50">
            <div className="text-xs uppercase tracking-widest text-slate-500 font-mono mb-3">
              Dense Retrieval + Reranker + Verifier
            </div>

            <div className="text-6xl font-black tracking-tighter text-sky-400">
              77.3%
            </div>
          </div>
        </div>

        {/* Error Analysis */}
        <div className="result-card space-y-8">
          <div>
            <h2 className="text-3xl font-display font-black tracking-tight text-white mb-2">
              Error Analysis
            </h2>

            <p className="text-slate-400">
              17 total errors out of 75 evaluated FEVER
              examples.
            </p>
          </div>

          <div className="space-y-6">
            {errorBreakdown.map((err) => {
              const pct =
                (err.count / totalErrors) * 100;

              return (
                <div
                  key={err.label}
                  className="space-y-2"
                >
                  <div className="flex justify-between text-sm">
                    <span className="text-slate-300">
                      {err.label}
                    </span>

                    <span className="font-mono text-white">
                      {err.count}
                    </span>
                  </div>

                  <div className="h-2 bg-slate-800 rounded-full overflow-hidden">
                    <motion.div
                      initial={{ width: 0 }}
                      animate={{ width: `${pct}%` }}
                      transition={{
                        duration: 1,
                      }}
                      className={`h-full ${err.color}`}
                    />
                  </div>
                </div>
              );
            })}
          </div>

          <div className="p-5 rounded-xl bg-sky-500/5 border border-sky-500/20 text-sm text-slate-300 leading-relaxed">
            Most remaining failures originate from the
            verifier stage rather than retrieval. The
            reranked retrieval pipeline successfully
            retrieves gold evidence for the majority of
            evaluated FEVER claims.
          </div>
        </div>
      </section>
      {/* LIVE WIKIPEDIA PIPELINE */}
        <section className="space-y-10">
        <div className="flex items-end justify-between flex-wrap gap-6">
            <div>
            <h1 className="text-5xl font-display font-black tracking-tighter text-white">
            Live Wikipedia Evaluation
            </h1>

            <p className="text-slate-400 max-w-3xl text-lg leading-relaxed">
                End-to-end evaluation using live Wikipedia retrieval
                instead of FEVER gold evidence. This setting is
                substantially harder because retrieval quality,
                page selection, and evidence completeness directly
                impact verifier performance.
            </p>
            </div>

            <div className="px-5 py-3 rounded-2xl border border-purple-500/20 bg-purple-500/10">
            <div className="text-[10px] font-mono uppercase tracking-widest text-purple-300 mb-1">
                Deployment Mode
            </div>

            <div className="text-xl font-black text-white">
                LIVE WIKIPEDIA
            </div>
            </div>
        </div>

        {/* Main Metrics */}
        <div className="grid grid-cols-1 md:grid-cols-4 gap-6">
            {[
            {
                label: 'Accuracy',
                value: '43.0%',
                color: 'text-purple-400',
            },
            {
                label: 'Supports Accuracy',
                value: '23.2%',
                color: 'text-emerald-400',
            },
            {
                label: 'Refutes Accuracy',
                value: '61.9%',
                color: 'text-red-400',
            },
            {
                label: 'NEI Accuracy',
                value: '73.9%',
                color: 'text-slate-300',
            },
            ].map((metric) => (
            <div
                key={metric.label}
                className="metric-card"
            >
                <div className="text-[10px] font-mono uppercase tracking-widest text-slate-500 mb-6">
                {metric.label}
                </div>

                <div
                className={`text-5xl font-black tracking-tighter ${metric.color}`}
                >
                {metric.value}
                </div>
            </div>
            ))}
        </div>

        {/* Prediction Distribution + Label Accuracy */}
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-12">
            {/* Prediction Distribution */}
            <div className="result-card space-y-8">
            <div>
                <h3 className="text-2xl font-display font-black tracking-tight text-white mb-2">
                Prediction Distribution
                </h3>

                <p className="text-slate-400">
                The live pipeline heavily biases toward
                NOT ENOUGH INFO predictions due to
                retrieval uncertainty and incomplete evidence.
                </p>
            </div>

            <div className="space-y-6">
                {[
                {
                    label: 'NOT ENOUGH INFO',
                    count: 57,
                    pct: 57,
                    color: 'bg-slate-400',
                },
                {
                    label: 'REFUTES',
                    count: 29,
                    pct: 29,
                    color: 'bg-red-500',
                },
                {
                    label: 'SUPPORTS',
                    count: 14,
                    pct: 14,
                    color: 'bg-emerald-500',
                },
                ].map((item) => (
                <div
                    key={item.label}
                    className="space-y-2"
                >
                    <div className="flex justify-between items-center">
                    <span className="text-sm text-slate-300">
                        {item.label}
                    </span>

                    <span className="font-mono text-white">
                        {item.count}
                    </span>
                    </div>

                    <div className="h-3 bg-slate-800 rounded-full overflow-hidden">
                    <motion.div
                        initial={{ width: 0 }}
                        animate={{
                        width: `${item.pct}%`,
                        }}
                        transition={{
                        duration: 1,
                        }}
                        className={`h-full ${item.color}`}
                    />
                    </div>
                </div>
                ))}
            </div>
            </div>

            {/* Per Label Accuracy */}
            <div className="result-card space-y-8">
            <div>
                <h3 className="text-2xl font-display font-black tracking-tight text-white mb-2">
                Per-Label Accuracy
                </h3>

                <p className="text-slate-400">
                REFUTES and NOT ENOUGH INFO are
                substantially easier for the verifier than
                SUPPORTS claims under live retrieval.
                </p>
            </div>

            <div className="space-y-8">
                {[
                {
                    label: 'SUPPORTS',
                    correct: 13,
                    total: 56,
                    acc: 23.2,
                    color: 'bg-emerald-500',
                },
                {
                    label: 'REFUTES',
                    correct: 13,
                    total: 21,
                    acc: 61.9,
                    color: 'bg-red-500',
                },
                {
                    label: 'NOT ENOUGH INFO',
                    correct: 17,
                    total: 23,
                    acc: 73.9,
                    color: 'bg-slate-400',
                },
                ].map((item) => (
                <div
                    key={item.label}
                    className="space-y-3"
                >
                    <div className="flex justify-between items-center">
                    <div>
                        <div className="text-white font-semibold">
                        {item.label}
                        </div>

                        <div className="text-xs text-slate-500 font-mono">
                        {item.correct}/{item.total} correct
                        </div>
                    </div>

                    <div className="text-2xl font-black text-white">
                        {item.acc}%
                    </div>
                    </div>

                    <div className="h-3 bg-slate-800 rounded-full overflow-hidden">
                    <motion.div
                        initial={{ width: 0 }}
                        animate={{
                        width: `${item.acc}%`,
                        }}
                        transition={{
                        duration: 1,
                        }}
                        className={`h-full ${item.color}`}
                    />
                    </div>
                </div>
                ))}
            </div>
            </div>
        </div>

        {/* Confusion Matrix */}
        <div className="result-card space-y-8">
            <div>
            <h3 className="text-2xl font-display font-black tracking-tight text-white mb-2">
                Confusion Matrix
            </h3>

            <p className="text-slate-400">
                Most SUPPORTS claims collapse into
                NOT ENOUGH INFO due to missing or weak
                retrieved evidence from live Wikipedia search.
            </p>
            </div>

            <div className="overflow-x-auto">
            <table className="w-full border-collapse">
                <thead>
                <tr className="border-b border-slate-800">
                    <th className="py-4 text-left text-xs uppercase tracking-widest text-slate-500 font-mono">
                    Gold Label
                    </th>

                    <th className="py-4 text-left text-xs uppercase tracking-widest text-slate-500 font-mono">
                    SUPPORTS
                    </th>

                    <th className="py-4 text-left text-xs uppercase tracking-widest text-slate-500 font-mono">
                    REFUTES
                    </th>

                    <th className="py-4 text-left text-xs uppercase tracking-widest text-slate-500 font-mono">
                    NOT ENOUGH INFO
                    </th>
                </tr>
                </thead>

                <tbody>
                {[
                    {
                    gold: 'SUPPORTS',
                    supports: 13,
                    refutes: 11,
                    nei: 32,
                    },
                    {
                    gold: 'REFUTES',
                    supports: 0,
                    refutes: 13,
                    nei: 8,
                    },
                    {
                    gold: 'NOT ENOUGH INFO',
                    supports: 1,
                    refutes: 5,
                    nei: 17,
                    },
                ].map((row) => (
                    <tr
                    key={row.gold}
                    className="border-b border-slate-900"
                    >
                    <td className="py-6 text-white font-semibold">
                        {row.gold}
                    </td>

                    <td className="py-6 font-mono text-slate-300">
                        {row.supports}
                    </td>

                    <td className="py-6 font-mono text-slate-300">
                        {row.refutes}
                    </td>

                    <td className="py-6 font-mono text-slate-300">
                        {row.nei}
                    </td>
                    </tr>
                ))}
                </tbody>
            </table>
            </div>
        </div>
        </section>
    </div>
  );
};