import { Send, Database, Layers, ShieldCheck } from 'lucide-react';
import { Evidence, PipelineStep } from './types';

export const MOCK_PREVIOUS_CLAIMS = [
  "The Roman Empire collapsed in 476 AD.",
  "Water boils at 100 degrees Celsius at sea level.",
  "Albert Einstein was born in Germany.",
  "The Great Wall of China is visible from the Moon."
];

export const MOCK_EVIDENCE: Evidence[] = [
  {
    text: "Roman Atwood is an American <span class='evidence-highlight'>YouTube personality</span>, comedian, vlogger, and <span class='evidence-highlight'>content creator</span> known for his prank videos.",
    page: "Roman_Atwood",
    sentence_id: 0,
    score: 0.92,
    rerank_score: 5.42
  },
  {
    text: "He began producing videos in 2006 and gained a massive following for his 'PrankvsPrank' collaborations, establishing him as a prominent <span class='evidence-highlight'>digital creator</span>.",
    page: "YouTube_Personalities",
    sentence_id: 42,
    score: 0.81,
    rerank_score: 4.15
  },
  {
    text: "Atwood established a vlogging channel in 2013, which currently has over 15 million subscribers who consume his daily <span class='evidence-highlight'>video content</span>.",
    page: "Roman_Atwood",
    sentence_id: 12,
    score: 0.78,
    rerank_score: 3.89
  },
  {
    text: "The term <span class='evidence-highlight'>content creator</span> refers to individuals who produce entertaining or educational material for digital media.",
    page: "Content_Creation",
    sentence_id: 2,
    score: 0.72,
    rerank_score: 3.21
  },
  {
    text: "Roman Atwood's career transition from pranks to family vlogging is often cited as a successful example of platform longevity as a <span class='evidence-highlight'>creator</span>.",
    page: "Digital_Media_Influencers",
    sentence_id: 110,
    score: 0.68,
    rerank_score: 2.95
  }
];

export const PIPELINE_STEPS: PipelineStep[] = [
  { id: 1, title: 'Claim Input', icon: Send, description: 'Natural language claim is normalized and embedded into a high-dimensional vector space.' },
  { id: 2, title: 'Dense Retrieval', icon: Database, description: 'FAISS searches 5M+ Wikipedia sentences to find top-K semantic candidates.' },
  { id: 3, title: 'Reranking', icon: Layers, description: 'A Cross-Encoder scores the alignment of each candidate against the query.' },
  { id: 4, title: 'Verification', icon: ShieldCheck, description: 'An NLI model predicts Entailment, Neutral, or Contradiction for the claim-evidence pair.' }
];

export const RETRIEVER_MODELS = [

  {
    id: 'minilm',
    name: 'MiniLM-v6',
    desc:
      'Fast lightweight dense retriever using all-MiniLM-L6-v2 embeddings.'
  },
  {
    id: 'bge',
    name: 'BGE-v1.5',
    desc:
      'Semantic dense retriever using BAAI/bge-small-en-v1.5 embeddings.'
  },
  {
    id: 'e5',
    name: 'E5-small-v2',
    desc:
      'Query-document optimized retriever using intfloat/e5-small-v2 embeddings.'
  }
];
