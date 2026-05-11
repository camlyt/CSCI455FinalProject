import { LucideIcon } from 'lucide-react';

export type Page = 'verify' | 'settings' | 'analytics' | 'tutorial';

export type Label = 'SUPPORTS' | 'REFUTES' | 'NOT ENOUGH INFO';

export type RetrievalMode = 'fever' | 'internet';

export interface Evidence {
  text: string;
  page: string;
  sentence_id: number;
  score: number;
  rerank_score: number;
}

export interface PipelineStep {
  id: number;
  title: string;
  icon: LucideIcon;
  description: string;
}

export interface AppSettings {
  topK: number;
  retriever: string;
  retrievalMode: RetrievalMode;
  useReranker: boolean;
}
