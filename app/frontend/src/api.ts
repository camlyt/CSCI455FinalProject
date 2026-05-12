import { AppSettings } from './types';

const API_BASE = 'http://127.0.0.1:8000';

export async function verifyClaim(
  claim: string,
  settings: AppSettings
) {

  const response = await fetch(`${API_BASE}/verify`, {
    method: 'POST',

    headers: {
      'Content-Type': 'application/json',
    },

    body: JSON.stringify({
      claim,
      top_k: settings.topK,
      retriever: settings.retriever,
      retrieval_mode: settings.retrievalMode,
      use_reranker: settings.useReranker,
    }),
  });

  if (!response.ok) {

    const errorText = await response.text();

    throw new Error(
      `Backend request failed: ${errorText}`
    );
  }

  return await response.json();
}

export async function fetchRandomClaim() {

  const response = await fetch(
    'http://127.0.0.1:8000/random-claim'
  )

  if (!response.ok) {
    throw new Error('Failed to fetch random claim')
  }

  return response.json()
}