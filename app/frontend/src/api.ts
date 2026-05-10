import { AppSettings } from "./types";

export async function verifyClaim(claim: string, settings: AppSettings) {
  const response = await fetch("http://127.0.0.1:8000/verify", {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify({
      claim: claim,
      top_k: settings.topK,
      threshold: settings.threshold,
    }),
  });

  if (!response.ok) {
    const errorText = await response.text();
    console.error("Backend error:", errorText);
    throw new Error(errorText);
  }

  return response.json();
}