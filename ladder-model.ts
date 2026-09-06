import type { ExtensionAPI } from "@earendil-works/pi-coding-agent";

export default function (pi: ExtensionAPI) {
  pi.registerProvider("ladder", {
    baseUrl: process.env.LLAMA_URL ?? "http://localhost:8258/v1",
    apiKey: "dummy",
    api: "openai-completions",
    models: [
      {
        id: process.env.LLAMA_MODEL ?? "laddermodel",
        name: "ladder xllamacpp",
        reasoning: true,
        input: ["text"],
        cost: {
          input: 0,
          output: 0,
          cacheRead: 0,
          cacheWrite: 0
        },
        contextWindow: 32768,
        maxTokens: 8192
      }
    ]
  });
}

