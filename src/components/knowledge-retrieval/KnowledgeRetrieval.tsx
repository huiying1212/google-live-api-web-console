/**
 * Copyright 2024 Google LLC
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
import { useEffect } from "react";
import { useLiveAPIContext } from "../../contexts/LiveAPIContext";
import {
  FunctionDeclaration,
  LiveServerToolCall,
  Type,
} from "@google/genai";

export const retrieveKnowledgeDeclaration: FunctionDeclaration = {
  name: "retrieve_knowledge",
  description: "Retrieves relevant knowledge from the local vector database using RAG (Retrieval-Augmented Generation). The knowledge base contains primarily ENGLISH content about design history. IMPORTANT: If the user's question is in Chinese, you MUST translate the search query to English before calling this function for better retrieval accuracy.",
  parameters: {
    type: Type.OBJECT,
    properties: {
      query: {
        type: Type.STRING,
        description: "The search query in ENGLISH to find relevant information in the knowledge base. If the user asks in Chinese, translate key terms to English first.",
      },
    },
    required: ["query"],
  },
};

interface KnowledgeRetrievalProps {
  apiUrl?: string;
  topK?: number;
  minScore?: number;
}

export function KnowledgeRetrieval({
  apiUrl = "http://localhost:8000/search/text",
  topK = 5,
  minScore = 0.3,
}: KnowledgeRetrievalProps) {
  const { client } = useLiveAPIContext();

  useEffect(() => {
    const onToolCall = async (toolCall: LiveServerToolCall) => {
      if (!toolCall.functionCalls) {
        return;
      }

      const knowledgeCalls = toolCall.functionCalls.filter(
        (fc) => fc.name === retrieveKnowledgeDeclaration.name
      );

      if (knowledgeCalls.length === 0) {
        return;
      }

      const responses: any[] = [];

      for (const fc of knowledgeCalls) {
        const query = (fc.args as any).query;
        try {
          // Call the local knowledge API
          const response = await fetch(apiUrl, {
            method: "POST",
            headers: {
              "Content-Type": "application/json",
            },
            body: JSON.stringify({
              query: query,
              top_k: topK,
              min_score: minScore,
              mode: "rag",
            }),
          });

          const result = await response.json();

          if (result.success) {
            responses.push({
              id: fc.id,
              name: fc.name,
              response: { 
                output: result.data,
              },
            });
          } else {
            responses.push({
              id: fc.id,
              name: fc.name,
              response: {
                output: {
                  error: "Failed to retrieve knowledge",
                  message: result.message,
                },
              },
            });
          }
        } catch (error) {
          console.error("Error calling knowledge API:", error);
          responses.push({
            id: fc.id,
            name: fc.name,
            response: {
              output: {
                error: "Failed to connect to knowledge API",
                details: String(error),
              },
            },
          });
        }
      }

      // Send responses for knowledge retrieval calls
      // No setTimeout needed for async operations - send immediately when ready
      if (responses.length > 0) {
        client.sendToolResponse({
          functionResponses: responses,
        });
      }
    };

    client.on("toolcall", onToolCall);
    return () => {
      client.off("toolcall", onToolCall);
    };
  }, [client, apiUrl, topK, minScore]);

  return null; // This component doesn't render anything
}

