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
  description: `Retrieves knowledge from the local knowledge base about design history. Supports three query types:
- "search": Semantic search for specific content (default). The knowledge base contains primarily ENGLISH content, so translate Chinese queries to English.
- "chapters": Get list of all available chapters with statistics (no query needed).
- "chapter_content": Get all content from a specific chapter (requires chapter_number).`,
  parameters: {
    type: Type.OBJECT,
    properties: {
      query_type: {
        type: Type.STRING,
        description: `The type of query to perform:
- "search": Search for specific content using semantic search (default)
- "chapters": Get the table of contents / list of all chapters
- "chapter_content": Get all content from a specific chapter`,
      },
      query: {
        type: Type.STRING,
        description: "The search query in ENGLISH. Required only when query_type is 'search'. If user asks in Chinese, translate key terms to English first.",
      },
      chapter_number: {
        type: Type.INTEGER,
        description: "The chapter number to retrieve. Required only when query_type is 'chapter_content'.",
      },
    },
    required: [],
  },
};

interface KnowledgeRetrievalProps {
  apiUrl?: string;
  topK?: number;
  minScore?: number;
}

export function KnowledgeRetrieval({
  apiUrl = "/search/text",
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
        const args = fc.args as any;
        const queryType = args.query_type || "search";
        const query = args.query;
        const chapterNumber = args.chapter_number;

        try {
          // Extract base URL from apiUrl (remove /search/text if present)
          const baseUrl = apiUrl.replace(/\/search\/text$/, "");
          
          let response: Response;
          let requestDescription: string;

          // Smart routing based on query_type
          if (queryType === "chapters") {
            // Get list of all chapters
            requestDescription = "获取章节列表";
            response = await fetch(`${baseUrl}/search/chapters`, {
              method: "GET",
              headers: {
                "Content-Type": "application/json",
              },
            });
          } else if (queryType === "chapter_content") {
            // Get content from specific chapter
            if (chapterNumber === undefined) {
              responses.push({
                id: fc.id,
                name: fc.name,
                response: {
                  output: {
                    error: "Missing chapter_number",
                    message: "chapter_number is required when query_type is 'chapter_content'",
                  },
                },
              });
              continue;
            }
            requestDescription = `获取第${chapterNumber}章内容`;
            response = await fetch(`${baseUrl}/search/chapter/${chapterNumber}`, {
              method: "GET",
              headers: {
                "Content-Type": "application/json",
              },
            });
          } else {
            // Default: semantic search
            if (!query) {
              responses.push({
                id: fc.id,
                name: fc.name,
                response: {
                  output: {
                    error: "Missing query",
                    message: "query is required when query_type is 'search'",
                  },
                },
              });
              continue;
            }
            requestDescription = `搜索: ${query}`;
            response = await fetch(`${baseUrl}/search/text`, {
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
          }

          console.log(`Knowledge API - ${requestDescription}`);
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

