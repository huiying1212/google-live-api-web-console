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
import { useEffect, useRef, useState, memo } from "react";
import vegaEmbed from "vega-embed";
import { useLiveAPIContext } from "../../contexts/LiveAPIContext";
import {
  FunctionDeclaration,
  LiveServerToolCall,
  Modality,
  Type,
} from "@google/genai";

const renderAltairDeclaration: FunctionDeclaration = {
  name: "render_altair",
  description: "Displays an altair graph in json format.",
  parameters: {
    type: Type.OBJECT,
    properties: {
      json_graph: {
        type: Type.STRING,
        description:
          "JSON STRING representation of the graph to render. Must be a string, not a json object",
      },
    },
    required: ["json_graph"],
  },
};

const retrieveKnowledgeDeclaration: FunctionDeclaration = {
  name: "retrieve_knowledge",
  description: "Retrieves relevant knowledge from the local vector database using RAG (Retrieval-Augmented Generation). Use this to search for information from the knowledge base.",
  parameters: {
    type: Type.OBJECT,
    properties: {
      query: {
        type: Type.STRING,
        description: "The search query to find relevant information in the knowledge base",
      },
    },
    required: ["query"],
  },
};

function AltairComponent() {
  const [jsonString, setJSONString] = useState<string>("");
  const { client, setConfig, setModel } = useLiveAPIContext();

  useEffect(() => {
    setModel("models/gemini-2.5-flash-native-audio-preview-09-2025");
    setConfig({
      responseModalities: [Modality.AUDIO],
      speechConfig: {
        voiceConfig: { prebuiltVoiceConfig: { voiceName: "Aoede" } },
      },
      systemInstruction: {
        parts: [
          {
            text: 'You are my helpful assistant. Any time I ask you for something you can call the "render_altair" function or retrieve_knowledge function I have provided you. Dont ask for additional information just make your best judgement. ',
          },
        ],
      },
      tools: [
        // there is a free-tier quota for search
        { googleSearch: {} },
        { functionDeclarations: [renderAltairDeclaration, retrieveKnowledgeDeclaration] },
      ],
    });
  }, [setConfig, setModel]);

  useEffect(() => {
    const onToolCall = async (toolCall: LiveServerToolCall) => {
      if (!toolCall.functionCalls) {
        return;
      }

      const responses: any[] = [];

      for (const fc of toolCall.functionCalls) {
        if (fc.name === renderAltairDeclaration.name) {
          const str = (fc.args as any).json_graph;
          setJSONString(str);
          responses.push({
            response: { output: { success: true } },
            id: fc.id,
            name: fc.name,
          });
        } else if (fc.name === retrieveKnowledgeDeclaration.name) {
          const query = (fc.args as any).query;
          try {
            // Call the local knowledge API
            const response = await fetch("http://localhost:8000/search/text", {
              method: "POST",
              headers: {
                "Content-Type": "application/json",
              },
              body: JSON.stringify({
                query: query,
                top_k: 5,
                min_score: 0.3,
                mode: "rag",
              }),
            });
            
            const result = await response.json();
            
            if (result.success) {
              responses.push({
                response: { output: result.data },
                id: fc.id,
                name: fc.name,
              });
            } else {
              responses.push({
                response: { output: { error: "Failed to retrieve knowledge", message: result.message } },
                id: fc.id,
                name: fc.name,
              });
            }
          } catch (error) {
            console.error("Error calling knowledge API:", error);
            responses.push({
              response: { output: { error: "Failed to connect to knowledge API", details: String(error) } },
              id: fc.id,
              name: fc.name,
            });
          }
        } else {
          responses.push({
            response: { output: { success: true } },
            id: fc.id,
            name: fc.name,
          });
        }
      }

      // Send responses for all tool calls
      if (responses.length > 0) {
        setTimeout(
          () =>
            client.sendToolResponse({
              functionResponses: responses,
            }),
          200
        );
      }
    };
    client.on("toolcall", onToolCall);
    return () => {
      client.off("toolcall", onToolCall);
    };
  }, [client]);

  const embedRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (embedRef.current && jsonString) {
      console.log("jsonString", jsonString);
      vegaEmbed(embedRef.current, JSON.parse(jsonString));
    }
  }, [embedRef, jsonString]);
  return <div className="vega-embed" ref={embedRef} />;
}

export const Altair = memo(AltairComponent);
