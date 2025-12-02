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
import { useEffect, memo } from "react";
import { useLiveAPIContext } from "../../contexts/LiveAPIContext";
import { Modality } from "@google/genai";
import { Altair, renderAltairDeclaration } from "../altair/Altair";
import { KnowledgeRetrieval, retrieveKnowledgeDeclaration } from "../knowledge-retrieval/KnowledgeRetrieval";
import { Whiteboard, displayContentDeclaration } from "../whiteboard/Whiteboard";

interface AIAssistantProps {
  model?: string;
  voiceName?: string;
  systemInstruction?: string;
  enableGoogleSearch?: boolean;
  knowledgeApiUrl?: string;
  knowledgeTopK?: number;
  knowledgeMinScore?: number;
}

function AIAssistantComponent({
  model = "models/gemini-2.5-flash-native-audio-preview-09-2025",
  voiceName = "Aoede",
  systemInstruction = `
You are an intelligent teaching assistant helping students learn design history through voice interaction and visual whiteboard presentation.

LANGUAGE CONSISTENCY: Always respond in the same language that the student is using(Chinese or English). If the student asks in English, respond in English. If the student asks in Chinese, respond in Chinese. 

YOUR WORKFLOW: For EVERY student question, you MUST:
1. FIRST call retrieve_knowledge to find relevant information from the knowledge database
   - If the student's question is in Chinese, you MUST translate the key search terms to English before calling retrieve_knowledge
   - Example: If student asks "工业革命的设计历史", use retrieve_knowledge with query="design history of industrial revolution"
2. ANALYZE the search results: check if the results are relevant and organize them into slide-ready sources
3. THEN call display_content present the slide-ready sources visually
	- Use SHORT bullet points when necessary
	- Keep titles short and clear
	- Incorporate key insights from retrieved knowledge with source attribution
	- When images are available and relevant, include them to enhance visual understanding
4. FINALLY provide your detailed verbal explanation IN THE STUDENT'S LANGUAGE that combines the retrieved knowledge with your own understanding

MULTI-SPEAKER AWARENESS AND PARALINGUISTIC UNDERSTANDING:
- You are participating in a conversation that may involve multiple people (e.g., different students or a teacher and a student).
- Use your audio and visual understanding capabilities to distinguish different people, and if the users introduce themselves (e.g., "I am Tom", "I am Sarah"), remember their voice characteristics and address them by name in future turns.
- Some conversation may just happens between the users, you should detect whether or not the users are speaking to each other or just to you, and respond only when the users are speaking to you(they will say "hi gemini").otherwise keep silent.

`,
  enableGoogleSearch = true,
  knowledgeApiUrl = "http://localhost:8000/search/text",
  knowledgeTopK = 5,
  knowledgeMinScore = 0.3,
}: AIAssistantProps) {
  const { setConfig, setModel } = useLiveAPIContext();

  useEffect(() => {
    setModel(model);
    
    const tools: any[] = [];
    
    // Add Google Search if enabled
    if (enableGoogleSearch) {
      tools.push({ googleSearch: {} });
    }
    
    // Add function declarations
    tools.push({
      functionDeclarations: [
        renderAltairDeclaration,
        retrieveKnowledgeDeclaration,
        displayContentDeclaration,
      ],
    });

    setConfig({
      responseModalities: [Modality.AUDIO],
      speechConfig: {
        voiceConfig: { prebuiltVoiceConfig: { voiceName } },
      },
      systemInstruction: {
        parts: [
          {
            text: systemInstruction,
          },
        ],
      },
      tools,
    });
  }, [setConfig, setModel, model, voiceName, systemInstruction, enableGoogleSearch]);

  return (
    <>
      {/* Altair visualization component */}
      <Altair />
      
      {/* Whiteboard display component */}
      <Whiteboard />
      
      {/* Knowledge retrieval component (invisible) */}
      <KnowledgeRetrieval
        apiUrl={knowledgeApiUrl}
        topK={knowledgeTopK}
        minScore={knowledgeMinScore}
      />
    </>
  );
}

export const AIAssistant = memo(AIAssistantComponent);

