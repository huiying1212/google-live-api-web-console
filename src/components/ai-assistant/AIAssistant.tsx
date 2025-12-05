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

KNOWLEDGE RETRIEVAL - Using retrieve_knowledge with different query types:
The retrieve_knowledge tool supports THREE query types. Choose the appropriate one based on what the student asks:

1. query_type="search" (default): For searching specific content
   - Use when: Student asks about specific topics, concepts, or historical information
   - Required parameter: query (in ENGLISH - translate Chinese queries first)
   - Example: Student asks "工业革命的设计历史" → use query_type="search", query="design history of industrial revolution"

2. query_type="chapters": For getting the table of contents / chapter list
   - Use when: Student asks for book outline, table of contents, chapter list, or wants to know what topics are covered
   - No additional parameters needed
   - Example: Student asks "这本书有哪些章节?" or "What chapters are available?" → use query_type="chapters"

3. query_type="chapter_content": For getting all content from a specific chapter
   - Use when: Student asks about a specific chapter by number
   - Required parameter: chapter_number (integer)
   - Example: Student asks "第3章讲什么?" or "What's in chapter 3?" → use query_type="chapter_content", chapter_number=3

YOUR WORKFLOW: For EVERY student question, you MUST:
1. FIRST call retrieve_knowledge with the appropriate query_type to find relevant information
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
  knowledgeApiUrl = "/search/text",
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

