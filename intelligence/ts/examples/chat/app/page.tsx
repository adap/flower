'use client';

import React, { useState, useEffect, useRef } from 'react';
import Image from 'next/image';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import { ArrowUpIcon } from '@heroicons/react/24/solid';
import { FlowerIntelligence, ChatResponseResult, Message, Progress } from '@flwr/flwr';

const fi: FlowerIntelligence = FlowerIntelligence.instance;

const history: Message[] = [{ role: 'system', content: 'You are a friendly assistant.' }];

interface ChatEntry {
  role: 'user' | 'bot';
  content: string;
  // For bot messages, store the model used when it was generated
  modelUsed?: string;
}

const availableModels = [
  'meta/llama3.2-1b/instruct-fp16',
  'meta/llama3.2-3b/instruct-q4',
  'meta/llama3.1-8b/instruct-q4',
  'deepseek/r1-distill-llama-8b/q4',
];

const isProduction = process.env.NODE_ENV === 'production';

// A simple collapsible component to show/hide internal reasoning.
const Collapsible: React.FC<{ content: string }> = ({ content }) => {
  const [isOpen, setIsOpen] = useState(false);
  return (
    <div className="mb-2">
      <button onClick={() => setIsOpen(!isOpen)} className="text-sm text-amber-400 underline mb-1">
        {isOpen ? 'Hide internal reasoning' : 'Show internal reasoning'}
      </button>
      {isOpen && (
        <div className="p-2 border-l-4 border-amber-400 bg-amber-50 text-sm text-justify italic">
          {content}
        </div>
      )}
    </div>
  );
};

// Helper function using guard clauses to determine remote handoff settings.
const getRemoteHandoffSettings = (allowRemote: boolean) => {
  if (isProduction || !allowRemote) return { remoteHandoff: false, apiKey: '' };
  return { remoteHandoff: true, apiKey: process.env.NEXT_PUBLIC_API_KEY || '' };
};

export default function ClientSideChatPage() {
  const [chatLog, setChatLog] = useState<ChatEntry[]>(
    history
      .filter((msg) => msg.role !== 'system')
      .map((msg) => ({ role: msg.role as 'user' | 'bot', content: msg.content }))
  );
  const [input, setInput] = useState('');
  const [loading, setLoading] = useState(false);

  // Reference for auto-scrolling chat window.
  const chatContainerRef = useRef<HTMLDivElement>(null);
  useEffect(() => {
    if (chatContainerRef.current) {
      chatContainerRef.current.scrollTop = chatContainerRef.current.scrollHeight;
    }
  }, [chatLog]);

  // Initialize model state as null; load the stored model on mount.
  const [model, setModel] = useState<string | null>(null);
  useEffect(() => {
    const stored = localStorage.getItem('selectedModel');
    setModel(stored || availableModels[0]);
  }, []);

  // Update localStorage whenever the model changes (and is not null).
  useEffect(() => {
    if (model !== null) {
      localStorage.setItem('selectedModel', model);
    }
  }, [model]);

  const [allowRemote, setAllowRemote] = useState(false);
  // In production, force allowRemote to be false.
  useEffect(() => {
    if (isProduction) {
      setAllowRemote(false);
    }
  }, []);

  // Keep track of which models have been loaded.
  const [loadedModels, setLoadedModels] = useState<string[]>([]);
  // Store the current model loading description (null when not loading).
  const [modelLoadingDescription, setModelLoadingDescription] = useState<string | null>(null);

  // Helper to render bot responses based on the model used.
  const renderAssistantContent = (entry: ChatEntry) => {
    const content = entry.content;
    if (!content) return 'Thinking...';
    // Use stored model if available, otherwise fall back to current model.
    const usedModel = entry.modelUsed || model || availableModels[0];
    if (usedModel === 'deepseek/r1-distill-llama-8b/q4') {
      // Extract <think> internal reasoning and main content.
      const regex = /<think>([\s\S]*?)<\/think>([\s\S]*)/;
      const match = content.match(regex);
      if (match) {
        const internalReasoning = match[1].trim();
        const mainContent = match[2].trim();
        return (
          <div className="markdown">
            <Collapsible content={internalReasoning} />
            <ReactMarkdown remarkPlugins={[remarkGfm]}>{mainContent}</ReactMarkdown>
          </div>
        );
      }
    }
    return (
      <div className="markdown">
        <ReactMarkdown remarkPlugins={[remarkGfm]}>{content}</ReactMarkdown>
      </div>
    );
  };

  const sendQuestion = async () => {
    if (loading) return; // Prevent new submissions while loading.
    if (!input.trim()) return;
    setLoading(true);

    // Use guard clause helper to set remote handoff settings.
    const { remoteHandoff, apiKey } = getRemoteHandoffSettings(allowRemote);
    fi.remoteHandoff = remoteHandoff;
    fi.apiKey = apiKey;

    // Append the user's message.
    setChatLog((prev) => [...prev, { role: 'user', content: input }]);
    history.push({ role: 'user', content: input });
    setInput('');

    // Check if the selected model is loaded; if not, fetch it.
    const currentModel = model || availableModels[0];
    if (!loadedModels.includes(currentModel)) {
      setModelLoadingDescription('Start to fetch params');
      await fi.fetchModel(currentModel, (progress: Progress) => {
        setModelLoadingDescription(progress.description ?? null);
      });
      setLoadedModels((prev) => [...prev, currentModel]);
      setModelLoadingDescription(null);
    }

    // Append a placeholder bot message with the current model stored.
    setChatLog((prev) => [...prev, { role: 'bot', content: '', modelUsed: currentModel }]);

    try {
      const response: ChatResponseResult = await fi.chat({
        messages: history,
        model: currentModel,
        stream: true,
        onStreamEvent: (event) => {
          setChatLog((prev) => {
            const updated = [...prev];
            const lastIndex = updated.length - 1;
            updated[lastIndex] = {
              ...updated[lastIndex],
              content: updated[lastIndex].content + event.chunk,
            };
            return updated;
          });
        },
      });
      if (!response.ok) {
        setChatLog((prev) => {
          const updated = [...prev];
          updated[updated.length - 1] = {
            ...updated[updated.length - 1],
            content: 'Failed to get a valid response.',
          };
          return updated;
        });
        return; // Exit early on error.
      }
      history.push(response.message);
    } catch {
      setChatLog(
        history
          .filter((msg) => msg.role !== 'system')
          .map((msg) => ({ role: msg.role as 'user' | 'bot', content: msg.content }))
      );
    } finally {
      setLoading(false);
    }
  };

  // Determine if the submit button should be disabled.
  const isSubmitDisabled = loading || !input.trim() || modelLoadingDescription !== null;

  const handleKeyPress = (e: React.KeyboardEvent<HTMLInputElement>) => {
    if (e.key === 'Enter' && !isSubmitDisabled) {
      e.preventDefault();
      sendQuestion();
    }
  };

  return (
    <div className="flex items-center min-h-[calc(90vh)]">
      <div className="flex flex-col max-h-[calc(80vh)] border rounded-2xl shadow bg-white mt-16 mx-auto mb-8 w-full max-w-7xl">
        {modelLoadingDescription !== null && (
          <div className="p-2 text-center bg-amber-500/50 text-zinc-600 rounded-t-2xl px-4">
            {modelLoadingDescription}
          </div>
        )}

        {/* Chat Messages */}
        <div ref={chatContainerRef} className="flex-grow p-4 overflow-auto mr-4">
          {chatLog.map((entry, index) => (
            <div
              key={index}
              className={`mb-4 flex text-pretty ${entry.role === 'user' ? 'justify-end' : 'justify-start'}`}
            >
              <div
                className={`p-4 rounded-lg ${
                  entry.role === 'user'
                    ? 'max-w-[75%] bg-zinc-100 text-zinc-900 rounded-tr-none'
                    : 'text-zinc-900 rounded-tl-none w-5/6'
                }`}
              >
                {entry.role === 'bot' ? (
                  renderAssistantContent(entry)
                ) : (
                  <ReactMarkdown remarkPlugins={[remarkGfm]}>{entry.content}</ReactMarkdown>
                )}
              </div>
            </div>
          ))}
        </div>

        {/* Updated Input Area */}
        <div className="px-4 py-4">
          <div className="relative">
            {/* Left Icon (Flower Logo) */}
            <div className="absolute inset-y-0 left-0 flex items-center pl-4 pointer-events-none">
              <Image src="/flwr-head.png" alt="Flower Icon" width={50} height={50} />
            </div>
            {/* Text Input */}
            <input
              type="text"
              placeholder="Type your question..."
              className="block w-full p-4 pl-20 text-xl text-gray-900 border border-gray-300 rounded-full bg-white focus:border-amber-400 focus:outline-amber-400"
              value={input}
              onChange={(e) => setInput(e.target.value)}
              onKeyPress={handleKeyPress}
            />
            {/* Send Button */}
            <button
              onClick={sendQuestion}
              disabled={isSubmitDisabled}
              className={`absolute right-4 top-1/2 transform -translate-y-1/2 rounded-full p-2 ${
                isSubmitDisabled
                  ? 'bg-gray-300 cursor-not-allowed'
                  : 'bg-zinc-900 hover:bg-zinc-700 focus:ring-2 focus:ring-amber-400 cursor-pointer'
              }`}
            >
              <ArrowUpIcon className="w-6 h-6 text-white font-bold" />
            </button>
          </div>

          {/* Additional Controls Below Input */}
          <div className="mt-4 flex flex-col md:flex-row items-center md:space-x-4 space-y-4 md:space-y-0">
            {/* Model Select */}
            <div className="p-2 border border-gray-300 rounded-full bg-white text-gray-800">
              <select
                className="w-full border-0 outline-none bg-transparent"
                value={model || availableModels[0]}
                onChange={(e) => setModel(e.target.value)}
              >
                {availableModels.map((modelName) => (
                  <option key={modelName} value={modelName}>
                    {modelName}
                  </option>
                ))}
              </select>
            </div>
            {/* Remote Handoff Toggle (only in non-production) */}
            {!isProduction && (
              <div>
                <label className="flex items-center space-x-2 text-md">
                  <input
                    type="checkbox"
                    checked={allowRemote}
                    onChange={(e) => setAllowRemote(e.target.checked)}
                  />
                  <span className="text-gray-800">Allow remote handoff</span>
                </label>
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}
