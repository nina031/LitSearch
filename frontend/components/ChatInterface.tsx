'use client';

import { useState } from 'react';
import { apiClient, ChatResponse } from '@/lib/api-client';

interface Message {
  role: 'user' | 'assistant';
  content: string;
  sources?: ChatResponse['sources'];
}

export default function ChatInterface() {
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState('');
  const [loading, setLoading] = useState(false);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!input.trim() || loading) return;

    const userMessage: Message = {
      role: 'user',
      content: input,
    };

    setMessages((prev) => [...prev, userMessage]);
    setInput('');
    setLoading(true);

    try {
      const response = await apiClient.chat(input);

      const assistantMessage: Message = {
        role: 'assistant',
        content: response.answer,
        sources: response.sources,
      };

      setMessages((prev) => [...prev, assistantMessage]);
    } catch (err) {
      console.error(err);
      const errorMessage: Message = {
        role: 'assistant',
        content: 'Sorry, I encountered an error. Please try again.',
      };
      setMessages((prev) => [...prev, errorMessage]);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="flex flex-col h-screen bg-gray-50">
      <div className="bg-white border-b border-gray-200 px-6 py-4">
        <h1 className="text-xl font-bold text-black">Research Chat</h1>
        <p className="text-sm text-black">Ask questions about your research corpus</p>
      </div>

      <div className="flex-1 overflow-y-auto px-6 py-4">
        {messages.length === 0 ? (
          <div className="text-center text-black mt-20">
            <p className="text-lg mb-2">Start asking questions!</p>
            <p className="text-sm">Try: "What are the main findings in this corpus?"</p>
          </div>
        ) : (
          <div className="space-y-4 max-w-4xl mx-auto">
            {messages.map((message, index) => (
              <div key={index} className={`flex ${message.role === 'user' ? 'justify-end' : 'justify-start'}`}>
                <div
                  className={`max-w-3xl rounded-lg px-4 py-3 ${
                    message.role === 'user'
                      ? 'bg-blue-600 text-white'
                      : 'bg-white border border-gray-200 text-black'
                  }`}
                >
                  <p className="whitespace-pre-wrap">{message.content}</p>

                  {message.sources && message.sources.length > 0 && (
                    <div className="mt-3 pt-3 border-t border-gray-200">
                      <p className="text-xs font-semibold text-black mb-3">Sources:</p>
                      <div className="space-y-3">
                        {message.sources.map((source, idx) => (
                          <div key={idx} className="bg-gray-50 rounded-lg p-3 border border-gray-100">
                            <div className="flex items-start justify-between gap-2">
                              <div className="flex-1 min-w-0">
                                <div className="flex items-center gap-2 mb-1">
                                  <a
                                    href={source.url}
                                    target="_blank"
                                    rel="noopener noreferrer"
                                    className="text-blue-600 hover:text-blue-800 hover:underline font-mono text-xs font-medium"
                                  >
                                    arXiv:{source.arxiv_id}
                                  </a>
                                  <span className={`inline-flex items-center px-2 py-0.5 rounded text-xs font-medium ${
                                    source.section === 'Abstract'
                                      ? 'bg-purple-100 text-purple-800'
                                      : 'bg-gray-200 text-gray-700'
                                  }`}>
                                    {source.section}
                                  </span>
                                  <span className="text-xs text-gray-500">
                                    {Math.round(source.score * 100)}% pertinent
                                  </span>
                                </div>
                                <p className="text-sm font-medium text-gray-900 truncate">
                                  {source.title}
                                </p>
                              </div>
                              <a
                                href={source.url}
                                target="_blank"
                                rel="noopener noreferrer"
                                className="flex-shrink-0 text-gray-400 hover:text-blue-600"
                                title="Ouvrir sur arXiv"
                              >
                                <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M10 6H6a2 2 0 00-2 2v10a2 2 0 002 2h10a2 2 0 002-2v-4M14 4h6m0 0v6m0-6L10 14" />
                                </svg>
                              </a>
                            </div>
                            <p className="mt-2 text-xs text-gray-600 italic leading-relaxed">
                              "{source.excerpt}"
                            </p>
                          </div>
                        ))}
                      </div>
                    </div>
                  )}
                </div>
              </div>
            ))}

            {loading && (
              <div className="flex justify-start">
                <div className="bg-white border border-gray-200 rounded-lg px-4 py-3">
                  <div className="flex space-x-2">
                    <div className="w-2 h-2 bg-gray-400 rounded-full animate-bounce" />
                    <div className="w-2 h-2 bg-gray-400 rounded-full animate-bounce" style={{ animationDelay: '0.1s' }} />
                    <div className="w-2 h-2 bg-gray-400 rounded-full animate-bounce" style={{ animationDelay: '0.2s' }} />
                  </div>
                </div>
              </div>
            )}
          </div>
        )}
      </div>

      <div className="bg-white border-t border-gray-200 px-6 py-4">
        <form onSubmit={handleSubmit} className="max-w-4xl mx-auto">
          <div className="flex space-x-4">
            <input
              type="text"
              value={input}
              onChange={(e) => setInput(e.target.value)}
              placeholder="Ask a question..."
              className="flex-1 px-4 py-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent text-black placeholder:text-gray-500"
              disabled={loading}
            />
            <button
              type="submit"
              disabled={loading || !input.trim()}
              className="bg-blue-600 hover:bg-blue-700 disabled:bg-gray-400 text-white font-medium px-6 py-3 rounded-lg transition-colors"
            >
              Send
            </button>
          </div>
        </form>
      </div>
    </div>
  );
}
