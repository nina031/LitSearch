'use client';

import { useState, useEffect } from 'react';
import { apiClient, StatusResponse } from '@/lib/api-client';
import { useRouter } from 'next/navigation';

type ViewState = 'form' | 'building';

export default function SubjectForm() {
  const [subject, setSubject] = useState('');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const [view, setView] = useState<ViewState>('form');
  const [status, setStatus] = useState<StatusResponse | null>(null);
  const router = useRouter();

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setError('');
    setLoading(true);

    try {
      await apiClient.enrichCorpus(subject);
      setView('building');
    } catch (err) {
      setError('Failed to start corpus enrichment. Please try again.');
      console.error(err);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    if (view !== 'building') return;

    const pollStatus = async () => {
      try {
        const result = await apiClient.getCorpusStatus();
        setStatus(result);

        if (result.status === 'ready') {
          router.push('/chat');
        }

        if (result.status === 'error') {
          setError(result.progress?.error || 'An error occurred');
          setView('form');
        }
      } catch (err) {
        console.error(err);
      }
    };

    const interval = setInterval(pollStatus, 2000);
    pollStatus();

    return () => clearInterval(interval);
  }, [view, router]);

  return (
    <div className="min-h-screen flex items-center justify-center bg-gray-50 px-4">
      <div className="max-w-2xl w-full">
        <div className="text-center mb-8">
          <h1 className="text-4xl font-bold text-gray-900 mb-2">
            LitSearch AI
          </h1>
          <p className="text-gray-600">
            Build a custom research corpus and chat with academic papers
          </p>
        </div>

        {view === 'form' ? (
          <form onSubmit={handleSubmit} className="bg-white rounded-lg shadow-md p-8">
            <label htmlFor="subject" className="block text-sm font-medium text-gray-700 mb-2">
              Research Subject
            </label>
            <textarea
              id="subject"
              value={subject}
              onChange={(e) => setSubject(e.target.value)}
              placeholder="e.g., Brain tumor detection using deep learning and MRI imaging"
              className="text-gray-700 w-full px-4 py-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent resize-none"
              rows={4}
              required
              disabled={loading}
            />

            {error && (
              <div className="mt-4 p-4 bg-red-50 border border-red-200 rounded-lg text-red-600 text-sm">
                {error}
              </div>
            )}

            <button
              type="submit"
              disabled={loading || !subject.trim()}
              className="mt-6 w-full bg-blue-600 hover:bg-blue-700 disabled:bg-gray-400 text-white font-medium py-3 px-6 rounded-lg transition-colors"
            >
              {loading ? 'Starting...' : 'Start Research'}
            </button>

            <p className="mt-4 text-sm text-gray-500 text-center">
              We'll fetch 100 relevant papers from ArXiv and build your research corpus
            </p>
          </form>
        ) : (
          <div className="bg-white rounded-lg shadow-md p-8">
            <h2 className="text-2xl font-bold text-gray-900 mb-6">
              Building Your Research Corpus
            </h2>

            <div className="flex flex-col items-center py-8">
              <div className="w-12 h-12 border-4 border-gray-200 border-t-blue-600 rounded-full animate-spin mb-6" />
              <p className="text-sm text-gray-500 text-center">
                This may take a few minutes.
              </p>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
