const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';

export interface EnrichResponse {
  job_id: string;
  keywords: string[];
}

export interface StatusResponse {
  status: 'extracting' | 'fetching' | 'parsing' | 'chunking' | 'embedding' | 'ready' | 'error' | null;
  progress: {
    step: string;
    current: number;
    total: number;
    error?: string;
  } | null;
}

export interface ChatResponse {
  answer: string;
  sources: {
    arxiv_id: string;
    title: string;
    section: string;
    excerpt: string;
    url: string;
    score: number;
  }[];
}

export const apiClient = {
  async enrichCorpus(subject: string): Promise<EnrichResponse> {
    const response = await fetch(`${API_BASE_URL}/corpus/enrich`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ subject }),
    });

    if (!response.ok) {
      throw new Error('Failed to enrich corpus');
    }

    return response.json();
  },

  async getCorpusStatus(): Promise<StatusResponse> {
    const response = await fetch(`${API_BASE_URL}/corpus/status`);

    if (!response.ok) {
      throw new Error('Failed to get corpus status');
    }

    return response.json();
  },

  async chat(question: string): Promise<ChatResponse> {
    const response = await fetch(`${API_BASE_URL}/chat`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ question }),
    });

    if (!response.ok) {
      throw new Error('Failed to get chat response');
    }

    return response.json();
  },
};
