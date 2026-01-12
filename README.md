# LitSearch AI

Build a custom research corpus from ArXiv and chat with academic papers using RAG.

## Stack

- **Frontend**: Next.js 14 + TypeScript + Tailwind
- **Backend**: FastAPI + Python
- **Database**: Supabase (PostgreSQL + pgvector)
- **LLM**: OpenAI GPT-4

## Setup

### 1. Backend

```bash
cd backend
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Create .env
OPENAI_API_KEY=your_key
DATABASE_URL=your_supabase_url

# Start
python main.py
```

### 2. Frontend

```bash
cd frontend
npm install

# Create .env.local
NEXT_PUBLIC_API_URL=http://localhost:8000

# Start
npm run dev
```

## Usage

1. Enter research subject → System fetches ~150 papers from ArXiv
2. Wait 5-10min for corpus building (fetch, parse, embed)
3. Chat with your corpus using RAG

## API

- `POST /research/create` - Create corpus
- `GET /research/{id}/status` - Check progress
- `POST /research/{id}/chat` - Ask questions
