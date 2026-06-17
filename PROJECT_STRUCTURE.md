# Project Structure

## Core
- `/agents/` - Agent definitions and workflows
  - `discovery_agent.py` (with Librosa audio processing for BPM, energy, key, etc.)
  - content_agent.py (planned)
  - ad_agent.py (planned)
  - orchestrator.py (planned)
- `/docs/` - Business and technical documentation
- `/frontend/` - Landing page and dashboard (planned)
- `/backend/` - API and orchestration (planned)

## Tech Stack
- Python + CrewAI for agents
- librosa + numpy for audio feature extraction
- LangChain/OpenAI for LLM reasoning
- Next.js for frontend
- Supabase or Firebase for DB
- Stripe for payments

## Next Steps
- Audio upload endpoint
- Full multi-agent crew
- Error handling and tests