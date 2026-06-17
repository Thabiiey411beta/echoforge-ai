# Agent Workflows

## Discovery Agent Workflow (Updated with Librosa)

1. **Audio Ingestion**: Receive MP3/WAV file.
2. **Librosa Feature Extraction**:
   - BPM/Tempo
   - Spectral features (centroid, energy)
   - Harmonic/percussive separation
   - Key/Chroma estimation
3. **LLM Analysis**: Feed features + metadata to CrewAI agent.
4. **Output**: JSON with personas, targeting, keywords.

Full implementation in `agents/discovery_agent.py`.

Other workflows to be added.