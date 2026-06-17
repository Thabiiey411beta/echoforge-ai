# Discovery Agent with Librosa Audio Processing

import os
from typing import Dict, Any, List
import librosa
import numpy as np
from crewai import Agent, Task, Crew
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()

class DiscoveryAgent:
    def __init__(self):
        self.llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.3)
        self.agent = self._create_agent()

    def _create_agent(self) -> Agent:
        return Agent(
            role='Music Discovery & Audio Analysis Specialist',
            goal='Analyze uploaded tracks for genre, mood, tempo, energy, and suggest optimal targeting for promotion.',
            backstory='Expert musicologist and data analyst using advanced audio processing to uncover hidden potential in unknown artists.',
            llm=self.llm,
            verbose=True,
            allow_delegation=False,
        )

    def analyze_audio(self, audio_path: str) -> Dict[str, Any]:
        """Process audio file with librosa and extract features."""
        try:
            # Load audio file
            y, sr = librosa.load(audio_path, sr=None)
            
            # Extract features
            features = {
                'duration': librosa.get_duration(y=y, sr=sr),
                'bpm': float(librosa.beat.beat_track(y=y, sr=sr)[0]),
                'spectral_centroid': float(np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))),
                'rms_energy': float(np.mean(librosa.feature.rms(y=y))),
                'zero_crossing_rate': float(np.mean(librosa.feature.zero_crossing_rate(y=y))),
            }
            
            # Tempo and beat
            tempo, beats = librosa.beat.beat_track(y=y, sr=sr)
            features['tempo'] = float(tempo)
            features['num_beats'] = len(beats)
            
            # Harmonic/Percussive
            y_harmonic, y_percussive = librosa.effects.hpss(y)
            features['harmonic_ratio'] = float(np.mean(np.abs(y_harmonic)) / (np.mean(np.abs(y)) + 1e-8))
            
            # Chroma for key estimation
            chroma = librosa.feature.chroma_cqt(y=y, sr=sr)
            features['key'] = int(np.argmax(np.mean(chroma, axis=1)))
            
            return features
        except Exception as e:
            return {'error': str(e)}

    def generate_analysis_task(self, audio_path: str, metadata: Dict) -> Task:
        audio_features = self.analyze_audio(audio_path)
        
        prompt = f"""
        Analyze this music track for promotion:
        Title: {metadata.get('title', 'Unknown')}
        Artist: {metadata.get('artist', 'Unknown Artist')}
        Genre: {metadata.get('genre', 'Unknown')}
        Bio: {metadata.get('bio', '')}
        
        Audio Features:
        {audio_features}
        
        Provide:
        1. Detailed mood and emotion profile
        2. Target audience personas (age, platforms, interests)
        3. Similar artists and subgenres
        4. Optimal platforms and regions for promotion
        5. Keywords, hashtags, and campaign themes
        6. Risk factors or unique selling points
        """
        
        return Task(
            description=prompt,
            agent=self.agent,
            expected_output="Structured JSON with analysis sections and recommendations",
        )

    def run_analysis(self, audio_path: str, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Run full discovery analysis."""
        task = self.generate_analysis_task(audio_path, metadata)
        crew = Crew(
            agents=[self.agent],
            tasks=[task],
            verbose=2
        )
        result = crew.kickoff()
        
        return {
            'audio_features': self.analyze_audio(audio_path),
            'llm_analysis': str(result),
            'recommendations': self._parse_recommendations(str(result))
        }

    def _parse_recommendations(self, text: str) -> Dict:
        """Basic parsing - improve with structured output."""
        return {'summary': text[:500] + '...' if len(text) > 500 else text}


# Example usage
if __name__ == "__main__":
    # Example
    agent = DiscoveryAgent()
    metadata = {
        "title": "Midnight Dreams",
        "artist": "Unknown Indie",
        "genre": "Lo-Fi Chillhop",
        "bio": "Emerging artist blending chill vibes with emotional lyrics."
    }
    # result = agent.run_analysis("path/to/track.mp3", metadata)
    print("Discovery Agent ready. Librosa integration added for audio feature extraction.")
