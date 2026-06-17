# Full Orchestrator Crew Implementation for EchoForge AI

import json
from typing import Dict, Any
from pydantic import BaseModel
from crewai import Crew, Task
from .discovery_agent import DiscoveryAgent
from .content_creation_agent import ContentCreationAgent

class CampaignOutput(BaseModel):
    discovery_brief: Dict[str, Any]
    content_assets: Dict[str, Any]
    next_steps: str
    status: str

class OrchestratorCrew:
    """Main orchestrator that coordinates Discovery and Content Creation Agents."""
    
    def __init__(self):
        self.discovery = DiscoveryAgent()
        self.content = ContentCreationAgent()
        
        # Define sequential tasks
        self.tasks = [
            Task(
                description="Perform full audio analysis and generate targeting brief using librosa features and LLM insights.",
                agent=self.discovery.agent if hasattr(self.discovery, 'agent') else None,
                expected_output="Structured JSON with BPM, mood, audience personas, keywords, etc."
            ),
            Task(
                description="Generate promotional assets: short video clips with moviepy, captions, thumbnails, hashtags based on discovery results.",
                agent=self.content.agent if hasattr(self.content, 'agent') else None,
                expected_output="Asset bundle ready for social posting and ads."
            )
        ]
        
        self.crew = Crew(
            agents=[self.discovery.agent, self.content.agent] if hasattr(self.discovery, 'agent') else [],
            tasks=self.tasks,
            verbose=True,
            memory=True,
            cache=True
        )
    
    def run_full_orchestration(self, audio_path: str, metadata: Dict) -> CampaignOutput:
        """Kick off the full Discovery -> Content pipeline."""
        print(f"🚀 Orchestrating campaign for {metadata.get('title', 'Unknown Track')}")
        
        # Run Discovery
        discovery_results = self.discovery.run_analysis(audio_path, metadata)
        
        # Pass to Content Creation
        content_results = self.content.create_content(discovery_results, audio_path)
        
        return CampaignOutput(
            discovery_brief=discovery_results,
            content_assets=content_results,
            next_steps="Ready for Outreach & Playlist Agent or Ad Optimization.",
            status="success"
        )

# Example usage / test
if __name__ == "__main__":
    import os
    orchestrator = OrchestratorCrew()
    test_metadata = {
        "title": "Midnight Dreams",
        "artist": "Echo Artist",
        "genre": "Lo-Fi Chillhop",
        "bio": "Emerging independent producer."
    }
    result = orchestrator.run_full_orchestration("uploads/sample.mp3", test_metadata)
    print(json.dumps(result.model_dump(), indent=2))