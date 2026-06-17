# Orchestrator Crew for EchoForge AI

# TODO: Implement full orchestrator
import os
from crewai import Crew
from agents.discovery_agent import DiscoveryAgent
from agents.content_creation_agent import ContentCreationAgent

class OrchestratorCrew:
    def __init__(self):
        self.discovery = DiscoveryAgent()
        self.content = ContentCreationAgent()
    
    def run_campaign(self, audio_path, metadata):
        print('Starting Orchestrator...')
        discovery_results = self.discovery.run_analysis(audio_path, metadata)
        content_results = self.content.create_content(discovery_results, audio_path)
        return {
            'discovery': discovery_results,
            'content': content_results,
            'status': 'Campaign initialized - Discovery + Content complete'
        }

if __name__ == "__main__":
    # Test
    orchestrator = OrchestratorCrew()
    result = orchestrator.run_campaign('sample.mp3', {'title': 'Test Track'})
    print(result)