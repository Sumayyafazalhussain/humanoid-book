import google.generativeai as genai
import os
from typing import List

class GeminiChatbot:
    def __init__(self):
        # Configure Gemini with your API key
        genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
        self.model = genai.GenerativeModel(
            'gemini-1.5-pro',
            safety_settings=[
                {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
                {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
                {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
                {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
            ]
        )
    
    def generate_response(self, full_prompt: str) -> str:
        """
        Generate response using the provided full prompt.
        """
        
        try:
            # Generate response
            response = self.model.generate_content(full_prompt)
            
            # Clean response
            answer = response.text.strip()
            
            # Ensure we always return a valid response
            if not answer or len(answer) < 10:
                return self.get_fallback_response("generic") # Use a generic fallback since we don't have user_query here
            
            return answer
            
        except Exception as e:
            print(f"Gemini error: {str(e)}")
            return self.get_fallback_response("generic") # Use a generic fallback for errors
    
    def get_fallback_response(self, query_or_type: str) -> str:
        """Always return a helpful response even if Gemini fails or for generic issues."""
        fallback_responses = {
            "ros": "🤖 **ROS 2** is the Robot Operating System, a middleware framework for robot software development. It enables communication between different robot components through nodes, topics, and services.",
            "gazebo": "🤖 **Gazebo** is a 3D robotics simulator that provides realistic physics simulation, sensor data, and environment modeling for testing robots before real-world deployment.",
            "isaac": "🤖 **NVIDIA Isaac** is a platform for AI-powered robotics that includes Isaac Sim for photorealistic simulation and Isaac ROS for hardware-accelerated perception.",
            "humanoid": "🤖 **Humanoid robots** are bipedal robots designed with human-like form and movement capabilities, allowing them to operate in environments built for humans.",
            "urdf": "🤖 **URDF** (Unified Robot Description Format) is an XML format used to describe a robot's physical structure including links, joints, and sensors.",
            "slam": "🤖 **SLAM** (Simultaneous Localization and Mapping) is a technique that allows robots to build a map of an unknown environment while tracking their position within it).",
            "generic": "🤖 I apologize, but I'm currently having trouble processing your request. Please try again later or rephrase your question."
        }
        
        query_lower = query_or_type.lower()
        for key, response in fallback_responses.items():
            if key in query_lower:
                return response
        
        return fallback_responses["generic"]

