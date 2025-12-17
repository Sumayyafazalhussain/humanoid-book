from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Physical AI textbook knowledge
TEXTBOOK_DB = {
    "ros": "🤖 **ROS 2** is robotic middleware for communication between robot components using nodes and topics.",
    "gazebo": "🤖 **Gazebo** is a 3D physics simulator for testing robots in virtual environments with URDF models.",
    "isaac": "🤖 **NVIDIA Isaac** is a platform for AI-powered robot perception and simulation.",
    "humanoid": "🤖 **Humanoid robots** are bipedal robots designed to operate in human environments.",
    "vla": "🤖 **VLA (Vision-Language-Action)** connects computer vision, language models, and robot actions.",
    "hardware": "🤖 **Required hardware**: RTX GPU, NVIDIA Jetson kit, RealSense camera, and Ubuntu OS.",
    "module1": "🤖 **Module 1**: ROS 2 fundamentals for robotic nervous system.",
    "module2": "🤖 **Module 2**: Gazebo & Unity for digital twin simulation.",
    "module3": "🤖 **Module 3**: NVIDIA Isaac for AI-robot brain.",
    "module4": "🤖 **Module 4**: VLA for vision-language-action integration."
}

@app.get("/health")
def health():
    return {"status": "healthy", "service": "Physical AI Chatbot"}

@app.post("/query")
def chatbot(request: dict):
    question = request.get("query", "").lower()
    
    # Always respond - never fail
    if not question:
        return {"answer": "🤖 Ask me about Physical AI, ROS 2, Gazebo, or humanoid robotics!"}
    
    # Check keywords
    for keyword, answer in TEXTBOOK_DB.items():
        if keyword in question:
            return {"answer": answer}
    
    # Fallback response
    return {"answer": f"🤖 Physical AI Assistant: I received your question about '{question}'. The textbook covers robotics, AI integration, and humanoid systems."}

if __name__ == "__main__":
    print("🚀 Starting SIMPLE Physical AI Chatbot...")
    uvicorn.run(app, host="0.0.0.0", port=8000, access_log=False)
