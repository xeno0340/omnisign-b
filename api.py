from fastapi import FastAPI, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from tts import speak_text
from gloss_dict import get_gloss
import json

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

clients = []

class SignInput(BaseModel):
    signs: list
    emotion: str = "neutral"
    language: str = "English"

class TextInput(BaseModel):
    text: str

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    clients.append(websocket)
    try:
        while True:
            await websocket.receive_text()
    except Exception:
        if websocket in clients:
            clients.remove(websocket)

async def broadcast(data: dict):
    for client in clients[:]:
        try:
            await client.send_json(data)
        except Exception:
            if client in clients:
                clients.remove(client)

@app.post("/signs-to-speech")
async def signs_to_speech(data: SignInput):
    print(f"RECEIVED: {data.signs} | emotion: {data.emotion}")
    signs = data.signs
    if not signs:
        return {"status": "empty"}
    text = " ".join(signs).lower().capitalize()
    print(f"SPEAKING: {text}")
    speak_text(text, data.language)
    await broadcast({
        "type": "sign",
        "text": text,
        "signs": signs,
        "emotion": data.emotion
    })
    return {"sentence": text, "status": "spoken"}

@app.post("/text-to-gloss")
async def text_to_gloss(data: TextInput):
    gloss = get_gloss(data.text)
    if not gloss:
        gloss = data.text.upper().split()
    await broadcast({"type": "gloss", "gloss": gloss})
    return {"gloss": gloss}

@app.post("/emergency")
async def emergency():
    speak_text("Emergency! Please help immediately!", "English")
    await broadcast({"type": "emergency"})
    return {"status": "emergency alert spoken"}

@app.get("/poses")
def get_poses():
    with open(r"C:\Projects\omnisign-b\hand_poses.json") as f:
        return json.load(f)

@app.get("/health")
def health():
    return {"status": "Member B API running"}