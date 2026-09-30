from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import uvicorn
import os

app = FastAPI(
    title="Marv-instruct API Render",
    description="API de raisonnement logique multi-étapes (<reflexion>)",
    version="1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

MODEL_ID = "Theplayboy117/marv-instruct-3b"

print("⏳ Téléchargement et chargement du modèle Marv-instruct depuis Hugging Face...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.float16,
    device_map="auto"
)
print("✅ Modèle chargé avec succès sur Render !")

class Message(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    messages: List[Message]
    max_tokens: Optional[int] = 400
    temperature: Optional[float] = 0.3
    top_p: Optional[float] = 0.9

class ChatChoice(BaseModel):
    index: int
    message: Message
    finish_reason: str

class ChatResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    choices: List[ChatChoice]

@app.get("/")
def home():
    return {"status": "online", "model": "marv-instruct-3b", "creator": "Marvens ZAMY"}

@app.post("/v1/chat/completions", response_model=ChatResponse)
def generate_chat(request: ChatRequest):
    try:
        full_prompt = ""
        for msg in request.messages:
            full_prompt += f"<|im_start|>{msg.role}\n{msg.content}<|im_end|>\n"

        full_prompt += "<|im_start|>assistant\n<reflexion>\n"

        inputs = tokenizer(full_prompt, return_tensors="pt").to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=request.max_tokens,
                do_sample=True,
                temperature=request.temperature,
                top_p=request.top_p,
                pad_token_id=tokenizer.eos_token_id,
            )

        input_length = inputs["input_ids"].shape[1]
        raw_text = tokenizer.decode(outputs[0][input_length:], skip_special_tokens=False)

        generated_content = "<reflexion>\n" + raw_text.replace("<|im_end|>", "").strip()

        return ChatResponse(
            id="chatcmpl-marv-render",
            choices=[
                ChatChoice(
                    index=0,
                    message=Message(role="assistant", content=generated_content),
                    finish_reason="stop",
                )
            ],
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    uvicorn.run("app:app", host="0.0.0.0", port=port, reload=False)
