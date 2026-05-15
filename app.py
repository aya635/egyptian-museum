from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse, StreamingResponse
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import google.generativeai as genai
import edge_tts
import asyncio
import os
import io
import tempfile
from huggingface_hub import hf_hub_download

app = FastAPI(title="Egyptian Museum TTS API")

# ==================== CONFIG ====================
HF_TOKEN = os.environ.get("HF_TOKEN", "")
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
HF_REPO = "aya-nasser/egyptian-museum-classifier"

# ==================== STATUES MAP ====================
STATUES = {
    "1":  {"name": "تمثال للملك تحتمس الأول مع آمون رع والملكة إعح", "gender": "male"},
    "2":  {"name": "تمثال حجرى ضخم للملك رمسيس الثاني", "gender": "male"},
    "3":  {"name": "لوحة خشبية ملونة لسيدة تدعى نس-خونسو", "gender": "female"},
    "4":  {"name": "تمثال باك-إن-خونسو الثالث", "gender": "male"},
    "5":  {"name": "لوحة لشخص يدعى ون-نفر", "gender": "male"},
    "6":  {"name": "تمثال كتلة لشخص يدعى با-نوب-إقر", "gender": "male"},
    "7":  {"name": "صندل لمسحتي", "gender": "male"},
    "8":  {"name": "قناع مومياء لمسحتي", "gender": "male"},
    "9":  {"name": "آنية برونزية لمسحتي", "gender": "male"},
    "10": {"name": "وعاء لمسحتي", "gender": "male"},
    "11": {"name": "محفة الملكة حتب-حرس الأولى", "gender": "female"},
    "12": {"name": "عصا خشبية لمسحتي", "gender": "male"},
    "13": {"name": "قلادة فضية لمسحتي", "gender": "male"},
    "14": {"name": "صولجان واس خشبي لمسحتي", "gender": "male"},
    "15": {"name": "إناء لحفظ الزيوت العطرية لمسحتي", "gender": "male"},
    "16": {"name": "مرآة برونزية لمسحتي", "gender": "female"},
    "17": {"name": "نموذج إناء حس لمسحتي", "gender": "male"},
    "18": {"name": "تمثال صغير من الألباستر لمسحتي", "gender": "male"},
    "19": {"name": "التابوت الخارجي لمسحتي", "gender": "male"},
    "20": {"name": "مسند رأس لمسحتي", "gender": "male"},
    "21": {"name": "صلاية على شكل صقر", "gender": "male"},
    "22": {"name": "تمثال للكاتب دوا-بتاح مع والده", "gender": "male"},
    "23": {"name": "تمثال الكاتب متري", "gender": "male"},
    "24": {"name": "نموذج لخادم يشوى إوزة", "gender": "male"},
    "25": {"name": "نموذج لسيدتين تصنعان الجعة", "gender": "female"},
    "26": {"name": "تمثال للكاتب الملكي رع-حتب", "gender": "male"},
    "27": {"name": "تمثال جالس للملك تحتمس الثالث", "gender": "male"},
    "28": {"name": "تمثال لاشيثي", "gender": "male"},
    "29": {"name": "الجزء العلوي لتمثال الملك تحتمس الرابع", "gender": "male"},
    "30": {"name": "تمثال لاشيثي الثاني", "gender": "male"},
}

CLASS_NAMES = [str(i) for i in sorted([int(k) for k in STATUES.keys()])]

# ==================== LOAD MODEL ====================
device = torch.device("cpu")
detection_model = None

def load_model():
    global detection_model
    print("Downloading model from Hugging Face...")
    model_path = hf_hub_download(
        repo_id=HF_REPO,
        filename="statue_classifier.pth",
        token=HF_TOKEN if HF_TOKEN else None
    )
    detection_model = models.resnet50()
    detection_model.fc = torch.nn.Linear(detection_model.fc.in_features, len(CLASS_NAMES))
    detection_model.load_state_dict(torch.load(model_path, map_location=device))
    detection_model.to(device)
    detection_model.eval()
    print("Model loaded successfully!")
@app.on_event("startup")
async def startup_event():
    if GEMINI_API_KEY:
        genai.configure(api_key=GEMINI_API_KEY)
    import threading
    threading.Thread(target=load_model, daemon=True).start()

# ==================== PREPROCESS ====================
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# ==================== DETECT ====================
def detect_artifact(image_bytes):
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img_t = preprocess(img)
    batch_t = torch.unsqueeze(img_t, 0).to(device)

    with torch.no_grad():
        outputs = detection_model(batch_t)
        _, preds = torch.max(outputs, 1)
        prob = torch.nn.functional.softmax(outputs, dim=1)[0]
        confidence = prob[preds[0]].item()

    predicted_class = CLASS_NAMES[preds[0]]
    info = STATUES.get(str(predicted_class), {"name": "قطعة أثرية", "gender": "male"})
    return predicted_class, info["name"], info["gender"], confidence

# ==================== STORY ====================
def generate_story(artifact_name, gender, language="ar"):
    gemini = genai.GenerativeModel("gemini-2.5-flash")
    if language == "ar":
        pronoun = "انتِ" if gender == "female" else "انتَ"
        prompt = (
            f"{pronoun} {artifact_name}، قطعة اثرية في المتحف المصري.\n"
            "احكِ قصتك في جملتين او ثلاثة بضمير المتكلم (انا).\n"
            "اذكر: اسمك، متى صُنعت، لماذا صُنعت، وجملة مشوقة عن سرك.\n"
            "بالعربي الفصيح، مختصر وجذاب.\n"
            "مهم جدا: اكتب الرد بالكلمات فقط بدون اي رموز او علامات تنسيق."
        )
    else:
        prompt = (
            f"You are {artifact_name}, an artifact in the Egyptian Museum.\n"
            "Tell your story in 2-3 sentences in first person (I am...).\n"
            "Include: your name, when and why you were made, and one captivating secret.\n"
            "Keep it short and elegant.\n"
            "IMPORTANT: Write only plain words, no symbols or formatting marks."
        )
    response = gemini.generate_content(prompt)
    return response.text

# ==================== TTS ====================
async def tts_async(text, gender="male", language="ar"):
    if language == "ar":
        voice = "ar-EG-ShakirNeural" if gender == "male" else "ar-EG-SalmaNeural"
    else:
        voice = "en-US-GuyNeural" if gender == "male" else "en-US-JennyNeural"

    communicate = edge_tts.Communicate(text, voice)
    tmp = tempfile.NamedTemporaryFile(suffix=".mp3", delete=False)
    await communicate.save(tmp.name)
    with open(tmp.name, "rb") as f:
        audio = f.read()
    os.unlink(tmp.name)
    return audio

# ==================== ENDPOINTS ====================
@app.get("/")
def home():
    return {"status": "running", "project": "Egyptian Museum TTS"}

@app.post("/predict")
async def predict(
    file: UploadFile = File(...),
    language: str = "ar"
):
    image_bytes = await file.read()
    predicted_class, name, gender, confidence = detect_artifact(image_bytes)

    story = generate_story(name, gender, language)
    audio = await tts_async(story, gender, language)

    return JSONResponse({
        "class_id": predicted_class,
        "artifact_name": name,
        "gender": gender,
        "confidence": round(confidence * 100, 1),
        "language": language,
        "story": story,
        "audio_url": f"/audio?class_id={predicted_class}&language={language}"
    })

@app.post("/predict/audio")
async def predict_audio(
    file: UploadFile = File(...),
    language: str = "ar"
):
    image_bytes = await file.read()
    predicted_class, name, gender, confidence = detect_artifact(image_bytes)
    story = generate_story(name, gender, language)
    audio = await tts_async(story, gender, language)

    return StreamingResponse(
        io.BytesIO(audio),
        media_type="audio/mpeg",
        headers={"Content-Disposition": "attachment; filename=story.mp3"}
    )
