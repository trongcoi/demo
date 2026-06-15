from flask import Flask, render_template, request, send_file
from faster_whisper import WhisperModel
import json
import os
import re
import subprocess

try:
    from deep_translator import GoogleTranslator
except ImportError:
    GoogleTranslator = None

try:
    import requests
except ImportError:
    requests = None

app = Flask(__name__)

UPLOAD_FOLDER = "uploads"
OUTPUT_FOLDER = "outputs"

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

model = WhisperModel("base", device="cpu")

TARGET_LANGUAGE = "vi"
OPENAI_API_URL = "https://api.openai.com/v1/responses"
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
DEEPSEEK_API_URL = "https://api.deepseek.com/chat/completions"
DEEPSEEK_MODEL = os.getenv("DEEPSEEK_MODEL", "deepseek-v4-flash")
POLISH_BATCH_SIZE = 25
NATURAL_VI_REPLACEMENTS = {
    "Tuy nhiên": "Nhưng",
    "tuy nhiên": "nhưng",
    "Bởi vì": "Vì",
    "bởi vì": "vì",
    "Do đó": "Vì vậy",
    "do đó": "vì vậy",
    "Điều đó": "Việc đó",
    "điều đó": "việc đó",
    "Tôi không biết": "Mình không biết",
    "tôi không biết": "mình không biết",
}

def format_time(seconds):
    ms = int((seconds % 1) * 1000)
    s = int(seconds) % 60
    m = (int(seconds) // 60) % 60
    h = int(seconds) // 3600
    return f"{h:02}:{m:02}:{s:02},{ms:03}"

def split_text_for_translation(text, max_chars=4500):
    words = text.split()
    chunks = []
    current = []
    current_length = 0

    for word in words:
        next_length = current_length + len(word) + (1 if current else 0)
        if current and next_length > max_chars:
            chunks.append(" ".join(current))
            current = [word]
            current_length = len(word)
        else:
            current.append(word)
            current_length = next_length

    if current:
        chunks.append(" ".join(current))

    return chunks or [text]

def translate_to_vietnamese(text):
    cleaned_text = text.strip()
    if not cleaned_text:
        return ""

    if GoogleTranslator is None:
        raise RuntimeError(
            "Chưa cài thư viện dịch. Hãy chạy: pip install -r requirements.txt"
        )

    translator = GoogleTranslator(source="auto", target=TARGET_LANGUAGE)
    translated_parts = [
        translator.translate(chunk)
        for chunk in split_text_for_translation(cleaned_text)
    ]
    translated_text = " ".join(part for part in translated_parts if part).strip()
    return make_vietnamese_more_natural(translated_text)

def translate_subtitle_texts_to_vietnamese(texts):
    translated_texts = [translate_to_vietnamese(text) for text in texts]
    return polish_vietnamese_subtitles(texts, translated_texts)

def polish_vietnamese_subtitles(original_texts, translated_texts):
    ai_config = get_ai_polisher_config()
    if ai_config is None or requests is None:
        return [make_vietnamese_more_natural(text) for text in translated_texts]

    polished_texts = []

    for start in range(0, len(translated_texts), POLISH_BATCH_SIZE):
        original_chunk = original_texts[start:start + POLISH_BATCH_SIZE]
        translated_chunk = translated_texts[start:start + POLISH_BATCH_SIZE]
        polished_texts.extend(
            polish_vietnamese_subtitle_batch(
                original_chunk,
                translated_chunk,
                ai_config
            )
        )

    return polished_texts

def get_ai_polisher_config():
    deepseek_api_key = os.getenv("DEEPSEEK_API_KEY")
    if deepseek_api_key:
        return {
            "provider": "deepseek",
            "api_key": deepseek_api_key,
            "api_url": DEEPSEEK_API_URL,
            "model": DEEPSEEK_MODEL
        }

    openai_api_key = os.getenv("OPENAI_API_KEY")
    if openai_api_key:
        return {
            "provider": "openai",
            "api_key": openai_api_key,
            "api_url": OPENAI_API_URL,
            "model": OPENAI_MODEL
        }

    return None

def polish_vietnamese_subtitle_batch(original_texts, translated_texts, ai_config):
    messages = [
        {
            "role": "system",
            "content": (
                "Bạn là biên tập viên phụ đề tiếng Việt. Viết lại bản dịch "
                "sao cho tự nhiên như người Việt nói hằng ngày, dễ hiểu, "
                "không văn dịch, không cứng máy móc. Giữ đúng nghĩa, tên riêng, "
                "sắc thái cảm xúc và độ ngắn phù hợp phụ đề."
            )
        },
        {
            "role": "user",
            "content": (
                "Dựa vào từng câu gốc và bản dịch thô, hãy trả về DUY NHẤT "
                "một JSON array các câu tiếng Việt đã được viết tự nhiên hơn. "
                "Số phần tử và thứ tự phải giữ nguyên.\n\n"
                f"{json.dumps(build_polish_items(original_texts, translated_texts), ensure_ascii=False)}"
            )
        }
    ]

    if ai_config["provider"] == "deepseek":
        payload = {
            "model": ai_config["model"],
            "messages": messages,
            "temperature": 0.35,
            "stream": False
        }
    else:
        payload = {
            "model": ai_config["model"],
            "input": messages,
            "temperature": 0.35
        }

    try:
        response = requests.post(
            ai_config["api_url"],
            headers={
                "Authorization": f"Bearer {ai_config['api_key']}",
                "Content-Type": "application/json"
            },
            json=payload,
            timeout=60
        )
        response.raise_for_status()
        response_data = response.json()
        polished_texts = parse_polished_texts(response_data, ai_config["provider"])

        if len(polished_texts) == len(translated_texts):
            return [make_vietnamese_more_natural(text) for text in polished_texts]
    except Exception:
        pass

    return [make_vietnamese_more_natural(text) for text in translated_texts]

def build_polish_items(original_texts, translated_texts):
    return [
        {
            "original": original_text,
            "draft_vi": translated_text
        }
        for original_text, translated_text in zip(original_texts, translated_texts)
    ]

def parse_polished_texts(response_data, provider):
    if provider == "deepseek":
        text = (
            response_data
            .get("choices", [{}])[0]
            .get("message", {})
            .get("content", "")
        )
    else:
        text = response_data.get("output_text", "")

    if provider == "openai" and not text:
        text_parts = []
        for output_item in response_data.get("output", []):
            for content_item in output_item.get("content", []):
                if content_item.get("type") in ("output_text", "text"):
                    text_parts.append(content_item.get("text", ""))
        text = "\n".join(text_parts)

    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    parsed = json.loads(text)

    if not isinstance(parsed, list):
        return []

    return [str(item).strip() for item in parsed]

def make_vietnamese_more_natural(text):
    text = re.sub(r"\s+", " ", text).strip()
    text = re.sub(r"\s+([,.!?;:])", r"\1", text)

    for formal_text, natural_text in NATURAL_VI_REPLACEMENTS.items():
        text = text.replace(formal_text, natural_text)

    return text

def build_subtitle_text(original_text, vietnamese_text, detected_language):
    original_text = original_text.strip()
    vietnamese_text = vietnamese_text.strip()

    if detected_language == TARGET_LANGUAGE or original_text == vietnamese_text:
        return original_text

    return f"{original_text}\n{vietnamese_text}"

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/upload", methods=["POST"])
def upload():

    video = request.files["video"]

    video_path = os.path.join(
        UPLOAD_FOLDER,
        video.filename
    )

    video.save(video_path)

    audio_path = video_path + ".wav"

    subprocess.run([
        "ffmpeg",
        "-i",
        video_path,
        "-ar",
        "16000",
        "-ac",
        "1",
        audio_path,
        "-y"
    ])

    segments, info = model.transcribe(audio_path)
    subtitle_segments = list(segments)
    detected_language = (getattr(info, "language", "") or "").lower()
    original_texts = [segment.text.strip() for segment in subtitle_segments]
    vietnamese_texts = (
        original_texts
        if detected_language == TARGET_LANGUAGE
        else translate_subtitle_texts_to_vietnamese(original_texts)
    )

    srt_path = os.path.join(
        OUTPUT_FOLDER,
        video.filename + ".srt"
    )

    with open(srt_path, "w", encoding="utf-8") as f:

        for idx, segment in enumerate(subtitle_segments, start=1):

            start = format_time(segment.start)
            end = format_time(segment.end)
            original_text = segment.text.strip()
            vietnamese_text = vietnamese_texts[idx - 1]
            subtitle_text = build_subtitle_text(
                original_text,
                vietnamese_text,
                detected_language
            )

            f.write(f"{idx}\n")
            f.write(f"{start} --> {end}\n")
            f.write(f"{subtitle_text}\n\n")

    return send_file(
        srt_path,
        as_attachment=True
    )

if __name__ == "__main__":
    app.run(debug=True)
