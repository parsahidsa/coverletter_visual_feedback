from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import JSONResponse
from pdf2image import convert_from_bytes
from openai import OpenAI
import io
import base64
import os
from dotenv import load_dotenv

load_dotenv()
app = FastAPI()

# ایجاد کلاینت OpenAI
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

@app.post("/analyze-resume-visual")
async def analyze_resume_visual(file: UploadFile = File(...), job_description: str = Form(...)):
    try:
        # چاپ job_description برای اطمینان از دریافت آن
        print(f"Job Description: {job_description}")
        
        pdf_bytes = await file.read()

        # تبدیل به تصویر - همه صفحات (یا فقط چند صفحه اول)
        images = convert_from_bytes(pdf_bytes, dpi=200)
        if not images:
            return JSONResponse(content={"error": "PDF conversion failed"}, status_code=400)

        # فقط ۳ صفحه اول رو بررسی کن (اگه کمتر بود، همونا)
        max_pages = min(3, len(images))
        image_messages = []

        for i in range(max_pages):
            buffered = io.BytesIO()
            images[i].save(buffered, format="PNG")
            img_base64 = base64.b64encode(buffered.getvalue()).decode("utf-8")
            image_messages.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{img_base64}"}
            })

        # ارسال به GPT-4v با همه تصاویر با متد جدید
        response = client.chat.completions.create(
            model="gpt-4.1",
            messages=[
                {"type": "text", "text": "You are a professional German application consultant with expertise in creating creative, personalized, and impactful cover letters (Anschreiben) for job applications in Germany. "
                "Follow the latest best practices for Anschreiben, making sure the text is NOT generic or repetitive. "
                "Adapt the content for the provided job description and the candidate's background from the attached resume image(s). "
                "Your cover letter should be tailored, original, and written in a confident and positive tone, connecting the applicant's skills and motivation with the job requirements and company profile. "
                "Do not simply repeat the resume; instead, use concrete examples from the candidate's experience and relate them to the position. "
                "Strictly avoid generic openings like 'Hiermit bewerbe ich mich...' or 'Mit großem Interesse...'. Start with a strong, individual introduction. "
                "Make sure the cover letter answers: Who am I? What can I do? Why do I want this job at this company? Why is the company better off hiring me? "
                "Keep sentences concise and easy to read, use synonyms to avoid repetition, and maintain professional yet warm language. "
                "Finish with a short closing, including possible starting date and a polite request for an interview. "
                "Output: Only the cover letter text in German, ready for direct use."},
                 {"type": "text", "text": job_description}, 
                    *image_messages
            ],
            max_tokens=8000
        )

        feedback = response.choices[0].message.content
        return {"success": True, "feedback": feedback}

    except Exception as e:
        print(f"Error: {e}")
        return JSONResponse(content={"error": str(e)}, status_code=500)


