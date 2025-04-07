from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Optional
from openai import OpenAI
import os
from dotenv import load_dotenv

from app.models import LoanRequest, LoanResponse

# Load environment variables
load_dotenv()

# Initialize FastAPI app
app = FastAPI(title="Loan Repayment Prediction API")

# Configure OpenAI client
# openai = OpenAI()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
# print('DDDD', os.getenv("OPENAI_API_KEY"))
# client = openai.OpenAI()


# Set your fine-tuned model name
FINE_TUNED_MODEL = os.getenv("FINE_TUNED_MODEL")



async def predict(dto: LoanRequest):
    """
    Predicts the loan repayment coefficient for an applicant
    using a fine-tuned OpenAI model.
    """
    try:
        # Format the input as per your fine-tuned model's requirements
        prompt = f"""User Details:
            - Marital Status: {dto.maritalStatus}
            - Gender: {dto.gender}
            - Location: {dto.state}, Nigeria
            - Age: {dto.age}
            - Loan Amount: {dto.loanAmount}
            - Loan Tenure: {dto.tenorInDays} days.

            Repayment coefficient is """
        
        system_message = "Task: Predict loan repayment coefficient (0-100)\n"
        system_message += "Guidelines:\n"
        system_message += "- Never predict exactly 100 (use 95-99 for perfect cases)\n"
        system_message += "- Typical range: 20-95\n"
        system_message += "- Lower values indicate higher risk"

        # Create the messages format for the OpenAI API
        messages = [
            {
                "role": "system",
                "content": system_message,
            },
            {
                "role": "user",
                "content": prompt
            },
            {
                "role": "assistant",
                "content": "Correct answer is"
            }
        ]

        print('FINE_TUNED_MODEL',FINE_TUNED_MODEL)
        # print('OPENAI_API_KEY',OPENAI_API_KEY)
        # print('OPENAI_API_KEY',client.api_key)
        print(messages)

        # Call the OpenAI API
        response = client.chat.completions.create(
            model=FINE_TUNED_MODEL,
            messages=messages,
            # temperature=0,
            seed=42,
            max_tokens=7
        )



        prediction_text = response.choices[0].message.content.strip() 
        print('prediction_text', prediction_text)
        
        # Process the result to extract just the number
        try:
            if prediction_text.lower().startswith("Correct answer is"):
                prediction_text = prediction_text[len("Correct answer is"):].strip()
            
            coefficient = int(prediction_text)
            
            # Ensure the coefficient is within the expected range
            coefficient = max(0, min(100, coefficient))

            return LoanResponse(
                repaymentCoefficient=f"{str(coefficient)}", 
                meta="Tested with 6,339 dataset. OpenAi Finetune model predicts correctly 40.0% of the time",
                message=f"This user has a {str(coefficient)}% chance of repaying ₦{dto.loanAmount} loan"
            )

        except ValueError:
            raise HTTPException(status_code=500, detail="Failed to parse model output correctly")
            
    except Exception as e:
        # print(e)
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

