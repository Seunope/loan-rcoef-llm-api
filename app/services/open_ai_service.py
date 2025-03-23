from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Optional
import openai
import os
from dotenv import load_dotenv

from app.models import LoanRequest, LoanResponse

# Load environment variables
load_dotenv()

# Initialize FastAPI app
app = FastAPI(title="Loan Repayment Prediction API")

# Configure OpenAI client
client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
# client = openai.OpenAI()


# Set your fine-tuned model name
FINE_TUNED_MODEL = os.getenv("FINE_TUNED_MODEL")



async def predict_repayment(dto: LoanRequest):
    """
    Predicts the loan repayment coefficient for an applicant
    using a fine-tuned OpenAI model.
    """
    try:
        # Format the input as per your fine-tuned model's requirements
        prompt = f"""User Details:
            - Marital Status: {dto.maritalStatus}
            - Gender: {dto.gender}
            - Location: {dto.state} Nigeria
            - Age: {dto.age}
            - Loan Amount: {dto.loanAmount}
            - Loan Tenure: {dto.tenorInDays} days.

            Repayment coefficient is """

        # Create the messages format for the OpenAI API
        messages = [
            {
                "role": "system",
                "content": "You predict the loan repayment coefficient (0-100). 100 implies a high likelihood of repaying a loan, and 0 implies a very low likelihood. Reply only with the repayment coefficient value, no explanation."
            },
            {
                "role": "user",
                "content": prompt
            },
            {
                "role": "assistant",
                "content": "Repayment Coefficient is"
            }
        ]

        print('FINE_TUNED_MODEL',FINE_TUNED_MODEL)
        # Call the OpenAI API
        response = client.chat.completions.create(
            model=FINE_TUNED_MODEL,
            messages=messages,
            # temperature=0,
            seed=42,
            max_tokens=7
        )
        #  response = openai.chat.completions.create(
        #         model=self.fine_tuned_model_name, 
        #         messages=self._messages_for_test(item),
        #         seed=42,
        #         max_tokens=7
        #     )

        # Extract the prediction from the response
        print("OPME", response.choices[0].message.content)
        prediction_text = response.choices[0].message.content.strip() or '0'
        print('prediction_text', prediction_text)
        
        # Process the result to extract just the number
        try:
            if prediction_text.lower().startswith("repayment coefficient is"):
                prediction_text = prediction_text[len("repayment coefficient is"):].strip()
            
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

