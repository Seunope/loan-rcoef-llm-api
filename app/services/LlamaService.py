import os
import modal
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from app.models import LoanRequest, LoanResponse

load_dotenv()

app = FastAPI(title="Loan Repayment Prediction API")

# Set your fine-tuned model name
FINE_TUNED_MODEL = os.getenv("FINE_TUNED_MODEL")



async def predict(dto: LoanRequest):

    try:
        userData = f"""User Details:
            - Marital Status: {dto.maritalStatus}
            - Gender: {dto.gender}
            - Location: {dto.state}, Nigeria
            - Age: {dto.age}
            - Loan Amount: {dto.loanAmount}
            - Loan Tenure: {dto.tenorInDays} days. """
        
        RepaymentPredictor = modal.Cls.lookup("lrcoef-service-v2", "RepaymentPredictor")
        predictor = RepaymentPredictor()
        reply = predictor.predict.remote(userData)
        print('DDDDD',reply)

        # Process the result to extract just the number
        try:
            # if prediction_text.lower().startswith("Correct answer is"):
            #     prediction_text = prediction_text[len("Correct answer is"):].strip()
            
            coefficient = int(reply)
            print('coefficient', coefficient)
            
            # Ensure the coefficient is within the expected range
            coefficient = max(0, min(100, coefficient))
            riskLevel = "high" if coefficient <= 40 else "medium" if coefficient <= 70 else "acceptable"

            return LoanResponse (
                repaymentProbabilityScore=f"{str(coefficient)}", 
                meta="Llama Finetune model predicts correctly 40.0% of the time",
                message=f"This user has a {str(coefficient)}% chance of repaying ₦{dto.loanAmount} loan",
                riskLevel=riskLevel,
                recommendation= "Available for only OpenAI models",
            )

        except ValueError:
            raise HTTPException(status_code=500, detail="Failed to parse model output correctly")
            
    except Exception as e:
        # print(e)
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

