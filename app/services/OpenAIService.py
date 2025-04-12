import json
import os
from openai import OpenAI
from dotenv import load_dotenv
from fastapi import HTTPException
from app.models import LoanRequest, LoanResponse


class OpenAIService:
    def __init__(self, dto: LoanRequest):

        load_dotenv()
        self.openai = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.FINE_TUNED_MODEL = os.getenv("FINE_TUNED_MODEL")
        self.dto = dto 

        self.prompt = f"""User Details:
            - Marital Status: {self.dto.maritalStatus}
            - Gender: {self.dto.gender}
            - Location: {self.dto.state}, Nigeria
            - Age: {self.dto.age}
            - Loan Amount: {self.dto.loanAmount}
            - Loan Tenure: {self.dto.tenorInDays} days.

            """

    async def predict(self):
        """
        Predicts the loan repayment coefficient for an applicant
        using a fine-tuned OpenAI model.
        """
        try:          

            system_message = """
                **Role**: Senior Loan Risk Analyst at a Tier-1 African Bank

                **Task**: Predict the repayment probability score (0-99 scale). Provide results in JSON format.

                **Key Rules**:
                - ABSOLUTE RULE: Never predict 100 (maximum allowed is 99)
                - Typical range: 20-95
                - Interpretation: Lower repayment probability scores indicate higher default risk
                - High risk (0-40),  medium risk (41-70) acceptable risk (71-99)

                **Output Format**:
                {
                    "repaymentProbabilityScore": [integer 0-99],
                    "riskLevel": [string: "high", "medium", or "acceptable"]
                }

                """

            messages = [
                {
                    "role": "system",
                    "content": system_message,
                },
                {
                    "role": "user",
                    "content": self.prompt
                },
                {
                    "role": "assistant",
                    "content": "Output is"
                }
            ]

            response = self.openai.chat.completions.create(
                model=self.FINE_TUNED_MODEL,
                messages=messages,
                # temperature=0,
                seed=42,
                max_tokens=20
            )

        
            prediction_output = json.loads(response.choices[0].message.content)
            # print('OutputX:', prediction_output)
            guess = int(prediction_output["repaymentProbabilityScore"])
            riskLevel = prediction_output["riskLevel"]

            recommendResult = await self._get_recommendation(guess)
            getRecommendation = json.loads(recommendResult)

            try:
            
                return LoanResponse(
                    repaymentProbabilityScore=f"{str(guess)}", 
                    meta="Tested with 6,339 dataset. OpenAi Finetune model predicts correctly 40.0% of the time",
                    message=f"This user has a {str(guess)}% chance of repaying ₦{self.dto.loanAmount} loan",
                    recommendation=getRecommendation["recommendation"],
                    riskLevel=riskLevel
                )

            except ValueError:
                raise HTTPException(status_code=500, detail="Failed to parse model output correctly")
                
        except Exception as e:
            # print(e)
            raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


    async def _get_recommendation(self, score):
        system_message = """
            **Role**: Senior Loan Risk Analyst at a Tier-1 African Bank

            **Style**:
            - Professional banker's tone - confident, concise, factual

            **Task**: Base on user loan data and repayment probability score (0-99 percent), give detail recommendation with explanation for a loan officer and provide results in JSON format.

            **Key Rules**:
            - Interpretation: Lower repayment probability scores indicate higher default risk
            - Note that repayment probability scores are based on previous users loan data and demography. Let this reflect in your recommendation.

            **Recommendation Guidelines**:
            - If high risk (0-40): Suggest specific loan amount (₦) and tenure 
            - If medium risk (41-70): Provide cautionary approval with conditions 
            - If acceptable risk (71-99): Provide positive reinforcement
        

            **Output Format**:
            {
                "repaymentProbabilityScore": [integer 0-99],
                "recommendation": [string based on risk level],
                "riskLevel": [string: "high", "medium", or "acceptable"]
            }
            
            """
        prompt = self.prompt + f"\n Repayment probability scores: {score}"

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
                    "content": "Output is"
                }
            ]

    
        response = self.openai.chat.completions.create(
            model="gpt-4o-mini", 
            messages=messages,
            seed=42,
            # max_tokens=5
        )
        reply = response.choices[0].message.content
        # print('Output:', reply)
        return reply
    




