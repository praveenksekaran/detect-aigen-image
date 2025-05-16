import base64
import os
import requests
from dotenv import load_dotenv
import json

load_dotenv()
header_auth = f"Bearer {os.getenv('NVIDIA_API_KEY')}"
invoke_url = os.getenv('NVIDIA_API_URL')

#import logging
        
#logging.basicConfig(level=logging.INFO)
#logger = logging.getLogger(__name__)
#logger.info(f"openrouter_results_returned: {'YES returned data' if openrouter_results else 'NO return data'}") 

def is_ai_generated_image(confidence_score):
    #Analyze if AI generated 
    if 0 < confidence_score < 0.5 :
        is_ai_generated = False
    elif confidence_score >= 0.5:
        is_ai_generated = True
    else:
        is_ai_generated = None

    return is_ai_generated

def detect_ai_generated_image(img_byte_arr, input_image_path):
    base64_image = base64.b64encode(img_byte_arr).decode('utf-8')

    try:
        confidence_score = -1
        analysis_details = None
        hive_analysis_status = None        
        
        # Initialize NVIDIA analyzer and get results        
        nvidia_results = hive_analyze_image(base64_image, input_image_path)
                
        if not nvidia_results or 'data' not in nvidia_results or not nvidia_results['data']:
            raise Exception("Invalid response from API")
        else:
            # Parse NVIDIA analysis results
            result_data = nvidia_results['data'][0]            
            hive_analysis_status = result_data.get('status', 'UNKNOWN')                 

            if hive_analysis_status == "SUCCESS":
                confidence_score = result_data.get('is_ai_generated', -1)           
                          
        result = {
                "is_ai_generated": is_ai_generated_image(confidence_score),
                "confidence_score": confidence_score              
        }          
       
        return result
        
    except Exception as e:
        error_response = {
            "error": str(e),
            "status": "failed"
        }
        return error_response

def get_detailed_analysis(img_byte_arr, input_image_path, is_ai_generated):
    try:
        base64_image = base64.b64encode(img_byte_arr).decode('utf-8')

        openrouter_results = openrouter_analyze_image(base64_image, input_image_path, is_ai_generated)                          
        
        if not openrouter_results or 'choices' not in openrouter_results or not openrouter_results['choices']:
            analysis_details = f"There seems to be a issue with detailed analysis. Try again later."     
        else:
            # Parse OpenRouter results
            analysis_details = openrouter_results['choices'][0]['message']['content']
    except Exception as e:
        error_response = {
            "error": str(e),
            "status": "failed"
        }  
    return analysis_details

# Nvidia function to create a upload URL if image size is greater 
def upload_asset(path, desc):        
    try:
        assets_url = "https://api.nvcf.nvidia.com/v2/nvcf/assets"
        headers = {
            "Content-Type": "application/json",
            "Authorization": header_auth,
            "accept": "application/json",
        }
        payload = {
            "contentType": "image/png",
            "description": desc
        }
        response = requests.post(assets_url, headers=headers, json=payload, timeout=30)
        
        if response.status_code != 200:
            raise Exception(f"Failed to create asset: {response.status_code}")

        response_data = response.json()
        current_pre_signed_url = response_data.get("uploadUrl")
        asset_id = response_data.get("assetId")

        if not current_pre_signed_url or not asset_id:
            raise Exception("Invalid response from asset creation")

        headers = {
            "Content-Type": "image/png",
            "x-amz-meta-nvcf-asset-description": desc,
        }

        with open(path, "rb") as input_data:
            upload_response = requests.put(
                current_pre_signed_url,
                data=input_data,
                headers=headers,
                timeout=300,
            )
            
            if upload_response.status_code != 200:
                raise Exception(f"Failed to upload asset: {upload_response.status_code}")

        return asset_id
    except Exception as e:
        raise

# Function to call Nvidia API for results 
def hive_analyze_image(image_b64, input_image_path):        
    try:
        if len(image_b64) < 180_000:
            payload = {
                "input": [f"data:image/png;base64,{image_b64}"]
            }
            headers = {
                "Content-Type": "application/json",
                "Authorization": header_auth,
                "Accept": "application/json",
            }
        else:
            asset_id = upload_asset(input_image_path, "Input Image")
            
            payload = {
                "input": [f"data:image/png;asset_id,{asset_id}"]
            }
            headers = {
                "Content-Type": "application/json",
                "NVCF-INPUT-ASSET-REFERENCES": asset_id,
                "Authorization": header_auth,
            }
        
        response = requests.post(invoke_url, headers=headers, json=payload)
        
        if response.status_code != 200:
            raise Exception(f"API request failed with status code: {response.status_code}")
            
        response_data = response.json()
        
        if not response_data or 'data' not in response_data:
            raise Exception("Invalid response format from NVIDIA API")
            
        return response_data
    
    except Exception as e:
        error_response = {
            "error": str(e),
            "status": "failed"
        }
        return error_response

def grok_analyze_image(image_b64, input_image_path, score):
    try:
        # Prepare headers for Grok API request
        headers = {
            "Authorization": f"Bearer {os.getenv('GROQ_API_KEY')}",
            "Content-Type": "application/json"
        }

        prompt = f"""You are an expert on image analysis. you are given a confidence score between 0 and 1, 
                    which indicates if the image was Human-Created or AI-Generated, with 0 being Human-Created  and 1 being AI-Generated.
                    You have to provide an explantion which justifies the confidence score, Highlight specific visual, statistical, or 
                    artifact-based evidence that contributed to the score. 
                    
                    confidence score for the given image is {score}
                    """

        # Prepare payload with base64 encoded image
        payload = {
            "messages": [
                {
                    "role": "user", 
                    "content": [
                        {
                            "type": "text",
                            "text": prompt
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{image_b64}"
                            }
                        }
                    ]
                }
            ],
            "model": "meta-llama/llama-4-scout-17b-16e-instruct"
        }

        # Make request to Grok API
        response = requests.post(
            "https://api.groq.com/v1/chat/completions",
            headers=headers,
            json=payload,
            timeout=30
        )

        if response.status_code != 200:
            raise Exception(f"API request failed with status code: {response.status_code}")

        response_data = response.json()

        if not response_data:
            raise Exception("Invalid response format from Grok API")

        print(response_data)
        return response_data

    except Exception as e:
        error_response = {
            "error": str(e),
            "status": "failed"
        }
        return error_response

def openrouter_analyze_image(base64_image, input_image_path, is_ai_generated):
    try: 
        # Construct the prompt
        prompt_ai = f"""You are an expert on image analysis. you are given a AI generated image, You have to provide an explanation and evidences to prove 
                    it is AI generated. Highlight specific visual, statistical, or artifact-based."""

        prompt_hu = f"""You are an expert on image analysis. you are given a human generated image i.e. not generated by AI.
                    You have to provide an explanation and evidences to prove it is human generated generated. 
                    Highlight specific visual, statistical, or artifact-based."""

        # Make request to OpenRouter API
        response = requests.post(
            url="https://openrouter.ai/api/v1/chat/completions",
            headers={
            "Authorization": f"Bearer {os.getenv('OPENROUTER_API_KEY')}",
            "Content-Type": "application/json"           
            },
            data=json.dumps({
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text", 
                                "text": prompt_ai if is_ai_generated else prompt_hu
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{base64_image}"
                                }
                            }
                        ]
                    }
                ],
                "model": "bytedance-research/ui-tars-72b:free"            
            })           
        )

        if response.status_code != 200:
            raise Exception(f"OpenRouter API request failed with status code: {response.status_code}")

        response_data = response.json()        

        if not response_data:
            raise Exception("Invalid response format from OpenRouter API")

        return response_data

    except Exception as e:
        error_response = {
            "error": str(e),
            "status": "failed"
        }
        return error_response

