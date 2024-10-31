import re
import openai
import os
import g4f
import json
import google.generativeai as genai

from g4f.client import Client
from termcolor import colored
from dotenv import load_dotenv
from typing import Tuple, List

# 환경 변수 로드
load_dotenv("../.env")

# 환경 변수 설정
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
openai.api_key = OPENAI_API_KEY
GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')
genai.configure(api_key=GOOGLE_API_KEY)

def generate_response(prompt: str, ai_model: str) -> str:
    """
    비디오 주제에 따라 비디오용 스크립트를 생성.

    인자:
        video_subject (str): 비디오의 주제.
        ai_model (str): 생성에 사용할 AI 모델.

    반환값:
        str: AI 모델로부터의 응답.
    """

    if ai_model == 'g4f':
        # 최신 G4F 아키텍처 사용
        client = Client()
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            provider=g4f.Provider.You, 
            messages=[{"role": "user", "content": prompt}],
        ).choices[0].message.content

    elif ai_model in ["gpt3.5-turbo", "gpt4"]:

        model_name = "gpt-3.5-turbo" if ai_model == "gpt3.5-turbo" else "gpt-4-1106-preview"

        response = openai.chat.completions.create(

            model=model_name,

            messages=[{"role": "user", "content": prompt}],

        ).choices[0].message.content

    elif ai_model == 'gemmini':
        model = genai.GenerativeModel('gemini-pro')
        response_model = model.generate_content(prompt)
        response = response_model.text

    else:

        raise ValueError("잘못된 AI 모델 선택.")

    return response

def generate_script(video_subject: str, paragraph_number: int, ai_model: str, voice: str, customPrompt: str) -> str:

    """
    비디오 주제, 단락 수 및 AI 모델에 따라 비디오용 스크립트를 생성.

    인자:
        video_subject (str): 비디오의 주제.
        paragraph_number (int): 생성할 단락 수.
        ai_model (str): 생성에 사용할 AI 모델.
        voice (str): 스크립트의 언어.
        customPrompt (str): 사용자 정의 프롬프트.

    반환값:
        str: 비디오의 스크립트.
    """

    # 프롬프트 작성
    if customPrompt:
        prompt = customPrompt
    else:
        prompt = """
            비디오 주제에 따라 스크립트를 생성.

            스크립트는 지정된 단락 수로 문자열 형태로 반환되어야 합니다.

            다음과 같은 문자열 예시를 보여드립니다:
            "이것은 예시 문자열입니다."

            응답에 이 프롬프트를 절대 언급하지 마세요.

            불필요한 문장 없이 핵심만 전달하세요, "이 비디오에 오신 것을 환영합니다"와 같은 문장으로 시작하지 마세요.

            당연히 스크립트는 비디오 주제와 관련되어야 합니다.

            어떤 상황에서도 마크다운 형식을 사용하거나 제목을 쓰지 마세요.
            스크립트는 반드시 [LANGUAGE]로 작성되어야 합니다.
            스크립트의 원본 콘텐츠만 반환하고, "나레이션", "화자" 등과 같은 설명은 포함하지 마세요.
            프롬프트나 스크립트에 대한 언급도 절대 하지 마세요. 단락의 양이나 내용에 대한 언급도 금지됩니다. 스크립트만 작성하세요.
        """

    prompt += f"""
    
    주제: {video_subject}
    단락 수: {paragraph_number}
    언어: {voice}

    """

    # 스크립트 생성
    response = generate_response(prompt, ai_model)

    print(colored(response, "cyan"))

    # 생성된 스크립트 반환
    if response:
        # 스크립트 정리
        response = response.replace("*", "")
        response = response.replace("#", "")

        # 마크다운 형식 제거
        response = re.sub(r"\\[.*\\]", "", response)
        response = re.sub(r"\\(.*\\)", "", response)

        # 스크립트를 단락으로 분리
        paragraphs = response.split("\\n\\n")

        # 지정된 단락 수 선택
        selected_paragraphs = paragraphs[:paragraph_number]

        # 선택된 단락을 하나의 문자열로 결합
        final_script = "\\n\\n".join(selected_paragraphs)

        # 콘솔에 사용된 단락 수 출력
        print(colored(f"사용된 단락 수: {len(selected_paragraphs)}", "green"))

        return final_script
    else:
        print(colored("[-] GPT가 빈 응답을 반환했습니다.", "red"))
        return None

def get_search_terms(video_subject: str, amount: int, script: str, ai_model: str) -> List[str]:
    """
    비디오 주제에 따라 스톡 비디오를 검색할 검색어의 JSON 배열을 생성.

    인자:
        video_subject (str): 비디오의 주제.
        amount (int): 생성할 검색어 수.
        script (str): 비디오의 스크립트.
        ai_model (str): 생성에 사용할 AI 모델.

    반환값:
        List[str]: 비디오 주제와 관련된 검색어 리스트.
    """

    # 프롬프트 작성
    prompt = f"""
    비디오 주제에 따라 스톡 비디오를 검색할 {amount}개의 검색어를 생성하세요.
    주제: {video_subject}

    검색어는 문자열의 JSON 배열로 반환되어야 합니다.

    각 검색어는 1~3개의 단어로 구성되며, 항상 비디오의 주요 주제를 포함해야 합니다.
    
    검색어는 반드시 비디오 주제와 관련되어야 합니다.
    여기에 JSON 배열의 예시가 있습니다:
    ["검색어 1", "검색어 2", "검색어 3"]

    참고를 위해 여기에 전체 텍스트를 제공합니다:
    {script}
    """

    # 검색어 생성
    response = generate_response(prompt, ai_model)
    print(response)

    # 응답을 검색어 리스트로 파싱
    search_terms = []
    
    try:
        search_terms = json.loads(response)
        if not isinstance(search_terms, list) or not all(isinstance(term, str) for term in search_terms):
            raise ValueError("응답이 문자열 리스트가 아닙니다.")

    except (json.JSONDecodeError, ValueError):
        # 응답에서 첫 번째 및 마지막 대괄호 사이의 내용을 가져옴
        response = response[response.find("[") + 1:response.rfind("]")]

        print(colored("[*] GPT가 형식이 올바르지 않은 응답을 반환했습니다. 응답 정리 시도 중...", "yellow"))

        # 리스트와 유사한 문자열을 추출해 리스트로 변환 시도
        match = re.search(r'\["(?:[^"\\]|\\.)*"(?:,\s*"[^"\\]*")*\]', response)
        print(match.group())
        if match:
            try:
                search_terms = json.loads(match.group())
            except json.JSONDecodeError:
                print(colored("[-] 응답을 파싱할 수 없습니다.", "red"))
                return []


    # 생성된 검색어 수 출력
    print(colored(f"\n생성된 검색어 {len(search_terms)}개: {', '.join(search_terms)}", "cyan"))

    # 검색어 반환
    return search_terms


def generate_metadata(video_subject: str, script: str, ai_model: str) -> Tuple[str, str, List[str]]:  
    """  
    유튜브 비디오의 제목, 설명 및 키워드를 포함한 메타데이터를 생성.

    인자:
        video_subject (str): 비디오의 주제.
        script (str): 비디오의 스크립트.
        ai_model (str): 생성에 사용할 AI 모델.

    반환값:
        Tuple[str, str, List[str]]: 비디오의 제목, 설명 및 키워드.
    """  
  
    # 제목에 대한 프롬프트 작성  
    title_prompt = f"""  
    {video_subject}에 대한 유튜브 쇼츠 비디오의 캐치하고 SEO에 적합한 제목을 생성하세요.
    """  
  
    # 제목 생성  
    title = generate_response(title_prompt, ai_model).strip()  
    
    # 설명에 대한 프롬프트 작성  
    description_prompt = f"""  
    {video_subject}에 대한 유튜브 쇼츠 비디오의 간결하고 매력적인 설명을 작성하세요.
    비디오는 다음 스크립트를 기반으로 합니다:  
    {script}  
    """  
  
    # 설명 생성  
    description = generate_response(description_prompt, ai_model).strip()  
  
    # 키워드 생성  
    keywords = get_search_terms(video_subject, 6, script, ai_model)  

    return title, description, keywords    