import os
from google import genai
from google.genai import types
from config.settings import GEMINI_API_KEY, PROMPTS_DIR, DEFAULT_MODEL

class BaseAgent:
    def __init__(self, agent_name: str, prompt_file: str, temperature: float = 0.0):
        self.agent_name = agent_name
        self.client = genai.Client(api_key=GEMINI_API_KEY)
        self.model_id = DEFAULT_MODEL
        self.temperature = temperature
        self.system_instruction = self._load_prompt(prompt_file)

    def _load_prompt(self, filename: str) -> str:
        """Carga el archivo XML de instrucciones del sistema."""
        filepath = os.path.join(PROMPTS_DIR, filename)
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                return f.read()
        except FileNotFoundError:
            print(f"[Warning] Prompt file no encontrado: {filepath}")
            return "Eres un asistente financiero de IA."

    def generate_response(self, prompt: str, tools: list = None, response_schema=None) -> str:
        """
        Envía un prompt al modelo y retorna la respuesta.
        Permite inyectar herramientas (Tool Calling) y forzar esquemas JSON.
        """
        config_dict = {
            "system_instruction": self.system_instruction,
            "temperature": self.temperature,
        }
        
        if tools:
            config_dict["tools"] = tools
            
        if response_schema:
            config_dict["response_mime_type"] = "application/json"
            config_dict["response_schema"] = response_schema

        config = types.GenerateContentConfig(**config_dict)
        
        print(f"[{self.agent_name}] Procesando solicitud...")
        response = self.client.models.generate_content(
            model=self.model_id,
            contents=prompt,
            config=config
        )
        return response.text