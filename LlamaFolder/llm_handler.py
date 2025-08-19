from typing import Optional, List
from time import time, gmtime
from langchain.llms.base import BaseLLM
from langchain.schema import LLMResult, Generation
from langchain.callbacks.manager import CallbackManagerForLLMRun
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
import requests
import json

class OllamaLLM(BaseLLM):
    
    # instance of LLM variables 
    model: str
    api_url: str = "http://localhost:11434/api/generate"
    temperature: float = 0.00
    
    def _call(self, prompt: str, stop: Optional[List[str]] = None, run_manager: Optional[CallbackManagerForLLMRun] = None) -> str:
        try:
            
            begin_time = time()
            print(f"LLM call time (hr|min): {gmtime().tm_hour - 5}:{gmtime().tm_min}")

            # post requests to API with passed prompt 
            response = requests.post(self.api_url, json={"model": self.model, "prompt": prompt, "stream": True, "num_predict": 4096, "temperature": self.temperature})
            
            # await response
            response.raise_for_status()
            
            end_time = time()
            duration = end_time - begin_time
            
            print("LLM response latency (s): ", duration)
             
            tokens = []
            for line in response.iter_lines():
                if line:
                    data = json.loads(line)
                    token = data.get("response", "")
                    tokens.append(token)

            return "".join(tokens).strip()


            #return "".join(json.loads(line).get("response", "") for line in response.iter_lines() if line).strip()
        except Exception as e:
            raise RuntimeError(f"Ollama API request failed: {e}")


    # generate response through iterated call
    def _generate(self, prompts: List[str], stop: Optional[List[str]] = None, run_manager: Optional[CallbackManagerForLLMRun] = None) -> LLMResult:
        
        generations = [[Generation(text=self._call(p, stop))] for p in prompts]
        
        # get the response
        return LLMResult(generations=generations)


    @property
    def _llm_type(self) -> str:
        # who am I?
        return "ollama"
