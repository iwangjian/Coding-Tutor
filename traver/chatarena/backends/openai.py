from typing import List
import re
from tenacity import retry, stop_after_attempt, wait_random_exponential
from openai import OpenAI

from .base import IntelligenceBackend
from ..message import Message, SYSTEM_NAME

END_OF_MESSAGE = "<EOS>"  # End of message token specified by us not OpenAI

class OpenAIChat(IntelligenceBackend):
    """
    Interface to the ChatGPT style model with system, user, assistant roles separation
    """
    stateful = False
    type_name = "openai-chat"

    def __init__(self, 
                 api_key: str,
                 base_url: str,
                 model: str,
                 temperature: float = 0.75, 
                 top_p: float = 0.95,
                 max_tokens: int = 500,
                 max_latest_messages: int = -1,
                 **kwargs):
        
        super().__init__(
            model=model, temperature=temperature, top_p=top_p, max_tokens=max_tokens,
            max_latest_messages=max_latest_messages, **kwargs)
        
        self.client = OpenAI(api_key=api_key, base_url=base_url)
    
        self.model = model
        self.temperature = temperature
        self.top_p = top_p
        self.max_tokens = max_tokens
        self.max_latest_messages = max_latest_messages

    @retry(stop=stop_after_attempt(5), wait=wait_random_exponential(min=1, max=60))
    def _get_response(self, messages, num_responses=1):
        completion = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=self.temperature,
            top_p=self.top_p,
            max_tokens=self.max_tokens,
            n=num_responses
        )
        if num_responses > 1:
            response = [c.message.content for c in completion.choices]
            response = [r.strip() for r in response]
        else:

            response = completion.choices[0].message.content
            response = response.strip()
        
        return response

    def query(self, agent_name: str, role_desc: str, history_messages: List[Message], global_prompt: str = None,
              request_msg: Message = None, num_responses: int = 1, *args, **kwargs) -> str:
        """
        format the input and call the ChatGPT/GPT-4 API
        args:
            agent_name: the name of the agent
            role_desc: the description of the role of the agent
            history_messages: the history of the conversation, or the observation for the agent
            global_prompt: the global prompt for the environment
            request_msg: the request from the system to guide the agent's next response
        """

        # Merge the role description and the global prompt as the system prompt for the agent
        if global_prompt is not None:  # Prepend the global prompt if it exists
            system_prompt = f"{global_prompt.strip()}\n\nYour name: {agent_name}\n\nYour role: {role_desc}"
        else:
            system_prompt = f"{role_desc}"
        
        if self.max_latest_messages > 0 and len(history_messages) > self.max_latest_messages:
            history_messages = history_messages[-self.max_latest_messages:]

        all_messages = [(SYSTEM_NAME, system_prompt)]
        for msg in history_messages:
            if msg.agent_name == SYSTEM_NAME:
                all_messages.append((SYSTEM_NAME, msg.content))
            else:  # non-system messages are suffixed with the end of message token
                all_messages.append((msg.agent_name, f"{msg.content}{END_OF_MESSAGE}"))

        if request_msg is not None:
            all_messages.append((SYSTEM_NAME, request_msg.content))
        else:  # The default request message that reminds the agent its role and instruct it to speak
            all_messages.append((SYSTEM_NAME, f"Now you speak, {agent_name}.{END_OF_MESSAGE}"))

        messages = []
        for i, msg in enumerate(all_messages):
            if i == 0:
                assert msg[0] == SYSTEM_NAME  # The first message should be from the system
                messages.append({"role": "system", "content": msg[1]})
            elif i == len(all_messages) - 1:
                assert msg[0] == SYSTEM_NAME  # The last message should be from the system
                messages[-1]["content"] = f"{messages[-1]['content']}\n\n[{msg[0]}]: {msg[1]}"
            else:
                if msg[0] == agent_name:
                    messages.append({"role": "assistant", "content": msg[1]})
                else:
                    messages.append({"role": "user", "content": f"[{msg[0]}]: {msg[1]}"})

        response = self._get_response(messages, num_responses, *args, **kwargs)

        if num_responses > 1:
            # Remove the agent name if the response starts with it
            response = [re.sub(rf"^\s*\[.*]:", "", r).strip() for r in response]
            response = [re.sub(rf"^\s*{re.escape(agent_name)}\s*:", "", r).strip() for r in response]
            # Remove the tailing end of message token
            response = [re.sub(rf"{END_OF_MESSAGE}$", "", r).strip() for r in response]
        else:
            # Remove the agent name if the response starts with it
            response = re.sub(rf"^\s*\[.*]:", "", response).strip()
            response = re.sub(rf"^\s*{re.escape(agent_name)}\s*:", "", response).strip()
            # Remove the tailing end of message token
            response = re.sub(rf"{END_OF_MESSAGE}$", "", response).strip()

        return response