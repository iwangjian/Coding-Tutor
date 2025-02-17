from typing import List
import re
from openai import OpenAI

from .base import IntelligenceBackend
from ..message import Message, SYSTEM_NAME

END_OF_MESSAGE = "<EOS>"  # End of message token specified by us not OpenAI
STOP_LLAMA3_INSTRUCT = ("<|end_of_text|>", "<|eot_id|>")


class VLLMChat(IntelligenceBackend):
    """
    Interface to the ChatGPT style model with system, user, assistant roles separation
    """
    stateful = False
    type_name = "vllm-chat"

    def __init__(self, 
                 vllm_api_key: str,
                 vllm_endpoint: str,
                 model_name_or_path: str,
                 temperature: float = 0.75, 
                 top_p: float = 0.95,
                 max_tokens: int = 500,
                 max_latest_messages: int = -1, 
                 **kwargs):

        super().__init__(
            model_name_or_path=model_name_or_path, 
            temperature=temperature, top_p=top_p, max_tokens=max_tokens,
            max_latest_messages=max_latest_messages, **kwargs)
        
        # create the OpenAI-like vllm client
        self.client = OpenAI(
            api_key=vllm_api_key,
            base_url=vllm_endpoint
        )
        self.model = model_name_or_path
        self.temperature = temperature
        self.top_p = top_p
        self.max_tokens = max_tokens
        self.max_latest_messages = max_latest_messages

    def _get_response(self, messages, num_responses=1):
        if "llama-3" in self.model.lower():
            completion = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=self.temperature,
                top_p=self.top_p,
                max_tokens=self.max_tokens,
                n=num_responses,
                stop=STOP_LLAMA3_INSTRUCT
            )
        else:
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
        if global_prompt:  # Prepend the global prompt if it exists
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
                # Since many LLMs do not support system message
                # set it to user message
                messages.append({"role": "user", "content": msg[1]})
            elif i == len(all_messages) - 1:
                assert msg[0] == SYSTEM_NAME  # The last message should be from the system
                messages[-1]["content"] = f"{messages[-1]['content']}\n\n{msg[1]}"
            else:
                if msg[0] == agent_name:
                    messages.append({"role": "assistant", "content": msg[1]})
                else:
                    if messages[-1]["role"] == "user":
                        messages[-1]["content"] = f"{messages[-1]['content']}\n\n[{msg[0]}]: {msg[1]}"
                    else:
                        messages.append({"role": "user", "content": f"[{msg[0]}]: {msg[1]}"})

        response = self._get_response(messages, num_responses, *args, **kwargs)

        if num_responses > 1:
            # Remove the agent name if the response starts with it
            response = [re.sub(rf"^\s*\[.*]:", "", r).strip() for r in response]
            response = [re.sub(rf"^\s*{re.escape(agent_name)}\s*:", "", r).strip() for r in response]
            # Remove the tailing end of message token
            response = [re.sub(rf"{END_OF_MESSAGE}$", "", r).strip() for r in response]
            for idx, r in enumerate(response):
                if END_OF_MESSAGE in r:
                    response[idx] = r.split(END_OF_MESSAGE)[0].strip()
        else:
            # Remove the agent name if the response starts with it
            response = re.sub(rf"^\s*\[.*]:", "", response).strip()
            response = re.sub(rf"^\s*{re.escape(agent_name)}\s*:", "", response).strip()
            # Remove the tailing end of message token
            response = re.sub(rf"{END_OF_MESSAGE}$", "", response).strip()
            if END_OF_MESSAGE in response:
                response = response.split(END_OF_MESSAGE)[0].strip()
        
        return response
