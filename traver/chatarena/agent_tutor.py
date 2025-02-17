from typing import List, Tuple, Union
import re
from tenacity import RetryError
import logging
import uuid
import torch
import copy
from abc import abstractmethod
import asyncio
from .backends import IntelligenceBackend
from .message import Message
from .config import AgentConfig, BackendConfig
from .agent import Player
from typing import Callable
from transformers import PreTrainedModel

# A special signal sent by the player to indicate that it is not possible to continue the conversation, and it requests to end the conversation.
# It contains a random UUID string to avoid being exploited by any of the players.
SIGNAL_END_OF_CONVERSATION = f"<<<<<<END_OF_CONVERSATION>>>>>>{uuid.uuid4()}"


class Tutor(Player):
    """
    Customed tutor agent.
    """

    def __init__(self, role_desc: str, KT_desc: str, 
                 backend: Union[BackendConfig, IntelligenceBackend],
                 global_prompt: str = None, 
                 request_prompt: str = None, 
                 verifier: PreTrainedModel = None,
                 verifier_data_builder: Callable = None,
                 num_responses: int = 1,
                 use_KT: bool = False,
                 **kwargs):
        name = "Tutor"
        # Register the fields in the _config
        super().__init__(name=name, role_desc=role_desc, backend=backend,
                         global_prompt=global_prompt, **kwargs)
        self.KT_desc = KT_desc
        self.backend = backend
        self.verifier = verifier
        self.verifier_data_builder = verifier_data_builder
        self.request_prompt = request_prompt
        self.num_responses = num_responses
        self.use_KT = use_KT
        self.KT_history = []

    def to_config(self) -> AgentConfig:
        return AgentConfig(
            name=self.name,
            role_desc=self.role_desc,
            KT_desc=self.KT_desc,
            backend=self.backend.to_config(),
            request_prompt=self.request_prompt,
            global_prompt=self.global_prompt,
        )

    def act(self, observation: List[Message]) -> Tuple:
        """
        Call the agents to generate a response (equivalent to taking an action).
        """
        try:
            all_messages = []
            for msg in observation:
                all_messages.append(f"{[msg.agent_name]}: {msg.content}")

            if self.use_KT and len(all_messages) > 0:
                # format the KT description based on the conversation history
                KT_desc = self.KT_desc.format(
                    conversation='\n'.join(all_messages),
                    previous_estimation=self.KT_history[-1] if len(self.KT_history) > 0 else "<empty>"
                )

                query_messages = [{"role": "user", "content": KT_desc}]
                student_knowledge = self.backend._get_response(query_messages)
                self.KT_history.append(student_knowledge)
                
                request_prompt = self.request_prompt.format(knowledge_tracing=student_knowledge)

                role_desc_list = self.role_desc.split("\n\n")
                role_desc_list[-1] = request_prompt
                role_desc = "\n\n".join(role_desc_list)

            else:
                role_desc = self.role_desc
                student_knowledge = None

            if self.verifier is not None and self.num_responses > 1:
                response_list = self.backend.query(agent_name=self.name, role_desc=role_desc,
                                                history_messages=observation, global_prompt=self.global_prompt,
                                                num_responses=self.num_responses)
                
                verifier_dataloader = self.verifier_data_builder.build_data(all_messages, response_list)
                
                verifier_scores = []
                with torch.no_grad():
                    for batch in verifier_dataloader:
                        input_ids = batch['input_ids'].to(self.verifier.device)
                        attention_mask = batch['attention_mask'].to(self.verifier.device)
                        outputs = self.verifier(input_ids=input_ids, attention_mask=attention_mask)
                        preds = outputs['score']
                        verifier_scores.extend(preds.tolist())
                # Select the response with the highest score
                response = response_list[verifier_scores.index(max(verifier_scores))]
                
            else:
                response_list = None
                verifier_scores = None
                response = self.backend.query(agent_name=self.name, role_desc=role_desc,
                                            history_messages=observation, global_prompt=self.global_prompt)

        except RetryError as e:
            logging.warning(f"Agent {self.name} failed to generate a response. "
                            f"Error: {e.last_attempt.exception()}. "
                            f"Sending signal to end the conversation.")
            response = SIGNAL_END_OF_CONVERSATION

        return (response, student_knowledge, response_list, verifier_scores)

    def __call__(self, observation: List[Message]) -> Tuple:
        return self.act(observation)

    def reset(self):
        self.backend.reset()
        self.KT_history = []