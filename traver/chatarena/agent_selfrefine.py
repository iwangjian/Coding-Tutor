from typing import List, Tuple, Union
from tenacity import RetryError
import logging
import uuid

from .backends import IntelligenceBackend
from .message import Message
from .config import AgentConfig, BackendConfig
from .agent import Player

# A special signal sent by the player to indicate that it is not possible to continue the conversation, and it requests to end the conversation.
# It contains a random UUID string to avoid being exploited by any of the players.
SIGNAL_END_OF_CONVERSATION = f"<<<<<<END_OF_CONVERSATION>>>>>>{uuid.uuid4()}"

FEEDBACK_PROMPT = '''{global_prompt}

Below is the ongoing dialogue context in a tutoring session:
{conversation}

Tutor's Response:
{response}

Please provide a brief feedback on the response according to the goal description.'''

REFINE_PROMPT = '''Tutor's Response:
{response}

Feedback:
{feedback}

Please refine the response based on the feedback, and output the refined response directly.'''


class SelfRefineTutor(Player):
    """
    Customed tutor agent with self-refine.
    """

    def __init__(self, name: str, role_desc: str, 
                 backend: Union[BackendConfig, IntelligenceBackend],
                 feedback_prompt: str = None, 
                 refine_prompt: str = None, 
                 num_iter: int = 1,
                 **kwargs):
        assert num_iter > 0, "num_iter should be greater than 0."

        # Register the fields in the _config
        super().__init__(name=name, role_desc=role_desc, backend=backend, **kwargs)
        self.backend = backend
        if feedback_prompt is None:
            feedback_prompt = FEEDBACK_PROMPT
        self.feedback_prompt = feedback_prompt
        if refine_prompt is None:
            refine_prompt = REFINE_PROMPT
        self.refine_prompt = refine_prompt

        self.num_iter = num_iter

    def to_config(self) -> AgentConfig:
        return AgentConfig(
            name=self.name,
            role_desc=self.role_desc,
            backend=self.backend.to_config()
        )

    def act(self, observation: List[Message]) -> Tuple:
        """
        Call the agents to generate a response (equivalent to taking an action).
        """
        try:
            all_messages = []
            for msg in observation:
                all_messages.append(f"{[msg.agent_name]}: {msg.content}")
            
            # generate initial response
            response = self.backend.query(agent_name=self.name, role_desc=self.role_desc,
                                        history_messages=observation)

            for iter in range(self.num_iter):
                # get feedback
                feedback_query_desc = self.feedback_prompt.format(
                    global_prompt=self.role_desc,
                    conversation='\n'.join(all_messages) if len(all_messages) > 0 else "<empty>", 
                    response=response
                )
                
                feedback_query = [{"role": "user", "content": feedback_query_desc}]
                feedback = self.backend._get_response(feedback_query)
        
                # refine response
                refine_query_desc = self.refine_prompt.format(
                    response=response,
                    feedback=feedback
                )
                refine_query = [{"role": "user", "content": refine_query_desc}]
                response = self.backend._get_response(refine_query)

        except RetryError as e:
            logging.warning(f"Agent {self.name} failed to generate a response. "
                            f"Error: {e.last_attempt.exception()}. "
                            f"Sending signal to end the conversation.")
            response = SIGNAL_END_OF_CONVERSATION

        return response

    def __call__(self, observation: List[Message]) -> Tuple:
        return self.act(observation)

    def reset(self):
        self.backend.reset()
    