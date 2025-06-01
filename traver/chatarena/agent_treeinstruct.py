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

STATE_ESTIMATION_PROMPT = '''
{role_desc}

Below is the ongoing dialogue context in a tutoring session:
{conversation}

Please evaluate the current tutoring state:
- Resolved Topics:
- Pending Topics:
- Next Steps Recommendations:
'''

QUESTION_GENERATION_PROMPT = '''Below is the tutoring state assessment:
{state_estimation}

Recent Questions Asked:
{previous_questions}

Please generate the following hierarchical questions:
- A main question (addressing the core concept/error)
- A follow-up question (clarifying understanding)
- A challenge question (testing deeper understanding)
'''


class TreeInstructTutor(Player):
    """
    Customed tutor agent with tree-instruct.
    """

    def __init__(self, name: str, role_desc: str, 
                 backend: Union[BackendConfig, IntelligenceBackend],
                 state_prompt: str = None, 
                 question_prompt: str = None,
                 **kwargs):
        # Register the fields in the _config
        super().__init__(name=name, role_desc=role_desc, backend=backend, **kwargs)
        self.backend = backend

        if state_prompt is None:
            state_prompt = STATE_ESTIMATION_PROMPT
        self.state_prompt = state_prompt
        if question_prompt is None:
            question_prompt = QUESTION_GENERATION_PROMPT
        self.question_prompt = question_prompt

        self.question_history = []  # Track previous questions

    def to_config(self) -> AgentConfig:
        return AgentConfig(
            name=self.name,
            role_desc=self.role_desc,
            backend=self.backend.to_config()
        )

    def act(self, observation: List[Message]) -> Tuple:
        """
        Call the agents to generate a response with adaptive feedback loop.
        """
        try:
            # State estimation
            state_query = self.state_prompt.format(
                role_desc=self.role_desc,
                conversation='\n'.join([f"{m.agent_name}: {m.content}" for m in observation]),
            )
            state_estimation = self.backend._get_response([{"role": "user", "content": state_query}])

            # Generate tree-based questions
            question_query = self.question_prompt.format(
                state_estimation=state_estimation,
                previous_questions='\n'.join(self.question_history[-3:]) if len(self.question_history) >= 3 else '\n'.join(self.question_history)
            )
            questions = self.backend._get_response([{"role": "user", "content": question_query}])

            # Store generated questions
            self.question_history.append(questions)
            
            # Generate response based on state and questions
            request_gen = Message(
                agent_name=self.name, 
                content=f"""State Assessment: {state_estimation}
                Adaptive Questions: {questions}
                Now you speak, {self.name}.""", 
                turn=-1
            )
            response = self.backend.query(
                agent_name=self.name,
                role_desc=self.role_desc,
                history_messages=observation,
                request_msg=request_gen
            )
        
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
        self.question_history = []
