from typing import List, Union
from dataclasses import dataclass
import time
from uuid import uuid1


@dataclass
class Thought:
    agent_name: str
    knowledge: str          # (traced) student knowledge
    candidates: List[str]   # candidate responses
    scores: List[float]     # scores for the candidate responses
    turn: int
    timestamp: int = time.time_ns()
    visible_to: Union[str, List[str]] = 'all'
    msg_type: str = "text"
    logged: bool = False  # Whether the thought is logged in the database


class ThoughtPool():
    """
    A thought pool to manage the thoughts. This allows a unified treatment of the visibility of the thoughts.
    Draft design:
    The thought pool is a list of (named) tuples, where each tuple has (turn, role, content).

    There should be two potential configurations for step definition: multiple players can act in the same turn (rock-paper-scissors).
    The agents can only see the thoughts that
    1) before the current turn, and
    2) visible to the current role
    """

    def __init__(self):
        self.conversation_id = str(uuid1())
        self._thoughts: List[Thought] = []  # TODO: for the sake of thread safety, use a queue instead
        self._last_thought_idx = 0

    def reset(self):
        self._thoughts = []

    def append_thought(self, thought: Thought):
        self._thoughts.append(thought)

    @property
    def last_turn(self):
        if len(self._thoughts) == 0:
            return 0
        else:
            return self._thoughts[-1].turn

    @property
    def last_thought(self):
        if len(self._thoughts) == 0:
            return None
        else:
            return self._thoughts[-1]

    def get_all_thoughts(self) -> List[Thought]:
        return self._thoughts

    def get_thoughts(self, agent_name, turn: int) -> List[Thought]:
        """
        get the thoughts that are yielded by the agent before the current turn
        """

        # Get the thoughts before the current turn
        prev_thoughts = [thought for thought in self._thoughts if thought.turn < turn]

        visible_thoughts = []
        for thought in prev_thoughts:
            if thought.agent_name == agent_name:
                visible_thoughts.append(thought)
        return visible_thoughts
