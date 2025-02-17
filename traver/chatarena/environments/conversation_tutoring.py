from typing import List, Union

from .base import TimeStep
from .conversation import Conversation
from ..message import Message, MessagePool
from ..thought import Thought, ThoughtPool
from ..agent import Moderator, SIGNAL_END_OF_CONVERSATION
from ..config import EnvironmentConfig, AgentConfig


class TutoringConversation(Conversation):
    """
    Tutoring conversation environment.
    Moderator is a special agent that can see all messages and can decide whether the conversation is over.
    """

    type_name = "tutoring_conversation"

    def __init__(self, player_names: List[str], moderator: Union[Moderator, AgentConfig],
                 parallel: bool = False, moderator_visibility="all", moderator_period="turn", **kwargs):

        super().__init__(player_names=player_names, parallel=parallel, **kwargs)

        if isinstance(moderator, AgentConfig):
            moderator_config = moderator
            moderator = Moderator.from_config(moderator_config)
        elif not isinstance(moderator, Moderator):
            raise ValueError("moderator must be either an AgentConfig or a Moderator instance.")

        # The thought pool is used to store the internal thoughts of the tutor agent
        self.thought_pool = ThoughtPool()

        self.moderator = moderator
        self.moderator_visibility = moderator_visibility
        self.moderator_period = moderator_period
    
    def reset(self):
        self._current_turn = 0
        self._next_player_idx = 0
        self.message_pool.reset()
        self.thought_pool.reset()

        init_timestep = TimeStep(observation=[],
                                 reward=self.get_zero_rewards(),
                                 terminal=False)
        return init_timestep

    def to_config(self) -> EnvironmentConfig:
        # This environment contains some speical config arguments that needs to be handle specially
        return EnvironmentConfig(env_type=self.type_name, player_names=self.player_names, parallel=self.parallel,
                                 moderator=self.moderator.to_config(), moderator_visibility=self.moderator_visibility,
                                 moderator_period=self.moderator_period)

    def get_thought(self, player_name=None) -> List[Thought]:
        """
        get thoughts for the player
        """
        if player_name is None:
            return self.thought_pool.get_all_thoughts()
        else:
            return self.thought_pool.get_thoughts(player_name, turn=self._current_turn)
    
    def step(self, player_name: str, action: str, 
             student_knowledge: str=None, 
             response_list: List[str]=None, 
             verifier_scores: List[float]=None
             ) -> TimeStep:
        """
        step function that is called by the arena
        Args:
            player_name: the name of the player that takes the action
            action: the action that the agents wants to take
        """
        message = Message(agent_name=player_name, content=action, turn=self._current_turn)
        self.message_pool.append_message(message)

        thought = Thought(agent_name=player_name, knowledge=student_knowledge,
                            candidates=response_list, scores=verifier_scores,
                            turn=self._current_turn)
        self.thought_pool.append_thought(thought)

        # Round-robin order for the next player
        self._next_player_idx = (self._next_player_idx + 1) % self.num_players

        if self.moderator_period == "turn" or \
                (self.moderator_period == "round" and self._next_player_idx == 0):
            # Moderator's turn
            moderator_history = self.message_pool.get_all_messages()

            # Moderator's response is not used
            #moderator_response = self.moderator(moderator_history)
            #moderator_message = Message(agent_name=self.moderator.name,
            #                            content=moderator_response,
            #                            turn=self._current_turn,
            #                            visible_to=self.moderator_visibility)
            #self.message_pool.append_message(moderator_message)

            # We only use Moderator to determine whether the conversation should be ended
            terminal = self.moderator.is_terminal(moderator_history) or self.is_terminal()
        else:
            terminal = self.is_terminal()

        # Update the counters
        if not self.parallel or self._next_player_idx == 0:
            self._current_turn += 1

        timestep = TimeStep(observation=self.get_observation(),
                            reward=self.get_zero_rewards(),
                            terminal=terminal)  # Return all the messages
        return timestep
