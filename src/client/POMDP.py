from proto.game_proto import GameMessage, game_proto
from proto.game_proto import ACT, OK, CHAL, BLOCK, SHOW, LOSE, COINS, DECK, CHOOSE, KEEP, HELLO, PLAYER, START, READY, TURN, EXIT, ILLEGAL
from .game.state_machine import PlayerState, Tag
from .game.core import *
from .player import InformedPlayer
import random
from loguru import logger

from itertools import combinations_with_replacement
from collections import Counter, defaultdict
import math



class GameState:
    def __init__(self, my_cards, num_players, my_id=1):
        self.CARD_TYPES = [ASSASSIN, AMBASSADOR, CAPTAIN, DUKE, CONTESSA]
        self.preferences = {
            ASSASSIN: 5,
            CAPTAIN: 4,
            DUKE: 3,
            CONTESSA: 2,
            AMBASSADOR: 1
        }
        assert len(my_cards) == 2, "Must have exactly 2 private cards."
        assert 2 <= num_players <= 6, "Coup supports 2–6 players."
        assert 1 <= my_id <= num_players, "my_id must be between 1 and num_players."

        self.num_players = num_players
        self.my_id = my_id
        self.my_cards = list(my_cards)
        self.coins = {pid: 2 for pid in range(1, num_players + 1)}
        self.n_cards = {pid: 2 for pid in range(1, num_players + 1)}

        self.full_deck_counts = Counter({c: 3 for c in self.CARD_TYPES})
        # Build remaining deck counts after removing your private cards
        self.remaining_deck_counts = self.full_deck_counts.copy()
        for c in self.my_cards:
            self.remaining_deck_counts[c] -= 1

        # Evidence
        self.known_deck_cards = Counter()
        self.deleted_cards = Counter()

        # Calculate initial beliefs
        self._calculate_beliefs()

    def add_evidence(self, card):
        self.known_deck_cards[card] += 1
        if self.remaining_deck_counts[card] > 0:
            self.remaining_deck_counts[card] -= 1
        else:
            raise ValueError(f"Cannot add evidence for card {card} that is not present.")
        self._calculate_beliefs()

    def remove_evidence(self):
        for card in self.CARD_TYPES:
            if self.known_deck_cards[card] > 0:
                self.remaining_deck_counts[card] += self.known_deck_cards[card]
                self.known_deck_cards[card] = 0
        self._calculate_beliefs()

    def remove_card_from_player(self, card, player_id):
        if self.remaining_deck_counts[card] > 0:
            self.n_cards[player_id] -= 1
            if player_id != self.my_id:
                self.remaining_deck_counts[card] -= 1
            self.deleted_cards[card] += 1
        else:
            raise ValueError(f"Cannot remove card {card} that is not present.")
        self._calculate_beliefs()

    def probability_opponent_has(self, card, player_id):
        probability = 0
        for deck_combo, prob in self.player_beliefs[self.n_cards[player_id]-1].items():
            if card in deck_combo:
                probability += prob
        return probability
    
    def probability_any_opponent_has(self, card):
        # Calculate the number of cards the adversary players have
        total_opponent_cards = sum(self.n_cards[pid] for pid in range(1, self.num_players + 1) if pid != self.my_id)

        hand_combinations = {}
        for hand_combo in combinations_with_replacement(self.CARD_TYPES, total_opponent_cards):
            hand_counts = Counter(hand_combo)
            if hand_counts[card] > self.remaining_deck_counts[card]:
                continue

            w = 1
            for c in self.CARD_TYPES:
                w *= math.comb(self.remaining_deck_counts[c], hand_counts[c])
            hand_combinations[tuple(sorted((c for c, count in hand_counts.items() for _ in range(count))))] = w

        # Normalize the probabilities
        total = sum(hand_combinations.values())
        for hand_combo in hand_combinations:
            hand_combinations[hand_combo] /= total

        # Calculate the probability that at least one opponent has the card
        probability = 0
        for hand_combo, prob in hand_combinations.items():
            if card in hand_combo:
                probability += prob
        return probability
                
    def _calculate_beliefs(self):
        # Calculate how many cards are supposed to be in the deck
        deck_size = sum(self.remaining_deck_counts.values()) - (sum(self.n_cards.values()) - self.n_cards[self.my_id]) + sum(self.known_deck_cards.values())

        # DECK BELIEF
        deck_combinations = {}
        for deck_combo in combinations_with_replacement(self.CARD_TYPES, deck_size - sum(self.known_deck_cards.values())):
            # Calculate all possible combinations of the remaining deck without the known cards
            deck_counts = Counter(deck_combo)

            if any(deck_counts[c] > self.remaining_deck_counts[c] for c in self.CARD_TYPES):
                continue

            complete_combo = deck_counts.copy()
            # Add known cards to the combo
            for card, count in self.known_deck_cards.items():
                complete_combo[card] += count
            complete_combo = tuple(sorted((card for card, count in complete_combo.items() for _ in range(count))))
            deck_combinations[complete_combo] = 0

            w = 1
            for c in self.CARD_TYPES:
                w *= math.comb(self.remaining_deck_counts[c], deck_counts[c])
            deck_combinations[complete_combo] += w
        
        # Normalize the probabilities
        total = sum(deck_combinations.values())
        for cob in deck_combinations:
            deck_combinations[cob] /= total
        self.deck_beliefs = deck_combinations

        # PLAYER BELIEF DOUBLE CARD
        hand_counts = defaultdict(int)
        total = 0
        for hand_combo in combinations_with_replacement(self.CARD_TYPES, 2):
            c1, c2 = hand_combo
            if c1 == c2:
                if self.remaining_deck_counts[c1] >= 2:
                    w = math.comb(self.remaining_deck_counts[c1], 2)
                else:
                    w = 0
            else:
                if self.remaining_deck_counts[c1] >= 1 and self.remaining_deck_counts[c2] >= 1:
                    w = self.remaining_deck_counts[c1] * self.remaining_deck_counts[c2]
                else:
                    w = 0
            if w > 0:
                hand_counts[tuple(sorted(hand_combo))] += w
                total += w

        player_beliefs_double = {hand: w/total for hand, w in hand_counts.items()}

        # PLAYER BELIEF SINGLE CARD
        total = sum(self.remaining_deck_counts[c] for c in self.CARD_TYPES)
        if total == 0:
            return {c: 0.0 for c in self.CARD_TYPES}

        player_beliefs_single = {c: self.remaining_deck_counts[c] / total for c in self.CARD_TYPES}

        self.player_beliefs = [player_beliefs_single, player_beliefs_double]

    def __str__(self):
        return f"GameState(my_cards={self.my_cards}, my_id={self.my_id}, coins={self.coins}, n_cards={self.n_cards}, remaining_deck={self.remaining_deck_counts}, known_deck_cards={self.known_deck_cards}, deleted_cards={self.deleted_cards})"

    def who_to_coup(self) -> int:
        targets = []
        for pid in range(1, self.num_players + 1):
            if pid != self.my_id and self.n_cards[pid] > 0:
                targets.append((pid, self.n_cards[pid], self.coins[pid]))
        
        # Sort by number of cards (descending) and then coins (descending)
        targets.sort(key=lambda x: (-x[1], -x[2]))
        
        if targets:
            return targets[0][0]  # Return the ID of the player with the most coins and cards
        return None

    def prefers(self, c1, c2):
        if self.preferences[c1] < self.preferences[c2]:
            return c1
        else:
            return c2

class CoupBot(InformedPlayer):
    def __init__(self):
        super().__init__()
        self.game_started = False

    def choose_message(self) -> None:
        if len(self.possible_messages) == 0:
            raise IndexError("No possible messages.")

        if not self.game_started:
            if self.state == PlayerState.R_OTHER_TURN or self.state == PlayerState.R_MY_TURN:
                self.game_started = True
                self.gs = GameState(self.deck, len(self.players) + 1, int(self.id))
            else:
                self.msg = GameMessage(random.choice(self.possible_messages))
            
        if self.game_started:

            who_to_target = self.gs.who_to_coup()
            if self.state == PlayerState.R_MY_TURN:
                if self.gs.coins[int(self.id)] >= 7:
                    self.msg = GameMessage(game_proto.ACT(self.id, COUP, who_to_target))
                    return
                elif ASSASSIN in self.gs.my_cards and self.gs.coins[int(self.id)] >= 3:
                    self.msg = GameMessage(game_proto.ACT(self.id, ASSASSINATE, who_to_target))
                    return
                elif DUKE in self.gs.my_cards:
                    self.msg = GameMessage(game_proto.ACT(self.id, TAX))
                    return
                elif AMBASSADOR in self.gs.my_cards:
                    self.msg = GameMessage(game_proto.ACT(self.id, EXCHANGE)) 
                    return
                elif CAPTAIN in self.gs.my_cards and self.gs.probability_opponent_has(CAPTAIN, who_to_target) < 0.3 and self.gs.probability_opponent_has(AMBASSADOR, who_to_target) < 0.3:
                    self.msg = GameMessage(game_proto.ACT(self.id, STEAL, who_to_target))
                    return
                elif self.gs.probability_any_opponent_has(DUKE) < 0.3:
                    self.msg = GameMessage(game_proto.ACT(self.id, FOREIGN_AID))
                    return
                else:
                    self.msg = GameMessage(game_proto.ACT(self.id, INCOME))
                    return
                
            if self.state == PlayerState.R_BLOCK_FAID:
                if self.gs.probability_opponent_has(DUKE, int(self.history[-2].ID1)) < 0.3:
                    self.msg = GameMessage(game_proto.CHAL(self.id))
                else:
                    self.msg = GameMessage(game_proto.OK())
                return

            if self.state == PlayerState.R_BLOCK_ASSASS:
                if self.gs.probability_opponent_has(CONTESSA, int(self.history[-2].ID1)) < 0.3:
                    self.msg = GameMessage(game_proto.CHAL(self.id))
                else:
                    self.msg = GameMessage(game_proto.OK())
                return

            if self.state == PlayerState.R_BLOCK_STEAL_B:
                if self.gs.probability_opponent_has(AMBASSADOR, int(self.history[-2].ID1)) < 0.3:
                    self.msg = GameMessage(game_proto.CHAL(self.id))
                else:
                    self.msg = GameMessage(game_proto.OK())
                return

            if self.state == PlayerState.R_BLOCK_STEAL_C:
                if self.gs.probability_opponent_has(CAPTAIN, int(self.history[-2].ID1)) < 0.3:
                    self.msg = GameMessage(game_proto.CHAL(self.id))
                else:
                    self.msg = GameMessage(game_proto.OK())
                return

            if self.state == PlayerState.R_OTHER_TURN:
                self.msg = GameMessage(random.choice(self.possible_messages))
                return

            if self.state == PlayerState.R_INCOME:
                self.msg = GameMessage(game_proto.OK())
                return

            if self.state == PlayerState.R_FAID:
                if DUKE in self.gs.my_cards:
                    self.msg = GameMessage(game_proto.BLOCK(self.id, DUKE))
                    return
                else:
                    if self.gs.probability_opponent_has(DUKE, int(self.history[-2].ID1)) < 0.3:
                        self.msg = GameMessage(game_proto.CHAL(self.id))
                        return
                    else:
                        self.msg = GameMessage(game_proto.OK())
                        return
            
            if self.state == PlayerState.R_TAX:
                if self.gs.probability_opponent_has(DUKE, int(self.history[-2].ID1)) < 0.3:
                    self.msg = GameMessage(game_proto.CHAL(self.id))
                    return 
                else:
                    self.msg = GameMessage(game_proto.OK())
                    return

            if self.state == PlayerState.R_EXCHANGE:
                if self.gs.probability_opponent_has(AMBASSADOR, int(self.history[-2].ID1)) < 0.3:
                    self.msg = GameMessage(game_proto.CHAL(self.id))
                    return
                else:
                    self.msg = GameMessage(game_proto.OK())
                    return
                
            if self.state == PlayerState.R_STEAL_ME:
                if AMBASSADOR in self.gs.my_cards:
                    self.msg = GameMessage(game_proto.BLOCK(self.id, AMBASSADOR))
                    return
                elif CAPTAIN in self.gs.my_cards:
                    self.msg = GameMessage(game_proto.BLOCK(self.id, CAPTAIN))
                    return
                elif self.gs.probability_opponent_has(CAPTAIN, int(self.history[-2].ID1)) < 0.3:
                    self.msg = GameMessage(game_proto.CHAL(self.id))
                    return
                else:
                    self.msg = GameMessage(game_proto.OK())
                    return

            if self.state == PlayerState.R_ASSASS_ME:
                if CONTESSA in self.gs.my_cards:
                    self.msg = GameMessage(game_proto.BLOCK(self.id, CONTESSA))
                    return
                elif self.gs.probability_opponent_has(ASSASSIN, int(self.history[-2].ID1)) < 0.3:
                    self.msg = GameMessage(game_proto.CHAL(self.id))
                    return
                else:
                    self.msg = GameMessage(game_proto.OK())
                    return
                
            if self.state == PlayerState.R_COUP_ME:
                self.msg = GameMessage(game_proto.OK())
                return                    
                
            if self.state == PlayerState.R_ASSASS:
                self.msg = GameMessage(game_proto.OK())
                return
            
            if self.state == PlayerState.R_STEAL:
                self.msg = GameMessage(game_proto.OK())
                return

            if self.state == PlayerState.R_COUP:
                self.msg = GameMessage(game_proto.OK())
                return                 

            if self.state == PlayerState.R_CHAL_A:
                self.msg = GameMessage(game_proto.OK())
                return
            
            if self.state == PlayerState.R_CHAL_B:
                self.msg = GameMessage(game_proto.OK())
                return
            
            if self.state == PlayerState.R_CHAL_C:
                self.msg = GameMessage(game_proto.OK())
                return
            
            if self.state == PlayerState.R_CHAL_D:
                self.msg = GameMessage(game_proto.OK())
                return
            
            if self.state == PlayerState.R_CHAL_E:
                self.msg = GameMessage(game_proto.OK())
                return
                
            if self.state == PlayerState.R_CHAL_MY_A:
                self.msg = GameMessage(game_proto.SHOW(self.id, ASSASSIN))
                return
            
            if self.state == PlayerState.R_CHAL_MY_B:
                self.msg = GameMessage(game_proto.SHOW(self.id, AMBASSADOR))
                return
            
            if self.state == PlayerState.R_CHAL_MY_C:
                self.msg = GameMessage(game_proto.SHOW(self.id, CAPTAIN))
                return
            
            if self.state == PlayerState.R_CHAL_MY_D:
                self.msg = GameMessage(game_proto.SHOW(self.id, DUKE))
                return
            
            if self.state == PlayerState.R_CHAL_MY_E:
                self.msg = GameMessage(game_proto.SHOW(self.id, CONTESSA))
                return

            if self.state == PlayerState.R_COINS:
                self.gs.coins[int(self.history[-1].ID1)] = int(self.history[-1].coins)
                self.msg = GameMessage(game_proto.OK())
                return
                
            if self.state == PlayerState.R_DECK:
                self.gs.my_cards = self.deck
                self.msg = GameMessage(game_proto.OK())
                self.gs.remove_evidence()
                return

            if self.state == PlayerState.R_LOSE:
                self.gs.remove_card_from_player(self.history[-1].card1, int(self.history[-1].ID1))
                self.msg = GameMessage(game_proto.OK())
                return

            if self.state == PlayerState.R_LOSE_ME:
                if len(self.deck) == 1:
                    self.msg = GameMessage(random.choice(self.possible_messages))
                else:
                    self.msg = GameMessage(game_proto.LOSE(self.id, self.gs.prefers(self.deck[0], self.deck[1])))
                    self.gs.remove_card_from_player(self.gs.prefers(self.deck[0], self.deck[1]), int(self.id))
                return

            if self.state == PlayerState.R_SHOW:
                if len(self.deck) == 1:
                    self.msg = GameMessage(random.choice(self.possible_messages))
                else:
                    self.msg = GameMessage(game_proto.LOSE(self.id, self.gs.prefers(self.deck[0], self.deck[1])))
                    self.gs.remove_card_from_player(self.gs.prefers(self.deck[0], self.deck[1]), int(self.id))
                return

            if self.state == PlayerState.R_CHOOSE:
                options = self.deck + self.exchange_cards
                options.sort(key=lambda x: self.gs.preferences[x], reverse=True)
                if self.gs.n_cards[int(self.id)] == 2:
                    self.msg = GameMessage(game_proto.KEEP(options[0], options[1]))
                    self.gs.add_evidence(options[2])
                    self.gs.add_evidence(options[3])
                else:
                    self.msg = GameMessage(game_proto.KEEP(options[0]))
                    self.gs.add_evidence(options[1])
                    self.gs.add_evidence(options[2])
                return

            if self.state == PlayerState.R_PLAYER:
                self.msg = GameMessage(random.choice(self.possible_messages))

            raise ValueError(f"Unknown state: {self.state}")

    


            
 