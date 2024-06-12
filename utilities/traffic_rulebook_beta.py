import math
from typing import List, Dict, Tuple, Callable

# Define the class TrafficRules
class TrafficRules:
    """A class to model traffic rules and calculate rewards based on ego vehicle state and trajectory."""
    RULEBOOK_ACTIVATION_DISTANCE = 10
    SPEED_THRESHOLD = 10

    # Rulebook global rewards
    COLLISION_PENALTY = -60   # Rule 0
    LANE_CHANGE_PENALTY = -1  # Rule 1
    OUT_ROAD_PENALTY = -1     # Rule 2

    # Rulebook eventual rewards

    # Ego basic rewards
    SPEED_PENALTY = -1
    NEAR_FINISH_REWARD = 10
    FINISH_REWARD = 40

    # Trajectory basic rewards
    PATH_NORMAL_REWARD = 1
    PATH_S_REWARD = 1
    PATH_CLEARANCE_REWARD = 1

    def __init__(self):
        """Initialize attributes to keep track of scores and penalties."""
        self.score = 0.0
        self.rule_book_reward = 0
        self.ego_vehicle_reward = 0
        self.rule_book_on = False  # activate rulebook or not
        self.finished = 0

    def hierarchy_1(self, data):
        """Calculate the reward for rule hierarchy 1."""
        return self.COLLISION_PENALTY if data[0] and not data[1] else 0

    def hierarchy_2(self, data):
        """Calculate the reward for rule hierarchy 2."""
        reward_1 = data[-1] * (self.LANE_CHANGE_PENALTY if abs(data[3]) > 1 else self.PATH_NORMAL_REWARD)
        reward_2 = data[-1] * self.OUT_ROAD_PENALTY if data[4] else 0
        return reward_1 + reward_2

    def get_rule_coefficients(self):
        """Get the rule and coefficient configurations."""
        return {
            self.hierarchy_1: 1,
            self.hierarchy_2: 0.1,
            # Add more hierachies and their coefficients here as needed 
        }

    def rulebook_score(self, done, finished, arrived_s, current_d, path_off_road, anomaly_s, anomaly_d, path_s_length):
        """Calculate the reward based on the rulebook."""
        data_instance = [done, finished, arrived_s, current_d, path_off_road, anomaly_s, anomaly_d, path_s_length / 2]
        rule_coefficients = self.get_rule_coefficients()
        self.rule_book_reward += data_instance[-1] * self.PATH_S_REWARD
        self.rule_book_reward += data_instance[-1] * (5.25 - abs(current_d - anomaly_d)) / 3.5 * self.PATH_CLEARANCE_REWARD

        for rule, coefficient in rule_coefficients.items():
            if self.rule_book_on and abs(arrived_s - anomaly_s) <= self.RULEBOOK_ACTIVATION_DISTANCE:
                self.rule_book_reward += data_instance[-1] * (5.25 - abs(data_instance[3] - data_instance[6])) / 3.5 * self.PATH_CLEARANCE_REWARD
                self.rule_book_reward += rule(data_instance) * coefficient  
            else:
                self.rule_book_reward += rule(data_instance)

        return self.rule_book_reward

    def ego_vehicle_score(self, v, finished, current_d):
        """Calculate the score based on the ego states."""
        if v < self.SPEED_THRESHOLD:
            self.ego_vehicle_reward += self.SPEED_PENALTY

        if finished:
            self.ego_vehicle_reward += self.NEAR_FINISH_REWARD
            self.finished = 0.5
            if abs(current_d) <= 0.5:
                self.ego_vehicle_reward += self.FINISH_REWARD
                self.finished = 1

        return self.ego_vehicle_reward

    def rule_book_check(self, arrived_s, prev_s, current_d, prev_d, path, path_lane_change, path_off_road, lane_change, off_road,
                        done, finished, v, anomaly_s, anomaly_d, path_s_length):
        """Calculate the score based on the ego states and rulebook."""
        self.score = 0.0
        self.rule_book_reward = 0
        self.ego_vehicle_reward = 0
        self.finished = 0
        self.rule_book_on = True

        self.score = self.ego_vehicle_score(v, finished, current_d) + self.rulebook_score(done, finished, arrived_s, current_d, off_road, anomaly_s, anomaly_d, path_s_length)
        return self.score, self.rule_book_reward, self.finished


           
