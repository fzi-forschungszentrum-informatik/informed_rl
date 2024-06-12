import math

def euclidean_dis(path_i, anomaly_pose):
    distance = math.sqrt(math.pow((path_i[0] - anomaly_pose[0]), 2) + math.pow((path_i[1] - anomaly_pose[1]), 2))
    return distance

class TrafficRules:
    """A class to model traffic rules and calculate rewards based on ego vehicle state and trajectory"""
    RULEBOOK_ACTIVATION_DISTANCE = 10

    def __init__(self):
        """Initialize attributes to keep track of scores and penalties"""

        # Score attributes
        self.score = 0.0
        self.rule_book_reward = 0
        self.ego_vehicle_reward = 0
        self.rule_book_on = False # activate rulebook or not
        self.collision_flag = False # collison or not
        self.SPEED_THRESHOLD = 10

        # Ego basic rewards
        self.LANE_CHANGE_PENALTY = -3
        self.OUT_ROAD_PENALTY = -3
        self.COLLISION_PENALTY = -40
        self.SPEED_PENALTY = -1
        self.WRONG_DIR_PENALTY = -10
        self.NOCOLLISION_STATE_REWARD = 5
        self.NEAR_FINISH_REWARD = 10
        self.FINISH_REWARD = 40

        # Trajectory basic rewards
        self.PATH_NOCOLLISION_STATE_REWARD = 10
        self.PATH_NORMAL_REWARD = 1
        self.PATH_S_REWARD = 1
        self.PATH_CLEARANCE_REWARD = 1
        
        

        # Rulebook rules basic rewards, for more rules, set basic reward based on thesis
        self.PATH_LANE_CHANGE_PENALTY = -1
        self.PATH_OUT_ROAD_PENALTY = -1

        # For more rules, set corresponding coefficients based on thesis
        self.rule_book_coeff = {"Lane keep": 0.1,
                                "No out road": 0.1}

    def rulebook_score(self, done, finished, arrived_s, current_d, path_off_road, anomaly_s, anomaly_d, path_s_length):
        """
        Calculate the score based on the rulebook.
        """
        self.rule_book_reward += path_s_length / 2 * self.PATH_S_REWARD # close to final reward

        # Check Collision
        if done and not finished:
            self.collision_flag = True
            self.rule_book_reward += self.COLLISION_PENALTY

        if self.rule_book_on and math.fabs(arrived_s - anomaly_s) <= self.RULEBOOK_ACTIVATION_DISTANCE: # rulebook activation
            self.rule_book_reward += path_s_length / 2 * (5.25 - abs(current_d - anomaly_d)) / 3.5 * self.PATH_CLEARANCE_REWARD # minimum clearance reward
            if math.fabs(current_d) > 1:
                self.rule_book_reward += self.rule_book_coeff["Lane keep"] * path_s_length / 2 * self.PATH_LANE_CHANGE_PENALTY # lane deviation penalty
            else:
                self.rule_book_reward += path_s_length / 2 * self.PATH_NORMAL_REWARD
            if path_off_road:
                self.rule_book_reward += self.rule_book_coeff["No out road"] * path_s_length / 2 * self.PATH_OUT_ROAD_PENALTY # out road penalty

        else: # away from anomaly
            if math.fabs(current_d) > 1:
                self.rule_book_reward += path_s_length / 2 * self.PATH_LANE_CHANGE_PENALTY # lane deviation penalty
            else:
                self.rule_book_reward += path_s_length / 2 * self.PATH_NORMAL_REWARD
            if path_off_road:
                self.rule_book_reward += path_s_length / 2 * self.PATH_OUT_ROAD_PENALTY # out road penalty
                    
        return self.rule_book_reward
    

    def ego_vehicle_score(self, v, finished, current_d):
        """
        Calculate the score based on the ego states.
        """
        # Check speed
        if v < self.SPEED_THRESHOLD:
            self.ego_vehicle_reward += self.SPEED_PENALTY

        # Check finish
        if finished:
            self.ego_vehicle_reward += self.NEAR_FINISH_REWARD
            self.finished = 0.5
            if math.fabs(current_d) <= 0.5:
                self.ego_vehicle_reward += self.FINISH_REWARD
                self.finished = 1
        
        return self.ego_vehicle_reward


    def rule_book_check(self, arrived_s, prev_s, current_d, prev_d, path, path_lane_change, path_off_road, lane_change,
                        off_road, done, finished, v, anomaly_s, anomaly_d, path_s_length):
        """
        Calculate the score based on the ego states and rulebook.
        """
        self.score = 0.0
        self.rule_book_reward = 0
        self.ego_vehicle_reward = 0
        self.collision_flag = False
        self.finished = 0
        self.rule_book_on = True
    
        self.score = self.ego_vehicle_score(v, finished, current_d) + self.rulebook_score(done, finished, arrived_s, current_d, off_road, anomaly_s, anomaly_d, path_s_length)
        return self.score, self.rule_book_reward, self.finished



