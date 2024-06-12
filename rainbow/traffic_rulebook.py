import math
import numpy


def euclidean_dis(path_i, anomaly_pose):
    distance = math.sqrt(math.pow((path_i[0] - anomaly_pose[0]), 2) + math.pow((path_i[1] - anomaly_pose[1]), 2))
    return distance

class TrafficRules:
    def __init__(self):
        self.score = 0.0
        self.rule_book_reward = 0

        self.LANE_CHANGE_PENALTY = -3
        self.OUT_ROAD_PENALTY = -3
        self.COLLISION_PENALTY = -60
        self.SPEED_PENALTY = -1
        self.WRONG_DIR_PENALTY = -10
        self.NOCOLLISION_STATE_REWARD = 5
        self.NEAR_FINISH_REWARD = 20
        self.FINISH_REWARD = 60

        self.PATH_NOCOLLISION_STATE_REWARD = 10

        self.PATH_NORMAL_REWARD = 1
        self.PATH_LANE_CHANGE_PENALTY = -1
        self.PATH_OUT_ROAD_PENALTY = -1
        self.PATH_S_REWARD = 1
        self.PATH_CLEARANCE_REWARD = 1
        
        self.SPEED_THRESHOLD_L = 10
        self.SPEED_THRESHOLD_H = 30

        self.rule_book_coeff = {"Lane keep": 0.1,
                                "No out road": 0.1,
                                "No collision": 1}

    def rulebook_score(self, arrived_s, prev_s, current_d, prev_d, lane_change, off_road, anomaly_s, anomaly_d):

        # if not self.collision_flag:
        # Straight
        # if not path_lane_change:
        #     self.rule_book_reward += self.PATH_S_REWARD # close to final reward
        #     print("s reward: ", self.PATH_S_REWARD)
        #     if math.fabs(arrived_s - anomaly_s) <= 10: # near the anomaly
        #         self.rule_book_reward += self.PATH_NORMAL_REWARD # normal reward
        #         print("n reward: ", self.PATH_NORMAL_REWARD)
        #         self.rule_book_reward += (5.25 - abs(current_d - anomaly_d)) / 3.5 * self.PATH_CLEARANCE_REWARD # minimum clearance reward
        #         print("c reward: ", (5.25 - abs(current_d - anomaly_d)) / 3.5 * self.PATH_CLEARANCE_REWARD)
        #         if math.fabs(current_d > 1):
        #             self.rule_book_reward += self.rule_book_coeff["Lane keep"] * abs(current_d) * self.PATH_LANE_CHANGE_PENALTY # lane deviation penalty
        #             print("d penalty: ", self.rule_book_coeff["Lane keep"] * abs(current_d) * self.PATH_LANE_CHANGE_PENALTY)
        #         if path_off_road:
        #             self.rule_book_reward += self.rule_book_coeff["No out road"] * self.PATH_OUT_ROAD_PENALTY # out road penalty
        #             print("o penalty: ", self.rule_book_coeff["No out road"] * self.PATH_OUT_ROAD_PENALTY)
        #     else: # away from anomaly
        #         if math.fabs(current_d > 1):
        #             self.rule_book_reward += abs(current_d) * self.PATH_LANE_CHANGE_PENALTY # lane deviation penalty
        #             print("d penalty: ", abs(current_d) * self.PATH_LANE_CHANGE_PENALTY)
        #         if path_off_road:
        #             self.rule_book_reward += self.PATH_OUT_ROAD_PENALTY # out road penalty
        #             print("o penalty: ", self.PATH_OUT_ROAD_PENALTY)
        # Curve
        # if path_lane_change:
        if arrived_s - prev_s > 0:
            self.rule_book_reward += self.PATH_S_REWARD # close to final reward
        else:
            self.rule_book_reward += -1 * self.PATH_S_REWARD
        if math.fabs(arrived_s - anomaly_s) <= 10: # near the anomaly
            self.rule_book_reward += self.PATH_NORMAL_REWARD # normal reward
            self.rule_book_reward += (5.25 - abs(current_d - anomaly_d)) / 3.5 * self.PATH_CLEARANCE_REWARD # minimum clearance reward
            if math.fabs(current_d) > 1.75:
                self.rule_book_reward += self.rule_book_coeff["Lane keep"] * self.PATH_LANE_CHANGE_PENALTY # lane deviation penalty
            if off_road:
                self.rule_book_reward += self.rule_book_coeff["No out road"] * self.PATH_OUT_ROAD_PENALTY # out road penalty
        else: # away from anomaly
            if math.fabs(current_d) > 1.75:
                self.rule_book_reward += self.PATH_LANE_CHANGE_PENALTY # lane deviation penalty
            else: 
                self.rule_book_reward += (1.75 - math.fabs(current_d)) / 1.75 * self.PATH_NORMAL_REWARD
            if off_road:
                self.rule_book_reward += self.PATH_OUT_ROAD_PENALTY # out road penalty

        # # Check if good state
        # if not path_lane_change and not self.collision_flag:
        #     if math.fabs(current_d) < 1.5:
        #         self.rule_book_reward += self.PATH_STRAIGHT_REWARD
        #     elif math.fabs(arrived_s - anomaly_s) <= 5 or math.fabs(prev_s - anomaly_s) <= 5:
        #         self.rule_book_reward += self.PATH_STRAIGHT_REWARD
        
        # # Check if off road
        # if path_off_road:
        #     self.rule_book_reward += self.PATH_OFF_ROAD_PENALTY * 5 if path_lane_change else self.PATH_OFF_ROAD_PENALTY
        
        # # Check if lane change for collision avoidance
        # if path_lane_change:
        #     if (math.fabs(arrived_s - anomaly_s) <= 5 and math.fabs(current_d - anomaly_d) >= 1.5 and math.fabs(prev_d) < 1.5) \
        #         or (math.fabs(prev_s - anomaly_s) <= 5 and math.fabs(prev_d - anomaly_d) >= 1.5 and math.fabs(current_d) < 1.5):
        #             self.rule_book_reward += self.PATH_NOCOLLISION_STATE_REWARD
        #             if path_off_road:
        #                 self.rule_book_reward -= self.PATH_OFF_ROAD_PENALTY * 5
        #     self.rule_book_reward += self.PATH_LANE_CHANGE_PENALTY
        #     self.score += (1 - math.fabs(current_d)) * 3
        #     if arrived_s == -1:
        #         self.score += self.WRONG_DIR_PENALTY
        #     self.score += 0.03 * arrived_s * 5
        # else:
        #     # s,d reward
        #     self.score += 1 - math.fabs(current_d)
        #     if arrived_s == -1:
        #         self.score += self.WRONG_DIR_PENALTY
        #     self.score += 0.03 * arrived_s
                    
        return self.rule_book_reward

    def rule_book_check(self, arrived_s, prev_s, current_d, prev_d, lane_change, off_road, 
                        collision_check, finished, v, anomaly_s, anomaly_d):
        
        self.score = 0.0
        self.rule_book_reward = 0
        self.collision_flag = False
        self.finished = 0

        # Collision
        if collision_check and not finished:
            self.collision_flag = True
            self.score += self.COLLISION_PENALTY

        # Check speed
        if v < self.SPEED_THRESHOLD_L or v > self.SPEED_THRESHOLD_H:
            self.score += 2 * self.SPEED_PENALTY

        # For rulebook comment these two penalty
        # if lane_change:
        #     self.score += self.LANE_CHANGE_PENALTY
        # if off_road:
        #     self.score += self.OFF_ROAD_PENALTY
        
        # Check finish
        if finished:
            self.score += self.NEAR_FINISH_REWARD
            self.finished = 0.5
            if math.fabs(current_d) <= 0.5:
                self.score += self.FINISH_REWARD
                self.finished = 1

        # if path_lane_change:
        #     # Lane keep reward
        #     self.score += -math.fabs(current_d) * 5

        #     # Distance to final point
        #     if arrived_s == -1:
        #         self.score += self.WRONG_DIR_PENALTY
        #     self.score += 0.02 * arrived_s * 5
        # else:
        #     self.score += -math.fabs(current_d)
        #     if arrived_s == -1:
        #         self.score += self.WRONG_DIR_PENALTY
        #     self.score += 0.02 * arrived_s

        self.score += self.rulebook_score(arrived_s, prev_s, current_d, prev_d, lane_change, off_road, anomaly_s, anomaly_d)
        return self.score, self.rule_book_reward, self.finished

