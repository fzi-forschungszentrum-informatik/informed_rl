import copy
import io
import threading

import carla
import gym
import numpy as np
from skimage.transform import resize
import random
import time
import math
import sys
import pathlib

sys.path.append(str(pathlib.Path(__file__).parent.parent/"utilities"))

import PIL.Image
from torchvision.transforms import ToTensor
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
import torch
import torchvision.utils as vutils
from torchvision.utils import make_grid

from clearml import Task

from frenet_optimal_trajectory import FrenetPlanner as MotionPlanner
from controller import VehiclePIDController
from traffic_rulebook_beta import TrafficRules

writer = SummaryWriter()


def euclidean_distance(v1, v2):
    return math.sqrt(sum([(a - b) ** 2 for a, b in zip(v1, v2)]))


def inertial_to_body_frame(ego_location, xi, yi, psi):
    xi2 = np.array([xi, yi])  # inertial frame
    r_psi_t = np.array([[np.cos(psi), np.sin(psi)],  # Rotation matrix transpose
                        [-np.sin(psi), np.cos(psi)]])
    xt = np.array([ego_location[0],  # Translation from inertial to body frame
                   ego_location[1]])
    xb = np.matmul(r_psi_t, xi2 - xt)
    return xb


def closest_wp_idx(ego_state, fpath, f_idx, w_size=10):
    """
    given the ego_state and frenet_path this function returns the closest WP in front of the vehicle that is within the w_size
    """

    min_dist = 300  # in meters (Max 100km/h /3.6) * 2 sn
    ego_location = [ego_state[0], ego_state[1]]
    closest_wp_index = 0  # default WP
    w_size = w_size if w_size <= len(fpath.x) - 1 - f_idx else len(fpath.x) - 1 - f_idx
    for i in range(w_size):
        temp_wp = [fpath.x[f_idx + i], fpath.y[f_idx + i]]
        temp_dist = euclidean_distance(ego_location, temp_wp)
        if temp_dist <= min_dist \
                and inertial_to_body_frame(ego_location, temp_wp[0], temp_wp[1], ego_state[2])[0] > 0.0:
            closest_wp_index = i
            min_dist = temp_dist

    return f_idx + closest_wp_index


def closest_route_idx(ego_state, route, f_idx, w_size=10):
    """
    given the ego_state and global route this function returns the closest WP in front of the vehicle that is within the w_size
    """

    min_dist = 300  # in meters (Max 100km/h /3.6) * 2 sn
    ego_location = [ego_state[0], ego_state[1]]
    closest_wp_index = 0  # default WP
    w_size = w_size if w_size <= len(route) - 2 - f_idx else len(route) - 2 - f_idx
    for i in range(w_size):
        temp_wp = [route[f_idx + i][0], route[f_idx + i][1]]
        temp_dist = euclidean_distance(ego_location, temp_wp)
        if temp_dist <= min_dist \
                and inertial_to_body_frame(ego_location, temp_wp[0], temp_wp[1], ego_state[2])[0] > 0.0:
            closest_wp_index = i
            min_dist = temp_dist

    return f_idx + closest_wp_index


class Carla:
    LOCK = threading.Lock()

    def __init__(
            self,
            world,
            settings,
            host='localhost',
            port=2000,
            action_repeat=1,
            size=(256, 256),
            grayscale=True,
            done=False,
    ):
        self.BEV_DISTANCE = 15
        self.N_ACTIONS = 9
        self.RESET_SLEEP_TIME = 0.5

        self.client = carla.Client(host, port)  # Connect to server
        print("Carla host: ", host)
        self.client.set_timeout(30.0)
        self.world = self.client.load_world(world)
        self.world.unload_map_layer(carla.MapLayer.Foliage)
        self.bp_lib = self.world.get_blueprint_library()
        self.map = self.world.get_map()

        self.grayscale = False
        self.size = size
        self.settings = settings

        self.random = np.random.RandomState(seed=None)
        self.actor_list = []
        # world params
        self.number_of_vehicles = 0
        self.max_time_episode = 1000
        # Set blueprint
        self.vehicle_bp = self.bp_lib.find('vehicle.tesla.model3')

        # Configure ss camera sensor
        self.ss_camera_bp = self.bp_lib.find('sensor.camera.semantic_segmentation')
        self.ss_camera_bp.set_attribute('image_size_x', f'{256}')
        self.ss_camera_bp.set_attribute('image_size_y', f'{256}')
        self.ss_camera_bp.set_attribute('fov', '110')
        self.ss_cam_location = carla.Location(10, 0, self.BEV_DISTANCE)
        self.ss_cam_rotation = carla.Rotation(-90, 0, 0)
        self.ss_cam_transform = carla.Transform(self.ss_cam_location, self.ss_cam_rotation)
        self.ss_camera_bp.set_attribute('sensor_tick', '0.02')

        # Configure RGB camera sensor
        self.rgb_camera_bp = self.bp_lib.find('sensor.camera.rgb')
        self.rgb_camera_bp.set_attribute('image_size_x', f'{256}')
        self.rgb_camera_bp.set_attribute('image_size_y', f'{256}')
        self.rgb_camera_bp.set_attribute('fov', '110')
        self.rgb_cam_location = carla.Location(x=20, z=self.BEV_DISTANCE)
        self.rgb_cam_rotation = carla.Rotation(pitch=90)
        self.rgb_cam_transform = carla.Transform(self.ss_cam_location, self.ss_cam_rotation)
        self.rgb_camera_bp.set_attribute('sensor_tick', '0.02')

        # Configure collision sensor
        self.col_sensor_bp = self.bp_lib.find('sensor.other.collision')
        self.col_sensor_location = carla.Location(0, 0, self.BEV_DISTANCE)
        self.col_sensor_rotation = carla.Rotation(0, 0, 0)
        self.col_sensor_transform = carla.Transform(self.col_sensor_location, self.col_sensor_rotation)

        self.collision_hist = []
        self.collision_hist_l = 1  # collision history length

        # Configure lane invation sensor
        self.lane_invation_sensor_bp = self.bp_lib.find('sensor.other.lane_invasion')
        self.lane_invation_transform = carla.Transform(carla.Location(0, 0, self.BEV_DISTANCE / 2),
                                                       carla.Rotation(0, 0, 0))
        self.lane_invation_list = []

        self._grayscale = grayscale

        # Record the time of total steps and resetting steps
        self.reset_step = 0
        self.total_step = 0

        # Motion planner AND Controller
        self.target_speed = 15 / 3.6
        self.init_s = None
        self.max_s = None
        self.track_length = None
        self.lane_change = False
        self.f_idx = 0
        self.loop_break = 30

        # For reward
        self.rule_score_list = []

        self.motionPlanner = MotionPlanner()
        self.trafficRule = TrafficRules()

    @property
    def observation_space(self):
        return gym.spaces.Dict(
            {
                "image": gym.spaces.Box(low=0, high=255, shape=(self.size[0], self.size[1], 3), dtype=np.uint8),
            }
        )

    @property
    def action_space(self):
        # return gym.spaces.Box(low=np.array([-1]), high=np.array([1]), dtype=np.float32)
        return gym.spaces.Discrete(7)

    def close(self):
        return self._env.close()

    def reset(self):
        print("RESET! New Episode.")
        if len(self.actor_list) > 0:
            print("actor num: ", len(self.actor_list))
            for actor in self.actor_list:
                    actor.destroy()
        self.scenario_index = random.randint(500, 520) # Set scenarios
        # self.scenario_index = 520
        print("Scenario index: ", self.scenario_index)
        self.scenario = self.settings.scenario_set[f"scenario_{self.scenario_index}"]

        counter = 0
        self.vehicle = None
        self.anomaly = None
        while self.vehicle == None or self.anomaly == None:
            # Spawn ego vehicle
            a_location = carla.Location(self.scenario.agent.spawn_point.location.x,
                                        self.scenario.agent.spawn_point.location.y,
                                        self.scenario.agent.spawn_point.location.z)
            a_rotation = carla.Rotation(self.scenario.agent.spawn_point.rotation.pitch,
                                        self.scenario.agent.spawn_point.rotation.yaw,
                                        self.scenario.agent.spawn_point.rotation.roll)
            start_waypoint = self.map.get_waypoint(a_location)
            a_location.z += 0.3
            a_transform = carla.Transform(a_location, a_rotation)
            self.vehicle = self.world.try_spawn_actor(self.vehicle_bp, a_transform)
            

            # Spawn anomaly
            anomaly = self.bp_lib.filter(self.scenario.anomaly.type)[0]
            if anomaly.has_attribute('is_invincible'):
                anomaly.set_attribute('is_invincible', 'false')

            # spawn anomaly at specific point
            anomaly_location = carla.Location(self.scenario.anomaly.spawn_point.location.x,
                                            self.scenario.anomaly.spawn_point.location.y,
                                            self.scenario.anomaly.spawn_point.location.z)
            anomaly_rotation = carla.Rotation(self.scenario.anomaly.spawn_point.rotation.pitch,
                                            self.scenario.anomaly.spawn_point.rotation.yaw,
                                            self.scenario.anomaly.spawn_point.rotation.roll)
            anomaly_transform = carla.Transform(anomaly_location, anomaly_rotation)
            # spawn anomaly. Note this can sometimes fail, therefore we iteratively try to spawn the car until it works
            self.anomaly = self.world.try_spawn_actor(anomaly, anomaly_transform)
            time.sleep(0.25)

            if counter > 10:
                if self.anomaly == None:
                    print("Spawning anomaly error, skip to next scenario!")
                if self.vehicle == None:
                    print("Spawning vehicle error, skip to next scenario!")
                self.scenario_index = random.randint(500, 550)
                print("Scenario index: ", self.scenario_index)
                self.scenario = self.settings.scenario_set[f"scenario_{self.scenario_index}"]
                counter = 0
            counter += 1

        self.col_sensor = None
        self.actor_list = []
        self.collision_hist = []
        self.rule_score_list_step = []
        self.camera_v_ls = []
        self.rule_paras_ls = []
        self.arrived_s = 0
        self.current_d = 0
        self.prev_s = 0
        self.prev_d = 0
        self.f_idx = 1

        self.actor_list.append(self.vehicle)
        self.actor_list.append(self.anomaly)
        self.vehicleController = VehiclePIDController(self.vehicle, args_lateral={'K_P': 1.5, 'K_D': 0.0, 'K_I': 0.0})

        # Attach ss_camera sensor
        self.ss_cam = self.world.spawn_actor(self.ss_camera_bp, self.ss_cam_transform, attach_to=self.vehicle,
                                             attachment_type=carla.AttachmentType.Rigid)
        self.actor_list.append(self.ss_cam)
        self.ss_cam.listen(lambda data: self.__process_sensor_data(data))
        time.sleep(self.RESET_SLEEP_TIME)

        # Attach rgb_camera sensor
        self.rgb_cam = self.world.spawn_actor(self.rgb_camera_bp, self.rgb_cam_transform, attach_to=self.vehicle,
                                              attachment_type=carla.AttachmentType.Rigid)
        self.actor_list.append(self.rgb_cam)
        self.rgb_cam.listen(lambda data: self.__rgb_process_sensor_data(data))
        time.sleep(self.RESET_SLEEP_TIME)

        # Attach collision sensors
        self.col_sensor = self.world.spawn_actor(self.col_sensor_bp, self.col_sensor_transform, attach_to=self.vehicle)
        self.actor_list.append(self.col_sensor)
        self.col_sensor.listen(lambda event: self.__process_collision_data(event))

        # Attach lane invasion sensor
        self.lane_invation_sensor = self.world.spawn_actor(self.lane_invation_sensor_bp, self.lane_invation_transform,
                                                           attach_to=self.vehicle)
        self.actor_list.append(self.lane_invation_sensor)
        self.lane_invation_sensor.listen(lambda event: self.__process_lane_invasion_data(event))

        # Set weather
        self.set_Weather()

        # Set reference route
        self.set_goalPoint()
        self.world.debug.draw_point(carla.Location(self.goalPoint.x, self.goalPoint.y, self.goalPoint.x),
                                    size=0.05, color=carla.Color(0, 255, 0), life_time=30)
        # Update timestamps
        self.episode_start = time.time()
        self.time_step = 0
        self.reset_step += 1

        # Set initial global route
        self.global_route = np.empty((0, 3))
        for waypoint in self.scenario.goal_trajectory:
            self.global_route = np.append(self.global_route, [[self.scenario.goal_trajectory[waypoint].x,
                                                               self.scenario.goal_trajectory[waypoint].y,
                                                               self.scenario.goal_trajectory[waypoint].z]], axis=0)
            
        np.save('road_map/global_route_town01', self.global_route)

        # Draw reference path
        for i in range(len(self.global_route)):
            self.world.debug.draw_point(carla.Location(self.global_route[i][0], self.global_route[i][1], 0.1),
                                        size=0.05, color=carla.Color(0, 255, 0), life_time=30)

        # Start & Reset of motion planner
        self.max_s = self.motionPlanner.start(route=self.global_route)
        self.motionPlanner.reset(0, 0)

        obs_image, _ = self.get_observation()
        obs = {"image": obs_image, "is_terminal": False,
               "is_first": True}
        return obs

    def step(self, action):
        # print("Step!")
        info = {}
        reward = 0

        """ 
        ================================================================================
        -------------------------------Motion Planner-----------------------------------
        ================================================================================ 
        """

        # Get parameter of vehicle
        v = self.vehicle.get_velocity()
        acc_cur = self.vehicle.get_acceleration()
        v_kmh = int(3.6 * math.sqrt(v.x ** 2 + v.y ** 2 + v.z ** 2))
        temp = [v, acc_cur]
        speed = math.sqrt(v.x ** 2 + v.y ** 2 + v.z ** 2)
        orientation = math.radians(self.vehicle.get_transform().rotation.yaw)
        ego_state = [self.vehicle.get_location().x, self.vehicle.get_location().y, speed, acc_cur, orientation, temp,
                     self.max_s]
        path, self.lane_change, path_off_road = self.motionPlanner.run_step_single_path(ego_state, self.f_idx,
                                                                                        df_n=action, Tf=4, Vf_n=0)

        wps_to_go = len(path.x) - 2
        self.f_idx = 1

        """ 
        ================================================================================
        -------------------------------Controller---------------------------------------
        ================================================================================ 
        """
        done = False
        finished = False

        if self.lane_change:
            for i in range(len(path.x) - 1):
                self.world.debug.draw_point(carla.Location(path.x[i], path.y[i], 0.3),
                                            size=0.05, color=carla.Color(0, 0, 255), life_time=5)
        else:
            for i in range(len(path.x) - 1):
                self.world.debug.draw_point(carla.Location(path.x[i], path.y[i], 0.3),
                                            size=0.05, color=carla.Color(0, 0, 255),
                                            life_time=self.motionPlanner.D_T / 5 + 0.5)

        elapsed_time = lambda previous_time: time.time() - previous_time
        path_start_time = time.time()
        loop_counter = 0
      
        while self.f_idx < wps_to_go and (elapsed_time(path_start_time) < self.motionPlanner.D_T / 5
                                          or self.lane_change):
            loop_counter += 1
            ego_state = [self.vehicle.get_location().x, self.vehicle.get_location().y,
                         math.radians(self.vehicle.get_transform().rotation.yaw), 0, 0, temp, self.max_s]
            ego_location = [self.vehicle.get_location().x,
                            self.vehicle.get_location().y]
            self.f_idx = closest_wp_idx(ego_state, path, self.f_idx)
            cmd_wp = [path.x[self.f_idx], path.y[self.f_idx]]
            cmd_wp2 = [path.x[self.f_idx + 1], path.y[self.f_idx + 1]]

            control = self.vehicleController.run_step_2_wp(self.target_speed, cmd_wp, cmd_wp2)  # calculate control
            self.vehicle.apply_control(control)  # apply control

            v = self.vehicle.get_velocity()
            speed = math.sqrt(v.x ** 2 + v.y ** 2 + v.z ** 2)
            self.arrived_s, self.current_d = self.motionPlanner.estimate_frenet_state_beta(ego_location)
            rule_paras = [reward, self.arrived_s, self.current_d]
            if loop_counter % 10 == 0:
                self.rule_paras_ls.append(rule_paras)
            if loop_counter % 100 == 0:
                _, rgb_image = self.get_observation()
                self.camera_v_ls.append(rgb_image)
            if len(self.collision_hist) != 0:
                done = True
                print("Collision!", "arrived length: ", self.arrived_s, "route length: ", self.max_s)
                break
            if self.arrived_s >= self.max_s - 4:
                done = True
                finished = True
                print("Arrived goal!", "arrived length: ", self.arrived_s, "route length: ", self.max_s)
                break
            if time.time() - self.episode_start > 30:
                done = True
                print("Max episode time!")
                break


        """ 
        ================================================================================
        -------------------------------Rulebook Reward ---------------------------------
        ================================================================================ 
        """
        # Convert state to frenet coordinate
        ego_location = [self.vehicle.get_location().x,
                        self.vehicle.get_location().y]

        anomaly_pos = [self.scenario.anomaly.spawn_point.location.x,
                       self.scenario.anomaly.spawn_point.location.y]
        anomaly_s, anomaly_d = self.motionPlanner.estimate_frenet_state_beta(anomaly_pos)

        # print("arrived s, current d:", self.arrived_s, self.current_d)

        path_s_length = self.arrived_s - self.prev_s
        
        # Check state
        if self.current_d <= -1.75 or self.current_d > 5.25:
            off_road = True
        else:
            off_road = False

        if math.fabs(self.current_d - self.prev_d) > 1.2:
            lane_change = True
            if self.current_d > 0:
                self.prev_d = self.current_d // 1.75 * 1.75 if abs(self.current_d % 1.75) < 0.875 else (self.current_d // 1.75 + 1) * 1.75
            else:
                self.prev_d = (self.current_d // 1.75 + 1) // 1.75 * 1.75 if abs(self.current_d % 1.75) > 0.875 else (self.current_d // 1.75) * 1.75
        else:
            lane_change = False        

        if self.time_step > self.max_time_episode:
            done = True
        
        reward, rule_score, finished_score = self.trafficRule.rule_book_check(self.arrived_s, self.prev_s, self.current_d, self.prev_d, path, self.lane_change, path_off_road,
                                                        lane_change, off_road, done, finished, v_kmh, anomaly_s, anomaly_d, path_s_length)
        self.rule_score_list_step.append(rule_score)
        self.prev_s = self.arrived_s

        rule_paras = [reward, self.arrived_s, self.current_d]
        self.rule_paras_ls.append(rule_paras)
        
        """ 
        ================================================================================
        -------------------------------Get Observation----------------------------------
        ================================================================================ 
        """

        obs_image, rgb_image = self.get_observation()
        obs = {}
        obs["image"] = obs_image
        obs["is_terminal"] = done
        obs["is_first"] = True if self.time_step == 0 else False
        self.camera_v_ls.append(rgb_image)
        info["arrived_s"] = self.arrived_s
        info["finished_score"] = finished_score
        info["camera_v"] = self.camera_v_ls
        info["rule_paras"] = self.rule_paras_ls
        info["anomaly"] = [anomaly_s, anomaly_d, self.max_s]

        # Update timestamps
        self.time_step += 1
        self.total_step += 1

        if done:
            rule_score = np.sum(self.rule_score_list_step)
            for actor in self.actor_list:
                actor.destroy()
            self.actor_list = []

        info["rule_score"] = rule_score

        return obs, reward, done, copy.deepcopy(info)

    def get_observation(self):
        """ Observations in PyTorch format BCHW """
        image = self.observation
        image_rgb = self.camera_rgb
        # image_rgb = image_rgb.astype(np.float32) / 255
        # image_rgb = self.arrange_colorchannels(image_rgb)
        image = image[:, :, None] if self.grayscale else image
        image_rgb = image_rgb[:, :, None] if self.grayscale else image_rgb
        camera = resize(image_rgb, (self.size[0], self.size[1])) * 255
        # writer.add_image("test_path", camera, dataformats='HWC')
        obs = camera
        return obs, image_rgb

    def __process_sensor_data(self, image):
        """ Observations directly viewable with OpenCV in CHW format """
        image.convert(carla.ColorConverter.CityScapesPalette)
        # array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
        array = np.array(image.raw_data)
        array = np.reshape(array, (image.height, image.width, 4))
        array = array[:, :, :3]
        # array = array[:, :, ::-1]
        self.observation = array

    def __rgb_process_sensor_data(self, image):
        """RGB camera"""
        # image.convert(carla.ColorConverter.CityScapesPalette)
        i = np.array(image.raw_data)
        i = np.reshape(i, (256, 256, 4))
        i = i[:, :, :3]
        # frame = i3.astype(np.float32) / 255
        # frame = self.arrange_colorchannels(frame)
        self.camera_rgb = i

    def arrange_colorchannels(self, image):
        mock = image.transpose(2, 1, 0)
        tmp = []
        tmp.append(mock[2])
        tmp.append(mock[1])
        tmp.append(mock[0])
        tmp = np.array(tmp)
        tmp = tmp.transpose(2, 1, 0)
        return tmp

    def __process_collision_data(self, event):
        impulse = event.normal_impulse
        intensity = np.sqrt(impulse.x ** 2 + impulse.y ** 2 + impulse.z ** 2)
        self.collision_hist.append(intensity)
        if len(self.collision_hist) > self.collision_hist_l:
            self.collision_hist.pop(0)

    def __process_lane_invasion_data(self, event):
        self.lane_invation_list.append(event)

    def _try_spawn_random_vehicle_at(self, transform, number_of_wheels=[4]):
        blueprint = self._create_vehicle_bluepprint('vehicle.*', number_of_wheels=number_of_wheels)
        blueprint.set_attribute('role_name', 'autopilot')
        vehicle = self.world.try_spawn_actor(blueprint, transform)
        if vehicle is not None:
            vehicle.set_autopilot()
            self.actor_list.append(vehicle)
            return True
        return False

    def _create_vehicle_bluepprint(self, actor_filter, color=None, number_of_wheels=[4]):
        blueprints = self.world.get_blueprint_library().filter(actor_filter)
        blueprint_library = []
        for nw in number_of_wheels:
            blueprint_library = blueprint_library + [x for x in blueprints if
                                                     int(x.get_attribute('number_of_wheels')) == nw]
        bp = random.choice(blueprint_library)
        if bp.has_attribute('color'):
            if not color:
                color = random.choice(bp.get_attribute('color').recommended_values)
            bp.set_attribute('color', color)
        return bp

    def spawn_anomaly(self):
        # select anomaly according to settings
        anomaly = self.bp_lib.filter(self.scenario.anomaly.type)[0]
        if anomaly.has_attribute('is_invincible'):
            anomaly.set_attribute('is_invincible', 'false')

            # spawn anomaly at specific point
        anomaly_location = carla.Location(self.scenario.anomaly.spawn_point.location.x,
                                          self.scenario.anomaly.spawn_point.location.y,
                                          self.scenario.anomaly.spawn_point.location.z)
        anomaly_rotation = carla.Rotation(self.scenario.anomaly.spawn_point.rotation.pitch,
                                          self.scenario.anomaly.spawn_point.rotation.yaw,
                                          self.scenario.anomaly.spawn_point.rotation.roll)
        anomaly_transform = carla.Transform(anomaly_location, anomaly_rotation)

        # spawn anomaly. Note this can sometimes fail, therefore we iteratively try to spawn the car until it works
        counter = 0
        player = None
        # while player is None:
        player = self.world.try_spawn_actor(anomaly, anomaly_transform)
        if player is not None:
            self.actor_list.append(player)
        return player

    def set_goalPoint(self):
        location = carla.Location(self.scenario.goal_point.location.x, self.scenario.goal_point.location.y,
                                  self.scenario.goal_point.location.z)
        self.latest_rotation = self.scenario.agent.spawn_point.rotation.yaw
        self.goalPoint = location

    def set_Weather(self):
        self.weather = carla.WeatherParameters(
            cloudiness=self.scenario.weather.cloudiness,
            precipitation=self.scenario.weather.precipitation,
            precipitation_deposits=self.scenario.weather.precipitation_deposits,
            wind_intensity=self.scenario.weather.wind_intensity,
            sun_altitude_angle=self.scenario.weather.sun_altitude_angle,
            fog_density=self.scenario.weather.fog_density,
            wetness=self.scenario.weather.wetness
        )
        self.world.set_weather(self.weather)

    def render(self, mode):
        pass


def main():
    task = Task.init(project_name="bogdoll/rl_traffic_rule_Jing", task_name="carla_path_debug",
                     reuse_last_task_id=True)
    carla_env = Carla(
        'Town01_Opt',
        host='ids-goodyear.fzi.de',
        port=2000,
        action_repeat=1,
        size=[256, 256],
        grayscale=False,
        done=False
    )

    carla_env.reset()
    for i in range(5):
        action = carla_env.action_space.sample()
        carla_env.step(action)
        tensor_images = torch.stack([torch.from_numpy(img) for img in carla_env.carla_video])
        tensor_images = tensor_images.unsqueeze(0)
        print(tensor_images.size())
        writer.add_video(f'carla_{i}_video', tensor_images, fps=360)
        writer.close()


if __name__ == "__main__":
    main()
