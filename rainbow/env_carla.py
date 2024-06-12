import carla
import copy

import random
import numpy as np
import math
import time
from skimage.transform import resize

import torch

from frenet_optimal_trajectory import FrenetPlanner as MotionPlanner
from traffic_rulebook import TrafficRules

IM_WIDTH = 256
IM_HEIGHT = 256

BEV_DISTANCE = 15

N_ACTIONS = 9

RESET_SLEEP_TIME = 1


FIXED_DELTA_SECONDS = 0.1
SUBSTEP_DELTA = 0.01
MAX_SUBSTEPS = 10
EPISODE_TIME = 30


class Environment:
    def __init__(
            self, 
            world, 
            settings, 
            host="tks-harper.fzi.de", 
            port=2000,
            size=(256, 256),
            grayscale=True
    ):
        self.client = carla.Client(host, port)  # Connect to server
        self.client.set_timeout(10.0)
        self.world = self.client.load_world(world)
        self.world.unload_map_layer(carla.MapLayer.Foliage)
        self.bp_lib = self.world.get_blueprint_library()
        self.map = self.world.get_map()
        # self.spawn_points = self.map.get_spawn_points()

        self.grayscale = False
        self.size = size
        self.settings = settings

        self.actor_list = []
        self.IM_WIDTH = IM_WIDTH
        self.IM_HEIGHT = IM_HEIGHT

        # Enable synchronous mode between server and client
        # self.settings = self.world.get_settings()
        # self.settings.synchronous_mode = True # Enables synchronous mode
        # self.world.apply_settings(self.settings)

        w_settings = self.world.get_settings()
        w_settings.synchronous_mode = True
        w_settings.fixed_delta_seconds = (
            FIXED_DELTA_SECONDS  # 10 fps | fixed_delta_seconds <= max_substep_delta_time * max_substeps
        )
        w_settings.substepping = True
        w_settings.max_substep_delta_time = SUBSTEP_DELTA
        w_settings.max_substeps = MAX_SUBSTEPS
        self.world.apply_settings(w_settings)
        self.fps_counter = 0
        self.max_fps = int(1 / FIXED_DELTA_SECONDS) * EPISODE_TIME

        print(
            f"~~~~~~~~~~~~~~\n## Simulator settings ##\nFrames: {int(1/FIXED_DELTA_SECONDS)}\nSubstep_delta: {SUBSTEP_DELTA}\nMax_substeps: {MAX_SUBSTEPS}\n~~~~~~~~~~~~~~"
        )

    def init_ego(self):
        self.vehicle_bp = self.bp_lib.find("vehicle.tesla.model3")
        self.ss_camera_bp = self.bp_lib.find("sensor.camera.semantic_segmentation")
        self.rgb_camera_bp = self.bp_lib.find('sensor.camera.rgb')
        self.col_sensor_bp = self.bp_lib.find("sensor.other.collision")

        # Configure sensors
        self.ss_camera_bp.set_attribute("image_size_x", f"{IM_WIDTH}")
        self.ss_camera_bp.set_attribute("image_size_y", f"{IM_HEIGHT}")
        self.ss_camera_bp.set_attribute("fov", "110")

        self.ss_cam_location = carla.Location(0, 0, BEV_DISTANCE)
        self.ss_cam_rotation = carla.Rotation(-90, 0, 0)
        self.ss_cam_transform = carla.Transform(self.ss_cam_location, self.ss_cam_rotation)

        self.col_sensor_location = carla.Location(0, 0, 0)
        self.col_sensor_rotation = carla.Rotation(0, 0, 0)
        self.col_sensor_transform = carla.Transform(self.col_sensor_location, self.col_sensor_rotation)

        # Configure rgb camera
        self.rgb_camera_bp = self.bp_lib.find('sensor.camera.rgb')
        self.rgb_camera_bp.set_attribute('image_size_x', f'{256}')
        self.rgb_camera_bp.set_attribute('image_size_y', f'{256}')
        self.rgb_camera_bp.set_attribute('fov', '110')
        self.rgb_cam_location = carla.Location(x=20, z=BEV_DISTANCE)
        self.rgb_cam_rotation = carla.Rotation(pitch=90)
        self.rgb_cam_transform = carla.Transform(self.ss_cam_location, self.ss_cam_rotation)

        self.collision_hist = []

        # For reward
        self.rule_score_list = []

        self.motionPlanner = MotionPlanner()
        self.trafficRule = TrafficRules()

    def reset(self):
        print("RESET! New Episode.")
        if len(self.actor_list) > 0:
            print("actor num: ", len(self.actor_list))
            for actor in self.actor_list:
                    actor.destroy()
        self.scenario_index = random.randint(500, 520)
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
                self.scenario_index = random.randint(500, 520)
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

        # Attach and listen to image sensor (BEV Semantic Segmentation)
        self.ss_cam = self.world.spawn_actor(
            self.ss_camera_bp, self.ss_cam_transform, attach_to=self.vehicle, attachment_type=carla.AttachmentType.Rigid
        )
        self.actor_list.append(self.ss_cam)
        self.ss_cam.listen(lambda data: self.__process_sensor_data(data))
        time.sleep(RESET_SLEEP_TIME)

        # Attach rgb_camera sensor
        self.rgb_cam = self.world.spawn_actor(self.rgb_camera_bp, self.rgb_cam_transform, attach_to=self.vehicle,
                                              attachment_type=carla.AttachmentType.Rigid)
        self.actor_list.append(self.rgb_cam)
        self.rgb_cam.listen(lambda data: self.__rgb_process_sensor_data(data))
        time.sleep(RESET_SLEEP_TIME)

        self.tick_world(times=10) # Let's see if we need this anymore in 0.9.14
        self.fps_counter = 0

        # Attach and listen to collision sensor
        self.col_sensor = self.world.spawn_actor(self.col_sensor_bp, self.col_sensor_transform, attach_to=self.vehicle)
        self.actor_list.append(self.col_sensor)
        self.col_sensor.listen(lambda event: self.__process_collision_data(event))

        # Set weather
        self.set_Weather()

        # Set reference route
        self.set_goalPoint()
        self.world.debug.draw_point(carla.Location(self.goalPoint.x, self.goalPoint.y, self.goalPoint.x),
                                    size=0.05, color=carla.Color(0, 255, 0), life_time=30)
        # Update timestamps
        self.episode_start = time.time()
        self.time_step = 0

         # Set initial global route
        self.global_route = np.empty((0, 3))
        # self.max_s = -1
        for waypoint in self.scenario.goal_trajectory:
            self.global_route = np.append(self.global_route, [[self.scenario.goal_trajectory[waypoint].x,
                                                               self.scenario.goal_trajectory[waypoint].y,
                                                               self.scenario.goal_trajectory[waypoint].z]], axis=0)
        # Draw reference path
        for i in range(len(self.global_route)):
            self.world.debug.draw_point(carla.Location(self.global_route[i][0], self.global_route[i][1], 0.1),
                                        size=0.07, color=carla.Color(0, 255, 0), life_time=30)

        # Start & Reset of motion planner
        self.max_s = self.motionPlanner.start(route=self.global_route)
        self.motionPlanner.reset(0, 0)
        
        obs, camera = self.get_observation()

        return obs, camera

    def step(self, action):
        # Set reward and 'done' flag
        done = False
        collision_check = False
        finished = False
        reward = 0
        info = {}

        # Easy actions: Steer left, center, right (0, 1, 2)
        if action == 0:
            self.vehicle.apply_control(carla.VehicleControl(throttle=1.0, steer=0))
        elif action == 1:
            self.vehicle.apply_control(carla.VehicleControl(throttle=1.0, steer=-1))
        elif action == 2:
            self.vehicle.apply_control(carla.VehicleControl(throttle=1.0, steer=1))
        elif action == 3:
            self.vehicle.apply_control(carla.VehicleControl(throttle=0.5, steer=0))
        elif action == 4:
            self.vehicle.apply_control(carla.VehicleControl(throttle=0.5, steer=-1))
        elif action == 5:
            self.vehicle.apply_control(carla.VehicleControl(throttle=0.5, steer=1))
        elif action == 6:
            self.vehicle.apply_control(carla.VehicleControl(throttle=0, steer=0))
        elif action == 7:
            self.vehicle.apply_control(carla.VehicleControl(throttle=0, steer=-1))
        elif action == 8:
            self.vehicle.apply_control(carla.VehicleControl(throttle=0, steer=1))

        self.tick_world()

        ego_location = [self.vehicle.get_location().x,
                        self.vehicle.get_location().y]
        self.arrived_s, self.current_d = self.motionPlanner.estimate_frenet_state_beta(ego_location)
        anomaly_pos = [self.scenario.anomaly.spawn_point.location.x,
                       self.scenario.anomaly.spawn_point.location.y]
        anomaly_s, anomaly_d = self.motionPlanner.estimate_frenet_state_beta(anomaly_pos)

        rule_paras = [reward, self.arrived_s, self.current_d]

        self.world.debug.draw_point(carla.Location(ego_location[0], ego_location[1], 0.3),
                                    size=0.03, color=carla.Color(0, 0, 255),
                                    life_time=30)
        
        self.rule_paras_ls.append(rule_paras)

        _, rgb_image = self.get_observation()
        self.camera_v_ls.append(rgb_image)

        # Get velocity of vehicle
        v = self.vehicle.get_velocity()
        v_kmh = int(3.6 * math.sqrt(v.x**2 + v.y**2 + v.z**2))

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
        
        # Check stop condition
        if len(self.collision_hist) != 0 or self.current_d <= -5.25 or self.current_d >= 8.75:
            done = True
            collision_check = True
            print("Collision!", "arrived length: ", self.arrived_s, "route length: ", self.max_s)
        if self.arrived_s >= self.max_s:
            done = True
            finished = True
            print("Arrived goal!", "arrived length: ", self.arrived_s, "route length: ", self.max_s)
        if time.time() - self.episode_start > 30:
            print("Max episode time!")
            done = True
        if self.time_step > 800:
            print("Max time step!")
            done = True

        # Reward function
        reward, rule_score, finished_score = self.trafficRule.rule_book_check(self.arrived_s, self.prev_s, self.current_d, self.prev_d,
                                                                              lane_change, off_road, collision_check, finished, v_kmh, anomaly_s, anomaly_d)
        self.rule_score_list_step.append(rule_score)
        self.prev_s = self.arrived_s

        rule_paras = [reward, self.arrived_s, self.current_d]
        self.rule_paras_ls.append(rule_paras)

        self.camera_v_ls.append(rgb_image)
        info["arrived_s"] = self.arrived_s
        info["finished_score"] = finished_score
        info["camera_v"] = self.camera_v_ls
        info["rule_paras"] = self.rule_paras_ls
        info["anomaly"] = [anomaly_s, anomaly_d, self.max_s]

        # # Set reward and 'done' flag
        # if len(self.collision_hist) != 0:
        #     print("Collided with v = " and v_kmh and " km/h")
        #     done = True
        #     reward = -200
        # elif v_kmh < 20:
        #     reward = -1
        # else:
        #     reward = 1
        self.time_step += 1

        if done:
            rule_score = np.sum(self.rule_score_list_step)
            for actor in self.actor_list:
                actor.destroy()
            self.actor_list = []

        info["rule_score"] = rule_score

        obs, _ = self.get_observation()

        return obs, reward, done, copy.deepcopy(info)

    def getFPS_Counter(self):
        return self.fps_counter

    def isTimeExpired(self):
        if self.fps_counter > self.max_fps:
            return True
        return False

    # perform a/multiple world tick
    def tick_world(self, times=1):
        for x in range(times):
            self.world.tick()
            self.fps_counter += 1

    def get_observation(self):
        """Observations in PyTorch format BCHW"""
        image_rgb = self.camera_rgb
        camera = image_rgb[:, :, None] if self.grayscale else image_rgb
        image_rgb = camera.transpose((2, 0, 1))  # from HWC to CHW
        image_rgb = np.ascontiguousarray(image_rgb, dtype=np.float32) / 255
        image_rgb = torch.from_numpy(image_rgb)
        image_rgb = image_rgb.unsqueeze(0)  # BCHW
        # camera = resize(image_rgb, (self.size[0], self.size[1])) * 255
        obs = image_rgb
        return obs, camera

    def close(self):
        print("Close")

    def __process_sensor_data(self, image):
        """Observations directly viewable with OpenCV in CHW format"""
        image.convert(carla.ColorConverter.CityScapesPalette)
        i = np.array(image.raw_data)
        i2 = i.reshape((IM_HEIGHT, IM_WIDTH, 4))
        i3 = i2[:, :, :3]
        self.observation = i3

    def __rgb_process_sensor_data(self, image):
        """RGB camera"""
        # image.convert(carla.ColorConverter.CityScapesPalette)
        i = np.array(image.raw_data)
        i = np.reshape(i, (256, 256, 4))
        i = i[:, :, :3]
        self.camera_rgb = i

    def __process_collision_data(self, event):
        self.collision_hist.append(event)

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
            # if counter > 1:
                # print("Spawning anomaly error: No anomaly this episode")
                # print(f"Actors: {len(self.world.get_actors())}")
                # break
            # counter += 1
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

    def __del__(self):
        for actor in self.actor_list:
            actor.destroy()
