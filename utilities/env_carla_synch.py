
##
# A snyched environment for sampling of the scenarios
##

import glob
import os
import sys
import json
import random
import numpy as np
import math
import time

import matplotlib.pyplot as plt
import cv2
import weakref

import torch

IM_WIDTH = 256
IM_HEIGHT = 256

BEV_DISTANCE = 20

N_ACTIONS = 9

RESET_SLEEP_TIME = 1

MIN_BBOX_SIZE = 0.5

EPISODE_TIME = 30
FIXED_DELTA_SECONDS = 0.05
SUBSTE_DELTA = 0.007
MAX_SUBSTEPS = 10

import carla

class Environment:

    def __init__(self, world=None, host='localhost', port=2000, s_width=IM_WIDTH, s_height=IM_HEIGHT,
                 cam_height=BEV_DISTANCE, cam_rotation=-90, cam_zoom=110, random_spawn=True, cam_x_offset=10.):
        weak_self = weakref.ref(self)
        self.client = carla.Client(host, port)            #Connect to server
        self.client.set_timeout(30.0)


        self.autoPilotOn = False
        self.random_spawn = random_spawn

        if not world == None: self.world = self.client.load_world(world)
        else: self.world = self.client.load_world("Town01_Opt")

        self.bp_lib = self.world.get_blueprint_library()
        self.map = self.world.get_map()
        self.goalPoint = None
        self.map_waypoints = self.map.generate_waypoints(3.0)
        self.spawn_points = self.map.get_spawn_points()
        self.spawn_point = None
        self.trajectory_list = None

        self.s_width = s_width
        self.s_height = s_height
        self.cam_height = cam_height
        self.cam_rotation = cam_rotation
        self.cam_zoom = cam_zoom
        self.cam_x_offset = cam_x_offset

        self.anomaly_point = None

        self.actor_list = []
        self.weather = carla.WeatherParameters(
            cloudiness=0.0,
            precipitation=0.0,
            precipitation_deposits= 0.0,
            wind_intensity=0.0,
            fog_density=0.0,   
            wetness=0.0,
            sun_altitude_angle=70.0)

        self.world.set_weather(self.weather)
        self.vehicle = None # important

        self.valid_anomalys = self.set_valid_anomalys()

        w_settings = self.world.get_settings()
        w_settings.synchronous_mode = True
        w_settings.fixed_delta_seconds = FIXED_DELTA_SECONDS # 10 fps | fixed_delta_seconds <= max_substep_delta_time * max_substeps
        w_settings.substepping = True
        w_settings.max_substep_delta_time = SUBSTE_DELTA
        w_settings.max_substeps = MAX_SUBSTEPS

        self.world.apply_settings(w_settings)
        self.fps_counter = 0
        self.max_fps = int(1/FIXED_DELTA_SECONDS) * EPISODE_TIME

    def init_ego(self):

        self.vehicle_bp = self.bp_lib.find('vehicle.tesla.model3')
        self.ss_camera_bp = self.bp_lib.find('sensor.camera.rgb')
        # self.ss_camera_bp_sg = self.bp_lib.find('sensor.camera.semantic_segmentation')
        self.col_sensor_bp = self.bp_lib.find('sensor.other.collision')

        # Configure rgb sensors
        self.ss_camera_bp.set_attribute('image_size_x', f'{self.s_width}')
        self.ss_camera_bp.set_attribute('image_size_y', f'{self.s_height}')
        self.ss_camera_bp.set_attribute('fov', str(self.cam_zoom))

        # Location for both sensors
        self.ss_cam_location = carla.Location(self.cam_x_offset,0,self.cam_height)
        self.ss_cam_rotation = carla.Rotation(self.cam_rotation,0,0)
        self.ss_cam_transform = carla.Transform(self.ss_cam_location, self.ss_cam_rotation)

        
        # collision sensor
        self.col_sensor_location = carla.Location(0,0,0)
        self.col_sensor_rotation = carla.Rotation(0,0,0)
        self.col_sensor_transform = carla.Transform(self.col_sensor_location, self.col_sensor_rotation)

        self.collision_hist = []



    def reset(self):

        self.deleteActors()
        
        self.actor_list = []
        self.collision_hist = []

        self.tick_world(times=5)

        # Spawn vehicle
        if self.random_spawn: 

            self.spawn_point = random.choice(self.spawn_points)
        else: self.spawn_point = self.spawn_points[1]

        self.vehicle = self.world.spawn_actor(self.vehicle_bp, self.spawn_point)
        if self.autoPilotOn: 
            self.vehicle.set_autopilot(self.autoPilotOn, self.tm_port)

        self.actor_list.append(self.vehicle)

        # Attach and listen to image sensor (RGB)
        self.ss_cam = self.world.spawn_actor(self.ss_camera_bp, self.ss_cam_transform, attach_to=self.vehicle, attachment_type=carla.AttachmentType.Rigid)
        self.actor_list.append(self.ss_cam)
        self.ss_cam.listen(lambda data: self.__process_sensor_data(data))

        # # Attach and listen to image sensor (Semantic Seg)
        # self.ss_cam_seg = self.world.spawn_actor(self.ss_camera_bp_sg, self.ss_cam_transform, attach_to=self.vehicle, attachment_type=carla.AttachmentType.Rigid)
        # self.actor_list.append(self.ss_cam_seg)
        # self.ss_cam_seg.listen(lambda data: self.__process_sensor_data_Seg(data))

        time.sleep(RESET_SLEEP_TIME)   # sleep to get things started and to not detect a collision when the car spawns/falls from sky.

        # Attach and listen to collision sensor
        self.col_sensor = self.world.spawn_actor(self.col_sensor_bp, self.col_sensor_transform, attach_to=self.vehicle)
        self.actor_list.append(self.col_sensor)
        self.col_sensor.listen(lambda event: self.__process_collision_data(event))

        self.tick_world(times=5)
        self.fps_counter = 0

        self.episode_start = time.time()
        return self.get_observation()

    def step(self, action):
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


        # Get velocity of vehicle
        v = self.vehicle.get_velocity()
        v_kmh = int(3.6 * math.sqrt(v.x**2 + v.y**2 + v.z**2))

        # Set reward and 'done' flag
        done = False
        if len(self.collision_hist) != 0:
            done = True
            reward = -100
        elif v_kmh < 20:
            reward = v_kmh / (80 - 3*v_kmh)
        else:
            reward = 1

        return self.get_observation(), reward, done, None

    
    def spawn_anomaly_ahead(self, distance=15):
        transform = self.get_Vehicle_transform() #get vehicle location and rotation (0-360 degrees)
        vec = transform.rotation.get_forward_vector()
        transform.location.x = transform.location.x + vec.x * distance
        transform.location.y = transform.location.y + vec.y * distance
        transform.location.z = transform.location.z + vec.z * distance
        self.spawn_anomaly(transform)


    def spawn_anomaly_alongRoad(self, max_numb, disposition_prob=0.7, max_lateral_disposition=2.5):
        count = 0
        spawn_failed = True
        while spawn_failed:
            if max_numb < 8: max_numb = 8
            ego_map_point = self.getEgoWaypoint() # closest map point to the spawn point
            wp_infront = [ego_map_point]
            for x in range(max_numb):
                wp_infront.append(wp_infront[-1].next(2.)[0])

            wp_infront = wp_infront[6:] # prevent spawning object on top of ego_vehicle
            s_index = random.randrange(0, len(wp_infront)-3)

            spawn =  wp_infront[s_index]
            spawn_location = spawn.transform.location

            if random.random() < disposition_prob:# and not spawn_location.distance(spawn.transform.location) == 0: # sometimes future waypoints are the same => no vector (weird)
                # calc orthogonal vector
                next_wp = wp_infront[s_index+2].transform.location
                dir_vector = np.array([spawn_location.x - next_wp.x, spawn_location.y - next_wp.y, spawn_location.z - next_wp.z])
                # dir_vector = dir_vector / np.linalg.norm(dir_vector) # normalize vector
                x_axis = (-dir_vector[1] - dir_vector[2]) / dir_vector[0] # v_1*v_2 = 0  =>  x_1 = (-x_2-x_3) / y_1 ||| y_2, y_3 = 1
                orthogonal_vector = np.array([x_axis*dir_vector[0], dir_vector[1], dir_vector[2]])
                orthogonal_vector = orthogonal_vector / np.linalg.norm(orthogonal_vector) # normalize vector
                # print(orthogonal_vector)
                # print(np.linalg.norm(orthogonal_vector))

                # disposition
                lateral_disposition = random.random() * max_lateral_disposition # strength of disposition
                if random.random() < 0.5: lateral_disposition = lateral_disposition * (-1) # left or right disposition
                orthogonal_vector = orthogonal_vector * lateral_disposition

                # apply to current location
                spawn_location.x += orthogonal_vector[0]
                spawn_location.y += orthogonal_vector[1]
                spawn_location.z += orthogonal_vector[2]

            rotation = spawn.transform.rotation
            spawn_location.z += 0.30 #prevent collision at start
            # print(ego_map_point)
            # print(carla.Transform(spawn_location, rotation))

            a_id, a_transform, a_object = self.spawn_anomaly(carla.Transform(spawn_location, rotation))
            count += 1
            if count > 100:
                print("Constantly failing to spawn object at different location! -> Abort")
                break
            if not a_object == None:
                spawn_failed = False

        return a_id, a_transform



    def spawn_anomaly(self, transform):
        anomaly_object = random.choice(self.valid_anomalys)
        # print(anomaly_object)
        player = self.world.try_spawn_actor(anomaly_object,transform)
        # self.get_valid_anomalys()


        if player == None: 
            print("!!! No anomaly spawned !!!")
        else: self.actor_list.append(player)

        self.anomaly_point = transform
        return anomaly_object.id, transform, player

    
    def get_valid_anomalys(self):
        valid_list = {}
        blueprints = self.bp_lib.filter('static.prop.*')
        transform = carla.Transform(carla.Location(0,0,0), carla.Rotation(0,0,0))
        counter = 0
        for object in blueprints:
            player = self.world.spawn_actor(object, transform)
            # print(player.attributes)
            if not player.attributes["role_name"] == "default" and not player.attributes["size"] == "tiny" and not player.attributes["size"] == "small":
                valid_list[f"object_{counter}"] = object.id
                counter += 1
            player.destroy()
        
        with open(f"anomaly_list.json", "w") as fp:
            json.dump(valid_list, fp, indent = 4)

    def set_valid_anomalys(self):
        with open('anomaly_list.json') as json_file:
            anomaly_list = json.load(json_file)
        
        bp_list = []
        for x in range(len(anomaly_list)): #len(anomaly_list)
            bp = self.bp_lib.filter(anomaly_list[f"object_{x}"])[0]
            bp_list.append(bp)

        print(f"~~~~~~~~~~~~~~\n# Utilizing {len(bp_list)} anomalies \n~~~~~~~~~~~~~~")
        return bp_list


    def set_goalPoint(self, max_numb):
        if max_numb < 4: max_numb = 4
        ego_map_point = self.getEgoWaypoint() # closest map point to the spawn point
        self.trajectory_list = [ego_map_point]
        for x in range(max_numb):
            self.trajectory_list.append(self.trajectory_list[-1].next(2.)[0])

        anomaly_spawn = self.trajectory_list[-1]
        location = anomaly_spawn.transform

        self.goalPoint = location

    
    def change_Weather(self):
        self.weather = carla.WeatherParameters(
            cloudiness=0.0,
            precipitation=70.0,
            precipitation_deposits= 0.0,
            wind_intensity=0.0,
            fog_density=70.0,
            fog_distance=3.0,
            wetness=0.0,
            sun_altitude_angle=70.0)

        self.world.set_weather(self.weather)

    def reset_Weather(self):
        self.weather = carla.WeatherParameters(
            cloudiness=0.0,
            precipitation=0.0,
            precipitation_deposits= 0.0,
            wind_intensity=0.0,
            fog_density=0.0,
            wetness=0.0,
            sun_altitude_angle=70.0)

        self.world.set_weather(self.weather)

    def makeRandomAction(self):
        v = random.random()
        if v <= 0.33333333:
            self.step(0)
        elif v <= 0.6666666:
            self.step(1)
        elif v <= 1.0:
            self.step(2)

    # plots the path from spawn to goal and anomaly boundingbox
    def plotTrajectory(self):
        lifetime=1.
        for x in range(len(self.trajectory_list)-1):
            w = self.trajectory_list[x]
            self.world.debug.draw_point(w.transform.location, size=0.2, life_time=lifetime, color=carla.Color(r=0, g=0, b=255))
        # color goal point in red   
        self.world.debug.draw_point(self.trajectory_list[-1].transform.location, size=0.3, life_time=lifetime, color=carla.Color(r=255, g=0, b=0))

        # color anomaly
        bounding_box_set = self.world.get_level_bbs(carla.CityObjectLabel.Dynamic) + self.world.get_level_bbs(carla.CityObjectLabel.Static) + self.world.get_level_bbs(carla.CityObjectLabel.Pedestrians) + self.world.get_level_bbs(carla.CityObjectLabel.Vehicles)
        best = 100000.
        bbox = None
        for box in bounding_box_set:
            distance = abs(box.location.distance(self.anomaly_point.location))
            if distance < best:
                best = distance
                bbox = box

        if bbox.extent.x < MIN_BBOX_SIZE: bbox.extent.x = MIN_BBOX_SIZE 
        if bbox.extent.y < MIN_BBOX_SIZE: bbox.extent.y = MIN_BBOX_SIZE 
        if bbox.extent.z < MIN_BBOX_SIZE: bbox.extent.z = MIN_BBOX_SIZE 


        # self.world.debug.draw_box(carla.BoundingBox(self.anomaly_point.get_transform().location,carla.Vector3D(3.5,3.5,4)),self.anomaly_point.get_transform().rotation, 0.3, carla.Color(255,140,0,0),-1.)
        self.world.debug.draw_box(bbox, self.anomaly_point.rotation, 0.15, carla.Color(r=0, g=0, b=0),lifetime)


    #Returns only the waypoints in one lane
    def single_lane(self, waypoint_list, lane):
        waypoints = []
        for i in range(len(waypoint_list) - 1):
            if waypoint_list[i].lane_id == lane:
                waypoints.append(waypoint_list[i])
        return waypoints

    def destroy_actor(self, actor):
        actor.destroy()

    def isActorAlive(self, actor):
        if actor.is_alive:
            return True
        return False
    
    def setAutoPilot(self, value):
        self.autoPilotOn = value
        print(f"### Autopilot: {self.autoPilotOn}")
        
# ==============================================================================
# -- Getters --------------------------------------------------------------------
# ==============================================================================

    def getSpawnPoint(self):
        return self.spawn_point

    def getGoalTrajectory(self):
        return self.trajectory_list
        
    def getGoalPoint(self):
        return self.goalPoint

    def get_Weather(self):
        wheather = self.world.get_weather()
        w_dict = {
            "cloudiness": wheather.cloudiness,
            "precipitation": wheather.precipitation,
            "precipitation_deposits": wheather.precipitation_deposits,
            "wind_intensity": wheather.wind_intensity,
            # "sun_azimuth_angle": wheather.sun_azimuth_angle,
            "sun_altitude_angle": wheather.sun_altitude_angle,
            "fog_density": wheather.fog_density,
            # "fog_distance": wheather.fog_distance,
            # "fog_falloff": wheather.fog_falloff,
            "wetness": wheather.wetness
            # "scattering_intensity": wheather.scattering_intensity,
            # "mie_scattering_scale": wheather.mie_scattering_scale,
            # "rayleigh_scattering_scale": wheather.rayleigh_scattering_scale
        }
        return w_dict

    def getEgoWaypoint(self):
        vehicle_loc = self.vehicle.get_location()
        wp = self.map.get_waypoint(vehicle_loc, project_to_road=True,
                      lane_type=carla.LaneType.Driving)

        return wp
    
    def getWaypoints(self):
        return self.map_waypoints

    #get vehicle location and rotation (0-360 degrees)
    def get_Vehicle_transform(self):
        return self.vehicle.get_transform()

    #get vehicle location
    def get_Vehicle_positionVec(self):
        position = self.vehicle.get_transform().location
        return np.array([position.x, position.y, position.z])

# ==============================================================================
# -- Sensor processing ---------------------------------------------------------
# ==============================================================================

    # perform a/multiple world tick
    def tick_world(self, times=1):
        for x in range(times):
            self.world.tick()
            self.fps_counter += 1

    # perform a/multiple world tick in seconds
    def tick_Seconds_world(self, seconds=1):
        times = int(seconds * int(1/FIXED_DELTA_SECONDS))
        for x in range(times):
            self.world.tick()
            self.fps_counter += 1

    def get_observation(self):
        """ Observations in PyTorch format BCHW """
        frame = self.observation
        frame = frame.astype(np.float32) / 255
        frame = self.arrange_colorchannels(frame)

        # seg = self.observation_seg
        # seg = seg.astype(np.float32)
        # seg = self.arrange_colorchannels(seg)
        # return frame, seg
        return frame,None

    def __process_sensor_data(self, image):
        """ Observations directly viewable with OpenCV in CHW format """
        # image.convert(carla.ColorConverter.CityScapesPalette)
        i = np.array(image.raw_data)
        i2 = i.reshape((self.s_height, self.s_width, 4))
        i3 = i2[:, :, :3]
        self.observation = i3

    # def __process_sensor_data_Seg(self, image):
    #     """ Observations directly viewable with OpenCV in CHW format """
    #     # image.convert(carla.ColorConverter.CityScapesPalette)
    #     i = np.array(image.raw_data)
    #     i2 = i.reshape((self.s_height, self.s_width, 4))
    #     i3 = i2[:, :, :3]
    #     self.observation_seg = i3

    def __process_collision_data(self, event):
        self.collision_hist.append(event)

    # changes order of color channels. Silly but works...
    def arrange_colorchannels(self, image):
        mock = image.transpose(2,1,0)
        tmp = []
        tmp.append(mock[2])
        tmp.append(mock[1])
        tmp.append(mock[0])
        tmp = np.array(tmp)
        tmp = tmp.transpose(2,1,0)
        return tmp
    
    def deleteActors(self):
        # if not self.vehicle == None:
        #     self.vehicle.set_autopilot(False, self.tm_port)

        for actor in self.actor_list:
            if self.isActorAlive(actor=actor):
                actor.destroy()       
        # self.client.apply_batch([carla.command.DestroyActor(x) for x in self.actor_list])
        # for actor in self.actor_list:
        #     if self.isActorAlive(actor=actor):
        #         print("!Actor destruction failed!")

    def __del__(self):
        print("__del__ called")