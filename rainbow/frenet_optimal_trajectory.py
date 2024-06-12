"""

Frenet optimal trajectory generator

author: Atsushi Sakai (@Atsushi_twi)

Ref:

- [Optimal Trajectory Generation for Dynamic Street Scenarios in a Frenet Frame](https://www.researchgate.net/profile/Moritz_Werling/publication/224156269_Optimal_Trajectory_Generation_for_Dynamic_Street_Scenarios_in_a_Frenet_Frame/links/54f749df0cf210398e9277af.pdf)

- [Optimal trajectory generation for dynamic street scenarios in a Frenet Frame](https://www.youtube.com/watch?v=Cj6tAQe7UCY)

"""

import numpy as np
import copy
import math
import cubic_spline_planner
# from config import cfg
import matplotlib.pyplot as plt

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


def euclidean_distance(v1, v2):
    return math.sqrt(sum([(a - b) ** 2 for a, b in zip(v1, v2)]))


def closest(lst, K):
    """
    Find closes value in a list
    """
    return lst[min(range(len(lst)), key=lambda i: abs(lst[i] - K))]


def frenet_to_inertial(s, d, csp):
    """
    transform a point from frenet frame to inertial frame
    input: frenet s and d variable and the instance of global cubic spline class
    output: x and y in global frame
    """
    ix, iy, iz = csp.calc_position(s)
    iyaw = csp.calc_yaw(s)
    x = ix + d * math.cos(iyaw + math.pi / 2.0)
    y = iy + d * math.sin(iyaw + math.pi / 2.0)

    return x, y, iz, iyaw


def update_frenet_coordinate(fpath, loc):
    """
    Finds best Frenet coordinates (s, d) in the path based on current position
    """

    min_e = float('inf')
    min_idx = -1
    for i in range(len(fpath.t)):
        e = euclidean_distance([fpath.x[i], fpath.y[i]], loc)
        if e < min_e:
            min_e = e
            min_idx = i

    if min_idx <= len(fpath.t) - 2:
        min_idx += 2  # +2 because if next wp gets too close to the ego, lat controller oscillates

    s, s_d, s_dd = fpath.s[min_idx], fpath.s_d[min_idx], fpath.s_dd[min_idx]
    d, d_d, d_dd = fpath.d[min_idx], fpath.d_d[min_idx], fpath.d_dd[min_idx]

    return s, s_d, s_dd, d, d_d, d_dd


class quintic_polynomial:

    def __init__(self, xs, vxs, axs, xe, vxe, axe, T):
        # calc coefficient of quintic polynomial
        self.xs = xs
        self.vxs = vxs
        self.axs = axs
        self.xe = xe
        self.vxe = vxe
        self.axe = axe

        self.a0 = xs
        self.a1 = vxs
        self.a2 = axs / 2.0

        A = np.array([[T ** 3, T ** 4, T ** 5],
                      [3 * T ** 2, 4 * T ** 3, 5 * T ** 4],
                      [6 * T, 12 * T ** 2, 20 * T ** 3]])
        b = np.array([xe - self.a0 - self.a1 * T - self.a2 * T ** 2,
                      vxe - self.a1 - 2 * self.a2 * T,
                      axe - 2 * self.a2])
        x = np.linalg.solve(A, b)

        self.a3 = x[0]
        self.a4 = x[1]
        self.a5 = x[2]

    def calc_point(self, t):
        xt = self.a0 + self.a1 * t + self.a2 * t ** 2 + \
             self.a3 * t ** 3 + self.a4 * t ** 4 + self.a5 * t ** 5

        return xt

    def calc_first_derivative(self, t):
        xt = self.a1 + 2 * self.a2 * t + \
             3 * self.a3 * t ** 2 + 4 * self.a4 * t ** 3 + 5 * self.a5 * t ** 4

        return xt

    def calc_second_derivative(self, t):
        xt = 2 * self.a2 + 6 * self.a3 * t + 12 * self.a4 * t ** 2 + 20 * self.a5 * t ** 3

        return xt

    def calc_third_derivative(self, t):
        xt = 6 * self.a3 + 24 * self.a4 * t + 60 * self.a5 * t ** 2

        return xt


class quartic_polynomial:

    def __init__(self, xs, vxs, axs, vxe, axe, T):
        # calc coefficient of quintic polynomial
        self.xs = xs
        self.vxs = vxs
        self.axs = axs
        self.vxe = vxe
        self.axe = axe

        self.a0 = xs
        self.a1 = vxs
        self.a2 = axs / 2.0

        A = np.array([[3 * T ** 2, 4 * T ** 3],
                      [6 * T, 12 * T ** 2]])
        b = np.array([vxe - self.a1 - 2 * self.a2 * T,
                      axe - 2 * self.a2])
        x = np.linalg.solve(A, b)

        self.a3 = x[0]
        self.a4 = x[1]

    def calc_point(self, t):
        xt = self.a0 + self.a1 * t + self.a2 * t ** 2 + \
             self.a3 * t ** 3 + self.a4 * t ** 4

        return xt

    def calc_first_derivative(self, t):
        xt = self.a1 + 2 * self.a2 * t + \
             3 * self.a3 * t ** 2 + 4 * self.a4 * t ** 3

        return xt

    def calc_second_derivative(self, t):
        xt = 2 * self.a2 + 6 * self.a3 * t + 12 * self.a4 * t ** 2

        return xt

    def calc_third_derivative(self, t):
        xt = 6 * self.a3 + 24 * self.a4 * t

        return xt


class Frenet_path:
    def __init__(self):
        self.id = None
        self.t = []
        self.d = []
        self.d_d = []
        self.d_dd = []
        self.d_ddd = []
        self.s = []
        self.s_d = []
        self.s_dd = []
        self.s_ddd = []
        self.cd = 0.0
        self.cv = 0.0
        self.cf = 0.0

        self.x = []
        self.y = []
        self.z = []
        self.yaw = []
        self.ds = []
        self.c = []

        self.v = []  # speed


class FrenetPlanner:
    def __init__(self):

        if float(0.1) > 0:
            self.dt = float(0.1)
        else:
            self.dt = 0.05

        # Parameters
        self.MAX_SPEED = 60.0 / 3.6  # maximum speed [m/s]
        self.MAX_ACCEL = 5.0  # maximum acceleration [m/ss]  || Tesla model 3: 6.878
        self.MAX_CURVATURE = 100.0  # maximum curvature [1/m]
        self.LANE_WIDTH = 3.5 # float(cfg.CARLA.DT)
        self.MAXT = 4.0  # max prediction time [m]
        self.MINT = 2.0  # min prediction time [m]
        self.D_T = 2.0  # prediction timestep length (s)
        self.D_T_S = 20.0 / 3.6  # target speed sampling length [m/s]
        self.N_S_SAMPLE = 1  # sampling number of target speed
        self.ROBOT_RADIUS = 2.0  # robot radius [m]
        self.MAX_DIST_ERR = 4.0  # max distance error to update frenet states based on ego states

        # cost weights
        self.KJ = 0.1
        self.KT = 0.1
        self.KD = 1.0
        self.KLAT = 1.0
        self.KLON = 1.0

        self.path = None  # current frenet path
        self.ob = []  # n obstacles [[x1, y1, z1], [x2, y2, z2], ... ,[xn, yn, zn]]
        self.csp = None  # cubic spline for global rout
        self.steps = 0  # planner steps

        self.targetSpeed = 5 #float(cfg.GYM_ENV.TARGET_SPEED)
        min_speed = 4 #float(cfg.LOCAL_PLANNER.MIN_SPEED)
        max_speed = 6 #float(cfg.LOCAL_PLANNER.MAX_SPEED)
        self.speed_center = (max_speed + min_speed) / 2
        self.speed_radius = (max_speed - min_speed) / 2

    def update_global_route(self, global_route):
        """
        fit an spline to the updated global route in inertial frame
        """
        wx = []
        wy = []
        wz = []
        for p in global_route:
            wx.append(p[0])
            wy.append(p[1])
            wz.append(p[2])
        self.csp = cubic_spline_planner.Spline3D(wx, wy, wz)
        return self.csp.s[-1]

    def update_obstacles(self, ob):
        self.ob = ob

    def estimate_frenet_state(self, ego_state, idx):
        """
        estimate the frenet state based on ego state and the current frenet state
        procedure: - initialize the estimation with the last frenet state
                   - check the error btw ego true position and the frenet's estimation of global position
                   - if error larger than threshold, update the frenet state

        """
        # Frenet state estimation [s, s_d, s_dd, d, d_d, d_dd]
        f_state = [self.path.s[idx], self.path.s_d[idx], self.path.s_dd[idx],
                   self.path.d[idx], self.path.d_d[idx], self.path.d_dd[idx]]

        if f_state[0] == 0:
            return f_state

        # update_frenet_coordinate(self.path, ego_state[:2])

        def normalize(vector):
            if sum(vector) == 0:
                return [0 for _ in range(len(vector))]
            return vector / np.sqrt(sum([n ** 2 for n in vector]))

        def magnitude(vector):
            return np.sqrt(sum([n ** 2 for n in vector]))

        # ------------------------ UPDATE S VALUE ------------------------------------ #
        # We calculate normal vector of s line and find error_s based on ego location. Note: This assumes error is small angle
        def update_s(current_s):
            s_yaw = self.csp.calc_yaw(current_s)
            s_x, s_y, s_z = self.csp.calc_position(current_s)
            ego_yaw = ego_state[4]
            s_norm = normalize([-np.sin(s_yaw), np.cos(s_yaw)])
            v1 = [ego_state[0] - s_x, ego_state[1] - s_y]
            v1_norm = normalize(v1)
            angle = np.arccos(np.clip(np.dot(s_norm, v1_norm),-1.0, 1.0))
            delta_s = np.sin(angle) * magnitude(
                v1)  # Since we use last coordinate of trajectory as possible ego location we know actual location is behind most of the time
            # print("delta_s:{}".format(delta_s))
            return delta_s

        estimated_s = self.path.s[idx] % ego_state[6]
        estimated_s -= update_s(estimated_s)
        if estimated_s < 0:
            estimated_s = -estimated_s
        estimated_s = estimated_s % ego_state[6]
        estimated_s += update_s(estimated_s)
        estimated_s = estimated_s % ego_state[6]

        # ------------------------- UPDATING D VALUE -------------------------------- #
        # after we update s value now we can update d value based on new coordinate

        s_yaw = self.csp.calc_yaw(estimated_s)
        s_norm = normalize([-np.sin(s_yaw), np.cos(s_yaw)])
        s_x, s_y, s_z = self.csp.calc_position(estimated_s)
        v1 = [ego_state[0] - s_x, ego_state[1] - s_y]
        v1_norm = normalize(v1)
        angle = np.arccos(np.clip(np.dot(s_norm, v1_norm),-1.0, 1.0))
        d = np.cos(angle) * magnitude(v1)

        # print("S_ego:{},S:{},angle:{}".format(f_state[0], f_state[0] + delta_s, angle))
        # print("d_ego:{}, d:{}".format(f_state[3], d))
        # ---------------------- UPDATE S_D D_D --------------------------------------- #
        a_v_norm = normalize([ego_state[5][0].x, ego_state[5][0].y])
        angle_vel = np.arccos(np.clip(np.dot(a_v_norm, s_norm),-1.0,1.0))
        s_d = np.sin(angle_vel) * ego_state[2]
        d_d = np.cos(angle_vel) * ego_state[2]
        # ---------------------- UPDATE S_DD D__DD -------------------------------------#
        # angle_acc = np.arccos(np.dot(normalize([ego_state[5][1].x, ego_state[5][1].y]), s_norm))
        # s_dd = np.sin(angle_acc) * ego_state[3]
        # d_dd = np.cos(angle_acc) * ego_state[3]

        # print("ego_d:{}, cal_d:{}".format([f_state[1], f_state[4]], [s_d, d_d]))
        # print("ego_dd:{}, cal_dd:{}".format([f_state[2], f_state[5]], [s_dd, d_dd]))
        # print("{}---{}".format(ego_yaw-s_yaw,angle_vel))

        f_state[0] = estimated_s #if estimated_s != None else 0
        f_state[3] = d #if d != None else 0
        f_state[1] = s_d #if s_d != None else 0
        f_state[4] = d_d #if d_d != None else 0
        # f_state[2] = s_dd
        # f_state[5] = d_dd

        return f_state
    
    def estimate_frenet_state_beta(self, ego_state):
        """
        estimate the frenet state based on ego state and the current frenet state

        """
        x = ego_state[0]
        y = ego_state[1]

        s = None
        min_distance = np.inf

        for s_value in np.linspace(0, self.csp.s[-1], num=1000):  # Adjust the num value for desired accuracy
            cx, cy, _ = self.csp.calc_position(s_value)
            distance = np.sqrt((cx - x) ** 2 + (cy - y) ** 2)

            if distance < min_distance:
                min_distance = distance
                s = s_value

        # Calculate orthogonal distance 'd'
        cx, cy, _ = self.csp.calc_position(s)
        dx = abs(x - cx)
        dy = abs(y - cy)
        yaw = self.csp.calc_yaw(s)
        d = dx * np.sin(yaw) + dy * np.cos(yaw)
        if s == 0 and min_distance > 0.5:
            s = -1
        if s > 0:
            cross_track_error = (x - cx) * np.sin(yaw) - (y - cy) * np.cos(yaw)
            if cross_track_error > 0:
                d = abs(d)
            else:
                d = -abs(d)
        return s, d

    def generate_single_frenet_path(self, f_state, df=0, Tf=4, Vf=30 / 3.6):
        """
        generate a single frenet path based on the current and terminal frenet state values
        input: ego's current frenet state and terminal frenet values (lateral displacement, time of arrival, and speed)
        output: single frenet path
        """
        s, s_d, s_dd, d, d_d, d_dd = f_state

        fp = Frenet_path()
        lat_qp = quintic_polynomial(d, d_d, d_dd, df, 0.0, 0.0, Tf)
        lon_qp = quartic_polynomial(s, s_d, s_dd, Vf, 0.0, Tf)

        for t in np.arange(0.0, Tf, self.dt):
            fp.t.append(t)
            fp.d.append(lat_qp.calc_point(t))
            fp.d_d.append(lat_qp.calc_first_derivative(t))
            fp.d_dd.append(lat_qp.calc_second_derivative(t))
            fp.d_ddd.append(lat_qp.calc_third_derivative(t))

            fp.s.append(lon_qp.calc_point(t))
            fp.s_d.append(lon_qp.calc_first_derivative(t))
            fp.s_dd.append(lon_qp.calc_second_derivative(t))
            fp.s_ddd.append(lon_qp.calc_third_derivative(t))

        fp = self.calc_global_paths([fp])[0]

        return fp

    def calc_frenet_paths(self, f_state, change_lane=0, target_speed=30 / 3.6):
        """
        generate lattices - discretized candidate frenet paths
        input: ego's current frenet state and actions
        output: list of candidate frenet paths
        """
        s, s_d, s_dd, d, d_d, d_dd = f_state

        # clip for feasible target lane numbers
        target_d = np.clip(d + change_lane * self.LANE_WIDTH, -self.LANE_WIDTH, 2 * self.LANE_WIDTH)
        frenet_paths = []

        # generate path to each offset goal
        path_id = 0
        for di in [d, target_d, -target_d]:

            # Lateral motion planning
            for Ti in np.arange(self.MINT, self.MAXT + self.D_T, self.D_T):
                fp = Frenet_path()
                lat_qp = quintic_polynomial(d, d_d, d_dd, di, 0.0, 0.0, Ti)

                for t in np.arange(0.0, Ti, self.dt):
                    fp.t.append(t)
                    fp.d.append(lat_qp.calc_point(t))
                    fp.d_d.append(lat_qp.calc_first_derivative(t))
                    fp.d_dd.append(lat_qp.calc_second_derivative(t))
                    fp.d_ddd.append(lat_qp.calc_third_derivative(t))

                # Longitudinal motion planning (Velocity keeping)
                for tv in np.arange(target_speed - self.D_T_S * self.N_S_SAMPLE,
                                    target_speed + self.D_T_S * self.N_S_SAMPLE, self.D_T_S):
                # for tv in [target_speed]:
                    tfp = copy.deepcopy(fp)
                    tfp.id = path_id
                    path_id += 1

                    lon_qp = quartic_polynomial(s, s_d, s_dd, tv, 0.0, Ti)

                    for t in tfp.t:
                        tfp.s.append(lon_qp.calc_point(t))
                        tfp.s_d.append(lon_qp.calc_first_derivative(t))
                        tfp.s_dd.append(lon_qp.calc_second_derivative(t))
                        tfp.s_ddd.append(lon_qp.calc_third_derivative(t))

                    Jp = sum(np.power(tfp.d_ddd, 2))  # square of jerk
                    Js = sum(np.power(tfp.s_ddd, 2))  # square of jerk

                    # square of diff from target speed
                    ds = (target_speed - tfp.s_d[-1]) ** 2

                    tfp.cd = self.KJ * Jp + self.KT * Ti + self.KD * (tfp.d[-1] - target_d) ** 2
                    tfp.cv = self.KJ * Js + self.KT * Ti + self.KD * ds
                    tfp.cf = self.KLAT * tfp.cd + self.KLON * tfp.cv

                    frenet_paths.append(tfp)
        return frenet_paths

    def calc_global_paths(self, fplist):
        """
        transform paths from frenet frame to inertial frame
        input: path list
        output: path list
        """
        for fp in fplist:

            # calc global positions
            for i in range(len(fp.s)):
                # print("fp.s[i]", i, fp.s[i])
                ix, iy, iz = self.csp.calc_position(fp.s[i])
                if ix is None:
                    break
                iyaw = self.csp.calc_yaw(fp.s[i])
                di = fp.d[i]
                fx = ix + di * math.cos(iyaw + math.pi / 2.0)
                fy = iy + di * math.sin(iyaw + math.pi / 2.0)
                fz = iz
                fp.x.append(fx)
                fp.y.append(fy)
                fp.z.append(fz)
                fp.yaw.append(iyaw)

        return fplist

    def calc_curvature_paths(self, fplist):
        """
        transform paths from frenet frame to inertial frame
        input: path list
        output: path list
        """
        for fp in fplist:

            # find curvature
            # source: http://www.kurims.kyoto-u.ac.jp/~kyodo/kokyuroku/contents/pdf/1111-16.pdf
            # and https://math.stackexchange.com/questions/2507540/numerical-way-to-solve-for-the-curvature-of-a-curve
            fp.c.append(0.0)
            for i in range(1, len(fp.t) - 1):
                a = np.hypot(fp.x[i - 1] - fp.x[i], fp.y[i - 1] - fp.y[i])
                b = np.hypot(fp.x[i] - fp.x[i + 1], fp.y[i] - fp.y[i + 1])
                c = np.hypot(fp.x[i + 1] - fp.x[i - 1], fp.y[i + 1] - fp.y[i - 1])

                # Compute inverse radius of circle using surface of triangle (for which Heron's formula is used)
                k = np.sqrt((a + (b + c)) * (c - (a - b)) * (c + (a - b)) * (
                        a + (b - c))) / 4  # Heron's formula for triangle's surface
                den = a * b * c  # Denumerator; make sure there is no division by zero.
                if den == 0.0:  # Very unlikely, but just to be sure
                    fp.c.append(0.0)
                else:
                    fp.c.append(4 * k / den)
            fp.c.append(0.0)

        return fplist

    def check_collision(self, fp, ob):
        """
        check if a frenet path makes collision with obstacles
        input: frenet path
        output: True/False
        """
        if len(ob) == 0:
            return True
        for i in range(len(ob)):
            d = [euclidean_distance([x, y, z], ob[i]) for (x, y, z) in zip(fp.x, fp.y, fp.z)]
            collision = any([di <= self.ROBOT_RADIUS ** 2 for di in d])

            if collision:
                return False

        return True

    def check_paths(self, fplist):
        """
        check for collisions
        input: list of frenet paths
        output: list of frenet paths - removed the infeasible ones
        """
        okind = []
        for i in range(len(fplist)):
            if any([v > self.MAX_SPEED for v in fplist[i].s_d]):  # Max speed check
                print('speed')
                continue
            elif any([abs(a) > self.MAX_ACCEL for a in fplist[i].s_dd]):  # Max accel check
                print('acc')
                continue
            elif any([abs(c) > self.MAX_CURVATURE for c in fplist[i].c]):  # Max curvature check
                print('cur')
                continue
            elif not self.check_collision(fplist[i], self.ob):
                print('col')
                continue

            okind.append(i)

        return [fplist[i] for i in okind]

    def frenet_optimal_planning(self, f_state, change_lane=0, target_speed=30 / 3.6):
        """
        input: current frenet state and actions
        output: candidate frenet paths and index of the optimal path
        process:
                - generate candidate frenet paths
                - calculate the inertial (global) trajectories
                - remove infeasible paths (those who make collisions)
                - find the optimal path based on cost values
        """

        fplist = self.calc_frenet_paths(f_state, change_lane=change_lane, target_speed=target_speed)
        fplist = self.calc_global_paths(fplist)
        fplist = self.calc_curvature_paths(fplist)
        fplist = self.check_paths(fplist)

        # find minimum cost path
        mincost = float("inf")
        bestpath_idx = None
        for i, fp in enumerate(fplist):
            if mincost >= fp.cf:
                mincost = fp.cf
                bestpath_idx = i

        return bestpath_idx, fplist

    def start(self, route):
        self.steps = 0
        max_s = self.update_global_route(route)
        return max_s

    def reset(self, s, d, df_n=0, Tf=4, Vf_n=0, optimal_path=False):
        # module_world reset should be executed beforehand to update the initial s and d values
        f_state = [s, 0, 0, d, 0, 0]

        if optimal_path:
            best_path_idx, fplist = self.frenet_optimal_planning(f_state)
            self.path = fplist[best_path_idx]
        else:
            # convert action values from range (-1, 1) to the desired range
            df = np.clip(np.round(df_n) * self.LANE_WIDTH + d, -self.LANE_WIDTH, 2 * self.LANE_WIDTH).item()

            speedRange = 10 / 3.6
            Vf = Vf_n * speedRange + self.targetSpeed

            self.path = self.generate_single_frenet_path(f_state, df=df, Tf=Tf, Vf=Vf)

    def run_step(self, ego_state, idx, change_lane=0, target_speed=30 / 3.6):
        """
        change lane: -1: go to left lane; 0: stay in current lane; 1: go to right lane;
        """
        self.steps += 1
        # t0 = time.time()

        f_state = self.estimate_frenet_state(ego_state, idx)

        # Frenet motion planning
        best_path_idx, fplist = self.frenet_optimal_planning(f_state, change_lane=change_lane,
                                                             target_speed=target_speed)
        print('best path index:', best_path_idx)
        self.path = fplist[best_path_idx]
        # print('trajectory planning time: {} s'.format(time.time() - t0))
        return self.path, fplist

    def run_step_single_path(self, ego_state, idx, df_n=0, Tf=4, Vf_n=0):
        """
        input: ego states, current frenet path's waypoint index, actions
        output: frenet path
        actions: final values for frenet lateral displacement (d), time, and speed
        """
        self.steps += 1

        # estimate frenet state
        f_state = self.estimate_frenet_state(ego_state, idx)
        d = f_state[3]
        # print("estimate f state", f_state[0], f_state[3])
        if abs(d) < 1.2:
            df_values = [0, 0.5, 1, -0.5, -1, -1.5, -2]
            if 0 <= df_n < len(df_values):
                df = df_values[df_n]
            else:
                df = 0
            _df = np.clip(df * self.LANE_WIDTH + d, -1.6 * self.LANE_WIDTH, 1.6 * self.LANE_WIDTH).item()
            df = closest([0.5 * self.LANE_WIDTH * lane_n for lane_n in range(-3, 4)], _df)
        else:
            if df_n == 0 or df_n == 1 or df_n == 2:
                df = d
            else:
                df = 0
        # print("d in motion planner: ", d)
        # print("df in motion planner:", df)
        # df = np.round(df_n[0]) * self.LANE_WIDTH + d  # allows agent to drive off the road
        # lanechange should be set true if there is a lane change
        lanechange = True if abs(df - d) >= 1.2 else False

        # off-the-road attempt is recorded
        off_the_road = True if df >= 1.75 or df <= -5.25 else False

        Vf = self.speed_radius * Vf_n + self.speed_center

        # Frenet motion planning
        self.path = self.generate_single_frenet_path(f_state, df=df, Tf=Tf, Vf=Vf)

        return self.path, lanechange, off_the_road


class threeDvector:
    def __init__(self, x, y, z):
        self.x = x
        self.y = y
        self.z = z


def main():
    velocity = threeDvector(0, 0, 0)
    acceleration = threeDvector(0, 0, 0)
    temp = [velocity, acceleration]
    speed = 0
    acc = 0
    psi = 1.5707971257432236 # math.pi
    max_s = 80
    init_route = np.load('/../road_map/global_route_town01.npy')
    init_route = init_route[::-1]
    x = 5
    # x = init_route[1][0] + 2
    y = 323
    # y = init_route[1][1]
    ego_state = [x, y, speed, acc, psi, temp,
                     max_s]

    for i in range(len(init_route)):
        plt.plot(init_route[i][0], init_route[i][1], 'bo', markersize=2)
    planner = FrenetPlanner()
    max_s = planner.start(init_route)

    planner.reset(0, 0)
    plt.plot(planner.path.x, planner.path.y)
    s, d = planner.estimate_frenet_state_beta(ego_state)
    print("initial route: ", init_route)
    print("s, d: ", [s, d])
    path, _, _ = planner.run_step_single_path(ego_state=ego_state, idx=1, df_n=1, Tf=4, Vf_n=-2)
    # path_opti, path_list = planner.run_step(ego_state, idx=0, change_lane=1)
    # for path in path_list:

    ego = [[x, y] for x, y in zip(path.x, path.y)]
    for ego_pos in ego:
        s, d = planner.estimate_frenet_state_beta(ego_pos)
        print("path x, y: ", ego_pos,"s, d: ", s, d)
    plt.plot(path.x, path.y, "-b")
    plt.axis('equal')
    plt.show()

    # fig, ax = plt.subplots(len(path_list), 1)
    # fig.suptitle('Path list')
    # for i in range(len(path_list)):
    #     ax[i].plot(path_list[i].x, path_list[i].y)
    # plt.show()


if __name__ == "__main__":
    main()
