#!/usr/bin/env python3

import rospy
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Path, Odometry
from geometry_msgs.msg import Twist
import dynamic_reconfigure.client

import os
import scipy
import numpy as np

import torch

from LfD_main import TrainingParams, LfD_2D_model

update_dt = 0.04
local_goal_dist = 1.5
local_path_dir_dist = 0.5
laser_max_range = 2.0

a_min = -180 # -135
a_max = 180 # 135
lidar_dim = 720
dt = 0.01
step = 20
lidar_tolerance = 0.00

a_min = a_min * np.pi / 180
a_max = a_max * np.pi / 180
da = (a_max - a_min) / (lidar_dim - 1)


class Predictor:
    def __init__(self, policy, params):
        self.policy = policy
        self.params = params

        self.global_path = []
        self.local_goal = None
        self.local_path_dir = None
        self.raw_scan = None
        self.clipped_scan = None

        self.turn_flag = 0
        self.v = 0
        self.w = 0

        # define the boundary of the vehicle
        self.width = 0.165 * 2
        self.length = 0.21 * 2
        self.y_top = 0.165
        self.y_bottom = -0.165
        self.x_right = 0.21
        self.x_left = -0.21
        n = 10
        self.boundary = np.zeros((4 * n, 2))
        xx = np.linspace(self.x_left, self.x_right, n)
        yy = np.linspace(self.y_bottom, self.y_top, n)
        # order is U, B, L, R
        self.boundary[:n][:, 0] = xx
        self.boundary[n : 2 * n][:, 0] = xx
        self.boundary[2 * n : 3 * n][:, 0] = self.x_left
        self.boundary[3 * n :][:, 0] = self.x_right
        self.boundary[:n][:, 1] = self.y_top
        self.boundary[n : 2 * n][:, 1] = self.y_bottom
        self.boundary[2 * n : 3 * n][:, 1] = yy
        self.boundary[3 * n :][:, 1] = yy

    def update_status(self, msg):
        # print("update_status")
        q1 = msg.pose.pose.orientation.x
        q2 = msg.pose.pose.orientation.y
        q3 = msg.pose.pose.orientation.z
        q0 = msg.pose.pose.orientation.w
        self.X = msg.pose.pose.position.x
        self.Y = msg.pose.pose.position.y
        self.PSI = np.arctan2(2 * (q0 * q3 + q1 * q2), (1 - 2 * (q2 ** 2 + q3 ** 2)))

    @staticmethod
    def transform_lg(gp, X, Y, PSI):
        R_r2i = np.matrix(
            [[np.cos(PSI), -np.sin(PSI), X], [np.sin(PSI), np.cos(PSI), Y], [0, 0, 1]]
        )
        R_i2r = np.linalg.inv(R_r2i)

        pi = np.concatenate([gp, np.ones_like(gp[:, :1])], axis=-1)
        pr = np.matmul(R_i2r, pi.T)
        return np.asarray(pr[:2, :]).T

    def update_global_path(self, msg):
        # print("update_global_path")

        gp = []
        for pose in msg.poses:
            gp.append([pose.pose.position.x, pose.pose.position.y])
        gp = np.array(gp)
        x = gp[:, 0]

        # print("update_global_path 1")

        # print('x: ', x)

        xhat = x

        # try:
        #     xhat = scipy.signal.savgol_filter(x, 19, 3)
        # except:
        #     xhat = x

        # print("update_global_path 2")


        y = gp[:, 1]
        # try:
        #     yhat = scipy.signal.savgol_filter(y, 19, 3)
        # except:
        #     yhat = y

        yhat = y

        # print("update_global_path 3")


        gphat = np.column_stack((xhat, yhat))
        gphat.tolist()
        self.global_path = self.transform_lg(gphat, self.X, self.Y, self.PSI)
        self.local_goal = self.get_local_goal(self.global_path)
        print("local_goal: ", self.local_goal)
        self.local_path_dir = self.get_local_path_dir(self.global_path)

    def get_local_goal(self, gp):
        local_goal = np.zeros(2)
        odom = np.zeros(2)
        if len(gp) > 0:
            if np.linalg.norm(gp[0] - odom) > 0.05:
                odom = gp[0]
            for wp in gp:
                dist = np.linalg.norm(wp - odom)
                if dist > self.params.local_goal_dist:
                    break
            local_goal = wp - odom
            local_goal /= np.linalg.norm(local_goal)

        return local_goal.astype(np.float32)

    def get_local_path_dir(self, gp):
        lp_dir = 0
        prev_wp = np.zeros(2)
        start = np.zeros(2)
        cum_dist = 0
        if len(gp) > 0:
            if np.linalg.norm(gp[0] - prev_wp) > 0.05:
                prev_wp = gp[0]
                start = gp[0]
            for wp in gp:
                wp_start_dist = np.linalg.norm(wp - start)
                if wp_start_dist > 0.1:
                    lp_dir += (wp - start) / wp_start_dist
                cum_dist += np.linalg.norm(wp - prev_wp)
                prev_wp = wp
                if cum_dist > local_path_dir_dist:
                    break
            if isinstance(lp_dir, np.ndarray):
                lp_dir = np.arctan2(lp_dir[1], lp_dir[0])

        return lp_dir

    def bch_safety_check(self, v, w, size, std):

        V = v + np.random.randn(size) * std * v
        W = w + np.random.randn(size) * std * w
        X = 0
        Y = 0
        PSI = 0
        for i in range(step):
            X_DOT = V * np.cos(PSI)
            Y_DOT = V * np.sin(PSI)
            PSI_DOT = W
            X = X + X_DOT * dt
            Y = Y + Y_DOT * dt
            PSI = PSI + PSI_DOT * dt

        # V [B, 1]
        # W [B, 1]
        # PSI [B, 1]
        # X [B, 1]
        # Y [B, 1]
        R_r2i = np.concatenate(
            [np.cos(PSI), -np.sin(PSI), np.sin(PSI), np.cos(PSI)]
        ).reshape(
            -1, 2, 2
        )  # [B, 2, 2]

        # R * x = (x^T R^T) => [1,3] x [3, 3]^T => [1, 3]
        N = self.boundary.shape[0]
        rotation = np.matmul(
            R_r2i.reshape(-1, 1, 2, 2), self.boundary.reshape(1, -1, 2, 1)
        ).reshape(
            -1, N, 2
        )  # [B, N, 2]
        translation = np.concatenate([X, Y], -1).reshape(-1, 1, 2)  # [B, 1, 2]
        boundary = rotation + translation  # [B, N, 2]

        XP, YP = boundary[:, :, 0], boundary[:, :, 1]
        beam_idx = ((np.arctan2(YP, XP) - a_min) // da).astype(np.int32)
        valid = ((beam_idx >= 0) & (beam_idx < lidar_dim)).astype(np.int32)
        beam_idx = beam_idx * valid
        RHO = np.sqrt(np.square(boundary)).sum(2)  # [B, N] the distance
        RHO_beam = self.raw_scan[beam_idx]  # [B, N]

        crash = np.sign(((RHO_beam < RHO + lidar_tolerance) * valid).sum(-1))  # [B]
        safety_percentage = 1 - crash.sum() / float(crash.shape[0])
        return safety_percentage

    def update_laser(self, msg):

        # interpolate from 512 to 720

        incoming_scan = np.array(msg.ranges)

        max_range = 5.0 # rospy.get_param("/laser/range")
        incoming_scan[np.isnan(incoming_scan)] = max_range
        incoming_scan[np.isinf(incoming_scan)] = max_range

        self.scan_size_des = 720
        self.scan_angle_increment_des = 2 * np.pi / (self.scan_size_des - 1)

        range_size = len(incoming_scan)
        angle_increment_orig = 2 * np.pi / (range_size - 1)

        self.scan_tmp = np.zeros(self.scan_size_des)
        for i in range(0, self.scan_size_des):
            theta = (i - self.scan_size_des/2) * self.scan_angle_increment_des
            
            idx_low = int(np.floor( (theta + np.pi) / angle_increment_orig))
            # print("orig idx_low: ", idx_low)
            idx_low = np.max([0, idx_low])
            # print("clip idx_low: ", idx_low)

            theta_low = (idx_low - range_size/2) * angle_increment_orig
            # print("theta_low: ", theta_low)

            idx_high = int(np.ceil( (theta + np.pi) / angle_increment_orig))
            # print("orig idx_high: ", idx_high)

            idx_high = np.min([range_size - 1, idx_high])
            # print("clip idx_high: ", idx_high)

            theta_high = (idx_high - range_size/2) * angle_increment_orig
            # print("theta_high: ", theta_high)

            # print("scan_data[idx_low]: ", scan_data[idx_low])
            # print("scan_data[idx_high]: ", scan_data[idx_high])

            self.scan_tmp[i] = incoming_scan[idx_low] +  \
                                ((theta - theta_low) / (0.000000000001 + theta_high - theta_low)) * (incoming_scan[idx_high] - incoming_scan[idx_low])
        self.raw_scan = self.scan_tmp

        # max_range = 5.0
        self.clipped_scan = np.minimum(
            self.raw_scan, max_range
        ).astype(np.float32)

    def update_cmd_vel(
        self,
    ):
        # print('update_cmd_vel')    

        if self.clipped_scan is None:
            return
        
        # print('clipped scan exists')    

        if self.local_goal is None:
            return

        # print('local goal exists')    

        # if needs hard turn
        try:
            turning_threshold = np.pi / 3  # this will depend on max_v
            stop_turning_threshold = np.pi / 18
            direction_angle = self.local_path_dir
            # print(direction_angle)
            if self.turn_flag == 0:
                if direction_angle > turning_threshold:
                    print("Hard Turn Left")
                    self.turn_flag = 1
                elif direction_angle < -turning_threshold:
                    print("Hard Turn Right")
                    self.turn_flag = -1
                else:
                    print("Normal Operation")
                    self.turn_flag = 0
            else:
                if abs(direction_angle) < stop_turning_threshold:
                    # print("Resume Normal Operation")
                    self.turn_flag = 0
        except:
            self.turn_flag = 0

        if self.turn_flag == 1:
            self.v = 0
            self.w = 2 * direction_angle
        elif self.turn_flag == -1:
            self.v = 0
            self.w = 2 * direction_angle
        else:
            scan = torch.from_numpy(self.clipped_scan[None]).to(self.params.device)
            local_goal = torch.from_numpy(self.local_goal[None]).to(self.params.device)

            # print("scan: ", scan)
            # print("local_goal: ", local_goal)


            cmd = self.policy(scan, local_goal)

            # print("cmd: ", cmd)

            cmd = cmd[0].detach().cpu().numpy()  # remove batch size
            if not np.any(np.isnan(cmd)):
                self.v, self.w = cmd

        # safety check
        ctr = 0  # how many recover count
        # while self.safety_check() == False:
        while self.bch_safety_check(self.v, self.w, 1, 0) == 0:
            # new recovery: just turn in place
            if ctr < 1:
                print("Recovery: Turn in Place")
                self.v = 0
                self.w = 2 * direction_angle
                # if self.w > 0 and self.w < 0.1:
                #     self.w = 0.1
                # elif self.w < 0 and self.w >-0.1:
                #     self.w = -0.1
                ctr += 1
            elif ctr >= 1:
                print("Recovery: Back up")
                self.v = -0.2
                self.w = 0
                break

        # print("[INFO] current v: {:4.2f}, w: {:5.2f}".format(self.v, self.w))


if __name__ == "__main__":

    print("LfLH main")

    repo_path = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
    folder_path = os.path.join("interesting_models", "LfD_2D", "2m_per_sec")
    params_path = os.path.join(repo_path, folder_path, "params.json")
    model_path = os.path.join(repo_path, folder_path, "trained_models", "model_1000")

    
    params = TrainingParams(params_path, train=False)
    device = torch.device("cpu")

    params.device = device
    params.local_goal_dist = local_goal_dist
    params.laser_max_range = laser_max_range

    model = LfD_2D_model(params).to(device)
    assert os.path.exists(model_path)
    model.load_state_dict(torch.load(model_path, map_location=device))

    predictor = Predictor(model, params)

    rospy.init_node("context_classifier", anonymous=True)

    move_base_node = str(rospy.get_param("~move_base_node", "move_base"))

    sub_robot = rospy.Subscriber("odom", Odometry, predictor.update_status)

    global_plan_topic = os.path.join(move_base_node, "TebLocalPlannerROS", "global_plan")
    print("global_plan_topic: ", global_plan_topic)
    sub_gp = rospy.Subscriber(global_plan_topic,
        Path,
        predictor.update_global_path,
        queue_size=1,
    )
    sub_scan = rospy.Subscriber(
        "scan", LaserScan, predictor.update_laser, queue_size=1
    )
    velocity_publisher = rospy.Publisher("cmd_vel", Twist, queue_size=1)

    client = dynamic_reconfigure.client.Client(
        os.path.join(move_base_node, "TebLocalPlannerROS")
    )
    client2 = dynamic_reconfigure.client.Client(
        os.path.join(move_base_node, "local_costmap", "inflation_layer")
    )

    prev_cmd_time = None
    while not rospy.is_shutdown():
        try:
            now = rospy.Time.now()
            if prev_cmd_time is None or (now - prev_cmd_time).to_sec() >= update_dt:
                predictor.update_cmd_vel()
                vel_msg = Twist()
                vel_msg.linear.x = predictor.v
                vel_msg.angular.z = predictor.w
                velocity_publisher.publish(vel_msg)

                prev_cmd_time = now
        except rospy.exceptions.ROSInterruptException:
            break
