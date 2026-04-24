#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
q_learning_node.py  --  Q-Learning wall follower
"""

import rospy
import numpy as np
import math
import os
import csv

from geometry_msgs.msg import Twist
from sensor_msgs.msg import LaserScan
from gazebo_msgs.srv import SetModelState
from gazebo_msgs.msg import ModelState
from geometry_msgs.msg import Pose, Point, Quaternion
from std_srvs.srv import Empty


# =====================================================================
#  STATE DISCRETISATION
# =====================================================================
# 3 sectors × 3 bins = 27 states

DIST_THRESHOLDS = [0.40, 0.90]
NUM_DIST_BINS   = 3

SECTORS = {
    'front':       (-20,   20),
    'front_right': (-60,  -20),
    'right':       (-110, -60),
}
SECTOR_NAMES = list(SECTORS.keys())
NUM_SECTORS  = len(SECTORS)
NUM_STATES   = NUM_DIST_BINS ** NUM_SECTORS  # 27

ACTIONS = [
    (0.02,  1.20),   # 0  hard-left
    (0.12,  0.50),   # 1  soft-left
    (0.22,  0.00),   # 2  straight
    (0.12, -0.50),   # 3  soft-right
    (0.02, -1.20),   # 4  hard-right
]
NUM_ACTIONS = len(ACTIONS)

COLLISION_DIST    = 0.18
DESIRED_DIST_LOW  = 0.40
DESIRED_DIST_HIGH = 0.90


def discretize_distance(d):
    for i, thresh in enumerate(DIST_THRESHOLDS):
        if d < thresh:
            return i
    return NUM_DIST_BINS - 1


def _sector_min(ranges, angle_min, angle_inc, lo_deg, hi_deg,
                rng_min=0.12, rng_max=3.5):
    lo_rad = math.radians(lo_deg)
    hi_rad = math.radians(hi_deg)
    best = rng_max
    for i, r in enumerate(ranges):
        ang = angle_min + i * angle_inc
        ang = math.atan2(math.sin(ang), math.cos(ang))
        if lo_rad <= ang <= hi_rad:
            if rng_min < r < rng_max and math.isfinite(r):
                best = min(best, r)
    return best


def scan_to_state(ranges, angle_min, angle_inc):
    sector_dists = {}
    levels = []
    for name in SECTOR_NAMES:
        lo, hi = SECTORS[name]
        d = _sector_min(ranges, angle_min, angle_inc, lo, hi)
        sector_dists[name] = d
        levels.append(discretize_distance(d))

    state = 0
    for i, lv in enumerate(levels):
        state += lv * (NUM_DIST_BINS ** (NUM_SECTORS - 1 - i))
    return state, sector_dists


# =====================================================================
#  Q-TABLE MANAGER
# =====================================================================

class QTableManager(object):

    def __init__(self, num_states=NUM_STATES, num_actions=NUM_ACTIONS):
        self.S = num_states
        self.A = num_actions
        self.q = np.zeros((num_states, num_actions), dtype=np.float64)

    def load_csv(self, path):
        if not os.path.isfile(path):
            return False
        try:
            with open(path, 'r') as f:
                reader = csv.reader(f)
                next(reader)
                for row in reader:
                    s = int(row[0])
                    vals = [float(v) for v in row[1:]]
                    self.q[s, :len(vals)] = vals[:self.A]
            return True
        except Exception as e:
            print('[QTableManager] load error: {}'.format(e))
            return False

    def save_csv(self, path):
        d = os.path.dirname(path)
        if d and not os.path.exists(d):
            os.makedirs(d)
        with open(path, 'w') as f:
            w = csv.writer(f)
            w.writerow(['state'] + ['a{}'.format(i) for i in range(self.A)])
            for s in range(self.S):
                w.writerow([s] + ['{:.6f}'.format(v) for v in self.q[s]])

    def best_action(self, s):
        return int(np.argmax(self.q[s]))

    def max_q(self, s):
        return float(np.max(self.q[s]))

    def get(self, s, a):
        return self.q[s, a]

    def set(self, s, a, v):
        self.q[s, a] = v


# =====================================================================
#  TRAINING SPAWNS
# =====================================================================

SPAWN_STRAIGHT = [
    ( 2.0,   0.5,  180.0),
    ( 2.0,  -0.5,  180.0),
    (-1.0,   1.8,  270.0),
    (-2.0,  -0.5,    0.0),
]

SPAWN_CORNERS = [
    ( 1.8,   1.8,  180.0),
    (-1.6,  -1.6,    0.0),
    ( 1.8,  -1.6,   90.0),
    (-1.6,   1.8,  270.0),
]

SPAWN_IBEAM = [
    (-1.0,  -1.26,    0.0),
    ( 0.5,  -1.26,    0.0),
    ( 1.8,  -1.26,    0.0),
    ( 2.3,  -2.0,   270.0),
    ( 1.0,  -2.74,  180.0),
    (-0.5,  -2.74,  180.0),
]

SPAWN_OPEN = [
    (0.0,  0.0,    0.0),
    (0.0,  0.0,   90.0),
    (0.0,  0.0,  180.0),
    (0.0,  0.0,  270.0),
]

TRAINING_SPAWNS = (
    SPAWN_STRAIGHT +
    SPAWN_CORNERS +
    SPAWN_IBEAM * 2 +
    SPAWN_OPEN
)


# =====================================================================
#  DEFAULTS
# =====================================================================
DEFAULT_ALPHA          = 0.10
DEFAULT_GAMMA          = 0.95
DEFAULT_EPSILON_START  = 0.90
DEFAULT_EPSILON_MIN    = 0.05
DEFAULT_EPSILON_DECAY  = 0.997
DEFAULT_NUM_EPISODES   = 300
MAX_STEPS_PER_EPISODE  = 600
CONTROL_HZ             = 10.0
SAVE_EVERY             = 50


# =====================================================================
#  UTILITY
# =====================================================================
def yaw_to_quaternion(yaw_deg):
    yr = math.radians(yaw_deg)
    q = Quaternion()
    q.x = 0.0
    q.y = 0.0
    q.z = math.sin(yr / 2.0)
    q.w = math.cos(yr / 2.0)
    return q


# =====================================================================
#  Q-LEARNING NODE
# =====================================================================
class QLearningNode(object):

    def __init__(self):
        rospy.init_node('q_learning_node', anonymous=False)

        self.mode        = rospy.get_param('~mode', 'train')
        default_qt_path  = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            'q_tables', 'q_table_qlearning.csv')
        self.qt_path     = rospy.get_param('~q_table_path', default_qt_path)
        self.model_name  = rospy.get_param('~model_name', 'triton')

        self.alpha         = rospy.get_param('~alpha',         DEFAULT_ALPHA)
        self.gamma         = rospy.get_param('~gamma',         DEFAULT_GAMMA)
        self.epsilon_start = rospy.get_param('~epsilon_start', DEFAULT_EPSILON_START)
        self.epsilon_min   = rospy.get_param('~epsilon_min',   DEFAULT_EPSILON_MIN)
        self.epsilon_decay = rospy.get_param('~epsilon_decay', DEFAULT_EPSILON_DECAY)
        self.num_episodes  = rospy.get_param('~num_episodes',  DEFAULT_NUM_EPISODES)

        rospy.loginfo('Mode:        {}'.format(self.mode))
        rospy.loginfo('Q-table:     {}'.format(self.qt_path))
        rospy.loginfo('Model name:  {}'.format(self.model_name))
        rospy.loginfo('States: {}  Actions: {}'.format(NUM_STATES, NUM_ACTIONS))

        self.cmd_pub = rospy.Publisher('/cmd_vel', Twist, queue_size=10)

        self.qtm = QTableManager()
        if os.path.isfile(self.qt_path):
            if self.qtm.load_csv(self.qt_path):
                rospy.loginfo('Loaded Q-table from {}'.format(self.qt_path))
            else:
                rospy.logwarn('Failed to load Q-table, starting from zeros')
        else:
            rospy.loginfo('No existing Q-table found, starting from zeros')

        rospy.loginfo('Waiting for /gazebo/set_model_state ...')
        rospy.wait_for_service('/gazebo/set_model_state', timeout=30.0)
        self.set_model_state = rospy.ServiceProxy(
            '/gazebo/set_model_state', SetModelState)

        try:
            rospy.wait_for_service('/gazebo/unpause_physics', timeout=5.0)
            self.unpause_physics = rospy.ServiceProxy(
                '/gazebo/unpause_physics', Empty)
            rospy.wait_for_service('/gazebo/pause_physics', timeout=5.0)
            self.pause_physics = rospy.ServiceProxy(
                '/gazebo/pause_physics', Empty)
            self.has_pause = True
        except rospy.ROSException:
            rospy.logwarn('Pause/unpause services not found')
            self.has_pause = False

        rospy.loginfo('Gazebo services ready.')
        self._last_scan = None

    def _wait_for_scan(self, timeout=3.0):
        try:
            msg = rospy.wait_for_message('/scan', LaserScan, timeout=timeout)
            self._last_scan = msg
            return msg
        except rospy.ROSException:
            rospy.logwarn('Timed out waiting for /scan')
            return None

    def _get_state(self):
        msg = self._wait_for_scan()
        if msg is None:
            return None, None
        return scan_to_state(msg.ranges, msg.angle_min, msg.angle_increment)

    def _publish_action(self, action_idx):
        lin, ang = ACTIONS[action_idx]
        t = Twist()
        t.linear.x  = lin
        t.angular.z = ang
        self.cmd_pub.publish(t)

    def _stop(self):
        self.cmd_pub.publish(Twist())

    def _teleport(self, x, y, yaw_deg):
        self._stop()
        rospy.sleep(0.2)
        state_msg = ModelState()
        state_msg.model_name = self.model_name
        state_msg.pose = Pose()
        state_msg.pose.position = Point(x, y, 0.01)
        state_msg.pose.orientation = yaw_to_quaternion(yaw_deg)
        state_msg.reference_frame = 'world'
        try:
            resp = self.set_model_state(state_msg)
            if not resp.success:
                rospy.logwarn('Teleport failed: {}'.format(resp.status_message))
        except rospy.ServiceException as e:
            rospy.logwarn('Teleport service call failed: {}'.format(e))
        rospy.sleep(0.5)

    @staticmethod
    def _reward(dists, action_idx):
        right = dists['right']
        front = dists['front']
        mn = min(dists.values())

        if mn < COLLISION_DIST:
            return -100.0, True

        reward = 0.0
        lin_vel = ACTIONS[action_idx][0]
        ang_vel = abs(ACTIONS[action_idx][1])

        if DESIRED_DIST_LOW <= right <= DESIRED_DIST_HIGH:
            reward += 15.0 * lin_vel
        elif right < DESIRED_DIST_LOW:
            reward -= 8.0
        else:
            reward -= 10.0

        if front < 0.35:
            reward -= 20.0
        elif front < 0.55:
            reward -= 8.0

        reward += 4.0 * lin_vel
        reward -= 3.0 * ang_vel
        reward -= 0.5

        return reward, False

    def _save_rewards(self, rewards):
        path = self.qt_path.replace('.csv', '_rewards.csv')
        d = os.path.dirname(path)
        if d and not os.path.exists(d):
            os.makedirs(d)
        with open(path, 'w') as f:
            w = csv.writer(f)
            w.writerow(['episode', 'reward'])
            for i, r in enumerate(rewards):
                w.writerow([i, '{:.4f}'.format(r)])

    def run_training(self):
        rospy.loginfo('=' * 60)
        rospy.loginfo('  Q-Learning TRAINING  ({} episodes, {} states)'.format(
            self.num_episodes, NUM_STATES))
        rospy.loginfo('=' * 60)

        epsilon = self.epsilon_start
        all_rewards = []

        for ep in range(self.num_episodes):
            if rospy.is_shutdown():
                break

            sx, sy, syaw = TRAINING_SPAWNS[ep % len(TRAINING_SPAWNS)]
            self._teleport(sx, sy, syaw)

            state, dists = self._get_state()
            if state is None:
                continue

            ep_reward = 0.0

            for step in range(MAX_STEPS_PER_EPISODE):
                if rospy.is_shutdown():
                    break

                if np.random.random() < epsilon:
                    action = np.random.randint(NUM_ACTIONS)
                else:
                    action = self.qtm.best_action(state)

                self._publish_action(action)
                rospy.sleep(1.0 / CONTROL_HZ)

                ns, nd = self._get_state()
                if ns is None:
                    break

                r, done = self._reward(nd, action)
                ep_reward += r

                old_q = self.qtm.get(state, action)
                td_target = r + self.gamma * self.qtm.max_q(ns)
                new_q = old_q + self.alpha * (td_target - old_q)
                self.qtm.set(state, action, new_q)

                if done:
                    self._stop()
                    break

                state = ns
                dists = nd

            all_rewards.append(ep_reward)
            epsilon = max(self.epsilon_min, epsilon * self.epsilon_decay)
            self._stop()

            rospy.loginfo(
                'Ep {:>4}/{:>4}  eps={:.3f}  R={:+8.1f}  avg100={:+8.1f}'.format(
                    ep + 1, self.num_episodes, epsilon, ep_reward,
                    np.mean(all_rewards[-100:])))

            if (ep + 1) % SAVE_EVERY == 0:
                self.qtm.save_csv(self.qt_path)
                self._save_rewards(all_rewards)
                rospy.loginfo('  >> Saved Q-table to {}'.format(self.qt_path))

        self.qtm.save_csv(self.qt_path)
        self._save_rewards(all_rewards)
        rospy.loginfo('Training complete! Q-table saved to {}'.format(self.qt_path))

    def run_testing(self):
        if not self.qtm.load_csv(self.qt_path):
            rospy.logerr('Cannot load Q-table from {}'.format(self.qt_path))
            return

        rospy.loginfo('=' * 60)
        rospy.loginfo('  Q-Learning TESTING  (greedy policy)')
        rospy.loginfo('=' * 60)

        rate = rospy.Rate(CONTROL_HZ)
        while not rospy.is_shutdown():
            state, dists = self._get_state()
            if state is None:
                continue
            action = self.qtm.best_action(state)
            self._publish_action(action)
            rate.sleep()

    def run(self):
        if self.mode == 'train':
            self.run_training()
        elif self.mode == 'test':
            self.run_testing()
        else:
            rospy.logerr('Unknown mode: "{}"'.format(self.mode))


def main():
    try:
        node = QLearningNode()
        node.run()
    except rospy.ROSInterruptException:
        rospy.loginfo('Interrupted -- saving Q-table')
        try:
            node.qtm.save_csv(node.qt_path)
        except Exception:
            pass


if __name__ == '__main__':
    main()
