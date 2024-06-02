import random

import numpy as np
import torch
from robobopy.Robobo import Robobo
from robobopy.utils.Color import Color
from robobosim.RoboboSim import RoboboSim
from robobopy.utils.IR import IR

from cognitive_system import CognitiveSystem
from params import Params, Colors


class Environment:
    def __init__(self, ip='localhost', debug=True):
        self.debug = debug
        self.ip = ip

        self.device = (
            "cuda"
            if torch.cuda.is_available()
            else "mps"
            if torch.backends.mps.is_available()
            else "cpu"
        )

        if self.debug: print(f"Using {self.device} device")
        if self.debug: print(f"Connecting to Robobo and RoboboSim servises on ip={self.ip}")

        self.sim = RoboboSim(self.ip)
        self.robobo = Robobo(self.ip)
        self.robobo.connect()
        self.sim.connect()
        self.cognitive = CognitiveSystem(device=self.device)
        self.initialize_env()

    def initialize_env(self):
        self.robobo.moveTiltTo(105, 100, wait=True)

    def reset(self):
        env.sim.resetSimulation()
        env.robobo.wait(1)
        env.robobo.moveTiltTo(105, 100)

    def get_env_state(self):
        loc = self.sim.getRobotLocation(0)
        y = loc['position']['x']
        x = loc['position']['z']
        theta = loc['rotation']['y']
        ir = self.robobo.readIRSensor(IR.FrontC)
        red_pos = self.robobo.readColorBlob(Color.RED).posx
        red_size = self.robobo.readColorBlob(Color.RED).size
        return [x, y, theta, ir, red_pos, red_size]

    def get_red_blob_pos(self):
        blob = self.robobo.readColorBlob(Color.RED)
        return blob.posx, blob.posy

    def compute_target_angle(self, action):
        if action == 0:
            return self.robobo.readOrientationSensor().yaw
        angle = self.robobo.readOrientationSensor().yaw
        xx = abs(action)
        sign = 1 if action < 0 else -1
        target = None

        range1 = (-180, 180 - xx)
        range2 = (180 - xx, 180)

        if action > 0:  # invert the ranges
            range1 = (-range1[1], -range1[0])
            range2 = (-range2[1], -range2[0])

        if range1[0] < angle <= range1[1]:
            target = angle + sign * xx
        elif range2[0] <= angle <= range2[1]:
            target = (-sign * 180) + sign * (sign * angle - (180 - xx))

        if target is None:
            raise ValueError('angle {} not in range ({},{}] or [{},{}]'.format(
                angle,
                range1[0],
                range1[1],
                range2[0],
                range2[1]
            ))

        return target

    def perform_action(self, action):
        speed_factor = 3
        # print('action:{}\tstart:{}\ttarget:{}\r'.format(action, robobo.readOrientationSensor().yaw, target), flush=True)
        action = self.cognitive.world.actions[action]

        if action == 0:
            self.robobo.moveWheels(10, 10)
            self.robobo.wait(1)
        else:
            target = self.compute_target_angle(action)
            sign = 1 if action < 0 else -1
            eps = Params.error_margin_rotation.value * speed_factor
            while abs(self.robobo.readOrientationSensor().yaw - target) > eps:
                # print(abs(robobo.readOrientationSensor().yaw - target), robobo.readOrientationSensor().yaw, target)
                self.robobo.moveWheels(sign * 2 * speed_factor, sign * (-2 * speed_factor))
                self.robobo.wait(0.1)

            # error = abs(robobo.readOrientationSensor().yaw - target)
            # print('error:{}\n'.format(error), flush=True)
            # robobo.stopMotors()

    def finish(self, real_state):
        blob_pos_x, blob_pos_y = env.get_red_blob_pos()
        return (15 <= blob_pos_x <= 85 and 85 <= blob_pos_y <= 100) or real_state[5] > 450

    def interact_nn(self):
        def can_insert(action_count, action):
            if action<2:
                action=0
            elif action>2:
                action=2
            else:
                action=1
            if action_count[action]-min(action_count) < 5 or True:
                action_count[action] += 1
                return True
            return True
        #
        #
        #    NEURAL WORLD MODEL
        #
        #
        if not Params.Extrinsic.train.value:
            print(f"{Colors.WARNING}!!Train not enabled for EXTRINSIC Module!!{Colors.ENDC}")

        if not Params.World.train.value:
            print(f"{Colors.WARNING}!!Train not enabled for WORLD Module!!{Colors.ENDC}")

        intrinsic = self.cognitive.intrinsic
        extrinsic = self.cognitive.extrinsic
        world = self.cognitive.world_nn
        env = self

        eps = Params.eps.value
        train_count = 0
        i = 0
        #world.memory_in.append([0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0])
        reached_times = 0
        path = []
        pre_real_state=None
        action_count=[0,0,0]
        append=False
        while True:
            real_state = env.get_env_state()
            intrinsic.memory.append(real_state)

            if append:
                #result = [a - b for a, b in zip(real_state, pre_real_state)]
                world.memory_out.append(real_state)

            if i != 0 and i % 50 == 0 and Params.World.train.value:
                env.robobo.stopMotors()
                print(len(world.memory_in), len(world.memory_out))
                world.train(world.memory_in, world.memory_out)
                world.save()

            input_states, predicted_states = world.predict(real_state)

            if random.random() < eps:
                action = intrinsic.get_action(predicted_states)
                print(f"Novelty {Colors.OKCYAN}{world.actions[action]}{Colors.ENDC}")
            else:
                action = extrinsic.get_action(predicted_states[3:6])
                print(f"Neural {Colors.OKCYAN}{world.actions[action]}{Colors.ENDC}")

            real_action=action
            if len(predicted_states) == 4 and action > 1:
                real_action+= 1

            print(action_count)
            if can_insert(action_count, real_action):
                world.memory_in.append(input_states[action])
                append=True

            else:
                append=False

            path.append(predicted_states[action][3:6])

            env.perform_action(real_action)

            pre_real_state=real_state[:]
            i += 1
            if i > Params.n_iterations_before_stop.value:
                i = 0
                env.reset()
                extrinsic.reset_memory()
            if env.finish(real_state):
                reached_times += 1
                env.robobo.stopMotors()
                print(f"Robobo Simulation Complete ({Colors.BOLD}{reached_times}{Colors.ENDC})\n\n\n")
                env.reset()
                eps -= Params.eps_decr.value

                # TRAINING EXTRINSIC MODULE
                if Params.Extrinsic.train.value:
                    extrinsic.memory_in.extend(path)
                    values = [(np.e ** (-x / 16)) for x in range(len(path), 0, -1)]

                    extrinsic.memory_out.extend(values)
                    X = extrinsic.memory_in
                    y = extrinsic.memory_out

                    extrinsic.train(X, y)
                    extrinsic.save(train_count)
                    train_count += 1
                    path = []

    def interact(self):
        #
        #
        #    MATHEMATICAL WORLD MODEL
        #
        #
        intrinsic = self.cognitive.intrinsic
        extrinsic = self.cognitive.extrinsic
        world = self.cognitive.world
        env = self

        if not Params.Extrinsic.train.value:
            print(f"{Colors.WARNING}!!Train not enabled for Extrinsic Module!!{Colors.ENDC}")

        eps = Params.eps.value
        train_count = 0
        i = 0
        reached_times = 0
        path = []

        while True:
            real_state = env.get_env_state()
            intrinsic.memory.append(real_state)
            _, predicted_states = world.predict(real_state)

            if random.random() < eps:
                action = intrinsic.get_action(predicted_states)
                print(f"Novelty {Colors.OKCYAN}{world.actions[action]}{Colors.ENDC}")
            else:
                action = extrinsic.get_action(predicted_states)
                print(f"Neural {Colors.OKCYAN}{world.actions[action]}{Colors.ENDC}")

            path.append(predicted_states[action][3:6])

            if len(predicted_states) == 4 and action >= Params.Action.forward.value:
                action += 1
            env.perform_action(action)

            i += 1
            if i > Params.n_iterations_before_stop.value:
                i = 0
                env.reset()
                extrinsic.reset_memory()
            if env.finish(real_state):
                reached_times += 1
                env.robobo.stopMotors()
                print(f"Robobo Simulation Complete ({Colors.BOLD}{reached_times}{Colors.ENDC})\n\n\n")
                env.reset()
                eps -= Params.eps_decr.value
                # TRAINING EXTRINSIC MODULE
                if Params.Extrinsic.train.value:
                    extrinsic.memory_in.extend(path)
                    values = [(np.e ** (-x / 16)) for x in range(len(path), 0, -1)]

                    extrinsic.memory_out.extend(values)
                    X = extrinsic.memory_in
                    y = extrinsic.memory_out

                    extrinsic.train(X, y)
                    extrinsic.save(train_count)
                    train_count += 1
                    path = []


if __name__ == '__main__':
    env = Environment()
    env.reset()
    if Params.use_neural_world_model.value:
        print(f'{Colors.OKBLUE}Using NEURAL WORLD MODEL!{Colors.ENDC}')
        env.interact_nn()
    else:
        print(f'{Colors.OKBLUE}Using MATH WORLD MODEL!{Colors.ENDC}')
        env.interact()
