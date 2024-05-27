import random
from collections import deque

import numpy as np
import torch
from robobopy.Robobo import Robobo
from robobopy.utils.Color import Color
from robobosim.RoboboSim import RoboboSim
from robobopy.utils.IR import IR

from cognitive2 import CognitiveSystem


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
        x = loc['position']['x']
        y = loc['position']['z']
        theta = loc['rotation']['y']
        ir = self.robobo.readIRSensor(IR.FrontC)
        red_pos = self.robobo.readColorBlob(Color.RED).posx
        red_size = self.robobo.readColorBlob(Color.RED).size
        print(red_pos)
        return [x, y, theta, ir, red_pos, red_size]

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
        action=self.cognitive.world.actions[action]

        if action == 0:
            self.robobo.moveWheels(10, 10)
            self.robobo.wait(1)
        else:
            target = self.compute_target_angle(action)
            sign = 1 if action < 0 else -1
            eps = 4 * speed_factor
            while abs(self.robobo.readOrientationSensor().yaw - target) > eps:
                # print(abs(robobo.readOrientationSensor().yaw - target), robobo.readOrientationSensor().yaw, target)
                self.robobo.moveWheels(sign * 2 * speed_factor, sign * (-2 * speed_factor))
                self.robobo.wait(0.1)

            # error = abs(robobo.readOrientationSensor().yaw - target)
            # print('error:{}\n'.format(error), flush=True)
            # robobo.stopMotors()

    def interact(self):
        intrinsic = self.cognitive.intrinsic
        extrinsic = self.cognitive.extrinsic
        world = self.cognitive.world
        env = self

        eps = 0.0
        train_count = 0
        i = 0
        #world.memory_in.append([0,0,0,0,0,0,0,0,1,0,0])
        finish=False
        predicted_state=[]
        append=True
        while True:

            real_state = env.get_env_state()
            #real_state.extend(action)
            #print("\n predicted state",predicted_state,"\n real state",real_state,"\n")
            if real_state[5] > 400:  ##finish
                finish = True

            if append:
                intrinsic.memory.append(real_state)
                #append=False

            if i%10==0:
                env.robobo.stopMotors()
                #world.train(world.memory_in, world.memory_out)
                #world.save()

            input_states,predicted_states = world.predict(real_state)
            # perception = [robobo.readIRSensor(IR.FrontC), robobo.readColorBlob(Color.RED).posx,robobo.readColorBlob(Color.RED).size]

            rand=random.random()
            print(rand)
            if rand < eps:
                #rand = np.random.random()
                action=intrinsic.get_action(predicted_states)    #action = 2 if len(predicted_states)==5 else np.random.random_integers(0,1)  #
                print("Novelty", action)
            else:
                action = extrinsic.get_action(predicted_states)
                print("Neural", action)



            if len(predicted_states)==5 and action==2:
                append = True
                #world.memory_in.append(input_states[action])

            elif len(predicted_states)==4 and rand<0.1:
                append = True
                #world.memory_in.append(input_states[action])
            elif len(predicted_states)==5 and rand<0.1 and action!=2:
                append = True
                #world.memory_in.append(input_states[action])


            extrinsic.memory.append(predicted_states[action][3:6])

            if len(predicted_states)==4 and action>1:
                action+=1
            env.perform_action(action)


            i += 1
            if i > 1000:
                i = 0
                env.reset()
                extrinsic.reset_memory()
            if finish:
                # robobo.wait(1)
                env.robobo.stopMotors()
                print("\n\n\n\nRobobo Simulation Complete\n\n\n")
                env.reset()
                eps -= 0.02
                train_count += 1
                print("train count",train_count)

                # TRAINING EXTRINSIC MODULE
                X = extrinsic.memory
                y = [x / len(X) for x in range(len(X))]
                extrinsic.train(X, y)
                extrinsic.save()

                extrinsic.reset_memory()
                finish = False






if __name__ == '__main__':
    env = Environment()
    env.interact()
