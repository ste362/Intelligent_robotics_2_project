from collections import deque

import torch
from robobopy.Robobo import Robobo
from robobopy.utils.Color import Color
from robobosim.RoboboSim import RoboboSim
import numpy as np
from robobopy.utils.IR import IR
import torch.nn as nn

from extrinsic import ExtrinsicModule

device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
print(f"Using {device} device")




IP = "localhost"
sim = RoboboSim(IP)
robobo = Robobo(IP)

robobo.connect()
sim.connect()

#sim.resetSimulation()
actions=[-90,-45,0,45,90]


##state x,y,theta
def word_model(state):
    predicted_state = []
    x,y,theta,_= state
    finish=False
    for action in actions:
        if action == 0:
            angle = np.deg2rad(theta)
            new_x=x+60*np.sin(angle)
            new_y=y+60*np.cos(angle)
            #print('sensor',robobo.readIRSensor(IR.FrontC))
            if robobo.readColorBlob(Color.RED).size > 400:
                predicted_state.append((new_x, new_y, theta, action))
                finish=True
            elif robobo.readIRSensor(IR.FrontC) < 20: #muro
                predicted_state.append((new_x,new_y,theta,action))

        else:
            new_angle=theta + action
            if new_angle>359:
                new_angle-=360
            elif new_angle<0:
                new_angle+=360
            predicted_state.append((x,y,new_angle,action))

    return predicted_state,finish



memory = deque(maxlen=500)
extrinsic_memory = deque(maxlen=50)
n=1/2
l=1
def calc_novelty(new_state):
    novelty=0
    for x in memory:
        x_diff,y_diff,_ = subtract(x,new_state)
        if x_diff < 10 and y_diff < 10:
            novelty += (0.05*np.abs(x[2]-new_state[2])) ** n
        else:
            novelty += ((x[0] - new_state[0])**l+(x[1]-new_state[1])**l)**n  #+ 0.0008*np.abs(x[2]-new_state[2])

    return novelty/len(memory)

def get_action(predict_state):
    novelty_array = []
    for p in predict_state:
        novelty_array.append(calc_novelty(p))
    #print(novelty_array)
    return predict_state[np.argmax(novelty_array)]




def compute_target_angle(action):
    if action == 0:
        return robobo.readOrientationSensor().yaw
    angle = robobo.readOrientationSensor().yaw
    xx = abs(action)
    sign = 1 if action < 0 else -1
    target = None

    range1 = (-180, 180-xx)
    range2 = (180-xx, 180)

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


def perform_action(action):
    speed_factor = 3
    #print('action:{}\tstart:{}\ttarget:{}\r'.format(action, robobo.readOrientationSensor().yaw, target), flush=True)

    if action == 0:
        robobo.moveWheels(10, 10)
        robobo.wait(1)
    else:
        target = compute_target_angle(action)
        sign = 1 if action < 0 else -1
        eps = 4 * speed_factor
        while abs(robobo.readOrientationSensor().yaw - target) > eps:
            # print(abs(robobo.readOrientationSensor().yaw - target), robobo.readOrientationSensor().yaw, target)
            robobo.moveWheels(sign * 2 * speed_factor, sign * (-2 * speed_factor))
            robobo.wait(0.1)

        error = abs(robobo.readOrientationSensor().yaw - target)
        #print('error:{}\n'.format(error), flush=True)
        #robobo.stopMotors()


    """
    match action:
        case -90:
            robobo.moveWheelsByTime(-10,10,2)
        case -45:
            robobo.moveWheelsByTime(-10, 10, 1)
        case 0:
            robobo.moveWheelsByTime(10, 10, 1)
        case 45:
            robobo.moveWheelsByTime(10, -10, 1)
        case 90:
            robobo.moveWheelsByTime(10, -10, 2)
    """


def subtract(t1, t2):
    return (round(abs(t1[0]-t2[0]),1), round(abs(t1[1]-t2[1]),1), round(abs(t1[2]-t2[2]),1))


def get_neural_action(perception):
    utility=[]
    for a in range(len(actions)):
        input_state=perception[:]
        input_state.append(a)
        input_state = torch.tensor(input_state,dtype=torch.float32)
        out = extrinsic.nn(input_state)
        utility.append(out.detach().numpy())
    return actions[np.argmax(utility)]

#robobo.wait(1)
robobo.moveTiltTo(105,100,wait=True)

predict_state=(0,0,0,0)
action=0
eps=0.99
extrinsic = ExtrinsicModule()
while True:
    loc = sim.getRobotLocation(0)
    real_state=(loc['position']['x'],loc['position']['z'],loc['rotation']['y'],action)
    #print("Error prediction (x,y,theta):",subtract(real_state,predict_state),'action:',action)
    memory.append(real_state)
    predict_states,finish=word_model(real_state)
    perception = [robobo.readIRSensor(IR.FrontC), robobo.readColorBlob(Color.RED).posx,
                  robobo.readColorBlob(Color.RED).size]
    if np.random.random() < eps:
        predict_state=get_action(predict_states)
        action=predict_state[3]
        print("Novelty",action)
    else:
        action=get_neural_action(perception)
        print("Neural",action)
    perception.append(action)
    extrinsic_memory.append(perception)

    perform_action(action)
    if finish:
        #robobo.wait(1)
        robobo.stopMotors()
        print("\n\n\n\nRobobo Simulation Complete\n\n\n\n")
        sim.resetSimulation()
        robobo.wait(1)
        robobo.moveTiltTo(105, 100)
        eps-=0.01
        X=torch.tensor(extrinsic_memory, dtype=torch.float32)
        y=torch.tensor([x/len(extrinsic_memory) for x in range(len(extrinsic_memory))], dtype=torch.float32)

        print("Start Train....\n")
        extrinsic.train(X,y)
        print("\nTrain Complete")
        torch.save(extrinsic.nn.state_dict(),"extrinsic_nn.pt")

