from collections import deque

import torch
from robobopy.Robobo import Robobo
from robobopy.utils.Color import Color
from robobosim.RoboboSim import RoboboSim
import numpy as np
from robobopy.utils.IR import IR

from extrinsic import ExtrinsicModule
from world_model import WorldModelNN

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
actions=[-30,-15,0,15,30]


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


def word_model_neural(state):
    predicted_states=[]
    for action in range(len(actions)):
        ## create one hot vector for the action
        one_hot=[0 for i in range(len(actions))]
        one_hot[action]=1

        input_state=state[:-1].extend(one_hot) ## remove the old action and add the one hot encoding

        input_state = torch.tensor(input_state, dtype=torch.float32)

        out = world_model.nn(input_state)
        predicted_states.append(out.detach().numpy())

    return predicted_states












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

        #error = abs(robobo.readOrientationSensor().yaw - target)
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


def get_neural_action(predicted_states):
    utility=[]
    for predicted_state in predicted_states:
        input_state=predicted_state[3:6]
        #if perception[2] > 0 and a!=2:
        #    input_state=[0,0,0]
        #if perception[2] == 0 and a != 2:
        #    input_state = [0, 20, 30]
        #input_state.append(a)
        #input_state = torch.tensor(input_state,dtype=torch.float32)
        out = extrinsic.nn(input_state)
        utility.append(out.detach().numpy())
    print("Neural output",utility)
    return np.argmax(utility)

def get_real_state(action):              #return the (x,y,theta,IR,RED_pos, RED_size, action)
    loc = sim.getRobotLocation(0)
    x=loc['position']['x']
    y=loc['position']['z']
    theta=loc['rotation']['y']
    ir=robobo.readIRSensor(IR.FrontC)
    red_pos=robobo.readColorBlob(Color.RED).posx
    red_size=robobo.readColorBlob(Color.RED).size
    return [x, y, theta, ir, red_pos, red_size, action]

#robobo.wait(1)
robobo.moveTiltTo(105,100,wait=True)



memory = deque(maxlen=500)
extrinsic_memory = deque(maxlen=50)
predicted_state_memory = deque(maxlen=5000)

predict_state=(0,0,0,0)
action=0
eps=0.2
extrinsic = ExtrinsicModule()
world_model = WorldModelNN()
extrinsic.load("extrinsic_nn.pt")
train_count=0
i=0
while True:
    real_state=get_real_state(action)

    #print("Error prediction (x,y,theta):",subtract(real_state,predict_state),'action:',action)

    if real_state[5] > 400: ##finish
        finish = True

    memory.append(real_state)
    predicted_states=word_model_neural(real_state)
    #perception = [robobo.readIRSensor(IR.FrontC), robobo.readColorBlob(Color.RED).posx,robobo.readColorBlob(Color.RED).size]

    if np.random.random() < eps:
        predict_state=get_action(predicted_states)
        action=predict_state[3]
        print("Novelty",action)
    else:
        action=get_neural_action(predicted_states)
        print("Neural",action,"train count:",train_count)

    extrinsic_memory.append(predicted_states[action][3:6])

    perform_action(action)
    i+=1
    if i>1000:
        i=0
        sim.resetSimulation()
        robobo.wait(1)
        robobo.moveTiltTo(105, 100)
        extrinsic_memory = deque(maxlen=50)
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

        extrinsic_memory = deque(maxlen=50)
        train_count+=1

