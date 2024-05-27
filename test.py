from robobopy.Robobo import Robobo
from robobopy.utils.LED import LED
from robobopy.utils.Color import Color
from robobopy.utils.IR import IR
from robobopy.utils.Wheels import Wheels
from robobosim.RoboboSim import RoboboSim

robobo = Robobo('localhost')
robobo.connect()

IP = "localhost"
sim = RoboboSim(IP)
sim.connect()
sim.wait(0.5)
# Get current location and print it
loc = sim.getRobotLocation(0)
print(loc["position"])
sim.wait(0.5)

# Move the Robot -20mm in the X axis
pos = loc['position']
pos["x"] -= 60
sim.setRobotLocation(0, loc['position'])
sim.wait(0.5)
loc = sim.getRobotLocation(0)
print(loc["position"]['x'])
sim.wait(3.5)

# Reset the simulation
sim.resetSimulation()

sim.disconnect()

"""

robobo.setActiveBlobs(True,False,False,False)
#robobo.moveWheelsByTime(20, 20, 2)
#robobo.moveWheelsByTime(30, -30, 3)
#robobo.moveWheelsByTime(-20, -20, 2)
#robobo.moveWheels(20,20)
#robobo.wait(2)
#robobo.stopMotors()
#robobo.movePanTo(50,20)
#robobo.moveTiltTo(70,20)

robobo.moveTiltTo(105,100)






def v1():
    robobo.moveWheels(10, 10)
    while robobo.readIRSensor(IR.FrontC) < 150:
        robobo.wait(0.01)
        if robobo.readColorBlob(Color.RED).posx == 0:
            robobo.moveWheels(-5, 5)
            while robobo.readColorBlob(Color.RED).posx < 20:
                robobo.wait(0.01)
            robobo.moveWheels(10, 10)
        elif robobo.readColorBlob(Color.RED).posx > 40:
            robobo.moveWheels(5, -5)
            while robobo.readColorBlob(Color.RED).posx > 20:
                robobo.wait(0.01)
            robobo.moveWheels(10, 10)
    robobo.stopMotors()
    robobo.moveWheels(20,20)
    robobo.wait(1)
    robobo.stopMotors()


def v2():
    # gira su se stesso
    robobo.moveWheels(-5,5)
    while robobo.readColorBlob(Color.RED).posx <= 10:
        robobo.wait(0.01)

    # va dritto seguendo un pid
    speed = 10
    while robobo.readIRSensor(IR.FrontC) < 150:
        p=int((robobo.readColorBlob(Color.RED).posx-30)/4)
        if p > 0:
            robobo.moveWheels(speed - p, speed + p)
        else:
            robobo.moveWheels(speed + p, speed - p)
        robobo.wait(0.15)
    robobo.moveWheels(20, 20)
    robobo.wait(1)
    robobo.stopMotors()

def getAngle():
    return robobo.readOrientationSensor().yaw




def move(action,speed=2):
    start_angle = getAngle()
    old_angle = start_angle
    print("start_angle",start_angle)
    match action:
        case 'right45':
            robobo.moveWheels(-speed, speed)
            while getAngle() - start_angle >= -45:
                print(getAngle() - start_angle)
                robobo.wait(0.01)

        case 'right90':
            robobo.moveWheels(-speed, speed)
            while getAngle() - start_angle >= -90:
                print(getAngle() - start_angle)
                robobo.wait(0.01)

        case 'left45':
            robobo.moveWheels(speed, -speed)
            t=45
            old_target=t
            while old_angle - start_angle < t:
                angle=getAngle()
                if abs(angle - old_angle) > 120:
                    start_angle = angle
                    t = old_target
                old_target = t - old_angle - start_angle
                old_angle = angle
                robobo.wait(0.01)

        case 'left90':
            robobo.moveWheels(speed, -speed)
            t=90
            old_target = t
            while old_angle - start_angle < t:
                angle = getAngle()
                if abs(angle - old_angle) > 120:
                    start_angle = angle
                    t = old_target
                old_target = t - old_angle - start_angle
                old_angle = angle
                robobo.wait(0.01)

        case '0':
            robobo.moveWheels(speed, speed)
            robobo.wait(0.01)

    robobo.stopMotors()


move('left45')
robobo.wait(1)
move('left45')
robobo.wait(1)
move('left45')
robobo.wait(1)
move('left45')
robobo.wait(1)
move('left45')
robobo.wait(1)
move('left45')
robobo.wait(1)
move('left45')
robobo.wait(1)
move('left45')

"""