from robobopy.Robobo import Robobo
from robobopy.utils.LED import LED
from robobopy.utils.Color import Color
from robobopy.utils.IR import IR

robobo = Robobo('localhost')
robobo.connect()
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



v2()