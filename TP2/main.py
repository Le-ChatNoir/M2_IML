#!/usr/bin/env python
# coding: utf-8

import cv2
import time
from qibullet import SimulationManager
from qibullet import PepperVirtual
import pybullet as p
import pybullet_data
import threading

def main():

	# create simulator

    simulation_manager = SimulationManager()
    client = simulation_manager.launchSimulation(gui=True)

	# env
    # create 3D object
    p.connect(p.DIRECT)
    cube_visual = p.createVisualShape(p.GEOM_BOX, halfExtents=[0.125,0.125,0.125])
    cube_collision = p.createCollisionShape(p.GEOM_BOX, halfExtents=[0.25,0.25,0.25])
    cube_body = p.createMultiBody(baseMass=0, baseCollisionShapeIndex=cube_collision, baseVisualShapeIndex=cube_visual, basePosition = [2,1, 0.725])

    # load 3D object
    p.loadURDF("./urdf/table/table.urdf", basePosition = [2,1,0], globalScaling = 1)
    p.loadURDF("./urdf/chair/chair.urdf", basePosition = [3,1,0], globalScaling = 1)
    p.loadURDF("./urdf/chair/chair.urdf", basePosition = [4,1,0], globalScaling = 1)

    p.setAdditionalSearchPath(pybullet_data.getDataPath())

    p.loadURDF(
        "duck_vhacd.urdf",
        basePosition=[1, -1, 0.5],
        globalScaling=10.0,
        physicsClientId=client)

	# robot
    pepper = simulation_manager.spawnPepper(client, spawn_ground_plane=True)
    pepper.goToPosture("Crouch", 0.6)
    time.sleep(3)
    pepper.goToPosture("Stand", 0.6)
    time.sleep(3)
    pepper.goToPosture("StandZero", 0.6)
    time.sleep(5)



	# camera
    handle = pepper.subscribeCamera(PepperVirtual.ID_CAMERA_BOTTOM)
    handle2 = pepper.subscribeCamera(PepperVirtual.ID_CAMERA_TOP)
    handle3 = pepper.subscribeCamera(PepperVirtual.ID_CAMERA_DEPTH)

    pepper.moveTo(3,3,0,frame=2,_async=False)

    # laser prep
    pepper.showLaser(True)
    pepper.subscribeLaser()

    # joints
    joint_parameters = list()

    for name, joint in pepper.joint_dict.items():
        if "Finger" not in name and "Thumb" not in name:
            joint_parameters.append((
                p.addUserDebugParameter(
                    name,
                    joint.getLowerLimit(),
                    joint.getUpperLimit(),
                    pepper.getAnglesPosition(name)),
                name))

    try:
        
        while True:
            # cam
            img = pepper.getCameraFrame(handle)
            cv2.imshow("bottom camera", img)
            
            img2 = pepper.getCameraFrame(handle2)
            cv2.imshow("top camera", img2)
            
            img3 = pepper.getCameraFrame(handle3)
            cv2.imshow("depth camera", img3)
            
            filename="ImageDepth.png"
            cv2.imwrite(filename,img3)
            im_gray = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
            im_color = cv2.applyColorMap(im_gray, cv2.COLORMAP_HSV)
            cv2.imshow("colorBar depth camera", im_color)
            
            cv2.waitKey(1)

            # lasers
            laser_list = pepper.getRightLaserValue()
            laser_list.extend(pepper.getFrontLaserValue())
            laser_list.extend(pepper.getLeftLaserValue())
            
            if all(laser == 5.6 for laser in laser_list):
                print("Nothing detected")
            else:
                print("Detected")
                pass

            # join
            for joint_parameter in joint_parameters:
                pepper.setAngles(
                    joint_parameter[1],
                    p.readUserDebugParameter(joint_parameter[0]), 1.0)


    except KeyboardInterrupt:
        simulation_manager.stopSimulation(client)
    

if __name__ == "__main__":
    main()