#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 11 15:35:02 2022
@author: bezzo
"""

import numpy as np
import math as m
from itertools import product
from bresenham import bresenham
import pandas as pd

class Nodes():
    pass
def posToMap(pos,origin,omap,res):
    mapPos = np.round((pos-origin)/res)
    return mapPos
def mapToPos(coord,origin,res):
    pos = coord*res+origin
    return pos

def get_obstacles(maze):
    obstacles = np.transpose(np.where(maze == 1))
    cardCells = [(1,0,0),(-1,0,0),(0,1,0),(0,-1,0),(0,0,1),(0,0,-1)]
    outerObs = []
    for obs in obstacles:
        nextDoor = obs + cardCells
        for nbh in nextDoor:
            obsNbh = 0
            try:
                obsNbh += 1-maze[nbh[0],nbh[1],nbh[2]]
            except:
                pass
        if obsNbh>0:
            outerObs.append(obs)
    return outerObs

def put_obs_on_map(obstacles,maze,origin,res,inflate,mapSize):
    for x in obstacles:
        obPos = posToMap(x[:2],origin,maze,res)
        leftx = int(max(obPos[0]-inflate,0))
        rightx = int(min(obPos[0]+inflate,mapSize))
        lefty = int(max(obPos[1]-inflate,0))
        righty = int(min(obPos[1]+inflate,mapSize))
        maze[leftx:rightx,lefty:righty] = 1
    return maze

def inflate_gates(gates,maze,gflate,angleFlate,origin,res):
    neighborCells = [d for d in product((-1, 0, 1), repeat=2) if any(d)]
    goalList = []
    for g in gates:
        height = 1 if g[6] == 0 else 0.525
        goal = np.array(g[:2])
        gyaw = m.atan2(m.sin(g[5]+m.pi/2),m.cos(g[5]+m.pi/2))
        goal = posToMap(goal, origin, maze, res)
        goalList.append(goal)
        goalNeighbors = [(i,j) for i in range(-gflate,gflate+1) for j in range(-gflate,gflate+1)]

        freeQueue = np.array([[0,0]])
        
        angles = gyaw + np.linspace(-angleFlate,angleFlate)
        endPts = []; 
        for ang in angles:
            front = np.round(gflate*np.array([m.cos(ang),m.sin(ang)]))
            endPts.append(front)
            back = np.round(-gflate*np.array([m.cos(ang),m.sin(ang)]))
            endPts.append(back)
            frontInt = front.astype(int)
            backInt = back.astype(int)
            goalInt = np.array([0,0]).astype(int)
            frontBres = list(bresenham(goalInt[0],goalInt[1],frontInt[0],frontInt[1]))
            frontBres = np.array([[elem[0],elem[1]] for elem in frontBres])
            backBres = list(bresenham(goalInt[0],goalInt[1],backInt[0],backInt[1]))
            backBres = np.array([[elem[0],elem[1]] for elem in backBres])
            freeQueue = np.concatenate([freeQueue,frontBres])
            freeQueue = np.concatenate([freeQueue,backBres])
            # freeQueue.append(backBres)
        freeQueue = np.unique(freeQueue,axis = 0)
        
        # for 
        neighborsFixed = []
        removeRows = []
        for idx,row in enumerate(goalNeighbors):
            boolCheck = any(np.equal(freeQueue,row).all(1))
            neighCheck = any(np.equal(neighborCells,row).all(1))
            if not boolCheck and not neighCheck:
                neighborsFixed.append(row)
            else:
                removeRows.append(row)
                           
        goalCells = goal + neighborsFixed
        otherCells = goal + goalNeighbors
        goalCells = goalCells.astype(int)
        for row in goalCells:
            maze[row[0],row[1]] = 1
    return maze,goalList

    

def edge_creation(vertices, maze):
    df_edges = pd.DataFrame({'Source' : [], 'Target' : [], 'Weight': []})
    neighborCells = np.array([d for d in product((-1, 0, 1), repeat=2) if any(d)])
    for vert in vertices:
        if maze[int(vert[0]),int(vert[1])]>0:
            continue
        for row in neighborCells:
            test = vert + row
            
            if np.all(test>=0) and np.all(test<maze.shape[0]):
                valTest = maze[int(test[0]),int(test[1])]
                if valTest<1:
                    weight = np.linalg.norm(vert-test)
                    source = tuple(np.array(vert, dtype=int))
                    sink = tuple(np.array(test, dtype=int))
                    df_edges.loc[len(df_edges.index)] = [source, sink, weight] 
    return df_edges
                
