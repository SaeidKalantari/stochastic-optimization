import numpy as np
from collections import (namedtuple, defaultdict)

from DriverAssignmentModel import contribution


class Graph:
    def __init__(self):
        self.drivernodes = list()
        self.driveramount = []
        
        self.demandnodes = list()
        self.demandamount = []
        self.demcontrib = {}
        
        self.demedges = defaultdict(list)
        self.demweights = {}
        
        
        
        self.supersink = None
        
        self.holdnodes = list()
        self.holdamount = []
        self.holdedges = defaultdict(list)
        self.holdweights = {}
        self.holdvbar = []
        
        self.parallelarr = {}
        self.varr = {}
        
        self.sqGrad = {} #this will store the sum of the squared gradients when using AdaGrad stepsizes.
        
    
    # supersink_node
    def add_supersinknode(self, name):
        self.supersink = name
        self.amount[name] = 0
    
    # node of type (bloodtype, age) for current blood inventory
    def add_bloodnode(self, name):
        self.bloodnodes.append(name)
        self.bloodamount.append(0)
    
    # node - (bloodtype, age)
    def add_demandnode(self, name):
        self.demandnodes.append(name)
        self.demandamount.append(0)
    
    # node - (bloodtype, age)
    def add_holdnode(self, name):
        self.holdnodes.append(name)
        self.holdamount.append(0)
    
    # create an edge between two nodes
    def add_demedge(self, from_node, to_node, weight):
        self.demedges[from_node].append(to_node)
        self.demweights[(from_node, to_node)] = weight
        
    # create an edge between two nodes
    def add_holdedge(self, from_node, to_node, weight):
        self.holdedges[from_node].append(to_node)
        self.holdweights[(from_node, to_node)] = weight 
    
    def add_parallel(self, t, from_node, to_node, parallelarray):
        self.parallelarr[(t, from_node, to_node)] =  parallelarray
            
    def add_varr(self, t, from_node, to_node, varr):
        self.varr[(t, from_node, to_node)] = varr
      
    def add_demcontribArr(self, bldnode,demcontribArr):
        self.demcontrib[bldnode] = demcontribArr
    
    def add_sqGradArr(self, t, bldnode,sqGradArr):
        self.sqGrad[(t,bldnode)] = sqGradArr


def create_dri_net(params):
    # create the network
    Dr_Net = Graph()
    Dr_Net.supersink = ('supersink', np.inf)
    # (BloodUnit, Age) pairs and respective hold nodes
    for i in params['Bloodtypes']:
        for j in params['Ages']:
            Dr_Net.add_bloodnode((i, str(j)))
            Dr_Net.add_holdnode((i, str(j)))

    # all possible demand nodes
    for i in params['Bloodtypes']:
        for j in params['Surgerytypes']:
            for k in params['Substitution']:
                Dr_Net.add_demandnode((i, j, k))

    #add edges from (bloodunit, age) pairs to suitable demand nodes
    for bld in Dr_Net.bloodnodes:
        for dmd in Dr_Net.demandnodes:
            weight = contribution(params,bld, dmd)
            Dr_Net.add_demedge(bld, dmd, weight)
    
    for bld in Dr_Net.bloodnodes:
        demcontribArr = [contribution(params,bld, dmd) for dmd in Dr_Net.demandnodes]
        Dr_Net.add_demcontribArr(bld,demcontribArr)


    # add edges from blood nodes to hold nodes
    for bld in Dr_Net.bloodnodes:
        for hld in Dr_Net.holdnodes:
            if bld[0] == hld[0] and bld[1] == hld[1]:
                Dr_Net.add_holdedge(bld, hld, 0)

    # add parallel edges from hold nodes to supersink
    for t in params['Times']:
        for hld in Dr_Net.holdnodes:
            parArr = np.zeros(params['NUM_PARALLEL_LINKS'])
            vArr = np.zeros(params['NUM_PARALLEL_LINKS'])
            Dr_Net.add_parallel(t, hld, Dr_Net.supersink, parArr)
            Dr_Net.add_varr(t, hld, Dr_Net.supersink, vArr)
            
            sqGradArr = np.zeros(params['NUM_PARALLEL_LINKS'])
            Dr_Net.add_sqGradArr(t, hld, sqGradArr)
            
            
            
            
    return(Dr_Net)

