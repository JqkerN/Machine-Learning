import monkdata as m
import matplotlib.pyplot as plt
import random
import time
from dtree import entropy, averageGain, mostCommon, buildTree, select, check, bestAttribute, allPruned
from drawtree_qt5 import drawTree


def ASSIGNMENT_1():
    print("ASSIGNMENT(1)")
    print("Entropy: ")
    print("Monk 1: %f" %entropy(m.monk1))
    print("Monk 2: %f" %entropy(m.monk2))
    print("Monk 3: %f" %entropy(m.monk3))

def ASSIGNMENT_3():
    print(" ")
    print("ASSIGNMENT(3)")
    print("GAIN: ")
    print("DATASET:%s       %s        %s        %s        %s        %s"%(m.attributes[0], 
                                         m.attributes[1], 
                                         m.attributes[2], 
                                         m.attributes[3], 
                                         m.attributes[4], 
                                         m.attributes[5]))
    print("Monk 1: %f %f %f %f %f %f" %(averageGain(m.monk1, m.attributes[0]), 
                                        averageGain(m.monk1, m.attributes[1]), 
                                        averageGain(m.monk1, m.attributes[2]), 
                                        averageGain(m.monk1, m.attributes[3]), 
                                        averageGain(m.monk1, m.attributes[4]), 
                                        averageGain(m.monk1, m.attributes[5])))
    print("Monk 2: %f %f %f %f %f %f" %(averageGain(m.monk2, m.attributes[0]), 
                                        averageGain(m.monk2, m.attributes[1]), 
                                        averageGain(m.monk2, m.attributes[2]), 
                                        averageGain(m.monk2, m.attributes[3]), 
                                        averageGain(m.monk2, m.attributes[4]), 
                                        averageGain(m.monk2, m.attributes[5])))
    print("Monk 3: %f %f %f %f %f %f" %(averageGain(m.monk3, m.attributes[0]), 
                                        averageGain(m.monk3, m.attributes[1]), 
                                        averageGain(m.monk3, m.attributes[2]), 
                                        averageGain(m.monk3, m.attributes[3]), 
                                        averageGain(m.monk3, m.attributes[4]), 
                                        averageGain(m.monk3, m.attributes[5])))

def ASSIGNMENT_5():
    print(" ")
    print("ASSIGNMENT(5)")
    print("ERROR:")
    t = buildTree(m.monk1, m.attributes)
    print("MONK-1      %f      %f" %(1-check(t,m.monk1),
                                     1-check(t,m.monk1test)))
    t = buildTree(m.monk2, m.attributes)
    print("MONK-2      %f      %f" %(1-check(t,m.monk2),
                                     1-check(t,m.monk2test)))
    t = buildTree(m.monk3, m.attributes)
    print("MONK-3      %f      %f" %(1-check(t,m.monk3),
                                     1-check(t,m.monk3test)))

def ASSIGNMENT_7():
    print(" ")
    print("ASSIGNMENT(7)")
    print("MONK-1:")
    get_pruned_tree_with_plots(m.monk1, m.monk1test)
    print(" ")
    print("MONK-3:")
    get_pruned_tree_with_plots(m.monk3, m.monk3test)

def PRINT_TREE_AT_LEVEL_2():
    # A5
    print(" ")
    print("LEVEL 1:")
    print(m.attributes[4])
    Att = [None]*4
    for value in range(1,5):
        Att[value-1] = select(m.monk1, m.attributes[4], value)

    print("LEVEL 2:")
    for A in Att:
        tmp = bestAttribute(A, m.attributes)
        print(tmp)
        if tmp == m.attributes[0]:
            for value in range(1,4):
                print(mostCommon( select(A, tmp, value)))
        if tmp == m.attributes[1]:
            for value in range(1,4):
                print(mostCommon( select(A, tmp, value)))
        if tmp == m.attributes[2]:
            for value in range(1,3):
                print(mostCommon( select(A, tmp, value)))
        if tmp == m.attributes[3]:
            for value in range(1,4):
                print(mostCommon( select(A, tmp, value)))
        if tmp == m.attributes[4]:
            for value in range(1,5):
                print(mostCommon( select(A, tmp, value)))
        if tmp == m.attributes[5]:
            for value in range(1,3):
                print(mostCommon( select(A, tmp, value)))
    print(" ")
    t = buildTree(m.monk1, m.attributes)
    drawTree(t)

def get_pruned_tree_with_plots(trainSET, validSET):
    fraction = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
    meanError = 0
    iterations = 1000
    savedError = []
    errorVec = []
    fracVec = []
    data = []
    E_tmp = []
    
    for f in fraction:
        for index in range(iterations):
            monktrain, monkval = partition(trainSET, f)
            motherTree = buildTree(monktrain, m.attributes)
            thres = check(motherTree, monkval)   
            tree = pruning(motherTree, monkval, thres)  
            meanError += 1 - check(tree, validSET) 
            errorVec.append(1 - check(tree, validSET))
            fracVec.append(f)
            E_tmp.append(1 - check(tree, validSET))
        print("Classification error: %f     with fraction %.1f" %(meanError/iterations, f) )
        savedError.append(meanError/iterations)
        meanError = 0
        data.append(E_tmp)
        E_tmp = []
    
    plt.subplot(2, 1, 1)
    plt.scatter(fracVec, errorVec, color = "red")
    plt.plot(fraction, savedError, 'k-')
    plt.axis([0.25, 0.85, 0, 0.6])
    plt.grid()
    plt.title("Error plot - Assignment 7")
    plt.ylabel("Error")
    plt.xlabel("Fraction")
    plt.legend(["Mean error", "Error data-points"])

    plt.subplot(2, 1, 2)
    plt.boxplot(data, positions=fraction, widths=0.08, showmeans=True)
    plt.axis([0.25, 0.85, 0, 0.6])
    plt.ylabel("Error")
    plt.xlabel("Fraction")
    plt.title("Error Boxplot - Assignment 7")
    plt.grid()
    plt.show()

def pruning(motherTree, validation, thres):
    newTree = motherTree
    for prunedTree in allPruned(motherTree):
        if check(prunedTree, validation) >= thres:
            return pruning(prunedTree, validation, thres)
    return newTree

def partition(data, fraction):
    ldata = list(data)
    random.shuffle(ldata)
    breakPoint = int(len(ldata) * fraction)
    return ldata[:breakPoint], ldata[breakPoint:]


def main():
    print("STARTING...")
    # PRINT_TREE_AT_LEVEL_2() # UNCOMMENT FOR TREE AT LEVEL 2.
    ASSIGNMENT_1()
    ASSIGNMENT_3()
    ASSIGNMENT_5()
    ASSIGNMENT_7()
    print("...FINNISHED")

if __name__ == '__main__':
    main()









