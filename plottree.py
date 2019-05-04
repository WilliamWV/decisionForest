import graphviz as grv
import copy
from time import localtime, strftime
import decisiontree as dt
import os

os.environ["PATH"] += os.pathsep + 'D:/ProgramFiles/VSUtilities/Library/bin/graphviz'

class PlotTree:

    #colors = ['aquamarine', 'snow4', 'bisque', 'lightblue2', 'brown', 'cyan3', 'darkorchid', 'darkslategray', 'gold', 'green4', 'darkseagreen1', 'hotpink3', 'goldenrod4']

    def __init__(self, tree):
        self.colors = []
        for i in range(9):
            color = "/set39/"+str(i+1)
            self.colors.append(color)
        self.colors_copy = copy.deepcopy(self.colors)
        self.tree = tree
        self.graph = grv.Digraph()
        self.graph.node_attr.update(style='filled')

    def chooseColor(self):
        if len(self.colors_copy) == 0:
            self.colors_copy = copy.deepcopy(self.colors)
        color = self.colors_copy[0]
        del self.colors_copy[0]
        return color

    def subtreesToGraphviz(self, tree, parentName, key, color):
        if tree.isAnswerNode():
            nodeName = parentName+" "+key+"->"
            nodeInfo = "Answer node with answer: " + tree.answer
            self.graph.node(nodeName, nodeInfo, color=color)
            self.graph.edge(parentName, nodeName, label=" "+key)
        else:
            nodeName = parentName+" "+key+"->"
            nodeInfo = "Decision node with best attribute: " + tree.questionAttr.attrName + "\n"
            nodeInfo += "Gain of \"" + tree.questionAttr.attrName + "\": {:.3f}".format(tree.nodeGain[0])
            self.graph.node(nodeName, nodeInfo, color=color)
            self.graph.edge(parentName, nodeName, label=" "+key)
            color = self.chooseColor()
            for key in tree.subtrees:
                if tree.questionAttr.isCategoric():
                    self.subtreesToGraphviz(tree.subtrees[key], nodeName, key, color)
                else:
                    if key == dt.NUM_GREATER:
                        self.subtreesToGraphviz(tree.subtrees[key], nodeName, " > " + "{:.3f}".format(tree.questionAttr.cutPoint), color)
                    else:
                        self.subtreesToGraphviz(tree.subtrees[key], nodeName, " <= " + "{:.3f}".format(tree.questionAttr.cutPoint), color)

    def decisionTreeToGraphviz(self, tree, color):
        nodeName = tree.questionAttr.attrName
        if tree.isAnswerNode():
            nodeInfo = "Answer node with answer: " + tree.answer        
            self.graph.node(nodeName, nodeInfo, color=color)
        else:
            nodeInfo = "Decision node with best attribute: " + tree.questionAttr.attrName + "\n"
            nodeInfo += "Gain of \"" + tree.questionAttr.attrName + "\": {:.3f}".format(tree.nodeGain[0])
            self.graph.node(nodeName, nodeInfo, color=color)
            color = self.chooseColor()
            for key in tree.subtrees:
                if tree.questionAttr.isCategoric():
                    self.subtreesToGraphviz(tree.subtrees[key], nodeName, key, color)
                else:
                    if key == dt.NUM_GREATER:
                        self.subtreesToGraphviz(tree.subtrees[key], nodeName, " > " + "{:.3f}".format(tree.questionAttr.cutPoint), color)
                    else:
                        self.subtreesToGraphviz(tree.subtrees[key], nodeName, " <= " + "{:.3f}".format(tree.questionAttr.cutPoint), color)

    def drawTree(self):
        time = strftime("%d_%m_%Y_%H_%M_%S", localtime())
        self.decisionTreeToGraphviz(self.tree, self.chooseColor())
        s = grv.Source(self.graph, filename="decisionTree_"+time, format="png")
        s.view()