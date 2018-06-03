import os
import numpy as np
import tensorflow as tf
import pickle
import re

config = tf.ConfigProto(allow_soft_placement = True)
config.gpu_options.allow_growth = True
config.gpu_options.per_process_gpu_memory_fraction = 0.5

class DataFormater(object):
    def __init__(self):
        self.embeddingArr = None
        self.wordNum = None
        self.wordIndDict = None
        self.wordSet = None
        self.ySet = None
        self.yNum = None
        self.yIndDict = None
        self.trainXList = None
        self.trainYList = None
        self.trainXQList = None
        self.maxSeqLen = None
        self.trainXArr = None
        self.trainYArr = None
        self.trainXQArr = None
        self.seqLenList = None
        self.seqLenArr = None
        self.embedDim = None

    def loadEmbedTable(self, fileName):
        wordList = []
        embeddingList = []
        
        with open(fileName) as inFile:
            for line in inFile:
                lineList = line.rstrip().split(' ')
                wordList += [lineList[0]]
                embeddingList += [[float(x) for x in lineList[1:]]]
        
        embeddingList += [[0.0] * len(embeddingList[0])]
        
        self.embedDim = len(embeddingList[0])
        self.embeddingArr = np.array(embeddingList, dtype = np.float64)
        self.wordNum = len(wordList)
        self.wordIndDict = {wordList[ind]: ind for ind in range(self.wordNum)}
        self.wordSet = set(list(self.wordIndDict.keys()))

    def loadTrainData(self, fileName):
        self.trainXList = []
        self.trainYList = []
        self.trainXQList = []
        self.seqLenList = []
        
        with open(fileName) as inFile:
            lineState = 0
            for line in inFile:
                if lineState == 0:
                    wordList = re.findall(r"[\w'<>/]+|[.,!?;]", line.rstrip().split('\t')[1][1:-1])
                    length = len(wordList)
                    self.seqLenList += [length]
                    
                    qList = [[0] for _ in range(length)]
                    for wordInd in range(len(wordList)):
                        if wordList[wordInd].startswith('<e'):
                            wordList[wordInd] = wordList[wordInd][4:-5]
                            qList[wordInd][0] = 1

                    self.trainXList += [np.array(wordList)]
                    self.trainXQList += [qList]

                elif lineState == 1:
                    self.trainYList += [[line.rstrip()]]

                lineState = (lineState + 1) % 4

        self.ySet = set([x[0] for x in self.trainYList])
        self.yNum = len(self.ySet)
        yList = list(self.ySet)
        self.yIndDict = {yList[ind]: ind for ind in range(self.yNum)}

    def tokenToInd(self):
        self.maxSeqLen = max([len(x) for x in self.trainXList])
        self.trainXArr = np.array([[self.wordIndDict.get(word, self.wordNum) for word in x] + [self.wordNum] * (self.maxSeqLen - len(x)) for x in self.trainXList])
        self.trainYArr = np.array([[self.yIndDict[y[0]]] for y in self.trainYList])
        self.trainXQArr = np.array([q + [[0] for _ in range(self.maxSeqLen - len(q))] for q in self.trainXQList])
        self.seqLenArr = np.array(self.seqLenList)

    def saveEmbedTable(self, fileName):
        with open(fileName, 'wb') as inFile:
            pickle.dump(self.embeddingArr, inFile)
        print ('embeddingArr is saved as %s' % fileName)

    def loadEmbedTableFile(self, fileName):
        with open(fileName, 'rb') as outFile:
            self.embeddingArr = pickle.load(outFile)

class BiDirRnnModel():
    def __init__(self):
        self.inputXPH = None
        self.inputYPH = None
        self.inputSeqLenPH = None
        self.inputXQPH = None
        self.loss = None

    def buildModel(self, embedDim, seqLen, yNum, wtStddev, biasStddev, embedTable, rnnHiddenDim):

        with tf.name_scope('input_PH'):
            self.inputXPH = tf.placeholder('int32', [None, seqLen])
            self.inputYPH = tf.placeholder('int32', [None, 1])
            self.inputSeqLenPH = tf.placeholder('int32', [None])
            self.inputXQPH = tf.placeholder('float64', [None, seqLen, 1])

        with tf.name_scope('embedding'):
            embedTable = tf.constant(embedTable, dtype = tf.float64)
            embeddedSeqPH = tf.nn.embedding_lookup(embedTable, self.inputXPH)
            fullEmbeddedSeqPH = tf.concat([embeddedSeqPH, self.inputXQPH], axis = 2)

        with tf.name_scope('bi-directional_rnn'):
            fwLstmCell = tf.nn.rnn_cell.BasicLSTMCell(rnnHiddenDim, state_is_tuple = False)
            bwLstmCell = tf.nn.rnn_cell.BasicLSTMCell(rnnHiddenDim, state_is_tuple = False)

            tupTup=tf.nn.bidirectional_dynamic_rnn(fwLstmCell, bwLstmCell, fullEmbeddedSeqPH, self.inputSeqLenPH, dtype='float64')
            (fwOutput, bwOutput), (fwState, bwState) = tupTup
            biOutput = tf.concat([fwOutput, bwOutput], axis = 2)
            biState = tf.concat([fwState, bwState], axis = 1)

        with tf.name_scope('fully_connected_layer'):
            outputPH = self.fcLayer(biState, yNum, wtStddev, biasStddev)
            lossPre = tf.nn.softmax_cross_entropy_with_logits(labels = tf.one_hot(self.inputYPH, yNum), logits = outputPH)
            self.loss = tf.reduce_sum(lossPre)

    def trainModel(self, epochNum, learningRate, batchSize, trainXArr, trainYArr, trainXQArr, seqLenArr):
        optimizer = tf.train.AdamOptimizer(learning_rate = learningRate).minimize(self.loss)
        feedDict = {self.inputXPH: trainXArr, self.inputYPH: trainYArr, self.inputXQPH: trainXQArr, self.inputSeqLenPH: seqLenArr}
        batchNum = len(trainXArr) // batchSize

        with tf.Session(config = config) as sess:
            sess.run(tf.global_variables_initializer())
            for epochInd in range(epochNum):
                print ('training epoch %d...' % epochInd)
                feedDict = self.shuffleDict(feedDict)

                for batchInd in range(batchNum):
                    batchFeedDict = self.getBatch(feedDict, batchSize, batchInd * batchSize)
                    sess.run(optimizer, feed_dict = batchFeedDict)
                
                print ('loss: ', sess.run(self.loss, feed_dict = feedDict))

    def shuffleDict(self, inDict):
        aKey = list(inDict.keys())[0]
        permutation = np.random.permutation(len(inDict[aKey]))
        return {k: v[permutation] for k, v in inDict.items()}

    def getBatch(self, inDict, batchSize, startInd):
        return {k: v[startInd: startInd + batchSize] for k, v in inDict.items()}

    def fcLayer(self, inPH, outDim, wtStddev, biasStddev):
        inDim = tf.cast(inPH.shape[1], tf.int32)
        wt = tf.Variable(tf.random_normal([inDim, outDim], stddev = wtStddev, dtype = 'float64'), dtype = 'float64')
        bias = tf.Variable(tf.random_normal([outDim], stddev = biasStddev, dtype = 'float64'), dtype = 'float64')
        
        return tf.matmul(inPH, wt) + bias

if __name__ == '__main__':
    df = DataFormater()
    df.loadEmbedTableFile('embedTable.pkl')
    df.loadEmbedTable('glove/glove.6B.50d.txt')
    df.loadTrainData('TRAIN_FILE.txt')
    df.tokenToInd()
    #df.saveEmbedTable('embedTable.pkl')

    bdmd = BiDirRnnModel()
    bdmd.buildModel(embedDim = df.embedDim, seqLen = df.maxSeqLen, yNum = df.yNum, wtStddev = 0.1, biasStddev = 0.01, embedTable = df.embeddingArr, rnnHiddenDim = 32)
    bdmd.trainModel(epochNum = 10000, learningRate = 0.001, batchSize = 128, trainXArr = df.trainXArr, trainYArr = df.trainYArr, trainXQArr = df.trainXQArr, seqLenArr = df.seqLenArr)
