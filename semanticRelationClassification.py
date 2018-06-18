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
        self.testXList = None
        self.testYList = None
        self.testXQList = None
        self.tesSeqLenList = None
        self.trainXArr = None
        self.trainYArr = None
        self.trainXQArr = None
        self.testXArr = None
        self.testYArr = None
        self.testXQArr = None
        self.testSeqLenArr = None
        self.seqLenList = None
        self.seqLenArr = None
        self.embedDim = None
        self.indToYDict = None

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

    def loadTestData(self, xFileName, yFileName):
        self.testXList = []
        self.testXQList = []
        self.testSeqLenList = []
        self.testYList = []

        with open(xFileName) as inFile:
            for line in inFile:
                wordList = re.findall(r"[\w<>/]+|[.,!?;]", line.rstrip().split('\t')[1][1:-1])
                length = len(wordList)
                self.testSeqLenList += [length]
                
                qList = []
                for wordInd in range(len(wordList)):
                    word = wordList[wordInd]
                    if word.startswith('<') and word.endswith('>'):
                        wordList[wordInd] = word[4:-5]
                        qList += [wordInd]
                    elif word.endswith('>'):
                        wordList[wordInd] = word[:-5]
                        qList += [wordInd]
                    elif word.startswith('<'):
                        wordList[wordInd] = word[4:]
                
                self.testXList += [np.array(wordList)]
                self.testXQList += [qList]
        
        with open(yFileName) as inFile:
            for line in inFile:
                self.testYList += [[line.rstrip().split('\t')[1]]]

        print ('maxSeqLen for test data: ', max([len(x) for x in self.testXList]))

    def loadTrainData(self, fileName):
        self.trainXList = []
        self.trainYList = []
        self.trainXQList = []
        self.seqLenList = []
        
        with open(fileName) as inFile:
            lineState = 0
            for line in inFile:
                if lineState == 0:
                    wordList = re.findall(r"[\w<>/]+|[.,!?;]", line.rstrip().split('\t')[1][1:-1])
                    length = len(wordList)
                    self.seqLenList += [length]
                    
                    qList = []
                    for wordInd in range(len(wordList)):
                        word = wordList[wordInd]
                        if word.startswith('<') and word.endswith('>'):
                            wordList[wordInd] = word[4:-5]
                            qList += [wordInd]
                        elif word.endswith('>'):
                            wordList[wordInd] = word[:-5]
                            qList += [wordInd]
                        elif word.startswith('<'):
                            wordList[wordInd] = word[4:]

                    self.trainXList += [np.array(wordList)]
                    self.trainXQList += [qList]

                elif lineState == 1:
                    self.trainYList += [[line.rstrip()]]

                lineState = (lineState + 1) % 4

        self.ySet = set([x[0] for x in self.trainYList])
        self.yNum = len(self.ySet)
        yList = list(self.ySet)
        self.yIndDict = {yList[ind]: ind for ind in range(self.yNum)}
        self.indToYDict = {v: k for k, v in self.yIndDict.items()}

    def tokenToInd(self):
        self.maxSeqLen = max([len(x) for x in self.trainXList])
        self.trainXArr = np.array([[self.wordIndDict.get(word, self.wordNum) for word in x] + [self.wordNum] * (self.maxSeqLen - len(x)) for x in self.trainXList])
        self.trainYArr = np.array([[self.yIndDict[y[0]]] for y in self.trainYList])
        self.trainXQArr = np.array(self.trainXQList)
        self.seqLenArr = np.array(self.seqLenList)
        print ('shape of trainXQArr: ', self.trainXQArr.shape)

        self.testXArr = np.array([[self.wordIndDict.get(word, self.wordNum) for word in x] + [self.wordNum] * (self.maxSeqLen - len(x)) for x in self.testXList])
        self.testYArr = np.array([[self.yIndDict[y[0]]] for y in self.testYList])
        self.testXQArr = np.array(self.testXQList)
        self.testSeqLenArr = np.array(self.testSeqLenList)

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

    def buildModel(self, embedDim, seqLen, yNum, wtStddev, biasStddev, embedTable, rnnHiddenDim, useRnnOutput, fcNum):

        with tf.name_scope('input_PH'):

            self.inputXPH = tf.placeholder('int32', [None, seqLen])
            self.inputYPH = tf.placeholder('int32', [None, 1])
            self.inputSeqLenPH = tf.placeholder('int32', [None])
            self.inputXQPH = tf.placeholder('int32', [None, 2])
            self.dropKeepProPH = tf.placeholder_with_default(1.0, shape = [])

        with tf.name_scope('embedding'):

            embedTable = tf.constant(embedTable, dtype = tf.float64)
            embeddedSeqPH = tf.nn.embedding_lookup(embedTable, self.inputXPH)
            qPH = tf.transpose(tf.one_hot(self.inputXQPH, seqLen, dtype = 'float64'), perm = [0, 2, 1])
            print ('qPH: ', qPH.shape)
            fullEmbeddedSeqPH = tf.concat([embeddedSeqPH, qPH], axis = 2)
            print ('fullEmbeddedSeqPH: ', fullEmbeddedSeqPH.shape)

        with tf.name_scope('bi-directional_rnn'):

            fwLstmCell = tf.nn.rnn_cell.BasicLSTMCell(rnnHiddenDim, state_is_tuple = False)
            bwLstmCell = tf.nn.rnn_cell.BasicLSTMCell(rnnHiddenDim, state_is_tuple = False)

            tupTup=tf.nn.bidirectional_dynamic_rnn(fwLstmCell, bwLstmCell, fullEmbeddedSeqPH, self.inputSeqLenPH, dtype='float64')
            (fwOutput, bwOutput), (fwState, bwState) = tupTup
            biOutput = tf.concat([fwOutput, bwOutput], axis = 2)
            biState = tf.concat([fwState, bwState], axis = 1)
            #print ('shape of fwState, bwState: ', fwState.get_shape(), bwState.get_shape())
            print ('shape of biOutput: ', biOutput.get_shape())
            print ('shape of biState: ', biState.get_shape())

            if useRnnOutput:

                batchSize = tf.shape(self.inputXPH)[0]
                rangePH = tf.range(batchSize, dtype = 'int32')
                indicesPHPre = tf.tile(tf.expand_dims(tf.expand_dims(rangePH, axis = 1), axis = 2), multiples = [1, 2, 1])
                indicesPH = tf.concat([indicesPHPre, tf.expand_dims(self.inputXQPH, axis = 2)], axis = 2)
                qBiOutput = tf.gather_nd(biOutput, indicesPH)
                
                qSh = qBiOutput.get_shape()
                fcInPH = tf.concat([biState, tf.reshape(qBiOutput, [-1, qSh[1] * qSh[2]])], axis = 1)

            else:
                fcInPH = biState

        with tf.name_scope('fully_connected_layer'):

            inNodeNum = 8 * rnnHiddenDim if useRnnOutput else 4 * rnnHiddenDim
            nodeNumList = self.getNodeNumList(inNodeNum, yNum, fcNum)
            
            for layerInd in range(fcNum):
                fcInPH = tf.nn.dropout(fcInPH, keep_prob = tf.cast(self.dropKeepProPH, 'float64')) if layerInd != 0 else fcInPH
                fcInPH = self.fcLayer(fcInPH, nodeNumList[layerInd], wtStddev, biasStddev)
                if layerInd != fcNum-1:
                    fcInPH = tf.nn.relu(fcInPH)
            outputPH = fcInPH

            self.predY = tf.argmax(outputPH, axis = 1)
            lossPre = tf.nn.softmax_cross_entropy_with_logits(labels = tf.one_hot(self.inputYPH, yNum), logits = outputPH)
            self.loss = tf.reduce_sum(lossPre)

    def trainModel(self, epochNum, learningRate, trainXArr, trainYArr, trainXQArr, seqLenArr, testXArr, testYArr, testXQArr, testSeqLenArr, indToYDict, outputFileName, batchSize, dropKeepPro):

        print ('fileName: ', outputFileName)

        optimizer = tf.train.AdamOptimizer(learning_rate = learningRate).minimize(self.loss)
        feedDict = {self.inputXPH: trainXArr, self.inputYPH: trainYArr, self.inputXQPH: trainXQArr, self.inputSeqLenPH: seqLenArr}
        testFeedDict = {self.inputXPH:testXArr,self.inputYPH:testYArr,self.inputXQPH:testXQArr,self.inputSeqLenPH:testSeqLenArr}
        batchNum = len(trainXArr) // batchSize

        with tf.Session(config = config) as sess:
            sess.run(tf.global_variables_initializer())
            for epochInd in range(epochNum):
                print ('training epoch %d...' % epochInd)
                feedDict = self.shuffleDict(feedDict)

                for batchInd in range(batchNum):
                    batchFeedDict = self.getBatch(feedDict, batchSize, batchInd * batchSize)
                    batchFeedDict[self.dropKeepProPH] = dropKeepPro
                    sess.run(optimizer, feed_dict = batchFeedDict)
                
                print ('loss: ', sess.run(self.loss, feed_dict = feedDict))
                print ('training accuracy: ', self.getAcc(sess, feedDict))
                print ('testing  accuracy: ', self.getAcc(sess, testFeedDict))

            self.outputAns(sess, outputFileName, indToYDict, testFeedDict, 8001)

    def outputAns(self, sess, fileName, indToYDict, feedDict, startNum):
        predY = sess.run(self.predY, feed_dict = feedDict)
        with open(fileName, 'w') as outFile:
            for ind in range(len(predY)):
                outFile.write(str(startNum + ind) + '\t' + indToYDict[predY[ind]] + '\n')

    def getAcc(self, sess, feedDict):
        predY = sess.run(self.predY, feed_dict = feedDict)
        return np.mean(np.equal(predY, np.squeeze(feedDict[self.inputYPH])))

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

    def getNodeNumList(self, inDim, outDim, layerNum):
        upList = [inDim]
        downList = [outDim]
        for layerInd in range(layerNum-1):
            if upList[-1] * 1.5 < downList[-1] * 2:
                upList += [upList[-1] * 1.5]
            else:
                downList += [downList[-1] * 2]
        return upList[1:] + downList[::-1]

if __name__ == '__main__':
    df = DataFormater()
    df.loadEmbedTableFile('embedTable.pkl')
    df.loadEmbedTable('glove/glove.6B.50d.txt')
    df.loadTrainData('TRAIN_FILE.txt')
    df.loadTestData('TEST_FILE.txt', 'answer_key.txt')
    df.tokenToInd()
    #df.saveEmbedTable('embedTable.pkl')

    bdmd = BiDirRnnModel()
    bdmd.buildModel(embedDim = df.embedDim, seqLen = df.maxSeqLen, yNum = df.yNum, wtStddev = 0.1, biasStddev = 0.01, embedTable = df.embeddingArr, rnnHiddenDim = 8, useRnnOutput = True, fcNum = 1)
    bdmd.trainModel(epochNum = 100, learningRate = 0.001, trainXArr = df.trainXArr, trainYArr = df.trainYArr, trainXQArr = df.trainXQArr, seqLenArr = df.seqLenArr, testXArr = df.testXArr, testYArr = df.testYArr, testXQArr = df.testXQArr, testSeqLenArr = df.testSeqLenArr, indToYDict = df.indToYDict, outputFileName = 'answer_hid8_useOutput_fc1_drop1.0.txt', batchSize = 128, dropKeepPro = 1.0)
