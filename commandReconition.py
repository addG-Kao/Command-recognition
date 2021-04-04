import os
from librosa.feature import mfcc
import librosa 
from hmmlearn import hmm
import numpy as np
import matplotlib.pyplot as plt


train_dir = os.getcwd() + '/adapt_train'
test_dir = os.getcwd() + '/adapt_test'

# utility
#   提取mfcc特徵值(目前未調整參，直接採用librosa的mfcc套件)
# input
#   要做mfcc的檔案路徑
# output
#   特徵值
def mfcc_process(path):
    
    sampling_freq, audio = librosa.load(path)
    mfcc_features = mfcc(sampling_freq,audio) # 提取MFCC特徵
    mfcc_features = mfcc_features.transpose() #維度0(mfcc coefisient係數) 和1(time)交換
    #print('\nNumber of windows =', mfcc_features.shape[0]) 
    #print('Length of each feature =', mfcc_features.shape[1])
    
    '''
    繪製特徵圖
    
    mfcc_features = mfcc_features.T 
    plt.matshow(mfcc_features) 
    plt.title('MFCC')
    '''
    return mfcc_features

# utility
#   建立dataset
# input
#   第幾個資料夾(int)
# output
#   dataset
def buildDataset(t_dir, f):
    dataset = {}
    for x in range(len(f)):
        if len(f[x])<=2 :    # 因為目前檔案很多 所以先找出檔名<2的來篩掉.wav
            dir = t_dir + '/' + f[x] # f[x]為編號
            print(dir)
            file_list = [file_number for file_number in os.listdir(dir) if os.path.splitext(file_number)[1] == '.wav']
            
            
            for file_name in file_list:
                feature =  mfcc_process(dir + '/' +file_name) #  呼叫mfcc提取特徵
                if f[x] not in  dataset.keys():
                    dataset[f[x]] = []
                    dataset[f[x]].append(feature)
                else:
                    dataset[f[x]].append(feature)
    return dataset

# utility
#   訓練模型
# input
#   dataset
# output
#   訓練完的hmm模型
def HMMTrain(dataset):
    
    hmmModels = {}
    
    for label in dataset.keys(): # 每個label都去train一個model
        
        # Prepare dataset
        trainData = np.array(dataset[label])
        dataSetLength = [i.shape[0] for i in trainData] # 取時間的長度
        # print([i.shape for i in trainData])
        trainData = np.concatenate(trainData, axis=0) # 用維度0(時間)去接
        
        hmmModel = hmm.GaussianHMM(n_components=4) # 建立model的架構
        hmmModel.fit(trainData, lengths=dataSetLength) # 訓練model(猜他應該是只能切維度0)
    
        hmmModels[label] = hmmModel
    
    return hmmModels

if __name__ == '__main__':
    f_train = [file_Number for file_Number in os.listdir(train_dir)]
    
    trainDataset = buildDataset(train_dir, f_train)
    

    print("Finish prepare the training data")
    hmmModels = HMMTrain(trainDataset)
    
    f_test = [file_Number for file_Number in os.listdir(test_dir)]
    testDataSet = buildDataset(test_dir, f_test)
    
    correctCnt = 0 # 記錄對的次數
    totalCnt = 0 
    for trueLabel in testDataSet.keys():
        singleTestDataSet = testDataSet[trueLabel]
        for testData in singleTestDataSet:
            
            print("True label: {}; Predict Score: ".format(trueLabel), end='')
            
            prob_list = []
            for predLabel in hmmModels.keys():
                model = hmmModels[predLabel]
                score = model.score(testData)
                
                # Output the resule
                print("{:.02f} ".format(score), end='')
                prob_list.append(score)
            idx = 0
            for i in range(len(prob_list)):
                if prob_list[i] > prob_list[idx]:
                    idx = i
            if(idx) == int(trueLabel):
                correctCnt = correctCnt+1
            totalCnt = totalCnt+1
            
            print("predict result :", int(idx) == int(trueLabel))
    print(round(correctCnt*100/totalCnt,3), '%');
    
