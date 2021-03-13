import os
import wave
from librosa.feature import mfcc 
import librosa
import matplotlib.pyplot as plt
from hmmlearn import hmm
# test_num = 4

'''
提取梅爾倒頻譜mfcc
'''
def mfcc_process(path):
    sampling_freq, audio = librosa.load(path)
    mfcc_features = mfcc(sampling_freq,audio) # 提取MFCC特徵
    print('\nNumber of windows =', mfcc_features.shape[0]) 
    print('Length of each feature =', mfcc_features.shape[1])
    
    '''
    繪製特稱圖
    mfcc_features = mfcc_features.T 
    plt.matshow(mfcc_features) 
    plt.title('MFCC')
    '''
    return mfcc_features

def buildDataSet(trainPath):
    fileList = [f for f in os.listdir(trainPath) if os.path.splitext(f)[1] == '.wav']
    dataset = {}
    
    for fileName in fileList:
        label = fileName.split('_')[0] # lable為各個指令編號
        feature = mfcc_process(trainPath+fileName) # mfcc特徵
        if label not in dataset.keys():
            dataset[label] = []
            dataset[label].append(feature)
        else:
            dataset[label].append(feature)
    return dataset
    
def HMMTrain(dataset):
    hmm_models = []
    model=hmm.GaussianHMM(n_components=4)
    hmm_models.append(model.fit(dataset))
    
    return hmm_models

    

def main():
    trainPath = (r'C:\Users\User\Desktop\畢專\MyHMM\train\\')
    #print(os.listdir(r'C:\Users\User\Desktop\畢專\MyHMM\train'))
    #filename=os.listdir()
    trainDataSet = buildDataSet(trainPath)
    print("Finish prepare the training data")

    hmmModels = HMMTrain(trainDataSet)
    # hmm_models.append((hmmModels, label))
    print("Finish training of the GMM_HMM models")
    
    testPath = (r'C:\Users\User\Desktop\畢專\MyHMM\test\\')
    testDataSet = buildDataSet(testPath)
    
    '''
    以下是計算正確率的部分
    '''
    score_cnt = 0
    for label in testDataSet.keys():
        
        scoreList = {}
        feature = testDataSet[label]
        for model_label in hmmModels.keys():
            model = hmmModels[model_label]
            score = model.score(feature[0]) # 為甚麼丟的是feature[0]?不是丟自己的feature?
            
            
            scoreList[model_label] = score
        predict = max(scoreList, key = scoreList.get)
        



if __name__ == '__main__':
    main()