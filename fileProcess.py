import os
import librosa 
import math
from ffmpeg import audio

train_dir = os.getcwd() + '/test'

# utility
#   讀取檔案
# input
#   路徑(str)、第幾個資料夾(int)
# output
#   每筆檔案的時間長度取上高斯
def readFile(tmp_dir,dir_num):
    dir = tmp_dir + '/' + dir_num
    wav_data = [file_number for file_number in os.listdir(dir)]
    time = 0
    num = 0
    
    for getfile in wav_data:
        # print(time,num)
        if getfile.endswith(".wav"):  #判斷是否為wav檔
            time += librosa.get_duration(filename = dir+"/"+getfile)
            num += 1
    # print(time/num, "\n")        
    return math.ceil(time/num) 

# utility
#   將調整音頻倍率使長度相同
# input
#   路徑(str)、第幾個資料夾(int)、音訊長度(float)
# output
#   以檔案形式輸出
def fileProcess(tmp_dir,dir_num,goal_time):
    goal_dir = os.getcwd()+'/adapt_test/'+dir_num
    if not os.path.isdir(goal_dir):
        os.makedirs(goal_dir)
    print(goal_dir)
    dir = tmp_dir + '/' + dir_num
    wav_data = [file_number for file_number in os.listdir(dir)]
    
    for getfile in wav_data:
        if getfile.endswith(".wav"):
            n = librosa.get_duration(filename = dir+"/"+getfile)/goal_time
            #print(n)
            audio.a_speed(dir+"/"+getfile, str(n) , goal_dir +'/'+ getfile)
       

if __name__ == '__main__':
    f = [file_Number for file_Number in os.listdir(train_dir)]
    for x in range(len(f)):
        if len(f[x])<=2 :    # 因為目前檔案很多 所以先找出檔名<2的來篩掉.wav
            avg_time = readFile(train_dir,f[x])
            fileProcess(train_dir, f[x], avg_time)