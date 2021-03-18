import os
import librosa 
import filetype


train_dir = os.getcwd() + '/train'

# utility
#   對音頻進行處理
# input
#   路徑(str)、第幾個資料夾(int)
# output
#   每筆檔案的時間長度
def fileProcess(tmp_dir,dir_num):
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
    return time/num        

if __name__ == '__main__':
    f = [file_Number for file_Number in os.listdir(train_dir)]
    for x in range(len(f)):
        if len(f[x])<=2 :    # 因為目前檔案很多 所以先找出檔名<2的來篩掉.wav
            avg_time = fileProcess(train_dir,f[x])
            print(avg_time)