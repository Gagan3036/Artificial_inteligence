import os
os.chdir('C:\\Users\\gagan\\Desktop\\Artificial_inteligence\\Image_classification\\Code\\simple_images\\harrypotter')
i=1
for file in os.listdir():
    src=file
    dst="0"+"_"+str(i)+".jpg"
    os.rename(src,dst)
    i+=1

