from PIL import Image
import os

os.chdir(os.getcwd() + "/Desktop/Praktikum_Tracking_and_Detection_in_CV/data_task1/init_texture1/")

image = Image.open(r'DSC_9750.png')
image.thumbnail((1000, 1000))
image.show()

print(image.size)
image.save('DSC_9750_new.png')
