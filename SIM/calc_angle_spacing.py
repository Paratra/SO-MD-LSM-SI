import numpy as np


def get_ang(x1,y1,x2,y2):
    ang = np.arctan2((x2-x1),(-y2+y1))
    return -ang

def get_spacing(x1,y1,x2,y2,dx,image_size):
    spacing = 1/(np.sqrt((x2-x1)**2+(y2-y1)**2) * 1/image_size/dx)
    return spacing

# print(get_ang(x1=64,y1=64,x2=40,y2=64))
# print(get_spacing(x1=64,y1=64,x2=40,y2=64,dx=0.135,image_size=128))

# print(get_ang(x1=64,y1=64,x2=75,y2=42))
# print(get_spacing(x1=64,y1=64,x2=75,y2=42,dx=0.135,image_size=128))

# print(get_ang(x1=64,y1=64,x2=52,y2=42))
# print(get_spacing(x1=64,y1=64,x2=52,y2=42,dx=0.135,image_size=128))

angs = [get_ang(x1=64,y1=64,x2=40,y2=64), get_ang(x1=64,y1=64,x2=75,y2=42), get_ang(x1=64,y1=64,x2=52,y2=42)]
spacings = [get_spacing(x1=64,y1=64,x2=40,y2=64,dx=0.135,image_size=128), get_spacing(x1=64,y1=64,x2=75,y2=42,dx=0.135,image_size=128), get_spacing(x1=64,y1=64,x2=52,y2=42,dx=0.135,image_size=128)]

print(angs)
print(spacings)