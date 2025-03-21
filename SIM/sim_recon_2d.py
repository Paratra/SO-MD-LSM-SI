import os, sys
# sys.path.append('C:\\Users\\kner\\Dropbox\\python\\sim\\')
# sys.path.append('C:\\Users\\kner\\Dropbox\\python\\sim\\NSIM\\')
# sys.path.append('C:\\Users\\kner\\Dropbox\\python\\simsxy\\')
import numpy as np
import tifffile as tf
import pylab as plt
import si_recon_2D_h as sir
from pdb import set_trace as st
import h5py
from get_angle_spacing import AngleSpacingCalculator
from PIL import Image

fft = np.fft.fft2
fftshift = np.fft.fftshift

def read_img(img_path):
    file_end = img_path.split('.')[-1]
    if file_end == 'tif' or file_end == 'tiff':
        imgs = tf.imread(img_path)
    elif file_end == 'h5':
        with h5py.File(img_path, 'r') as file:
            # def print_name(name):
            #     print(name)
            # file.visit(print_name)
            # st()
            datasets = []
            for i in range(len(file)):  
                dataset_name = f'{i}'
                if dataset_name in file:
                    datasets.append(file[dataset_name][:])

            imgs = np.stack(datasets, axis=0)

    return imgs


def main():
    # fn = r'../../../../Data_For_Use/data/20240320/worm5_sim9px0000_128.tif'
    # fn = r'./data/beads100nm_9px_sim0256.tif'
    # fn = r'../../../../Data_For_Use/data/20240326/heart_organoid_12px_sim0256.tif'
    # fn = r'../../../../Data_For_Use/data/20240402/beads_sim12px_50um_50ms_512_tif.tif'
    # fn = r'../../../../Data_For_Use/data/20240404/test.tif'
    # fn = r'../../../../Data_For_Use/data/Usable_Data/worm/stack/test_recon/worm13_3_test.tif'
    fn = r'./beads_sim12px_10um_50ms_200nm.tif'
    imgs = read_img(fn)
    
    N_angles = 3
    N_phases = 3

    imgstack = imgs.reshape(-1,N_angles*N_phases,imgs.shape[-2],imgs.shape[-1])
    
    peak_corrs = []
    final_recon_stack = []
    
    for ind, each_plane in enumerate(imgstack):
        # st()
    
        p = sir.si2D(each_plane)
        p.dx = 0.135
        p.na = 0.8
        # p.mu = 0.1 # control the wiener filter
        p.psf = p.getpsf()
        q = sir.phaseest(each_plane)
        # st()
        
        # if ind == 0:
            
        angles, spacings, mags, phases = [], [], [], []
        for i in range(len(q)):
            selector = AngleSpacingCalculator(q[i],dx=p.dx)
            if len(peak_corrs) < N_angles:
                selector.show()
                # brightest_pixel = selector.get_brightest_pixel()
                ang, spacing = selector.get_ang_spacing()
                
                # st()
                # print("angle spacing are:", ang, spacing)
                angles.append(ang)
                spacings.append(spacing)
                peak_corrs.append(selector.get_brightest_pixel())
            else:
                # print(peak_corrs[i])
                ang, spacing = selector.get_ang_spacing_given_brightest_pixel(peak_corrs[i])  
                angles.append(ang)
                spacings.append(spacing)
            # print(angles,spacings)
            # print
        
        # st()
        r_ang = p.r_ang
        r_sp = p.r_sp
        for i in range(len(q)):       
            p.r_ang = r_ang
            p.r_sp = r_sp
            ang = angles[i]
            spacing = spacings[i]
            out = p.mapoverlap2(ang, spacing, ind=i, marr=True) # 12px -- 0.732
            while np.round(ang,3) != np.round(out[0],3) or np.round(spacing,3) != np.round(out[1],3):
                print(f'Re-calculatuion for angle {i}')
                ang = out[0]
                spacing = out[1]
                out = p.mapoverlap2(ang, spacing, ind=i, marr=True)


            print(f'Finer searching for angle {i}')
            p.r_ang = p.r_ang/4
            p.r_sp = p.r_sp/4
            out = p.mapoverlap2(ang, spacing, ind=i, marr=True)
            phase, _ = p.viewoverlap(out[0], out[1], ind=i)
            angles[i] = out[0]
            spacings[i] = out[1]
            # angles.append(out[0])
            # spacings.append(out[1])
            mags.append(out[2])
            phases.append(phase)
            # print(out)    
            # st()
        # print(mags)
    
    
        
        
        
        mags = [1.8,1.8,1.8]
        # for mu_ in np.arange(0.1,1,0.1):
        # st()
        for mu_ in [0.2]:
            save_path = os.path.join('./result/beads/beads_new_prisms/',f'mu_{mu_}')
            os.makedirs(save_path, exist_ok = True) 
            
            save_otf_path = os.path.join('./result/beads/beads_new_prisms/',f'mu_otf_{mu_}')
            os.makedirs(save_otf_path, exist_ok = True) 
            
            # p.mu = 0.1 # control the wiener filter
            p.mu = mu_
            
            
            p.recon(angles, spacings, mags, phases)
            # tf.imwrite('./otf.tif',np.abs(np.fft.fftshift(p.S)))
            # tf.imwrite(f'{save_path}/recon{ind}.tif',abs(p.finalimage))
            image_otf = Image.fromarray(np.abs(np.fft.fftshift(p.S)))
            image_otf.save(f'{save_otf_path}/recon{ind}_otf.tif', format='TIFF')
            
            image = Image.fromarray(abs(p.finalimage))
            image.save(f'{save_path}/recon{ind}.tif', format='TIFF')

            # plt.show()
            # final_recon_stack.append()
            plt.close('all')
            
            
if __name__ == '__main__':
    main()