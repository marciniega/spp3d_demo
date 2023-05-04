#!/usr/bin/env python
import numpy as np
import sys

def write_xplor_file(name,grid,last,cell_size):
    npts = grid.shape
    out_file=open('%s.xplor'%name,'w')
    out_file.write('\n')
    out_file.write('       2 !NTITLE \n')
    out_file.write(' REMARKS FILENAME="" \n')
    out_file.write(' REMARKS DATE:       created by user: common  \n')
    out_file.write('%8s%8s%8s%8s%8s%8s%8s%8s%8s\n'%(npts[0]-1,0,npts[0]-1,
                                                    npts[1]-1,0,npts[1]-1,
                                                    npts[2]-1,0,npts[2]-1))
    out_file.write('{:12.5E}{:12.5E}{:12.5E}{:12.5E}{:12.5E}{:12.5E}\n'.format(cell_size[0],
                                                                               cell_size[1],
                                                                               cell_size[2],
                                                                               90.,90.,90.))
    out_file.write('ZYX\n')
    for c in range(npts[2]):
        out_file.write('%8s\n'%c)
        count = 0
        for b in range(npts[1]):
            for a in range(npts[0]):
                out_file.write('{:12.5E}'.format(grid[a,b,c]))
                count += 1
                if count%6==0 or count==last:
                     out_file.write('\n')
                else:
                    continue
    out_file.write('%8s\n'%(-9999))
    out_file.write('{:12.4E}{:12.4E}\n'.format(np.mean(grid),np.std(grid)))

def read_dens_file(filename):
   with open(filename,'r') as datfil:
        datafile=datfil.readlines()
        gridsize=[ int(i) for i in datafile[0].split(',')]
        dens_grid_file = np.zeros(gridsize)
        for line in datafile[2:]:
            if line[0:7]=="CHANNEL":
                chnl=int(line[7:])
                continue
            elif line[0:3]=="END":
                continue
            else:
                line=line.split(',')
                x,y,z=int(line[0]),int(line[1]),int(line[2])
                dens=float(line[3])
                dens_grid_file[chnl,x,y,z]=dens
        return dens_grid_file

def read_npz_file(filename):
    with np.load(filename,'r') as df:
         return df['arr_0']

filetarget=sys.argv[1]
g_size=np.array([int(i) for i in sys.argv[2].split(',') ])
outname=filetarget
step=0.5

dens_grid=read_npz_file("npzs/%s.npz"%filetarget)       
g_length=dens_grid.shape[1:]
n_channels=dens_grid.shape[0]
path="/home/marciniega/Things_ulises"
folder_xpls=path+"/xpls"
last=g_length[0]*g_length[1]
write_xplor_file(folder_xpls+"/"+outname,
                 dens_grid[0],
                 last,
                 g_size)
