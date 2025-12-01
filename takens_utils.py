from sklearn.metrics import pairwise_distances
#from utils import NearestNeighbor, lin_, DelayEmbed
from scipy.optimize import curve_fit
import numpy as np
import matplotlib
from matplotlib import pyplot as plt
import logging
import scipy as sp

logger = logging.getLogger("coordinates_embedding")
from matplotlib.ticker import MaxNLocator

def lin_(x, a, b):
    return a * x + b



def coordinates_embedding(
        t: list, 
        x: list, 
        imdim: int=None, 
        over_embedding: int=0,
        force_embedding: bool=False,
        time_stepping: int=1,
        shift_steps: int=1
    ):
    """
    Returns the n-dim. time series x into a time series of properly embedded
    coordinate system y of dimension p. Optional inputs to be specified as
    'field_name','field value'
        
    Parameters:
    t : list of time vectors
    x : list of observed trajectories 
    imdim - dimension of the invariant manifold to learn
        
    over_embedding (optional): augment the minimal embedding dimension with a number of
                     time delayed measurements, default 0
    force_embedding (optional): force the embedding in the states of x, default false
    time_stepping   (optional): time stepping in the time series, default 1
    shift_steps     (optional): number of timesteps passed between components (but 
                     subsequent measurements are kept intact), default 1

    Returns:
    t_y : list of time vectors

    y : cell array of dimension (N_traj,2) where the first column contains
        time instances (1 x mi each) and the second column the trajectories
        (p x mi each)
    opts_embdedding : options containing the embedding information

    """
    if not imdim:
        raise RuntimeError("imdim not specified for coordinates embedding")
    n_observables = x[0].shape[0] 
    n_n = int(np.ceil( (2*imdim + 1)/n_observables) + over_embedding)

    # Construct embedding coordinate system
    if n_n > 1 and force_embedding != 1:
        p = n_n * n_observables
        # Augment embdedding dimension with time delays
        if n_observables == 1:
            logger.info((
                f'The {str(p)} embedding coordinates consist of the ' +
                f'measured state and its {str(n_n-1)} time-delayed measurements.'
            ))
        else:
            logger.info((
                f'The {str(p)} embedding coordinates consist of the {str(n_observables)} ' +
                f'measured states and their {str(n_n-1)} time-delayed measurements.'
            ))
        t_y = []
        y = []
        for i_traj in range(len(x)):
            t_i = t[i_traj]
            x_i = x[i_traj]

            subsample = np.arange(start=0, stop=len(t_i), step=time_stepping)

            y_i = x_i[:, subsample]
            y_base = x_i[:, subsample]

            for i_rep in range(1, n_n):
                y_i = np.concatenate(
                    (
                        y_i,
                        np.roll(y_base, -i_rep) 
                    )   
                )
            
            y.append(
                y_i[:, :-n_n+1]
            )
            t_y.append(
                t_i[
                    subsample[:-n_n+1]
                ]
            )

    else:
        p = n_observables

        if time_stepping > 1:
            logger.info('The embedding coordinates consist of the measured states.')
            t_y = []
            y = []
            for i_traj in range(len(x)):
                t_i = t[i_traj]
                x_i = x[i_traj]
                subsample = np.arange(start=0, stop=len(t_i), step=time_stepping)
                t_y.append(t_i[subsample])
                y.append(x_i[:, subsample])

        else:
            t_y = t
            y = x

    opts_embdedding = {
        'imdim' : imdim,
        'over_embedding': over_embedding,
        'force_embedding': force_embedding,
        'time_stepping' : time_stepping,
        'shift_steps' : shift_steps,
        'embedding_space_dim': p
    }
    

    return t_y, y, opts_embdedding

def computeCorrDim(data_matrix, ells, el_to_fit,ax, colors = ['darkcyan', 'orange'],):

    # ells is the array with the candidate l values, el_to_fit is the array of indices of l values to use for the fit

    M = np.shape(data_matrix)[0]
    N = np.shape(data_matrix)[1]


    # Calculate pairwise distances
    distances = pairwise_distances(data_matrix)
    # Normalize to 1
    distances = distances / distances.max()

    # To compare results with reduced dynamics


    c_eps = np.zeros(len(ells))
    c_eps_reduced = np.zeros(len(ells))
    for l,ell in enumerate(ells):
        c_eps[l]=(np.sum(distances < ell)-M) /M**2

      
    # Use first len(ells)-el_to_fit values of l to find the params

    # To estimate nu with best linear fit (to the log of C(l))
    popt_full, pcov_full = curve_fit(lin_, np.log10(ells[el_to_fit]), np.log10(c_eps[el_to_fit]))
    error_full = np.sqrt(np.diag(pcov_full))


    ax.loglog(ells, c_eps, '.', c= colors[0])
    ax.loglog(ells, 10**(lin_(np.log10(ells), *popt_full)),'--',
          linewidth=1,  c= colors[0],  label = r'$\nu=%s\pm%s$' %(round(popt_full[0],3), round(error_full[0],3)))

    ax.xaxis.set_major_locator(MaxNLocator(nbins=3))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=3))
    ax.legend( )
    ax.grid(which = 'both')
    ax.set_xlabel(r'$\ell$',)
    ax.set_ylabel(r'$C(\ell)$', )
    return popt_full, error_full

def plot_spectrum(ax, s, fs, range_to_plot=(0,5), nseg =10000):
    data_windowed = s.T-np.mean(s, axis = 1).T
    f, t, Zxx = sp.signal.stft(data_windowed.T, fs, nperseg=nseg)
    ax.pcolormesh(t, f, np.average(np.abs(Zxx), axis = 0),  shading='gouraud')
    ax.xaxis.set_major_locator(MaxNLocator(nbins=3))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=3))
    ax.set_title('STFT Magnitude')
    ax.set_ylabel('Frequency [Hz]')
    ax.set_xlabel('Time [sec]')
    ax.set_ylim(*range_to_plot)


def plot_traj(sol_pc, y_pc, subfig, proj = '3d'):
    if proj=='3d':
        axLeft = subfig.add_subplot(1,2,1, projection = '3d')
        axRight = subfig.add_subplot(1,2,2, projection = '3d')
    axLeft.plot(*sol_pc[:, ],'-', color = 'black')
    axRight.plot(*y_pc[:, ], '-',color = 'black')
    
    for ax in [axLeft, axRight]:
            ax.xaxis.set_major_locator(MaxNLocator(nbins=3))
            ax.yaxis.set_major_locator(MaxNLocator(nbins=3))
            ax.zaxis.set_major_locator(MaxNLocator(nbins=3))
            ax.set_xlabel('$x$')
            ax.set_ylabel('$y$')
            ax.set_zlabel('$z$')


def compare_diagnostics(sol, y, y_pc, fs, ells_full = np.logspace(-3.5,-2.5, 100), ells_delay =np.logspace(-4,-3.5, 100) ):
    fig = plt.figure(figsize = (7,10), constrained_layout = True)
    subfigs = fig.subfigures(3, 1, hspace=0.07,)


    plot_traj(sol, y_pc, subfigs[2])
    axTop = subfigs[0].subplots(1,2)
    plot_spectrum(axTop[0], sol[:,:], fs)
    axTop[0].set_title('Full')
    plot_spectrum(axTop[1], y[:,:], fs)

    axTop[1].set_title('Delay Embedding')

    axCenter = subfigs[1].subplots(1,2)

    el_to_fit = np.arange(0,len(ells_full)-1, 1, dtype= int) # Use first len(ells)-el_to_fit values of l to find the params
    data_cd = sol[:,::10]

    computeCorrDim(data_cd.T, ells_full, el_to_fit, axCenter[0])
   
    data_cd = y[:,::10]
    el_to_fit = np.arange(0,len(ells_delay)-1, 1, dtype= int) # Use first len(ells)-el_to_fit values of l to find the params
    
    computeCorrDim(data_cd.T, ells_delay, el_to_fit,axCenter[1])
    #fig.tight_layout()