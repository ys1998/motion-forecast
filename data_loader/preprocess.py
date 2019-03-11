import numpy as np
import matplotlib.pyplot as plt

def get_motion_bands(seq, k=3, show=False):
    """
    org_seq : [batch_size, seq_len]
    k       : number of motion bands to extract
    show    : whether to display plot

    Returns [batch_size, k+2, seq_len] shaped array containing the
    extracted motion bands.
    """
    laplacian = []
    init_seq = seq
    for _ in range(k):
        smoothed_seq = np.apply_along_axis(
            lambda r: np.convolve(r, np.ones(3)/3., 'same'),
            arr = init_seq, axis=1
        )
        laplacian.append(init_seq - smoothed_seq)
        init_seq = smoothed_seq
    if show:
        _, ax = plt.subplots(k+3,1, sharex=True)
        x = np.arange(0, seq.shape[1])
        lp = sum(laplacian)
        for b in range(seq.shape[0]):
            ax[0].plot(x, seq[b])
            ax[1].plot(x, init_seq[b])
            for i in range(2,k+2):
                ax[i].plot(x, laplacian[i-2][b])
            ax[-1].plot(x, np.abs(seq[b]-init_seq[b]-lp[b]))
        ax[0].set_title("Original")
        ax[1].set_title("Smoothed (base)")
        for i in range(2,k+2):
            ax[i].set_title("k = "+str(i-1))
        ax[-1].set_title("Absolute error")
        plt.show(False)
    return np.stack(laplacian + [init_seq, seq], axis=0).transpose(1,0,2)

def to_euler_angle(q):
    """
    Convert joint parameters to euler angle representation.
    q : [..., 4] quaternion representation (w, x, y, z)
    
    Returns [..., 3] euler angle representation.
    """
    phi = np.arctan2(2*(q[...,0]*q[...,1] + q[...,2]*q[...,3]), 1-2*(q[...,1]**2 + q[...,2]**2))
    theta = np.arcsin(2*(q[...,0]*q[...,2]-q[...,3]*q[...,1]))
    psi = np.arctan2(2*(q[...,0]*q[...,3] + q[...,1]*q[...,2]), 1-2*(q[...,2]**2 + q[...,3]**2))
    return np.stack([phi, theta, psi], axis=-1)

