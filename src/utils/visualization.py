import numpy as np
import matplotlib.pyplot as plt


def plot_trues(*, tps, tps_tfs, gt, labels, tps_err=None, tps_tfs_err=None):
    x = np.arange(len(tps_tfs))
    
    
    plt.bar(x, tps_tfs, yerr=tps_tfs_err, width=0.2, label='TP  + FP')        
    plt.bar(x + 0.3, tps, yerr=tps_err, width=0.2, label='TP')
    
    for widths, heights, heights_errors in zip([x, x+0.3], [tps_tfs, tps], [tps_tfs_err, tps_err]):
        for width, height, err in zip(widths, heights, heights_errors):
            plt.text(width + 0.05, height + 0.05, f'{height} ± {round(err, 2)}')
    

    plt.axhline(gt, color='red', label='Ground TP')
    plt.text(-.5, gt+1, gt)
    
    plt.xticks(x, labels, rotation=30)
    plt.legend()
