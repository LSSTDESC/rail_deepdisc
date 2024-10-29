import os

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from IPython.display import Markdown
from matplotlib import gridspec
from qp import interp
from qp.ensemble import Ensemble
from qp.metrics.pit import PIT
import matplotlib.colors as colors
from sklearn.utils import resample



def make_rail_cat(filename,dcat,columns):
    with h5py.File(filename,'w') as f:
        grp = f.create_group("photometry")

        for c in columns:
            grp[c] = dcat[c].values


class Sample(Ensemble):
    """Expand qp.Ensemble to append true redshifts
    array, metadata, and specific plots."""

    def __init__(
        self, pdfs, zgrid, ztrue, photoz_mode=None, code="", name="", n_quant=100
    ):
        """Class constructor

        Parameters
        ----------
        pdfs: `ndarray`
            photo-z PDFs array, shape=(Ngals, Nbins)
        zgrid: `ndarray`
            PDF bins centers, shape=(Nbins,)
        ztrue: `ndarray`
            true redshifts, shape=(Ngals,)
        photoz_mode: `ndarray`
            photo-z (PDF mode), shape=(Ngals,)
        code: `str`, (optional)
            algorithm name (for plot legends)
        name: `str`, (optional)
            sample name (for plot legends)
        """

        super().__init__(interp, data=dict(xvals=zgrid, yvals=pdfs))
        self._pdfs = pdfs
        self._zgrid = zgrid
        self._ztrue = ztrue
        self._photoz_mode = photoz_mode
        self._code = code
        self._name = name
        self._n_quant = n_quant
        self._pit = None
        self._qq = None

    @property
    def code(self):
        """Photo-z code/algorithm name"""
        return self._code

    @property
    def name(self):
        """Sample name"""
        return self._name

    @property
    def ztrue(self):
        """True redshifts array"""
        return self._ztrue

    @property
    def zgrid(self):
        """Redshift grid (binning)"""
        return self._zgrid

    @property
    def photoz_mode(self):
        """Photo-z (mode) array"""
        return self._photoz_mode

    @property
    def n_quant(self):
        return self._n_quant

    @property
    def pit(self):
        if self._pit is None:
            pit_array = np.array(
                [self[i].cdf(self.ztrue[i])[0][0] for i in range(len(self))]
            )
            self._pit = pit_array
        return self._pit

    @property
    def qq(self, n_quant=100):
        q_theory = np.linspace(0.0, 1.0, n_quant)
        q_data = np.quantile(self.pit, q_theory)
        self._qq = (q_theory, q_data)
        return self._qq

    def __len__(self):
        if len(self._ztrue) != len(self._pdfs):
            raise ValueError("Number of pdfs and true redshifts do not match!!!")
        return len(self._ztrue)

    def __str__(self):
        code_str = f"Algorithm: {self._code}"
        name_str = f"Sample: {self._name}"
        line_str = "-" * (max(len(code_str), len(name_str)))
        text = str(
            line_str
            + "\n"
            + name_str
            + "\n"
            + code_str
            + "\n"
            + line_str
            + "\n"
            + f"{len(self)} PDFs with {len(self.zgrid)} probabilities each \n"
            + f"qp representation: {self.gen_class.name} \n"
            + f"z grid: {len(self.zgrid)} z values from {np.min(self.zgrid)} to {np.max(self.zgrid)} inclusive"
        )
        return text

    def plot_pdfs(self, gals, show_ztrue=True, show_photoz_mode=False):
        colors = plot_pdfs(
            self, gals, show_ztrue=show_ztrue, show_photoz_mode=show_photoz_mode
        )
        return colors

    def plot_old_valid(self, gals=None, colors=None):
        old_metrics_table = plot_old_valid(self, gals=gals, colors=colors)
        return old_metrics_table

    def plot_pit_qq(
        self,
        bins=None,
        label=None,
        title=None,
        show_pit=True,
        show_qq=True,
        show_pit_out_rate=True,
        savefig=False,
    ):
        """Make plot PIT-QQ as Figure 2 from Schmidt et al. 2020."""
        fig_filename = plot_pit_qq(
            self,
            bins=bins,
            label=label,
            title=title,
            show_pit=show_pit,
            show_qq=show_qq,
            show_pit_out_rate=show_pit_out_rate,
            savefig=savefig,
        )
        return fig_filename

    
def plot_metrics(ens,ztrue, point_est='mode', code='', zgrid = np.linspace(0, 5, 200), rnge=[[0,3.2],[0,3.2]], savefig=False, path='./plot'):
        
    #pitobj = PIT(res, truth)
    #pit_out_rate = pitobj.evaluate_PIT_outlier_rate()
    
    gs = gridspec.GridSpec(ncols=2, nrows=2, height_ratios=[3, 1], width_ratios=[1,1.2],
                          hspace=0.05,wspace=0.4)
    
    fig = plt.figure(figsize=[10, 5], constrained_layout=True)
   
    #pdfs = res.objdata()["yvals"]
    
    gs, _ = custom_plot_pit_qq(
        ens,
        zgrid,
        ztrue,
        gs,
        title="",
        code="DeepDISC Swin NG3 Weighted",
        pit_out_rate=None,
        #savefig=True,
    )
    
    gals = np.where(ztrue!=0)
    
    ax_point = plt.subplot(gs[0:,1])
    
    if point_est == 'mode':
        points = ens.mode(zgrid)
    elif point_est == 'mean':
        points = ens.mean()

        
    met = point_metrics(ztrue, points[:,0])
    
    threesig = 3.0*met[3]
    cutcriterion = np.maximum(0.06,threesig)
    mask= (np.fabs(met[0])>cutcriterion)
    
    ztmax = rnge[0][1]
    zpmax1 = cutcriterion * (1+ztmax) + ztmax
    zpmax2 = -cutcriterion * (1+ztmax) + ztmax
    
    df = pd.DataFrame(np.array([ztrue,points[:,0]]).T, columns=['ztrue','zmode'])
    
    label = f"Bias: {met[1]:.4f}"    
    label += f"\n$\sigma_{{IQR}}$: {met[3]:.4f}"    
    label += f"\nOutlier Frac: {met[4]:.4f}"    

    
    h= sns.histplot(
        data=df, x="ztrue", y="zmode", fill=True,  norm=colors.LogNorm(), 
        vmin=None, vmax=None, cmap='plasma', cbar=True, bins=300, binwidth=0.01, 
        alpha=1.0, kde=True
    )
    #ax_point.set_position(ax.figbox)    
    #plt.gca().set_aspect('equal');
    im = ax_point.plot(rnge[0],rnge[1],color='black', label=label, linestyle='--', linewidth=1.5)
    ax_point.plot(rnge[0],[cutcriterion,zpmax1],color='black', linestyle='-', linewidth=1.5)
    ax_point.plot(rnge[0],[-cutcriterion,zpmax2],color='black', linestyle='-', linewidth=1.5)
    #ax_point.set_xlabel('True Redshift' , fontsize=14)
    #ax_point.set_ylabel('Predicted Redshift', fontsize=14)
    ax_point.set_xlabel('True Redshift' , fontsize=14)
    ax_point.set_ylabel(f'Predicted Redshift ({point_est})', fontsize=14)
    #plt.colorbar(h[3],ax=ax_point)
    
    ax_point.set_aspect('equal')
    
    ax_point.set_xlim(rnge[0][0],rnge[0][1])
    ax_point.set_ylim(rnge[1][0],rnge[1][1])

    leg = ax_point.legend(handlelength=0, handletextpad=0, fancybox=True, framealpha=0.99)
    
    plt.suptitle(code, fontsize=16)
    
    if savefig:
        plt.savefig(path)
        
        
        
def plot_PIT(ens,ztrue, point_est='mode', code='', title='', zgrid = np.linspace(0, 5, 200), rnge=[[0,3.2],[0,3.2]], savefig=False, path='./plot'):
        
    #pitobj = PIT(res, truth)
    #pit_out_rate = pitobj.evaluate_PIT_outlier_rate()
    
    gs = gridspec.GridSpec(ncols=1, nrows=2, height_ratios=[3,1], width_ratios=[1],
                          hspace=0.05,wspace=0.4)
    
    fig = plt.figure(figsize=[5, 7], constrained_layout=True)
   
    #pdfs = res.objdata()["yvals"]
    
    gs, _ = custom_plot_pit_qq(
        ens,
        zgrid,
        ztrue,
        gs,
        title=title,
        code=code,
        pit_out_rate=None,
        #savefig=True,
    )
    
    gals = np.where(ztrue!=0)
    
    #plt.suptitle(code, fontsize=16)
    
    if savefig:
        plt.savefig(path, bbox_inches='tight')
    

def plot_point_metrics(res,ztrue, point_est='mode', code='', zgrid = np.linspace(0, 5, 200), rnge=[[0,3.2],[0,3.2]], savefig=False, path='./plot'):
        
    #pitobj = PIT(res, truth)
    #pit_out_rate = pitobj.evaluate_PIT_outlier_rate()
    
    gs = gridspec.GridSpec(ncols=1, nrows=1, height_ratios=[1], width_ratios=[1])#, hspace=0.05,wspace=0.4)
    
    fig = plt.figure(figsize=[7, 7], constrained_layout=True)
    
    
    ax_point = plt.subplot(gs[0])
    
    
    if point_est == 'mode':
        points = res.mode(zgrid)
    elif point_est == 'mean':
        points = res.mean()

        
    met = point_metrics(ztrue, points[:,0])
    
    threesig = 3.0*met[3]
    cutcriterion = np.maximum(0.06,threesig)
    mask= (np.fabs(met[0])>cutcriterion)
    
    ztmax = rnge[0][1]
    zpmax1 = cutcriterion * (1+ztmax) + ztmax
    zpmax2 = -cutcriterion * (1+ztmax) + ztmax
    
    df = pd.DataFrame(np.array([ztrue,points[:,0]]).T, columns=['ztrue','zmode'])
    
    label = f"Bias: {met[1]:.4f}"    
    label += f"\n$\sigma_{{IQR}}$: {met[3]:.4f}"    
    label += f"\n $\eta$: {met[4]:.4f}"    

    
    h= sns.histplot(
        data=df, x="ztrue", y="zmode", fill=True,  norm=colors.LogNorm(), 
        vmin=None, vmax=None, cmap='plasma', cbar=True, bins=300, binwidth=0.01, 
        alpha=1.0, kde=True
    )
    
    
    #ax_point.set_position(ax.figbox)    
    #plt.gca().set_aspect('equal');
    im = ax_point.plot(rnge[0],rnge[1],color='black', label=label, linestyle='--', linewidth=1.5)
    ax_point.plot(rnge[0],[cutcriterion,zpmax1],color='black', linestyle='-', linewidth=1.5)
    ax_point.plot(rnge[0],[-cutcriterion,zpmax2],color='black', linestyle='-', linewidth=1.5)
    #ax_point.set_xlabel('True Redshift' , fontsize=14)
    #ax_point.set_ylabel('Predicted Redshift', fontsize=14)
    ax_point.set_xlabel('True Redshift' , fontsize=16)
    ax_point.set_ylabel(f'Predicted Redshift ({point_est})', fontsize=16)
    ax_point.tick_params(axis='both', which='major', labelsize=14)
    
    ax_point.set_aspect('equal')

    leg = ax_point.legend(handlelength=0, handletextpad=0, fancybox=True, framealpha=0.99, fontsize=12)
    
    #plt.suptitle(code, fontsize=16)
    
    plt.xlim(rnge[0][0],rnge[0][1])
    plt.ylim(rnge[1][0],rnge[1][1])

    cbar = h.collections[0].colorbar
    cbar.ax.tick_params(which='both', labelsize=14)
    
    if savefig:
        plt.savefig(path,bbox_inches='tight')
        
    plt.show()
        
def plot_point_metrics_mult(res_list,ztrue_list, point_est='mode', codes=None, zgrid = np.linspace(0, 3, 300),
                            rnge=[[0,3.2],[0,3.2]], vmin=None, vmax=None, savefig=False, path='./plot'):
        
    #pitobj = PIT(res, truth)
    #pit_out_rate = pitobj.evaluate_PIT_outlier_rate()
    
    numcodes = len(res_list)
    if codes is None:
        codes=['']*numcodes
    
    gs = gridspec.GridSpec(ncols=numcodes, nrows=1, height_ratios=[1], width_ratios=[0.91]*(numcodes-1)+[1])#, hspace=0.05,wspace=0.4)
    
    fig = plt.figure(figsize=[7*numcodes, 7], tight_layout=True)


    for i, (res,ztrue) in enumerate(zip(res_list,ztrue_list)):
        if i>0:
            ax_point = plt.subplot(gs[i],sharey=ax_point)
        else:
            ax_point = plt.subplot(gs[i])


        if point_est == 'mode':
            points = res.mode(zgrid)
        elif point_est == 'mean':
            points = res.mean()


        met = point_metrics(ztrue, points[:,0])

        threesig = 3.0*met[3]
        cutcriterion = np.maximum(0.06,threesig)
        mask= (np.fabs(met[0])>cutcriterion)

        ztmax = rnge[0][1]
        zpmax1 = cutcriterion * (1+ztmax) + ztmax
        zpmax2 = -cutcriterion * (1+ztmax) + ztmax

        df = pd.DataFrame(np.array([ztrue,points[:,0]]).T, columns=['ztrue','zmode'])

        label = f"Bias: {met[1]:.4f}"    
        label += f"\n$\sigma_{{IQR}}$: {met[3]:.4f}"    
        label += f"\n $\eta$: {met[4]:.4f}"    

        if i==numcodes-1:
            cbardict = {'label': ' ', 'fraction':0.046, 'pad':0.04, 'use_gridspec':True}
            #cbardict = {'label': ' ', 'fraction':0.1, 'pad':0.04}

            #cax = fig.add_axes([ax_point.get_position().x1+0.01,ax_point.get_position().y0,0.02,ax_point.get_position().height])    
            h= sns.histplot(
                data=df, x="ztrue", y="zmode", fill=True,  norm=colors.LogNorm(vmin=vmin,vmax=vmax), 
                vmin=None, vmax=None, cmap='plasma', cbar=True, cbar_kws=cbardict, bins=300, binwidth=0.01, 
                alpha=1.0, kde=True
            )
        else:
            h= sns.histplot(
                data=df, x="ztrue", y="zmode", fill=True,  norm=colors.LogNorm(vmin=vmin,vmax=vmax), 
                vmin=None, vmax=None, cmap='plasma', cbar=False, bins=300, binwidth=0.01, 
                alpha=1.0, kde=True
            )                    


        #ax_point.set_position(ax.figbox)    
        #plt.gca().set_aspect('equal');
        im = ax_point.plot(rnge[0],rnge[1],color='black', label=label, linestyle='--', linewidth=1.5)
        ax_point.plot(rnge[0],[cutcriterion,zpmax1],color='black', linestyle='-', linewidth=1.5)
        ax_point.plot(rnge[0],[-cutcriterion,zpmax2],color='black', linestyle='-', linewidth=1.5)
        #ax_point.set_xlabel('True Redshift' , fontsize=14)
        if i==0:
            ax_point.set_ylabel(r'$z_{\rm phot}$', fontsize=20)
        ax_point.set_xlabel(r'$z_{\rm true}$' , fontsize=20)
        ax_point.tick_params(axis='both', which='major', labelsize=20)
        ax_point.set_title(codes[i],fontsize=24)
        ax_point.set_aspect('equal')

        leg = ax_point.legend(handlelength=0, handletextpad=0, fancybox=True, framealpha=0.99, fontsize=16)
    
        ax_point.set_xlim(rnge[0][0],rnge[0][1])
        ax_point.set_ylim(rnge[1][0],rnge[1][1])
        
        ax_point.set_xticks(np.arange(0,3.5,0.5))
        ax_point.tick_params(axis='x', labelrotation=45)
        
        if i>0:
            plt.setp(ax_point.get_yticklabels(), visible=False)
            #h.set(yticklabels=[])
            h.set(ylabel=None)


    cbar = h.collections[0].colorbar
    cbar.ax.tick_params(which='both', labelsize=20)
    
    if savefig:
        plt.savefig(path,bbox_inches='tight')
        
    plt.show()
    
    
def custom_plot_pit_qq(
    ens,
    zgrid,
    ztrue,
    gs,
    bins=None,
    title=None,
    code=None,
    show_pit=True,
    show_qq=True,
    pit_out_rate=None,
    savefig=False,
) -> str:
    """Quantile-quantile plot
    Ancillary function to be used by class Metrics.

    Parameters
    ----------
    pit: `PIT` object
        class from metrics.py
    bins: `int`, optional
        number of PIT bins
        if None, use the same number of quantiles (sample.n_quant)
    title: `str`, optional
        if None, use formatted sample's name (sample.name)
    label: `str`, optional
        if None, use formatted code's name (sample.code)
    show_pit: `bool`, optional
        include PIT histogram (default=True)
    show_qq: `bool`, optional
        include QQ plot (default=True)
    pit_out_rate: `ndarray`, optional
        print metric value on the plot panel (default=None)
    savefig: `bool`, optional
        save plot in .png file (default=False)
    """

    if bins is None:
        bins = 100
    if title is None:
        title = ""

    if code is None:
        code = ""
        label = ""
    else:
        label = code + "\n"

    if pit_out_rate is not None:
        try:
            label += "PIT$_{out}$: "
            label += f"{float(pit_out_rate):.4f}"
        except:
            print("Unsupported format for pit_out_rate.")

    #plt.figure(figsize=[4, 5])
    #gs = gridspec.GridSpec(2, 1, height_ratios=[3, 1])
    ax0 = plt.subplot(gs[0,0])
    
    #sample = Sample(pdfs, zgrid, ztrue)

    pitobj = PIT(ens,ztrue)
    pit_vals = np.array(pitobj.pit_samps)
    q_theory = np.linspace(0.0, 1.0, 100)
    q_data = np.quantile(pit_vals, q_theory)
    qq = (q_theory, q_data)

    
    if show_qq:
        ax0.plot(
            qq[0], qq[1], c="r", linestyle="-", linewidth=3, label=label)
        ax0.plot([0, 1], [0, 1], color="k", linestyle="--", linewidth=2)
        ax0.set_ylabel(r"Q$_{\rm data}$", fontsize=20)
        plt.ylim(-0.001, 1.001)
        ax0.tick_params(axis='y', labelsize=18)   

    plt.xlim(-0.001, 1.001)
    plt.title(title,fontsize=18)
    if show_pit:
        #fzdata = Ensemble(interp, data=dict(xvals=zgrid, yvals=pdfs))
        #pitobj = PIT(fzdata, ztrue)
        #pit_vals = np.array(pitobj.pit_samps)
        pit_out_rate = pitobj.evaluate_PIT_outlier_rate()

        try:
            y_uni = float(len(pit_vals)) / float(bins)
        except:
            y_uni = float(len(pit_vals)) / float(len(bins))
        if not show_qq:
            ax0.hist(pit_vals, bins=bins, alpha=0.7, label=label)
            ax0.set_ylabel("Number")
            ax0.hlines(y_uni, xmin=0, xmax=1, color="k")
            plt.ylim(
                0,
            )  # -0.001, 1.001)
        else:
            ax1 = ax0.twinx()
            ax1.hist(pit_vals, bins=bins, alpha=0.7, color='dimgray')
            #ax1.set_ylabel("Number")
            ax1.hlines(y_uni, xmin=0, xmax=1, color="k")
            ax1.set_xticklabels([])
    
        ax1.tick_params(axis='y', labelsize=18)   
        ax1.ticklabel_format(axis='y', style='sci', scilimits=(0,0))
        tx = ax1.yaxis.get_offset_text()
        tx.set_fontsize(15)

    #leg = ax0.legend(handlelength=0, handletextpad=0, fancybox=True)
    #for item in leg.legendHandles:
    #    item.set_visible(False)
    if show_qq:
        ax2 = plt.subplot(gs[1,0])
        ax2.plot(
            qq[0],
            (qq[1] - qq[0]),
            c="r",
            linestyle="-",
            linewidth=3,
        )
        plt.ylabel("$\Delta$Q", fontsize=18)
        ax2.plot([0, 1], [0, 0], color="k", linestyle="--", linewidth=2)
        plt.xlim(-0.001, 1.001)
        plt.ylim(
            np.min([-0.12, np.min(qq[1] - qq[0]) * 1.05]),
            np.max([0.12, np.max(qq[1] - qq[0]) * 1.05]),
        )
        ax2.tick_params(axis='x', labelsize=18)   
        ax2.tick_params(axis='y', labelsize=18)   

    if show_pit:
        if show_qq:
            plt.xlabel(r"Q$_{\rm theory}$ / PIT Value", fontsize=20)
        else:
            plt.xlabel("PIT Value", fontsize=20)
    else:
        if show_qq:
            plt.xlabel(r"Q$_{\rm theory}$", fontsize=20)
    if savefig:
        fig_filename = str("plot_pit_qq_" + f"{(code).replace(' ', '_')}.png")
        plt.savefig(fig_filename)
    else:
        fig_filename = None

    return gs, fig_filename


def point_metrics(z, pred):
    """
        Returns the desired performance metrics.
        
        Arguments:
            z (numpy array): true redschift
            pred (numpy array): the redshift prediction
             
        Returns:
            dz (ndarray): Residuals for every test image.
            pred_bias (float): The prediction bias, mean of dz.
            smad (float):The MAD deviation.
            out_frac (float): The fraction of outliers.
    """
    
    ez = (pred - z) / (1 + z)
    nans = np.isnan(ez)
    ez = ez[~nans]
    
    pred_bias = np.median(ez)
    MAD = np.median(np.abs(ez - np.median(ez)))
    smad = 1.4826 * MAD
    
    x75,x25 = np.percentile(ez,[75.,25.])
    iqr = x75-x25
    sigma_iqr = iqr/1.349        
    
    #out_frac_1 = np.sum(np.abs(dz) > 0.05) / float(len(z))
    
    
    threesig = 3.0*sigma_iqr
    cutcriterion = np.maximum(0.06,threesig)
    mask= (np.fabs(ez)>cutcriterion)
    outlier = np.sum(mask)
    out_frac = float(outlier)/float(len(z))
    
    return ez, pred_bias, smad, sigma_iqr, out_frac


def bias_per_quantity(res, df, key, bins, zgrid=np.linspace(0,3,300), log=False, return_bins=False):

    if key =='pzmode':
        quant = res.mode(zgrid)
    else:
        fininds = np.isfinite(df[key].values)
        quant = df[key].values[fininds]

    if log:
        quant = np.log(quant)
        
    binsi = np.digitize(quant,bins=bins)
    
    modes= res.mode(zgrid)[fininds]
    ztrue = res.ancil['true_zs'][fininds]

    mets = []
    ezs=[]
    resbins=[]
    for i, bbin in enumerate(bins):
        resbin = np.where(binsi-1==i)[0]
        if len(resbin)==0:
            continue
        modesb = modes[:,0][resbin]
        ztrueb = ztrue[resbin]
        ez, pred_bias, smad, sigma_iqr, out_frac = point_metrics(ztrueb, modesb)
       
        ezs.append(ez)
        
        mets.append([pred_bias,smad,sigma_iqr,out_frac])
        resbins.append(resbin)
    
    biases = [np.median(ez) for ez in ezs]

    if return_bins:
        return ezs, biases, np.array(mets), quant, resbins
    else:
        return ezs, biases, np.array(mets), quant
    

def bootstrap_binned_bias(ezs):
    
    #ml = len(ezs[0])
    #for ez in ezs[1:]:
    #    if len(ez)<ml:
    #        ml = len(ez)

    
    all_medians = []
    all_siqrs = []
    all_ofs = []
    for ez in ezs:
        ml = len(ez)
        medians =[]
        siqrs=[]
        ofs = []
        for N in range(1000):
            ez_sample = resample(ez,replace=True,n_samples=ml)

            pred_bias = np.median(ez_sample)
            MAD = np.median(np.abs(ez_sample - np.median(ez_sample)))
            smad = 1.4826 * MAD

            x75,x25 = np.percentile(ez_sample,[75.,25.])
            iqr = x75-x25
            sigma_iqr = iqr/1.349            

            threesig = 3.0*sigma_iqr
            cutcriterion = np.maximum(0.06,threesig)
            mask= (np.fabs(ez_sample)>cutcriterion)
            outlier = np.sum(mask)
            out_frac = float(outlier)/float(len(ez_sample))

            medians.append(pred_bias)
            siqrs.append(sigma_iqr)
            ofs.append(out_frac)

        all_medians.append(np.array(medians))
        all_siqrs.append(np.array(siqrs))
        all_ofs.append(np.array(ofs))
    

    bias_mean_bs = [all_median.mean() for all_median in all_medians]
    bias_std_bs = [all_median.std() for all_median in all_medians]

    siqrs_mean_bs = [all_siqr.mean() for all_siqr in all_siqrs]
    siqrs_std_bs = [all_siqr.std() for all_siqr in all_siqrs]

    ofs_mean_bs = [all_of.mean() for all_of in all_ofs]
    ofs_std_bs = [all_of.std() for all_of in all_ofs]
        
        
    return bias_std_bs, siqrs_std_bs, ofs_std_bs


def all_metrics(resfull, df, quantf, bins, log=False):
    all_bias_stds = []
    all_siqrs_stds = []
    all_ofs_stds = []
    
    all_biases = []
    all_siqrs = []
    all_ofs = []
    
    all_quants = []
    
    inds = np.arange(0,resfull.npdf)
    for i in range(1):
        print(i)
        #indsi = np.random.choice(inds,100000)
        #res = resfull[indsi]
        res = resfull
        ezs, biases, mets, quant = bias_per_quantity(res,df,quantf, bins, log=log)
        
        bias_std_bs, siqrs_std_bs, ofs_std_bs = bootstrap_binned_bias(ezs)
        
        all_bias_stds.append(bias_std_bs)
        all_biases.append(biases)
        
        all_siqrs_stds.append(siqrs_std_bs)
        all_siqrs.append(mets[:,2])
        
        all_ofs_stds.append(ofs_std_bs)
        all_ofs.append(mets[:,3])
        
        all_quants.append(quant)
        
    return all_biases, all_bias_stds, all_siqrs, all_siqrs_stds, all_ofs, all_ofs_stds, all_quants
