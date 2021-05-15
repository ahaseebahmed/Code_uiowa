They are all single channel 128 fid points and has dynamic (64 time segments) healpix with navigator vieworder #26:

    Shared_Champaign_Iowa/Scan_Data/Jan-2019/20190131/silent_grad_26.txt

Description of '.mat' data object:

    Shared_Champaign_Iowa/Scan_Data/Jan-2019/20190131/ngfnRecon_object.txt

Static Data p-files (no movement)

    Shared_Champaign_Iowa/Scan_Data/Jan-2019/20190131/P10752.7    PD/T1 weighted
    Shared_Champaign_Iowa/Scan_Data/Jan-2019/20190131/P11264.7    T1 prep
    Shared_Champaign_Iowa/Scan_Data/Jan-2019/20190131/P11776.7    T2 prep

Static data loaded from P-file into ngfnRecon object:

    Shared_Champaign_Iowa/Scan_Data/Jan-2019/20190131/P10752.mat
    Shared_Champaign_Iowa/Scan_Data/Jan-2019/20190131/P11264.mat
    Shared_Champaign_Iowa/Scan_Data/Jan-2019/20190131/P11776.mat
   

Static dataset reconstructions (previews, logs and nifti files):

    Shared_Champaign_Iowa/Scan_Data/Jan-2019/20190131/P10752_recon_20200909T135951
    Shared_Champaign_Iowa/Scan_Data/Jan-2019/20190131/P11264_recon_20200909T140230
    Shared_Champaign_Iowa/Scan_Data/Jan-2019/20190131/P11776_recon_20200909T140441

Static dataset reconstructions in ngfnRecon object:

    Shared_Champaign_Iowa/Scan_Data/Jan-2019/20190131/P10752_recon.mat
    Shared_Champaign_Iowa/Scan_Data/Jan-2019/20190131/P11264_recon.mat
    Shared_Champaign_Iowa/Scan_Data/Jan-2019/20190131/P11776_recon.mat

Dynamic data p-files (movement):

    Shared_Champaign_Iowa/Scan_Data/Jan-2019/20190131/P12288.7    PD/T1 weighted
    Shared_Champaign_Iowa/Scan_Data/Jan-2019/20190131/P12800.7    T1 prep
    Shared_Champaign_Iowa/Scan_Data/Jan-2019/20190131/P13312.7    T2 prep

Dynamic data loaded from P-file into ngfnRecon object:

    Shared_Champaign_Iowa/Scan_Data/Jan-2019/20190131/P12288.mat
    Shared_Champaign_Iowa/Scan_Data/Jan-2019/20190131/P12800.mat
    Shared_Champaign_Iowa/Scan_Data/Jan-2019/20190131/P13312.mat


The main items in the .mat files are:

    obj.fid                        the complex fid data as fid( readout, views, segments, channels) 128*192*64*1
    obj.fid_nav                 data for navigator segments    128*64*64*1

    obj.kspaceRad            normalized (to approximately 1 at edge) radial k-space coordinates for gridding, 128*192*64*1
    obj.kspaceRad_nav    k=space coordinates of periodic navigators,  128*64*64*1
