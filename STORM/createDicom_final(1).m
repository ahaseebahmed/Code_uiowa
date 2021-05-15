
% keySet =   {'PT1_V1', 'PT1_V2', 'PT2_V1', 'PT2_V2', 'PT3_V1', 'PT3_V2', 'PT4_V1', 'PT4_V2','PT5_V2', 'PT6_V1', 'PT6_V2','PT7_V1', 'PT7_V2', 'PT8_V1', 'PT8_V2', 'PT9_V1'};
% valueSet = [463,457,630,622,375,388,194,195,322,142,151,365,368,453,447,583];
% TransSet = [2,1,1,2,1,2,0,1,1,0,1,1,2,0,1,0];
% mapObj = containers.Map(keySet,valueSet);
% mapObj2 = containers.Map(keySet,TransSet);

keySet =   {'204', '119'};
valueSet = [185, 365];
TransSet = [1,0];
mapObj = containers.Map(keySet,valueSet);
mapObj2 = containers.Map(keySet,TransSet);

num_fr = 800;
nav = 2;
N=512;

for i=1:1%length(keySet)
    
    key = keySet(i);
    key = key{1};
    
    fileStr = ['meas_MID00',num2str(mapObj(key)),'_FID1']
    
    rawFolder = './../../../Codes/Data/RadialData_27Feb19_UIOWA/prashantData/';
    liste = rdir([rawFolder,fileStr,'*.dat'],'',rawFolder);
    filesRaw = {liste.name};
    
    twix_obj = mapVBVD([rawFolder,filesRaw{1}]);
    reconFolder = strcat('/Users/ahhmed/storm_radial/Storm_Basis/STORM/ReconData/',filesRaw{1},'/');
    liste = rdir([reconFolder,fileStr,'*.mat'],'',reconFolder);
    files = {liste.name};    
    
    for j=1:length(files)
        
        fprintf('Reading from %s\n',files{j});
        
        load([reconFolder,files{j}]);
        
        recon = reshape((U1)*D',[512,512,size(D,1)]);
        recon = flipud(fftshift(fftshift(recon,1),2));
        
        recon = uint16(1024.*abs(recon)./max(abs(recon(:))));
        
        plot(D(:,end-1));
        hold on;
        h1 = impoint();
        pt1=getPosition(h1);
        h2 = impoint();
        pt2=getPosition(h2); 
        recon_inp=recon(N/2-N/4+1:N/2+N/4,N/2-N/4+1:N/2+N/4,pt1(1,1):pt2(1,1))
        plot(D(:,end-1));
        hold on;
        h1 = impoint();
        pt1=getPosition(h1);
        h2 = impoint();
        pt2=getPosition(h2); 
        recon_exp=recon(N/2-N/4+1:N/2+N/4,N/2-N/4+1:N/2+N/4,pt1(1,1):pt2(1,1))
        
        %% Copying information from Siemens Prop format to dicom Header
        load info
        
        dicomHDr = twix_obj{1}.hdr;
        
        info.InstitutionAddress = dicomHDr.Dicom.InstitutionAddress;
        info.InstitutionName = dicomHDr.Dicom.InstitutionName;
        info.LongModelName = dicomHDr.Dicom.LongModelName;
        info.Manufacturer = dicomHDr.Dicom.Manufacturer;
        info.ManufacturersModelName = dicomHDr.Dicom.ManufacturersModelName;
        info.Modality = dicomHDr.Dicom.Modality;
        info.SoftwareVersions = dicomHDr.Dicom.SoftwareVersions;
        info.TransmittingCoil = dicomHDr.Dicom.TransmittingCoil;
        info.adFlipAngleDegree = dicomHDr.Dicom.adFlipAngleDegree;
        info.dSliceResolution = dicomHDr.Dicom.dSliceResolution;
        if(dicomHDr.Dicom.lPatientSex == 1)
            info.PatientSex= 'M';
        else
            info.PatientSex= 'F';
        end
        
        info.PatientAge = ['0',num2str(dicomHDr.Dicom.flPatientAge),'Y'];
        info.tBodyPartExamined = dicomHDr.Dicom.tBodyPartExamined;
        info.tGradientCoil = dicomHDr.Dicom.tGradientCoil;
        info.tSequenceOwner = dicomHDr.Dicom.tSequenceOwner;
        info.ulVersion = dicomHDr.Dicom.ulVersion;
        info.StudyComments = 'PROJECT:MJ-DEV,SUBJECT';
       % info.PerformedProcedureStepDescription = 'SG SCAN';
        
        pixelsize = dicomHDr.Config.ReadoutFOV/512;
        info.PixelSpacing = [pixelsize,pixelsize];
        
        %%
        strt_ind = 1;
        end_ind = 800;
        
        newdirname = key(1:3);
        key_short = key(1:3);
        
        DicomFolder =  strcat(reconFolder,'/Dicom/')%'/nfs/s-iibi60/users/spoddar/Documents/MATLAB/STORM/MRM/DICOM/Nov20/';
        
        if(~exist([DicomFolder,newdirname],'dir'))
            mkdir([DicomFolder,newdirname]);
        end
        
        for ii=1:size(recon_exp,3)
            data=recon_exp(:,:,ii);
%             trans = mapObj2(key);
%             if trans == 1
%                 data = fliplr(rot90(data,-1));
%             elseif trans == 2 
%                  data = fliplr(rot90(data,-2));
%             end
%             info.AcquisitionNumber = mapObj(key);
%             info.SeriesDescription = 'MJ Dev';
%             info.PatientID = key_short;
%             info.SliceLocation = ii-strt_ind+1;
%             info.InstanceNumber = ii-strt_ind+1;
%             info.SliceThickness = 1;
%             info.PatientName.FamilyName = 'subject_';
%             info.PatientName.GivenName = key_short;
%             
% %             info.StudyDescription = [key,'param',num2str(j)];
%             info.StudyDescription = [key,'Fr',num2str(num_fr),'Nav',num2str(nav)];
%             info.StudyID = num2str(mapObj(key));
%             info.SOPInstanceUID=[info.SOPInstanceUID,key_short];
%             info.AccessionNumber = num2str(mapObj(key));
%             %info.SeriesDescription = [num2str(mapObj(key)),num2str(j)];
%             info.SeriesDescription = [num2str(mapObj(key)),'Fr',num2str(num_fr),'Nav',num2str(nav)];
% %             info.SeriesNumber = str2double([num2str(mapObj(key)),num2str(j)]);
%             info.SeriesNumber = str2double([num2str(mapObj(key)),num2str(num_fr),num2str(nav)]);
%             info.PatientID = key_short;
%             
%             info.SequenceName = ['data',key_short];
%             %info.ProtocolName = [key,num2str(j)];
%             info.ProtocolName = [key,'Fr',num2str(num_fr),'Nav',num2str(nav)];
%             info.ImagePositionPatient = [ii-strt_ind+1;1;1];
            %dicomwrite(data,[DicomFolder,newdirname,'/',key,'Frame',num2str(ii-strt_ind+1),'Param',num2str(j),'.dcm'], info);
            dicomwrite(data,[DicomFolder,newdirname,'/',key,'Frame',num2str(ii-1+1),'Fr',num2str(size(recon_exp,3)),'Nav',num2str(nav),'.dcm'], info);
        end
        for ii=1:size(recon_inp,3)
            data=recon_inp(:,:,ii);

            dicomwrite(data,[DicomFolder,newdirname,'/',key,'Frame',num2str(ii-1+1),'Fr',num2str(size(recon_inp,3)),'Nav',num2str(nav),'.dcm'], info);
        end
    end
end