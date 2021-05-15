function  a = read_psc_header( my_file,rdbm_rev )
if rdbm_rev == 20.006 || rdbm_rev == 21.001
 
% Ashoks changes %

 a.mps_freq = fread(my_file, 1, 'int64');  % What is the mapping for s64 %
 a.aps_freq = fread(my_file, 1, 'int64');  % What is the mapping for s64 %
 for id = 1:34
  a.buff_s64(id) = fread(my_file, 1, 'int64'); % What is the mapping for s64 %
 end 

 for id = 1 : 128
  a.rec_std(id) = fread(my_file, 1, 'float32');
 end
 for id = 1 : 128
  a.rec_mean(id) = fread(my_file, 1, 'float32');
 end

 a.scalei = fread(my_file, 1, 'float32');
 a.scaleq = fread(my_file, 1, 'float32');

 a.dd_q_phase_delay_qd = fread(my_file, 1, 'float32');
 a.dd_q_phase_delay_from_qd = fread(my_file, 1, 'float32'); 

 for id = 1: 8 
 a.phaseOffset(id) = fread(my_file, 1, 'float32');
 end 

 for id = 1:14
  a.buffer(id) = fread(my_file, 1, 'float32');
 end

 a.command = fread(my_file, 1, 'int32');

 a.mps_r1 = fread(my_file, 1, 'int32');
 a.mps_r2 = fread(my_file, 1, 'int32');
 a.mps_tg = fread(my_file, 1, 'int32');
 a.mps_freq_deprec = fread(my_file, 1, 'uint32');  % what is the mapping for n32 %
 

 a.aps_r1 = fread(my_file, 1, 'int32');
 a.aps_r2 = fread(my_file, 1, 'int32');
 a.aps_tg = fread(my_file, 1, 'int32');

 a.aps_freq_deprec = fread(my_file, 1, 'uint32');  % what is the mapping for n32 %
 a.snr_warning = fread(my_file, 1, 'int32');
 a.aps_or_mps = fread(my_file, 1, 'int32');
 a.mps_bitmap = fread(my_file, 1, 'int32');
 a.filler1 = fread(my_file, 1, 'int32');
 a.filler2 = fread(my_file, 1, 'int32'); 
 a.autoshim_status = fread(my_file, 1, 'int32');
 a.line_width = fread(my_file, 1, 'int32');
 a.ws_flip = fread(my_file, 1, 'int32');
 a.supp_lvl = fread(my_file, 1, 'int32');
 a.psc_reuse = fread(my_file, 1, 'int32');

 a.psc_ta = fread(my_file, 1, 'int32');
 a.phase_correction_status = fread(my_file, 1, 'int32');
 a.broad_band_select = fread(my_file, 1, 'int32');

 for id = 1:24
  a.buffer_s32(id) = fread(my_file, 1, 'int32');
 end
 
 a.xshim = fread(my_file, 1, 'int16');
 a.yshim = fread(my_file, 1, 'int16');
 a.zshim = fread(my_file, 1, 'int16');
 a.recon_enable = fread(my_file, 1, 'int16');
 a.dd_q_ta_offset_qd = fread(my_file, 1, 'int16');
 a.dd_q_ta_offset_from_qd = fread(my_file, 1, 'int16');
 a.dd_mode = fread(my_file, 1, 'int16');
 a.dummy_for_32bit_align = fread(my_file, 1, 'int16');
 a.txMode = fread(my_file, 1, 'int16');

 for id = 1:8
  a.taOffset(id) = fread(my_file, 1, 'int16');
 end

 
 for id = 1:27
  a.buff_16(id) = fread(my_file, 1, 'int16');
 end


for id = 1:256
  a.powerspec(id) = fread(my_file, 1, 'char');
 end
 for id = 1:52
  a.psc_reuse_string(id) = fread(my_file, 1, 'char');
 end 
 for id = 1:52
  a.buffer(id) = fread(my_file, 1, 'char');
 end
else
   a = struct();
end
