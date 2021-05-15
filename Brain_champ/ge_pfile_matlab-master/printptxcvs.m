function printptxcvs( rdb_hdr )

usercvs_msg = ' Rdbm header User cvs :   ';

msg = sprintf( ' %s = %f', 'user10',rdb_hdr.user10);
usercvs_msg = strcat(usercvs_msg, msg);

msg = sprintf( ' %s = %f', 'user11',rdb_hdr.user11);
usercvs_msg = strcat(usercvs_msg, msg);

msg = sprintf( ' %s = %f', 'user12',rdb_hdr.user12);
usercvs_msg = strcat(usercvs_msg, msg);

msg = sprintf( ' %s = %f', 'user13',rdb_hdr.user13);
usercvs_msg = strcat(usercvs_msg, msg);

msg = sprintf( ' %s = %f', 'user14',rdb_hdr.user14);
usercvs_msg = strcat(usercvs_msg, msg);

msg = sprintf( ' %s = %f', 'user15',rdb_hdr.user15);
usercvs_msg = strcat(usercvs_msg, msg);

msg = sprintf( ' %s = %f', 'user16',rdb_hdr.user16);
usercvs_msg = strcat(usercvs_msg, msg);

msg = sprintf( ' %s = %f', 'user17',rdb_hdr.user17);
usercvs_msg = strcat(usercvs_msg, msg);

disp(usercvs_msg);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

usercvs_msg = ' Rdbm header Ptx Cvs :   ';

msg = sprintf( ' %s = %f', 'ptx',rdb_hdr.ptx);
usercvs_msg = strcat(usercvs_msg, msg);

msg = sprintf( ' %s = %f', 'ptx_blochsiegert_constant',rdb_hdr.ptx_blochsiegert_constant);
usercvs_msg = strcat(usercvs_msg, msg);

msg = sprintf( ' %s = %f', 'ptx_blochsiegert_b1',rdb_hdr.ptx_blochsiegert_b1);
usercvs_msg = strcat(usercvs_msg, msg);

msg = sprintf( ' %s = %f', 'ptx_b1map_txmode',rdb_hdr.ptx_b1map_txmode);
usercvs_msg = strcat(usercvs_msg, msg);

msg = sprintf( ' %s = %f', 'ptx_delta_te',rdb_hdr.ptx_delta_te);
usercvs_msg = strcat(usercvs_msg, msg);

msg = sprintf( ' %s = %f', 'xmtaddaps1',rdb_hdr.xmtaddaps1);
usercvs_msg = strcat(usercvs_msg, msg);

msg = sprintf( ' %s = %f', 'tx_nchannels',rdb_hdr.tx_nchannels);
usercvs_msg = strcat(usercvs_msg, msg);

msg = sprintf( ' %s = %f', 'xmtaddScan',rdb_hdr.xmtaddScan);
usercvs_msg = strcat(usercvs_msg, msg);

disp( usercvs_msg );
                   
end

