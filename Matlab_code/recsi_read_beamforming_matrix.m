% 简化版：仅处理单个 .pcapng 文件
clear all;
clc;
for i=1:9 
    %% 配置
    file_path = ['P:\DeepCSI\split\SU_matrix\frontback\frontback', num2str(i), '.pcapng']; % 单个 .pcapng 文件路径
    output_folder_beamf_angles = 'P:\DeepCSI\processed_data\beamf_angles'; % 结果保存路径
    output_folder_exclusive_beamf_reports = 'P:\DeepCSI\processed_data\exclusive_beamf_reports';
    output_folder_time_vector = 'P:\DeepCSI\processed_data\time_vector';
    output_folder_vtilde_matrices = 'P:\DeepCSI\processed_data\vtilde_matrices';
    
    % 参数配置
    Nc = 2; % 用户数量
    Nr=4;
    phi_numbers = 5;
    psi_numbers = 5;
    phi_bit = 6;
    psi_bit = 4;
    tot_angles = phi_numbers + psi_numbers;
    tot_bytes = ceil((phi_numbers * phi_bit + psi_numbers * psi_bit) / 8)-1;
    
    BW = 80;                 
    NSUBC = 256;
    subcarrier_idxs = linspace(1, NSUBC, NSUBC) - NSUBC/2 - 1;
    pilot_subcarriers = [25, 53, 89, 117, 139, 167, 203, 231];
    num_pilots = numel(pilot_subcarriers);
    subcarrier_idxs(252:end) = [];
    subcarrier_idxs(231) = [];
    subcarrier_idxs(203) = [];
    subcarrier_idxs(167) = [];
    subcarrier_idxs(139) = [];
    subcarrier_idxs(128:130) = [];
    subcarrier_idxs(117) = [];
    subcarrier_idxs(89) = [];
    subcarrier_idxs(53) = [];
    subcarrier_idxs(25) = [];
    subcarrier_idxs(1:6) = [];
    
    
    NSUBC_VALID = numel(subcarrier_idxs);
    length_angles_users = NSUBC_VALID*6.25;
    length_report_users = ((NSUBC_VALID + num_pilots)/2 + 1)/2*Nc;
    
    length_angles = length_angles_users(1);
    length_report = length_report_users(1);
    
    
    payload_length = 1462;
    order_angles = ['phi_11', 'phi_21', 'phi_31', 'psi_21', 'psi_31', 'psi_41', 'phi_22', 'phi_32', 'psi_32', 'psi_42']; 
    order_bits = [phi_bit, phi_bit, phi_bit, psi_bit, psi_bit, psi_bit, phi_bit, phi_bit, psi_bit, psi_bit];%S
    
    % 初始化结果存储
    beamf_angles = {};
    time_vector = {};
    excl_beamf_reports = {}; % 存储 exclusive beamforming report
    vtilde_matrices = {};    % 存储 vtilde matrix
    
    %% 处理文件
    skip_start = 263;
    %skip_start_sample = 25;
    disp(['正在处理文件: ', file_path]);
    p = readpcap_beamf(); % 调用外部的解析函数
    
    p.open(file_path, skip_start)
    
    % 解析文件
    k = 1;
    k_start =1;
    j = 0;
    while true
        if j == 0
            skip_start_sample = 25;
            f = p.next(payload_length, skip_start_sample); % 读取单个数据包
            j = j + 1;
        else 
            skip_start_sample = 34;
            f = p.next(payload_length, skip_start_sample); % 读取单个数据包
        end
        
        
        if isempty(f.payload)
            disp('所有数据包处理完成');
            break;
        end
        if size(f.payload, 1) < payload_length
            disp('警告: 数据包长度不匹配');
            break;
        end
    

        
        % 提取时间戳（用自定义函数替代 de2bi）
        %disp(f.header)
        %disp(f.header.radiotap_header)
        timestamp_dec = flip(f.header.radiotap_header(9:16));
        timestamp_bin = de2bi(timestamp_dec, 'left-msb', 8); % 使用自定义函数
        timestamp_bin = reshape(timestamp_bin.', 1, []);
        timestamp = bi2de(timestamp_bin, 'left-msb');
        time_vector{k-k_start+1} = timestamp;

        snr = flip(f.payload(1:Nc));
    
        % 提取波束成形角度
        start_angles = Nc+1;
        end_angles =ceil(length_angles-1);%1460;% ;
        angle_values = f.payload(start_angles:end_angles);
        beamforming_angles = zeros(NSUBC_VALID, tot_angles);
        for s_i = 1:NSUBC_VALID
            start_idx = (s_i-1)*tot_bytes + 1;
            end_idx = s_i*tot_bytes;
            angles_subc_dec = (angle_values(start_idx:end_idx));
            angles_subc = de2bi(angles_subc_dec, 'right-msb', 8);
            angles_subc = reshape(angles_subc.', 1, []);
            i_curs = 1;
            for a_i = 1:7
                num_b = order_bits(a_i);
                angle_val = bi2de(angles_subc(i_curs:i_curs+num_b-1), 'right-msb');
                beamforming_angles(s_i, a_i) = angle_val;
                i_curs = i_curs + num_b;
            end
        end
        beamf_angles{k-k_start+1} = beamforming_angles;
    
        % 提取 exclusive beamforming report
%         start_report = ceil(start_angles+length_angles);
%         end_report = ceil(start_report+length_report-1);
%         
%         length_payload = numel(f.payload);
%         if end_report > length_payload
%             disp('警告: 数据包中 exclusive beamforming report 超出范围');
%             continue;
%         end
%     
%         exclusive_beamf_report = f.payload(start_report:end_report);
%         %disp(excl_beamf_report);
%         excl_beamf_reports{k-k_start+1} = exclusive_beamf_report;
    
    
        % 计算 vtilde matrix
        if Nc == 2
            vtilde_matrix = Vtilde_NSS2(beamforming_angles, Nc, Nr, NSUBC_VALID, phi_bit, psi_bit);
        elseif Nc == 1
            vtilde_matrix = Vtilde_NSS1(beamforming_angles, Nc, Nr, NSUBC_VALID, phi_bit, psi_bit);
            %disp(vtilde_matrices);
        end
        vtilde_matrices{k-k_start+1} = vtilde_matrix;
        %disp(vtilde_matrix)
        
    
        k = k + 1;
    end
    
    
    
    [~, file_name, ~] = fileparts(file_path);
    
    
    char_position = 1:2; 
    
    if ~isempty(file_name)  % 确保 file_name 不是空字符串

        extracted_str = file_name(char_position);
    else
        error('文件名为空，无法提取字符');
    end

    save(fullfile(output_folder_time_vector, [extracted_str, '_time_vector', num2str(i), '.mat']), 'time_vector');
    save(fullfile(output_folder_beamf_angles, [extracted_str, '_beamf_angles',num2str(i), '.mat']), 'beamf_angles');
    save(fullfile(output_folder_exclusive_beamf_reports, [extracted_str, '_exclusive_beamf_reports',num2str(i), '.mat']), 'excl_beamf_reports');
    save(fullfile(output_folder_vtilde_matrices, [extracted_str, '_vtilde_matrices',num2str(i), '.mat']), 'vtilde_matrices');
    
    disp('文件处理完成，结果已保存');

end
%% 自定义函数：custom_de2bi
function bin = custom_de2bi(dec, len)
    bin = zeros(numel(dec), len); % 初始化矩阵
    for i = 1:numel(dec)
        bin(i, :) = bitget(dec(i), len:-1:1); % 逐位获取二进制值
    end
end