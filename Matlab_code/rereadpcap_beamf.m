classdef rereadpcap_beamf < handle
    
    properties
        fid; % 文件句柄
    end
    
    methods
        % 打开文件并跳过初始字节
        function open(obj, filename, skip_start)
            obj.fid = fopen(filename, 'rb');
            if obj.fid == -1
                error('无法打开文件: %s', filename);
            end
            
            % 检查文件大小
            fseek(obj.fid, 0, 'eof'); % 移动到文件末尾
            file_size = ftell(obj.fid); % 获取文件大小
            fseek(obj.fid, 0, 'bof'); % 返回文件开头
            
            if skip_start > file_size
                error('skip_start 超过文件大小: %d bytes', file_size);
            end
            
            % 跳过指定字节
            fread(obj.fid, skip_start, '*uint8');
        end
        
        % 读取下一个数据包
        function frame = next(obj, payload_length, skip_start_sample)
            % 检查文件结束
            if feof(obj.fid)
                disp('文件读取到末尾');
                frame.payload = {};
                return;
            end

            % 跳过指定的样本字节
            fread(obj.fid, skip_start_sample, '*uint8');

            % 读取头部长度信息
            fread(obj.fid, 2, '*uint8'); % 跳过 2 字节

            % 获取当前文件指针位置
            position_before_header_length = ftell(obj.fid);
            disp(['当前读取 header_length 位置: ', num2str(position_before_header_length)]);
            
            % 读取 header_length
            frame.header.header_length = fread(obj.fid, 1, '*uint8');
            disp(['读取到的 header_length: ', num2str(frame.header.header_length)]);
            
            % 判断 header_length 是否正确
            if frame.header.header_length ~= 60
                error(['错误: header_length 不是 60, 而是 ', num2str(frame.header.header_length)]);
            end
            
            % 验证 header_length 合法性
            if frame.header.header_length < 3
                disp('header_length 不合法，跳过当前数据包');
                frame.payload = {};
                return;
            end
            
            % 读取 radiotap_header
            try
                frame.header.radiotap_header = fread(obj.fid, frame.header.header_length - 3, '*uint8');
            catch
                disp('读取 radiotap_header 时发生错误');
                frame.payload = {};
                return;
            end

            % 继续读取其他头部信息
            frame.header.control_field = fread(obj.fid, 2, '*uint8');
            frame.header.duration = fread(obj.fid, 2, '*uint8');
            frame.header.destination_mac = dec2hex(fread(obj.fid, 6, '*uint8'));
            frame.header.source_mac = dec2hex(fread(obj.fid, 6, '*uint8'));
            frame.header.bss_id = dec2hex(fread(obj.fid, 6, '*uint8'));
            frame.header.seq_frag_number = fread(obj.fid, 2, '*uint8');
            frame.header.category = fread(obj.fid, 1, '*uint8');
            fread(obj.fid, 1, '*uint8'); % 跳过 1 字节
            frame.header.mimo_control = fread(obj.fid, 3, '*uint8');

            % 读取 payload
            frame.payload = fread(obj.fid, payload_length, '*uint8');
            
            % 检查 payload 长度
            if numel(frame.payload) < payload_length
                disp('payload 长度不足，可能已到文件末尾');
                frame.payload = {};
                return;
            end

            % 读取 FCS
            frame.FCS = fread(obj.fid, 4, '*uint8');
        end
        
        % 关闭文件
        function close(obj)
            if obj.fid > 0
                fclose(obj.fid);
            end
        end
    end
end
