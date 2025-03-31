import os
import threading
import time
import numpy as np
import pyshark
import math
from textwrap import wrap
from visualization import update_plot
from utils import hex2dec, flip_hex
from bfi_angles import bfi_angles
from vmatrices import vmatrices
import shutil

# 设置LSB参数
LSB = True

class DataProcessor(threading.Thread):
    def __init__(self, ui, model, args, log_function, update_function, ax, canvas):
        threading.Thread.__init__(self)
        self.daemon = True
        
        # 保存引用
        self.ui = ui
        self.model = model
        self.args = args
        self.log_message = log_function
        self.update_prediction = update_function
        self.ax = ax
        self.canvas = canvas
        
        # 从参数中提取
        self.standard = args.standard
        self.mimo = args.mimo
        self.config = args.config
        self.bw = int(args.bw)
        self.MAC = args.MAC
        self.exp_dir = '/home/ggbo/Wi-BFI/Demo/' + args.dir + '/'
        print(self.exp_dir)
        self.step = args.seconds_interval
        
        # 初始化
        self.processed_files = set()
        self.default_v_matrix = None
        
    def run(self):
        """线程主函数，运行数据处理循环"""
        try:
            # 确保目录存在
            os.makedirs(self.exp_dir + 'vmatrix', exist_ok=True)
            os.makedirs(self.exp_dir + 'bfa', exist_ok=True)
            
            # 设置参数
            self.setup_parameters()
            
            # 等待start_time.txt文件
            start_time = self.wait_for_start_time()
            
            # 主循环
            self.main_processing_loop(start_time)
            
        except Exception as e:
            self.log_message(f"Error in data processor: {str(e)}", error=True)
    
    def setup_parameters(self):
        """设置处理参数"""
        # 根据标准设置参数
        if self.standard == "AC":
            if self.bw == 80:
                self.subcarrier_idxs = np.arange(-122, 123)
                self.pilot_n_null = np.array([-104, -76, -40, -12, -1, 0, 1, 10, 38, 74, 102])
                self.subcarrier_idxs = np.setdiff1d(self.subcarrier_idxs, self.pilot_n_null)
                
        self.NSUBC_VALID = len(self.subcarrier_idxs)
        
        # 根据配置设置参数
        if self.config == "4x2":
            self.Nc_users = 2  # 空间流数量
            self.Nr = 4  # 发送天线数量
            self.phi_numbers = 5
            self.psi_numbers = 5
            self.order_angles = ['phi_11', 'phi_21', 'phi_31', 'psi_21', 'psi_31', 'psi_41', 'phi_22', 'phi_32', 'psi_32', 'psi_42']
        
        self.log_message(f"Parameters set: NSUBC_VALID={self.NSUBC_VALID}, Nr={self.Nr}, Nc_users={self.Nc_users}")
    
    def wait_for_start_time(self):
        """等待start_time.txt文件或使用默认值"""
        self.log_message(f"Looking for start_time.txt in: {self.exp_dir}")
        
        start_time_path = self.exp_dir + 'start_time.txt'
        if os.path.exists(start_time_path):
            try:
                with open(start_time_path) as f:
                    content = f.read().strip()
                    self.log_message(f"start_time.txt content: '{content}'")
                    start_time = int(content)
                    self.log_message(f"Found start time: {start_time}")
                    return start_time
            except Exception as e:
                self.log_message(f"Error reading start_time.txt: {str(e)}", error=True)
        
        # 如果文件不存在，使用当前时间
        import time
        current_time = int(time.time())
        self.log_message(f"start_time.txt not found, using current time: {current_time}")
        
        # 将当前时间写入文件以便后续使用
        try:
            with open(start_time_path, 'w') as f:
                f.write(str(current_time))
            self.log_message(f"Created start_time.txt with time: {current_time}")
        except Exception as e:
            self.log_message(f"Failed to create start_time.txt: {e}", error=True)
        
        return current_time

    
    def main_processing_loop(self, start_time):
        """主数据处理循环"""
        self.log_message("Starting main processing loop...")
        
        while True:
            try:
                # 获取目录中所有数字文件名并排序
                all_files = [f for f in os.listdir(self.exp_dir) if f.isdigit()]
                all_files.sort(key=int)
                
                # 跳过已处理的文件
                unprocessed_files = [f for f in all_files if f not in self.processed_files]
                
                if not unprocessed_files:
                    time.sleep(1)
                    continue
                    
                # 获取最早的未处理文件
                current_file = unprocessed_files[0]
                file_path = self.exp_dir + current_file
                self.log_message(f"Processing file: {current_file}")
                
                # 添加延迟确保文件完全写入
                time.sleep(0.5)
                
                # 检查文件是否存在且大小稳定
                if not os.path.exists(file_path):
                    time.sleep(1)
                    continue
                    
                # 检查文件大小是否稳定
                file_size_1 = os.path.getsize(file_path)
                time.sleep(0.5)
                file_size_2 = os.path.getsize(file_path)
                
                if file_size_1 != file_size_2:
                    self.log_message(f"File {current_file} is still being written, waiting...")
                    time.sleep(1)
                    continue
                    
                # 处理文件
                self.process_file(file_path, current_file)
                
            except Exception as e:
                self.log_message(f"Error in main loop: {str(e)}", error=True)
                time.sleep(1)
    
    def process_file(self, file_path, file_name):
        """处理单个文件"""
        # 创建临时文件副本
        temp_file = file_path + "_temp"
        try:
            os.system(f"cp {file_path} {temp_file}")
        except Exception as e:
            self.log_message(f"Error creating temp file: {str(e)}", error=True)
            return
            
        # 初始化列表以存储反馈角度和v矩阵
        bfi_angles_all_packets = []
        v_matrices_all = []
        
        try:
            # 根据选择的标准从pcap文件中读取数据包
            if self.standard == "AC":
                reader = pyshark.FileCapture(
                    input_file=temp_file,
                    display_filter=f'wlan.vht.mimo_control.feedbacktype=={self.mimo} && wlan.addr=={self.MAC}',
                    use_json=True,
                    include_raw=True
                )._packets_from_tshark_sync()
                
            # 处理每个数据包
            packet_count = 0
            valid_packets = 0
            
            while True:
                try:
                    packet = reader.__next__().frame_raw.value
                    packet_count += 1
                except StopIteration:
                    break
                except Exception as e:
                    self.log_message(f"Error reading packet: {str(e)}", error=True)
                    continue
                    
                try:
                    # 从原始帧数据中提取头信息
                    Header_rivision_dec = hex2dec(flip_hex(packet[0:2]))
                    Header_pad_dec = hex2dec(flip_hex(packet[2:4]))
                    Header_length_dec = hex2dec(flip_hex(packet[4:8]))
                    i = Header_length_dec * 2
                    
                    # 从帧中提取各种字段
                    Frame_Control_Field_hex = packet[i:(i + 4)]
                    packet_duration = packet[(i + 4):(i + 8)]
                    packet_destination_mac = packet[(i + 8):(i + 20)]
                    packet_sender_mac = packet[(i + 20):(i + 32)]
                    packet_BSS_ID = packet[(i + 32):(i + 44)]
                    packet_sequence_number = packet[(i + 44):(i + 48)]
                    packet_HE_category = packet[(i + 48):(i + 50)]
                    packet_CQI = packet[(i + 50):(i + 52)]
                    
                    # 提取AC标准的特定字段
                    if self.standard == "AC":
                        packet_mimo_control = packet[(i + 52):(i + 58)]
                        packet_mimo_control_binary = ''.join(format(int(char, 16), '04b') for char in flip_hex(packet_mimo_control))
                        codebook_info = packet_mimo_control_binary[13]
                        packet_snr = packet[(i + 58):(i + 60)]
                        frame_check_sequence = packet[-8:]
                        
                    # 根据mimo类型设置角度位
                    if self.mimo == "SU":
                        if codebook_info == "1":
                            psi_bit = 4
                            phi_bit = psi_bit + 2
                        else:
                            psi_bit = 2
                            phi_bit = psi_bit + 2
                            
                    if self.config == "4x2":
                        order_bits = [phi_bit, phi_bit, phi_bit, psi_bit, psi_bit, psi_bit, phi_bit, phi_bit, psi_bit, psi_bit]
                        tot_angles_users = self.phi_numbers + self.psi_numbers
                        tot_bits_users = self.phi_numbers * phi_bit + self.psi_numbers * psi_bit
                        matrix_shape = (4, 2)
                        
                    # 设置有效子载波的常数
                    length_angles_users_bits = self.NSUBC_VALID * tot_bits_users
                    length_angles_users = math.floor(length_angles_users_bits / 8)
                    
                    # 提取反馈角度
                    if self.standard == "AC":
                        Feedback_angles = packet[(i + 58 + 2 * int(self.config[-1])):(len(packet) - 8)]
                        Feedback_angles_splitted = np.array(wrap(Feedback_angles, 2))
                        Feedback_angles_bin = ""
                        
                    # 将反馈角度转换为二进制格式
                    for j in range(0, len(Feedback_angles_splitted)):
                        bin_str = str(format(hex2dec(Feedback_angles_splitted[j]), '08b'))
                        if LSB:
                            bin_str = bin_str[::-1]
                        Feedback_angles_bin += bin_str
                        
                    Feed_back_angles_bin_chunk = np.array(wrap(Feedback_angles_bin[:(tot_bits_users * self.NSUBC_VALID)], tot_bits_users))
                    
                    if Feed_back_angles_bin_chunk.shape[0] != self.NSUBC_VALID:
                        self.log_message(f"Bandwidth mismatch: expected {self.NSUBC_VALID}, got {Feed_back_angles_bin_chunk.shape[0]}")
                        continue
                        
                    # 处理角度和V矩阵
                    bfi_angles_single_pkt = bfi_angles(Feed_back_angles_bin_chunk, LSB, self.NSUBC_VALID, order_bits)
                    v_matrix_all = vmatrices(
                        bfi_angles_single_pkt,
                        phi_bit,
                        psi_bit,
                        self.NSUBC_VALID,
                        self.Nr,
                        self.Nc_users,
                        self.config
                    )
                    
                    v_matrices_all.append(v_matrix_all)
                    bfi_angles_all_packets.append(bfi_angles_single_pkt)
                    valid_packets += 1
                    
                except Exception as e:
                    self.log_message(f"Error processing packet data: {str(e)}", error=True)
                    continue
                    
            # 关闭读取器和清理临时文件
            reader.close()
            if os.path.exists(temp_file):
                os.remove(temp_file)
                
            # 将处理过的文件添加到已处理集合
            self.processed_files.add(file_name)
            
            # 日志记录处理结果
            self.log_message(f"Processed {packet_count} packets, found {valid_packets} valid ones")
            
            # 处理提取的数据
            if v_matrices_all:
                # 保存 v 矩阵和角度
                vmatrix_path = self.exp_dir + 'vmatrix/' + file_name + '_vmatrix.npy'
                angles_path = self.exp_dir + 'bfa/' + file_name + '_angles.npy'
                
                np.save(vmatrix_path, v_matrices_all)
                np.save(angles_path, bfi_angles_all_packets)
                
                # 更新图表
                self.default_v_matrix = v_matrices_all[-1]
                update_plot(v_matrices_all[-1], self.ax)
                self.canvas.draw()
                
                # 如果模型已加载，对保存的v矩阵文件进行预测
                if self.model and self.model.is_loaded:
                    # 确保文件已经被写入
                    time.sleep(0.1)
                    if os.path.exists(vmatrix_path):
                        result = self.model.predict(vmatrix_path)
                        if result:
                            # 在主线程中更新UI
                            self.ui.after(0, lambda: self.update_prediction(result))
            else:
                self.log_message(f"No valid data extracted from {file_name}")
                # 如果有默认矩阵，则使用它绘制
                if self.default_v_matrix is not None:
                    update_plot(self.default_v_matrix, self.ax)
                    self.canvas.draw()
                    
        except Exception as e:
            self.log_message(f"Error processing file {file_path}: {str(e)}", error=True)
            if os.path.exists(temp_file):
                os.remove(temp_file)


