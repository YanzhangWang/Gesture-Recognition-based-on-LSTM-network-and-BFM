import threading
import queue
import time
import os
import numpy as np
import shutil
import pyshark
import math
from textwrap import wrap
from bfi_angles import bfi_angles
from vmatrices import vmatrices
from utils import hex2dec, flip_hex

class FileProcessorThread(threading.Thread):
    """File processing thread - Consumer"""
    def __init__(self, task_queue, result_queue, data_processor, thread_id):
        threading.Thread.__init__(self)
        self.daemon = True
        self.task_queue = task_queue  # Task queue
        self.result_queue = result_queue  # Result queue
        self.dp = data_processor  # Data processor reference
        self.thread_id = thread_id  # Thread ID
        self.running = True  # Control thread running
        
    def run(self):
        """Thread main function"""
        self.dp.log_message(f"Processing thread {self.thread_id} has started")
        
        while self.running:
            try:
                # Get file from task queue, wait 1 second if queue is empty
                file_data = self.task_queue.get(timeout=1)
                
                if file_data is None:  # Termination signal
                    self.running = False
                    break
                    
                # Unpack file data
                file_path, file_name = file_data
                
                # Process file
                self.dp.log_message(f"Thread {self.thread_id} starts processing file: {file_name}")
                start_time = time.time()
                
                try:
                    # Call processing function
                    result = self.process_file(file_path, file_name)
                    
                    # If there is a result, put it in the result queue
                    if result:
                        self.result_queue.put((file_name, result))
                        processing_time = time.time() - start_time
                        self.dp.log_message(f"Thread {self.thread_id} completed processing file {file_name} (Time: {processing_time:.2f} seconds)")
                    else:
                        self.dp.log_message(f"Thread {self.thread_id} processed file {file_name} with no valid results")
                        
                except Exception as e:
                    self.dp.log_message(f"Thread {self.thread_id} error processing file {file_name}: {str(e)}", error=True)
                
                finally:
                    # Mark task as complete
                    self.task_queue.task_done()
                    
            except queue.Empty:
                # Queue is empty, continue waiting
                pass
                
            except Exception as e:
                self.dp.log_message(f"Processing thread {self.thread_id} encountered an error: {str(e)}", error=True)
                
        self.dp.log_message(f"Processing thread {self.thread_id} has stopped")
    
    def process_file(self, file_path, file_name):
        """Process a single file - Implement file processing logic"""
        temp_file = file_path + f"_temp_{self.thread_id}"
        
        try:
            # Copy file
            shutil.copy2(file_path, temp_file)
            
            # Initialize lists to store feedback angles and v matrices
            bfi_angles_all_packets = []
            v_matrices_all = []
            
            # Get parameters from data processor
            standard = self.dp.standard
            mimo = self.dp.mimo
            config = self.dp.config
            MAC = self.dp.MAC
            NSUBC_VALID = self.dp.NSUBC_VALID
            Nr = self.dp.Nr
            Nc_users = self.dp.Nc_users
            
            # Set angle configuration
            if config == "4x2":
                phi_numbers = 5
                psi_numbers = 5
                
            # Set LSB parameter
            LSB = True
            
            try:
                # Read packets from pcap file according to selected standard
                if standard == "AC":
                    reader = pyshark.FileCapture(
                        input_file=temp_file,
                        display_filter=f'wlan.vht.mimo_control.feedbacktype=={mimo} && wlan.addr=={MAC}',
                        use_json=True,
                        include_raw=True
                    )._packets_from_tshark_sync()
                else:
                    raise ValueError(f"Unsupported standard: {standard}")
            except Exception as e:
                self.dp.log_message(f"Thread {self.thread_id} failed to create pyshark reader: {str(e)}", error=True)
                raise
                
            # Process each packet
            packet_count = 0
            valid_packets = 0
            
            while True:
                try:
                    # Get next packet
                    packet = reader.__next__().frame_raw.value
                    packet_count += 1
                    
                    # Extract frame header information
                    Header_rivision_dec = hex2dec(flip_hex(packet[0:2]))
                    Header_pad_dec = hex2dec(flip_hex(packet[2:4]))
                    Header_length_dec = hex2dec(flip_hex(packet[4:8]))
                    i = Header_length_dec * 2
                    
                    # Extract various fields
                    Frame_Control_Field_hex = packet[i:(i + 4)]
                    packet_duration = packet[(i + 4):(i + 8)]
                    packet_destination_mac = packet[(i + 8):(i + 20)]
                    packet_sender_mac = packet[(i + 20):(i + 32)]
                    packet_BSS_ID = packet[(i + 32):(i + 44)]
                    packet_sequence_number = packet[(i + 44):(i + 48)]
                    packet_HE_category = packet[(i + 48):(i + 50)]
                    packet_CQI = packet[(i + 50):(i + 52)]
                    
                    # Extract AC standard specific fields
                    if standard == "AC":
                        packet_mimo_control = packet[(i + 52):(i + 58)]
                        packet_mimo_control_binary = ''.join(format(int(char, 16), '04b') for char in flip_hex(packet_mimo_control))
                        codebook_info = packet_mimo_control_binary[13]
                        packet_snr = packet[(i + 58):(i + 60)]
                        frame_check_sequence = packet[-8:]
                        
                    # Set angle bits
                    if mimo == "SU":
                        if codebook_info == "1":
                            psi_bit = 4
                            phi_bit = psi_bit + 2
                        else:
                            psi_bit = 2
                            phi_bit = psi_bit + 2
                    
                    # Set parameters according to configuration
                    if config == "4x2":
                        order_bits = [phi_bit, phi_bit, phi_bit, psi_bit, psi_bit, psi_bit, phi_bit, phi_bit, psi_bit, psi_bit]
                        tot_angles_users = phi_numbers + psi_numbers
                        tot_bits_users = phi_numbers * phi_bit + psi_numbers * psi_bit
                        matrix_shape = (4, 2)
                    
                    # Set valid subcarrier constants
                    length_angles_users_bits = NSUBC_VALID * tot_bits_users
                    length_angles_users = math.floor(length_angles_users_bits / 8)
                    
                    # Extract feedback angles
                    if standard == "AC":
                        Feedback_angles = packet[(i + 58 + 2 * int(config[-1])):(len(packet) - 8)]
                        Feedback_angles_splitted = np.array(wrap(Feedback_angles, 2))
                        Feedback_angles_bin = ""
                    
                    # Convert feedback angles to binary format
                    for j in range(0, len(Feedback_angles_splitted)):
                        bin_str = str(format(hex2dec(Feedback_angles_splitted[j]), '08b'))
                        if LSB:
                            bin_str = bin_str[::-1]
                        Feedback_angles_bin += bin_str
                    
                    Feed_back_angles_bin_chunk = np.array(wrap(Feedback_angles_bin[:(tot_bits_users * NSUBC_VALID)], tot_bits_users))
                    
                    if Feed_back_angles_bin_chunk.shape[0] != NSUBC_VALID:
                        self.dp.log_message(f"Bandwidth mismatch: Expected {NSUBC_VALID}, actually got {Feed_back_angles_bin_chunk.shape[0]}")
                        continue
                    
                    # Process angles and V matrices
                    bfi_angles_single_pkt = bfi_angles(Feed_back_angles_bin_chunk, LSB, NSUBC_VALID, order_bits)
                    v_matrix_all = vmatrices(
                        bfi_angles_single_pkt,
                        phi_bit,
                        psi_bit,
                        NSUBC_VALID,
                        Nr,
                        Nc_users,
                        config
                    )
                    
                    v_matrices_all.append(v_matrix_all)
                    bfi_angles_all_packets.append(bfi_angles_single_pkt)
                    valid_packets += 1
                    
                except StopIteration:
                    # Packet reading complete
                    break
                except Exception as e:
                    # Error processing a single packet, continue to next
                    self.dp.log_message(f"Thread {self.thread_id} error processing packet: {str(e)}", error=True)
                    continue
            
            # Clean up resources
            reader.close()
            if os.path.exists(temp_file):
                os.remove(temp_file)
            
            # Record processing results
            self.dp.log_message(f"Thread {self.thread_id} processed {packet_count} packets, found {valid_packets} valid packets")
            
            # Return processing results
            if v_matrices_all:
                return {
                    'v_matrices': v_matrices_all,
                    'bfi_angles': bfi_angles_all_packets,
                    'thread_id': self.thread_id,
                    'valid_packets': valid_packets,
                    'total_packets': packet_count
                }
            else:
                self.dp.log_message(f"Thread {self.thread_id} did not extract valid data from file {file_name}")
                return None
            
        except Exception as e:
            self.dp.log_message(f"Thread {self.thread_id} error processing file {file_path}: {str(e)}", error=True)
            return None
            
        finally:
            # Clean up temporary files
            if os.path.exists(temp_file):
                try:
                    os.remove(temp_file)
                except Exception as e:
                    self.dp.log_message(f"Thread {self.thread_id} error deleting temporary file: {str(e)}", error=True)
    
    def stop(self):
        """Stop thread"""
        self.running = False


class ResultProcessorThread(threading.Thread):
    """Result processing thread"""
    def __init__(self, result_queue, data_processor):
        threading.Thread.__init__(self)
        self.daemon = True
        self.result_queue = result_queue  # Result queue
        self.dp = data_processor  # Data processor reference
        self.running = True  # Control thread running
        
    def run(self):
        """Thread main function"""
        self.dp.log_message("Result processing thread has started")
        
        while self.running:
            try:
                # Get processing result from result queue, wait 1 second if queue is empty
                result_data = self.result_queue.get(timeout=1)
                
                if result_data is None:  # Termination signal
                    self.running = False
                    break
                    
                # Unpack result data
                file_name, result = result_data
                
                # Process result
                self.process_result(file_name, result)
                
                # Mark result processing as complete
                self.result_queue.task_done()
                
            except queue.Empty:
                # Queue is empty, continue waiting
                pass
                
            except Exception as e:
                self.dp.log_message(f"Result processing thread encountered an error: {str(e)}", error=True)
                
        self.dp.log_message("Result processing thread has stopped")
    
    def process_result(self, file_name, result):
        """Process a single result - e.g., save file, update UI, etc."""
        try:
            # Extract processed data from the result
            v_matrices = result.get('v_matrices', [])
            bfi_angles = result.get('bfi_angles', [])
            thread_id = result.get('thread_id', 'unknown')
            
            self.dp.log_message(f"Result from processing thread {thread_id}: File {file_name}")
            
            # Save result to file
            if v_matrices and bfi_angles:
                # Save v matrices
                vmatrix_path = os.path.join(self.dp.exp_dir, 'vmatrix', f'{file_name}_vmatrix.npy')
                np.save(vmatrix_path, v_matrices)
                
                # Save angle data
                angles_path = os.path.join(self.dp.exp_dir, 'bfa', f'{file_name}_angles.npy')
                np.save(angles_path, bfi_angles)
                
                # Update UI (using main thread)
                if v_matrices:
                    # Update default matrix
                    self.dp.default_v_matrix = v_matrices[-1]
                    
                    # Update chart in main thread
                    self.dp.ui.after(0, lambda: self.dp.update_visualization(v_matrices[-1]))
                    
                    # Use model for prediction
                    if self.dp.model and self.dp.model.is_loaded:
                        # Ensure file is written
                        time.sleep(0.1)
                        
                        try:
                            # Use model to predict
                            result = self.dp.model.predict_with_static_detection(vmatrix_path, verbose=False)
                            
                            # Update prediction results in main thread
                            if result:
                                self.dp.ui.after(0, lambda r=result: self.dp.update_prediction(r))
                        except Exception as e:
                            self.dp.log_message(f"Error predicting file {file_name}: {str(e)}", error=True)
            else:
                self.dp.log_message(f"File {file_name} has no valid processing results")
                
        except Exception as e:
            self.dp.log_message(f"Error processing result: {str(e)}", error=True)
    
    def stop(self):
        """Stop thread"""
        self.running = False