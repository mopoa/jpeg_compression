import socket
import numpy as np
from PIL import Image



huffman_ac = {
    (0, 0): "1010",
    (0, 1): "11000",
    (0, 2): "11001",
    (0, 3): "11100",
    # Add more entries as needed
}


q_table_y = np.array([[16, 11, 10, 16, 24, 40, 51, 61],
                      [12, 12, 14, 19, 26, 58, 60, 55],
                      [14, 13, 16, 24, 40, 57, 69, 56],
                      [14, 17, 22, 29, 51, 87, 80, 62],
                      [18, 22, 37, 56, 68, 109, 103, 77],
                      [24, 35, 55, 64, 81, 104, 113, 92],
                      [49, 64, 78, 87, 103, 121, 120, 101],
                      [72, 92, 95, 98, 112, 100, 103, 99]])


q_table_iq = np.array([[17, 18, 24, 47, 99, 99, 99, 99],
                       [18, 21, 26, 66, 99, 99, 99, 99],
                       [24, 26, 56, 99, 99, 99, 99, 99],
                       [47, 66, 99, 99, 99, 99, 99, 99],
                       [99, 99, 99, 99, 99, 99, 99, 99],
                       [99, 99, 99, 99, 99, 99, 99, 99],
                       [99, 99, 99, 99, 99, 99, 99, 99],
                       [99, 99, 99, 99, 99, 99, 99, 99]])


def receive_strings_from_network(host, port):
    try:
        # Create a socket object
        server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        
        # Bind the socket to the host and port
        server_socket.bind((host, port))
        
        # Listen for incoming connections
        server_socket.listen(1)
        print("Waiting for a connection...")
        
        # Accept a connection
        client_socket, client_address = server_socket.accept()
        print("Connected to:", client_address)
        
        # Receive strings
        received_strings = []
        while True:
            data = client_socket.recv(1024)  # Receive data (up to 1024 bytes)
            if not data:
                break
            received_string = data.decode('utf-8')  # Decode bytes to string
            received_strings.append(received_string)
        
        print("Received strings:", received_strings)
        
    except Exception as e:
        print("Error:", e)
    finally:
        # Close the client socket
        client_socket.close()
        
        # Close the server socket
        server_socket.close()
    return received_strings



def huffman_decode(encoded_bits, huffman_table):
    decoded_coefficients = []
    current_code = ""
    
    for bit in encoded_bits:
        current_code += bit
        for (run_length, coef), code in huffman_table.items():
            if current_code == code:
                decoded_coefficients.append((run_length, coef))
                current_code = ""
                break
    
    return decoded_coefficients

def run_length_decode(encoded_coefficients):
    decoded_coefficients = []
    
    for run_length, coef in encoded_coefficients:
        if coef == 0 and run_length == 0:
            break  # Reached end-of-block marker (EOB)
        
        if coef == 0:
            decoded_coefficients.extend([0] * run_length)
        else:
            decoded_coefficients.append(coef)
    
    return decoded_coefficients


def dequantize(block_quantized, q_table):
    return block_quantized * q_table


def idct2(block_dct):
    block_idct = np.fft.ifft2(block_dct).real.astype(int)
    return block_idct + 128


def yiq2rgb(yiq_img):
    y, i, q = yiq_img[:, :, 0], yiq_img[:, :, 1], yiq_img[:, :, 2]
    r = y + 0.956 * i + 0.621 * q
    g = y - 0.272 * i - 0.647 * q
    b = y - 1.106 * i + 1.703 * q
    return np.dstack((r, g, b)).astype(np.uint8)





def save_rgb_image_to_ppm(image_data, file_path):
    # Create a new RGB image using Pillow
    image = Image.fromarray(image_data, mode='RGB')
    
    # Save the image in PPM format
    image.save(file_path)

# Usage example
# Assuming image_data is a NumPy array containing RGB image data
# and file_path is the path where you want to save the PPM file
# image_data = ...  # Your RGB image data
# file_path = "output_image.ppm"  # File path to save the PPM file
# save_rgb_image_to_ppm(image_data, file_path)


host = "192.168.1.107"  
port = 9595  
rcs=receive_strings_from_network(host, port)

encoded_rle_dct_y = rcs[0]
encoded_rle_dct_i = rcs[1]
encoded_rle_dct_q = rcs[2]


rle_dct_y = huffman_decode(encoded_rle_dct_y,huffman_ac)
rle_dct_i = huffman_decode(encoded_rle_dct_i,huffman_ac)
rle_dct_q = huffman_decode(encoded_rle_dct_q,huffman_ac)

quantized_dct_y = run_length_decode(rle_dct_y)
quantized_dct_i = run_length_decode(rle_dct_i)
quantized_dct_q = run_length_decode(rle_dct_q)

dct_y = dequantize(quantized_dct_y,q_table_y)
dct_i = dequantize(quantized_dct_i,q_table_iq)
dct_q = dequantize(quantized_dct_q,q_table_iq)

y, i, q= idct2(dct_y), idct2(dct_i), idct2(dct_q)

yiq_img = np.dstack((y, i, q))

rgb_img=yiq2rgb(yiq_img)

save_rgb_image_to_ppm(rgb_img,"img/rcvtest.ppm")