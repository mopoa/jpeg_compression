from collections import Counter
from colorama import Fore
from PIL import Image
import numpy as np
import colorama
import socket
import time


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


def rgb2yiq(img):
    r, g, b = img[:, :, 0], img[:, :, 1], img[:, :, 2]
    y = 0.299 * r + 0.587 * g + 0.114 * b
    i = 0.596 * r - 0.275 * g - 0.321 * b
    q = 0.212 * r - 0.523 * g + 0.311 * b
    return np.dstack((y, i, q))


def dct2(block):
    return np.round(np.fft.fft2(block - 128), decimals=0).astype(int)


def quantize(block, q_table):
    return np.round(block / q_table).astype(int)


def run_length_encode(coefficients):
    encoded_coefficients = []
    count_zeros = 0
    
    for coef in coefficients:
        if coef == 0:
            count_zeros += 1
        else:
            encoded_coefficients.append((count_zeros, coef))
            count_zeros = 0
    
    # Add end-of-block marker (EOB)
    encoded_coefficients.append((0, 0))
    
    return encoded_coefficients


def huffman_encode(coefficients, huffman_table):
    encoded_bits = ""
    
    for run_length, coef in coefficients:
        # Encode run-length and coefficient using Huffman table
        encoded_bits += huffman_table[(run_length, coef)] if (run_length, coef) in huffman_table else ""
    
    return encoded_bits



# change it to byteStream
def send_strings_over_network(strings, host, port):
    try:
        # Create a socket object
        client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        
        # Connect to the server
        client_socket.connect((host, port))
        
        # Send each string in the list
        for string in strings:
            # Encode the string as bytes before sending
            string_bytes = string.encode('utf-8')
            # Send the encoded bytes
            client_socket.sendall(string_bytes)
            print(f"String '{string}' sent successfully!")
        
    except Exception as e:
        return e
    finally:
        # Close the socket
        client_socket.close()


image_path = "img/testimage.ppm"
img = Image.open(image_path)


# save as jpeg in file
img_arr = np.array(img)
Image.fromarray(img_arr).save("testoutput.jpg","JPEG",quality=64)


yiq_img = rgb2yiq(img_arr)
y, i, q = yiq_img[:, :, 0], yiq_img[:, :, 1], yiq_img[:, :, 2]


dct_y = dct2(y)
dct_i = dct2(i)
dct_q = dct2(q)


quantized_dct_y = quantize(dct_y, q_table_y)
quantized_dct_i = quantize(dct_i, q_table_iq)
quantized_dct_q = quantize(dct_q, q_table_iq)


# Run-length encode quantized DCT coefficients
rle_dct_y = run_length_encode(quantized_dct_y.flatten())
rle_dct_i = run_length_encode(quantized_dct_i.flatten())
rle_dct_q = run_length_encode(quantized_dct_q.flatten())


# Encode run-length encoded coefficients using Huffman tables
encoded_rle_dct_y = huffman_encode(rle_dct_y, huffman_ac)
encoded_rle_dct_i = huffman_encode(rle_dct_i, huffman_ac)
encoded_rle_dct_q = huffman_encode(rle_dct_q, huffman_ac)


# huffman_dc = {
#     0: "00",
#     1: "010",
#     2: "011",
#     3: "100",
#     # Add more entries as needed
# }


# encoded_dct_y = huffman_encode(quantized_dct_y.flatten(), huffman_dc)
# encoded_dct_i = huffman_encode(quantized_dct_i.flatten(), huffman_ac)
# encoded_dct_q = huffman_encode(quantized_dct_q.flatten(), huffman_ac)



# def calculate_huffman_table(coefficients):
#     frequencies = Counter(coefficients)
#     pq = [(freq, symbol) for symbol, freq in frequencies.items()]
#     heapq.heapify(pq)
    
#     while len(pq) > 1:
#         freq1, symbol1 = heapq.heappop(pq)
#         freq2, symbol2 = heapq.heappop(pq)
#         merged_symbol = (symbol1, symbol2)
#         merged_freq = freq1 + freq2
#         heapq.heappush(pq, (merged_freq, merged_symbol))
    
#     huffman_table = {}
#     def generate_huffman_codes(node, code=""):
#         if isinstance(node, tuple):
#             generate_huffman_codes(node[0], code + "0")
#             generate_huffman_codes(node[1], code + "1")
#         else:
#             huffman_table[node] = code
    
#     if pq:
#         root = pq[0][1]
#         generate_huffman_codes(root)
    
#     return huffman_table
# def generate_huffman_codes(node, code="", huffman_table={}):
#     if node.symbol is not None:  # Leaf node (symbol)
#         huffman_table[node.symbol] = code
#     else:  # Internal node
#         if node.left is not None:
#             generate_huffman_codes(node.left, code + "0", huffman_table)
#         if node.right is not None:
#             generate_huffman_codes(node.right, code + "1", huffman_table)

# # Calculate Huffman codes and build Huffman tables
# def calculate_huffman_table(coefficients):
#     frequencies = Counter(coefficients)
#     pq = [Node(symbol=symbol, frequency=frequency) for symbol, frequency in frequencies.items()]
#     heapq.heapify(pq)

#     while len(pq) > 1:
#         left_node = heapq.heappop(pq)
#         right_node = heapq.heappop(pq)
#         merged_node = Node(frequency=left_node.frequency + right_node.frequency)
#         merged_node.left = left_node
#         merged_node.right = right_node
#         heapq.heappush(pq, merged_node)

#     root = pq[0]
#     huffman_table = {}
#     generate_huffman_codes(root, "", huffman_table)
#     return huffman_table


# huffman_dc_y = calculate_huffman_table(quantized_dct_y.flatten())
# huffman_ac_y = calculate_huffman_table(quantized_dct_y.flatten())
# huffman_dc_i = calculate_huffman_table(quantized_dct_i.flatten())
# huffman_ac_i = calculate_huffman_table(quantized_dct_i.flatten())
# huffman_dc_q = calculate_huffman_table(quantized_dct_q.flatten())
# huffman_ac_q = calculate_huffman_table(quantized_dct_q.flatten())



# # Print the Huffman-encoded bits
# print("Huffman-encoded DCT Y (Luminance):\n", encoded_dct_y)
# print("Huffman-encoded DCT I (In-phase Chrominance):\n", encoded_dct_i)
# print("Huffman-encoded DCT Q (Quadrature-phase Chrominance):\n", encoded_dct_q)

print("Y component (luminance):\n", y)
print("Cb component (blue-difference chrominance):\n", i)
print("Cr component (red-difference chrominance):\n", q)

# Print the quantized DCT matrices to the terminal
print("Quantized DCT Y (Luminance):\n", quantized_dct_y)
print("Quantized DCT I (In-phase Chrominance):\n", quantized_dct_i)
print("Quantized DCT Q (Quadrature-phase Chrominance):\n", quantized_dct_q)


# Print the Huffman-encoded bits
print(colorama.Fore.BLUE+"Huffman-encoded RLE DCT Y (Luminance):\n"+colorama.Fore.RESET, encoded_rle_dct_y)
print(colorama.Fore.BLUE+"Huffman-encoded RLE DCT I (In-phase Chrominance):\n"+colorama.Fore.RESET, encoded_rle_dct_i)
print(colorama.Fore.BLUE+"Huffman-encoded RLE DCT Q (Quadrature-phase Chrominance):\n"+colorama.Fore.RESET, encoded_rle_dct_q)


strings_to_send = [encoded_rle_dct_y,encoded_rle_dct_i,encoded_rle_dct_q]
host = "192.168.1.107"  
port = 9595
print("trying to connect...",end="")
i=10
while i!=0:
    try:
        send_strings_over_network(strings_to_send, host, port)
        i-=1
    except Exception as e:
        print("connection refused!!!")
    
    print(".",end="")
    time.sleep(1)


print()