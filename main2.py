import pycuda.autoinit
import pycuda.driver as cuda
from pycuda.compiler import SourceModule
import numpy as np
import time
import sys
import argparse
import struct
import random
import threading
from queue import Queue

def int_to_bigint_np(val):
    """Convert integer to BigInt numpy array"""
    bigint_arr = np.zeros(8, dtype=np.uint32)
    for j in range(8):
        bigint_arr[j] = (val >> (32 * j)) & 0xFFFFFFFF
    return bigint_arr

def bigint_np_to_int(bigint_arr):
    """Convert BigInt numpy array to integer"""
    val = 0
    for j in range(8):
        val |= int(bigint_arr[j]) << (32 * j)
    return val

def load_target_pubkeys(filename):
    """Load target public keys from file"""
    targets_bin = bytearray()
    is_text_file = filename.lower().endswith('.txt')

    with open(filename, 'r' if is_text_file else 'rb') as f:
        if is_text_file:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if len(line) == 66 and line[:2] in ('02', '03'):
                    try:
                        pubkey_bytes = bytes.fromhex(line)
                        targets_bin.extend(pubkey_bytes)
                    except ValueError:
                        print(f"[!] Warning: Invalid hex format on line {line_num}: {line}")
                elif line and not line.startswith('#'):
                    print(f"[!] Warning: Invalid format on line {line_num}: {line}")
        else:
            while True:
                chunk = f.read(33)
                if not chunk:
                    break
                if len(chunk) == 33:
                    targets_bin.extend(chunk)

    return targets_bin

def decompress_pubkey(compressed):
    """Decompress compressed public key"""
    if len(compressed) != 33:
        raise ValueError("Compressed public key should be 33 bytes")

    prefix = compressed[0]
    x_bytes = compressed[1:]

    if prefix not in [0x02, 0x03]:
        raise ValueError("Invalid prefix for compressed public key")

    # Convert x bytes to integer
    x_int = int.from_bytes(x_bytes, 'big')

    # Calculate y^2 = x^3 + 7 mod p
    p = 0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEFFFFFC2F
    x3 = pow(x_int, 3, p)
    y_sq = (x3 + 7) % p

    # Calculate square root of y_sq mod p
    y = pow(y_sq, (p + 1) // 4, p)

    # Check parity
    is_even = (y % 2 == 0)
    if (prefix == 0x02 and not is_even) or (prefix == 0x03 and is_even):
        y = p - y

    return x_int, y

def init_secp256k1_constants(mod, device_id):
    """Initialize secp256k1 curve constants in GPU constant memory"""
    # Set device context
    cuda.Context.synchronize()
    
    # Prime modulus p
    p_data = np.array([
        0xFFFFFC2F, 0xFFFFFFFE, 0xFFFFFFFF, 0xFFFFFFFF,
        0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF
    ], dtype=np.uint32)
    const_p_gpu = mod.get_global("const_p")[0]
    cuda.memcpy_htod(const_p_gpu, p_data)

    # Curve order n
    n_data = np.array([
        0xD0364141, 0xBFD25E8C, 0xAF48A03B, 0xBAAEDCE6,
        0xFFFFFFFE, 0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF
    ], dtype=np.uint32)
    const_n_gpu = mod.get_global("const_n")[0]
    cuda.memcpy_htod(const_n_gpu, n_data)

    # Base point G in Jacobian coordinates
    g_x = np.array([
        0x16F81798, 0x59F2815B, 0x2DCE28D9, 0x029BFCDB,
        0xCE870B07, 0x55A06295, 0xF9DCBBAC, 0x79BE667E
    ], dtype=np.uint32)
    g_y = np.array([
        0xFB10D4B8, 0x9C47D08F, 0xA6855419, 0xFD17B448,
        0x0E1108A8, 0x5DA4FBFC, 0x26A3C465, 0x483ADA77
    ], dtype=np.uint32)
    g_z = np.array([1, 0, 0, 0, 0, 0, 0, 0], dtype=np.uint32)
    g_infinity = np.array([False], dtype=np.bool_)

    # Create structured array for ECPointJac
    ecpoint_jac_dtype = np.dtype([
        ('X', np.uint32, 8),
        ('Y', np.uint32, 8),
        ('Z', np.uint32, 8),
        ('infinity', np.bool_)
    ])
    g_jac = np.zeros(1, dtype=ecpoint_jac_dtype)
    g_jac['X'], g_jac['Y'], g_jac['Z'], g_jac['infinity'] = g_x, g_y, g_z, g_infinity

    const_G_gpu = mod.get_global("const_G_jacobian")[0]
    cuda.memcpy_htod(const_G_gpu, g_jac)

def run_precomputation(mod):
    """Run GPU precomputation kernel"""
    precompute_kernel = mod.get_function("precompute_G_table_kernel")
    precompute_kernel(block=(1, 1, 1))
    cuda.Context.synchronize()

def optimize_bloom_filter_size(num_items, false_positive_rate=0.001):
    """Calculate optimal Bloom filter size"""
    m = - (num_items * np.log(false_positive_rate)) / (np.log(2) ** 2)
    return int(2 ** np.ceil(np.log2(m)))

class GPUWorker(threading.Thread):
    """Worker thread untuk menjalankan pencarian di GPU tertentu"""
    
    def __init__(self, gpu_id, mod, target_bin, args, shared_vars):
        threading.Thread.__init__(self)
        self.gpu_id = gpu_id
        self.mod = mod
        self.target_bin = target_bin
        self.args = args
        self.shared_vars = shared_vars
        self.found = False
        self.result = None
        
    def run(self):
        """Jalankan pekerjaan GPU"""
        try:
            # Set GPU context
            device = cuda.Device(self.gpu_id)
            context = device.make_context()
            
            print(f"[GPU{self.gpu_id}] Initializing GPU...")
            
            # Inisialisasi konstanta
            init_secp256k1_constants(self.mod, self.gpu_id)
            
            # Load target points
            num_targets = len(self.target_bin) // 33
            target_compressed = self.target_bin[:33]
            target_x, target_y = decompress_pubkey(target_compressed)

            # Setup target point untuk DP
            ecpoint_jac_dtype = np.dtype([
                ('X', np.uint32, 8), ('Y', np.uint32, 8),
                ('Z', np.uint32, 8), ('infinity', np.bool_)
            ])
            target_jac = np.zeros(1, dtype=ecpoint_jac_dtype)
            target_jac['X'] = int_to_bigint_np(target_x)
            target_jac['Y'] = int_to_bigint_np(target_y)
            target_jac['Z'] = int_to_bigint_np(1)
            target_jac['infinity'] = False

            const_target_jac_gpu = self.mod.get_global("const_target_jacobian")[0]
            cuda.memcpy_htod(const_target_jac_gpu, target_jac)

            print(f"[GPU{self.gpu_id}] Running precomputation...")
            run_precomputation(self.mod)

            # Inisialisasi kernel
            find_pubkey_kernel = self.mod.get_function("find_pubkey_kernel_sequential")
            generate_dp_kernel = self.mod.get_function("generate_dp_table_kernel")

            # Inisialisasi memory GPU
            d_target_pubkeys = cuda.mem_alloc(len(self.target_bin))
            cuda.memcpy_htod(d_target_pubkeys, np.frombuffer(self.target_bin, dtype=np.uint8))

            # Allocate 64 bytes for result (32 for private key, 32 for multiplier)
            d_result = cuda.mem_alloc(64)
            d_found_flag = cuda.mem_alloc(4)
            cuda.memset_d32(d_result, 0, 16)
            cuda.memset_d32(d_found_flag, 0, 1)

            # Inisialisasi DP table
            bloom_size = optimize_bloom_filter_size(self.args.num_dp, self.args.bloom_fp_rate)
            dp_table_size = self.args.num_dp

            dp_table_dtype = np.dtype([('fp', np.uint64), ('reduction_factor', np.uint64)])
            d_dp_table = cuda.mem_alloc(dp_table_size * dp_table_dtype.itemsize)
            d_bloom_filter = cuda.mem_alloc((bloom_size + 31) // 32 * 4)
            cuda.memset_d32(d_bloom_filter, 0, (bloom_size + 31) // 32)

            print(f"[GPU{self.gpu_id}] Generating {self.args.num_dp} DP points...")
            block_size_dp = 256
            grid_size_dp = (dp_table_size + block_size_dp - 1) // block_size_dp

            generate_dp_kernel(
                d_dp_table, d_bloom_filter,
                np.uint32(dp_table_size), np.uint64(bloom_size), np.uint64(self.args.reduction_step),
                block=(block_size_dp, 1, 1), grid=(grid_size_dp, 1)
            )
            cuda.Context.synchronize()

            # Sort DP table
            dp_table_host = np.zeros(dp_table_size, dtype=dp_table_dtype)
            cuda.memcpy_dtoh(dp_table_host, d_dp_table)
            dp_table_host.sort(order='fp')
            cuda.memcpy_htod(d_dp_table, dp_table_host)
            print(f"[GPU{self.gpu_id}] DP table ready.")

            # MAIN LOOP untuk GPU ini
            range_min_np = int_to_bigint_np(self.args.range_min)
            range_max_np = int_to_bigint_np(self.args.range_max)

            # Bagi workload berdasarkan GPU ID
            if self.gpu_id == 0:
                # GPU 0: start scalar genap
                current_start = self.args.start if self.args.start % 2 == 0 else self.args.start - 1
            else:
                # GPU 1: start scalar ganjil  
                current_start = self.args.start if self.args.start % 2 == 1 else self.args.start - 1

            start_scalars_tried = 0
            found_flag_host = np.zeros(1, dtype=np.int32)
            range_size = self.args.range_max - self.args.range_min + 1

            print(f"[GPU{self.gpu_id}] Starting search with start scalar: {hex(current_start)}")

            while (current_start >= 1 and 
                   start_scalars_tried < self.args.max_start_scalars and 
                   not self.shared_vars['found_global'] and
                   not self.found):

                # Reset found flag untuk start scalar baru
                cuda.memset_d32(d_found_flag, 0, 1)
                found_flag_host[0] = 0

                start_scalar_np = int_to_bigint_np(current_start)
                iteration_offset = 0
                total_iterations_current = 0

                # Loop untuk range saat ini
                while (iteration_offset < range_size and 
                       found_flag_host[0] == 0 and 
                       not self.shared_vars['found_global']):

                    iterations_left = range_size - iteration_offset
                    iterations_this_launch = min(self.args.keys_per_launch, iterations_left)

                    if iterations_this_launch <= 0:
                        break

                    block_size = 256
                    grid_size = (iterations_this_launch + block_size - 1) // block_size

                    # Jalankan kernel pencarian sequential
                    find_pubkey_kernel(
                        cuda.In(start_scalar_np), cuda.In(range_min_np), cuda.In(range_max_np),
                        np.uint64(iteration_offset),
                        d_target_pubkeys, np.int32(num_targets),
                        d_result, d_found_flag,
                        d_bloom_filter, np.uint64(bloom_size),
                        d_dp_table, np.uint32(dp_table_size),
                        block=(block_size, 1, 1), grid=(grid_size, 1)
                    )

                    cuda.Context.synchronize()

                    total_iterations_current += iterations_this_launch
                    iteration_offset += iterations_this_launch

                    cuda.memcpy_dtoh(found_flag_host, d_found_flag)

                    # Update progress
                    with self.shared_vars['lock']:
                        self.shared_vars['total_iterations'] += iterations_this_launch
                        total_iterations_all = self.shared_vars['total_iterations']
                        elapsed = time.time() - self.shared_vars['start_time']
                        speed = total_iterations_all / elapsed if elapsed > 0 else 0
                        
                        progress_str = (f"[GPU{self.gpu_id}] Start: {hex(current_start)} | "
                                      f"Progress: {iteration_offset:,}/{range_size:,} | "
                                      f"Speed: {speed:,.0f} it/s")
                        sys.stdout.write('\r' + progress_str.ljust(120))
                        sys.stdout.flush()

                # Cek hasil untuk start scalar ini
                if found_flag_host[0] == 1 and not self.shared_vars['found_global']:
                    with self.shared_vars['lock']:
                        if not self.shared_vars['found_global']:
                            self.shared_vars['found_global'] = True
                            self.found = True
                            
                            # Baca hasil dari GPU
                            result_buffer = np.zeros(16, dtype=np.uint32)
                            cuda.memcpy_dtoh(result_buffer, d_result)

                            # Extract private key (first 8 uint32 = 32 bytes)
                            found_privkey_np = result_buffer[:8]
                            privkey_int = bigint_np_to_int(found_privkey_np)

                            # Extract multiplier (next 8 uint32 = 32 bytes)
                            found_multiplier_np = result_buffer[8:16]
                            multiplier_int = bigint_np_to_int(found_multiplier_np)

                            self.result = {
                                'privkey': privkey_int,
                                'multiplier': multiplier_int,
                                'start_scalar': current_start,
                                'gpu_id': self.gpu_id
                            }
                            
                            sys.stdout.write('\n')
                            print(f"\n[GPU{self.gpu_id}] COLLISION FOUND!")
                            print(f"    Private Key: {hex(privkey_int)}")
                            print(f"    Multiplier: {hex(multiplier_int)}")
                            print(f"    Start scalar: {hex(current_start)}")
                            print(f"    Found by GPU: {self.gpu_id}")

                else:
                    # Tidak ditemukan di start scalar ini, lanjut ke berikutnya
                    current_start += 2  # Loncat 2 karena pembagian genap/ganjil
                    start_scalars_tried += 1

                    if start_scalars_tried % 10 == 0:  # Beri jeda setiap 10 start scalar
                        time.sleep(5)

            # Cleanup
            d_target_pubkeys.free()
            d_result.free()
            d_found_flag.free()
            d_dp_table.free()
            d_bloom_filter.free()
            
        except Exception as e:
            print(f"\n[GPU{self.gpu_id}] Error: {e}")
            import traceback
            traceback.print_exc()
        finally:
            context.pop()

def main():
    parser = argparse.ArgumentParser(description='CUDA Sequential Scalar Multiplication dengan 2 GPU Support')
    parser.add_argument('--start', type=lambda x: int(x, 0), required=True, help='Skalar awal perkalian')
    parser.add_argument('--range-min', type=lambda x: int(x, 0), required=True, help='Batas bawah range pengali')
    parser.add_argument('--range-max', type=lambda x: int(x, 0), required=True, help='Batas atas range pengali')
    parser.add_argument('--file', required=True, help='File target public keys')
    parser.add_argument('--keys-per-launch', type=int, default=2**20, help='Jumlah iterasi per batch GPU')
    parser.add_argument('--reduction-step', type=int, default=1, help='Langkah pengurangan untuk DP')
    parser.add_argument('--num-dp', type=int, default=248435455, help='Jumlah DP points')
    parser.add_argument('--bloom-fp-rate', type=float, default=0.001, help='False positive rate Bloom Filter')
    parser.add_argument('--max-start-scalars', type=int, default=162259276829213363391578010288128, help='Maksimal jumlah start scalar yang akan dicoba')

    args = parser.parse_args()

    # Validasi range
    if args.range_min >= args.range_max:
        print("[!] ERROR: range-min harus lebih kecil dari range-max")
        sys.exit(1)

    range_size = args.range_max - args.range_min + 1
    print(f"[*] Range size: {range_size:,} kemungkinan multiplier")

    print("[*] Memuat dan mengkompilasi kernel CUDA...")
    try:
        with open('main.cu', 'r') as f:
            full_cuda_code = f.read()
    except FileNotFoundError:
        print("[!] FATAL: File 'main.cu' tidak ditemukan.")
        sys.exit(1)

    # Compile kernel CUDA
    mod = SourceModule(full_cuda_code, no_extern_c=False, options=['-std=c++11', '-arch=sm_75'])

    # Load target points
    target_bin = load_target_pubkeys(args.file)
    num_targets = len(target_bin) // 33
    if num_targets == 0:
        print(f"[!] Tidak ada public key valid di {args.file}.")
        sys.exit(1)

    # Shared variables untuk koordinasi antara GPU
    shared_vars = {
        'found_global': False,
        'total_iterations': 0,
        'start_time': time.time(),
        'lock': threading.Lock()
    }

    # Buat dan jalankan worker threads untuk 2 GPU
    print("[*] Starting 2 GPU workers...")
    workers = []
    for gpu_id in range(2):
        worker = GPUWorker(gpu_id, mod, target_bin, args, shared_vars)
        workers.append(worker)
        worker.start()
        time.sleep(2)  # Jeda antara inisialisasi GPU

    # Tunggu semua worker selesai
    try:
        for worker in workers:
            worker.join()
    except KeyboardInterrupt:
        print(f"\n\n[!] Dihentikan oleh pengguna.")
        shared_vars['found_global'] = True
        for worker in workers:
            worker.join(timeout=5)

    # Tampilkan hasil akhir
    elapsed = time.time() - shared_vars['start_time']
    print(f"\n\n[*] Pencarian selesai.")
    print(f"    Total iterasi: {shared_vars['total_iterations']:,}")
    print(f"    Waktu pencarian: {elapsed:.2f} detik")
    print(f"    Kecepatan rata-rata: {shared_vars['total_iterations']/elapsed:,.0f} it/detik")

    # Cek jika ada yang menemukan
    found_worker = None
    for worker in workers:
        if worker.found and worker.result:
            found_worker = worker
            break

    if found_worker:
        result = found_worker.result
        print(f"\n[+] COLLISION DITEMUKAN OLEH GPU{result['gpu_id']}!")
        print(f"    Private Key: {hex(result['privkey'])}")
        print(f"    Multiplier: {hex(result['multiplier'])}")
        print(f"    Start scalar: {hex(result['start_scalar'])}")

        # Verifikasi
        n = 0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEBAAEDCE6AF48A03BBFD25E8CD0364141
        expected = (result['start_scalar'] * result['multiplier']) % n
        if expected == result['privkey']:
            print(f"    [VERIFIED] Perhitungan valid")
        else:
            print(f"    [WARNING] Perhitungan tidak valid")

        with open("2gpu.txt", "w") as f:
            f.write(f"Private Key: {hex(result['privkey'])}\n")
            f.write(f"Multiplier: {hex(result['multiplier'])}\n")
            f.write(f"Start scalar: {hex(result['start_scalar'])}\n")
            f.write(f"Found by GPU: {result['gpu_id']}\n")
            f.write(f"Total iterations: {shared_vars['total_iterations']}\n")
            f.write(f"Search time: {elapsed:.2f} seconds\n")
    else:
        print(f"\n[+] Tidak ada collision yang ditemukan.")

if __name__ == '__main__':
    main()
