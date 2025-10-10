import pycuda.autoinit
import pycuda.driver as cuda
from pycuda.compiler import SourceModule
import numpy as np
import time
import sys
import argparse
import struct
import random

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

def init_secp256k1_constants(mod):
    """Initialize secp256k1 curve constants in GPU constant memory"""
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
    print("[*] Precomputation table selesai.")

def optimize_bloom_filter_size(num_items, false_positive_rate=0.001):
    """Calculate optimal Bloom filter size"""
    m = - (num_items * np.log(false_positive_rate)) / (np.log(2) ** 2)
    return int(2 ** np.ceil(np.log2(m)))

def main():
    parser = argparse.ArgumentParser(description='CUDA Sequential Scalar Multiplication dengan DP dan Auto-Loop Start Scalar')
    parser.add_argument('--start', type=lambda x: int(x, 0), required=True, help='Skalar awal perkalian')
    parser.add_argument('--range-min', type=lambda x: int(x, 0), required=True, help='Batas bawah range pengali')
    parser.add_argument('--range-max', type=lambda x: int(x, 0), required=True, help='Batas atas range pengali')
    parser.add_argument('--file', required=True, help='File target public keys')
    parser.add_argument('--keys-per-launch', type=int, default=2**20, help='Jumlah iterasi per batch GPU')
    parser.add_argument('--reduction-step', type=int, default=2**20, help='Langkah pengurangan untuk DP')
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
        print("[!] FATAL: File 'step.cu' tidak ditemukan.")
        sys.exit(1)

    mod = SourceModule(full_cuda_code, no_extern_c=False, options=['-std=c++11', '-arch=sm_75'])
    init_secp256k1_constants(mod)

    # Load target points
    target_bin = load_target_pubkeys(args.file)
    num_targets = len(target_bin) // 33
    if num_targets == 0:
        print(f"[!] Tidak ada public key valid di {args.file}.")
        sys.exit(1)

    # Setup target point untuk DP
    target_compressed = target_bin[:33]
    target_x, target_y = decompress_pubkey(target_compressed)

    ecpoint_jac_dtype = np.dtype([
        ('X', np.uint32, 8), ('Y', np.uint32, 8),
        ('Z', np.uint32, 8), ('infinity', np.bool_)
    ])
    target_jac = np.zeros(1, dtype=ecpoint_jac_dtype)
    target_jac['X'] = int_to_bigint_np(target_x)
    target_jac['Y'] = int_to_bigint_np(target_y)
    target_jac['Z'] = int_to_bigint_np(1)
    target_jac['infinity'] = False

    const_target_jac_gpu = mod.get_global("const_target_jacobian")[0]
    cuda.memcpy_htod(const_target_jac_gpu, target_jac)

    print("[*] Menjalankan precomputation...")
    run_precomputation(mod)

    # Pilih kernel
    find_pubkey_kernel = mod.get_function("find_pubkey_kernel_sequential")
    generate_dp_kernel = mod.get_function("generate_dp_table_kernel")
    print("[*] Mode sequential dengan DP diaktifkan")

    # Inisialisasi memory GPU
    d_target_pubkeys = cuda.mem_alloc(len(target_bin))
    cuda.memcpy_htod(d_target_pubkeys, np.frombuffer(target_bin, dtype=np.uint8))

    # Allocate 64 bytes for result (32 for private key, 32 for multiplier)
    d_result = cuda.mem_alloc(64)
    d_found_flag = cuda.mem_alloc(4)
    cuda.memset_d32(d_result, 0, 16)
    cuda.memset_d32(d_found_flag, 0, 1)

    # Inisialisasi DP table
    bloom_size = optimize_bloom_filter_size(args.num_dp, args.bloom_fp_rate)
    dp_table_size = args.num_dp

    dp_table_dtype = np.dtype([('fp', np.uint64), ('reduction_factor', np.uint64)])
    d_dp_table = cuda.mem_alloc(dp_table_size * dp_table_dtype.itemsize)
    d_bloom_filter = cuda.mem_alloc((bloom_size + 31) // 32 * 4)
    cuda.memset_d32(d_bloom_filter, 0, (bloom_size + 31) // 32)

    print(f"[*] Membuat {args.num_dp} DP points di GPU...")
    block_size_dp = 256
    grid_size_dp = (dp_table_size + block_size_dp - 1) // block_size_dp

    generate_dp_kernel(
        d_dp_table, d_bloom_filter,
        np.uint32(dp_table_size), np.uint64(bloom_size), np.uint64(args.reduction_step),
        block=(block_size_dp, 1, 1), grid=(grid_size_dp, 1)
    )
    cuda.Context.synchronize()

    # Sort DP table
    dp_table_host = np.zeros(dp_table_size, dtype=dp_table_dtype)
    cuda.memcpy_dtoh(dp_table_host, d_dp_table)
    dp_table_host.sort(order='fp')
    cuda.memcpy_htod(d_dp_table, dp_table_host)
    print("[*] Tabel DP siap.")

    # MAIN LOOP UNTUK PENCARIAN SEQUENTIAL DENGAN AUTO-LOOP START SCALAR
    total_iterations_all = 0
    start_time = time.time()
    found_flag_host = np.zeros(1, dtype=np.int32)

    range_min_np = int_to_bigint_np(args.range_min)
    range_max_np = int_to_bigint_np(args.range_max)

    current_start = args.start
    start_scalars_tried = 0
    found = False

    print(f"[*] Memulai pencarian sequential dengan auto-loop start scalar:")
    print(f"    Start scalar awal: {hex(args.start)}")
    print(f"    Range pengali: {hex(args.range_min)} - {hex(args.range_max)}")
    print(f"    Range size: {range_size:,} kemungkinan per start scalar")
    print(f"    Maksimal start scalar: {args.max_start_scalars}")
    print(f"    DP table size: {dp_table_size:,}")
    print(f"    Bloom filter size: {bloom_size:,}")

    try:
        while current_start >= 1 and start_scalars_tried < args.max_start_scalars and not found:
            start_scalars_tried += 1

            # Reset found flag untuk start scalar baru
            cuda.memset_d32(d_found_flag, 0, 1)
            found_flag_host[0] = 0

            start_scalar_np = int_to_bigint_np(current_start)
            iteration_offset = 0
            total_iterations_current = 0

            print(f"\n[*] Mencoba dengan start scalar: {hex(current_start)} ({start_scalars_tried}/{args.max_start_scalars})")

            # Loop untuk range saat ini
            while iteration_offset < range_size and found_flag_host[0] == 0:
                iterations_left = range_size - iteration_offset
                iterations_this_launch = min(args.keys_per_launch, iterations_left)

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
                total_iterations_all += iterations_this_launch
                iteration_offset += iterations_this_launch

                cuda.memcpy_dtoh(found_flag_host, d_found_flag)

                elapsed = time.time() - start_time
                speed = total_iterations_all / elapsed if elapsed > 0 else 0
                progress_current = 100 * iteration_offset / range_size
                progress_total = 100 * start_scalars_tried / args.max_start_scalars

                progress_str = (f"[+] Start: {hex(current_start)} | "
                              f"Progress: {iteration_offset:,}/{range_size:,} ({progress_current:.1f}%) | "
                              f"Total: {start_scalars_tried}/{args.max_start_scalars} ({progress_total:.1f}%) | "
                              f"Speed: {speed:,.0f} it/s | "
                              f"Running: {elapsed:.0f}s")
                sys.stdout.write('\r' + progress_str.ljust(120))
                sys.stdout.flush()

            # Cek hasil untuk start scalar ini
            if found_flag_host[0] == 1:
                found = True
                sys.stdout.write('\n')

                # Baca hasil dari GPU
                result_buffer = np.zeros(16, dtype=np.uint32)
                cuda.memcpy_dtoh(result_buffer, d_result)

                # Extract private key (first 8 uint32 = 32 bytes)
                found_privkey_np = result_buffer[:8]
                privkey_int = bigint_np_to_int(found_privkey_np)

                # Extract multiplier (next 8 uint32 = 32 bytes)
                found_multiplier_np = result_buffer[8:16]
                multiplier_int = bigint_np_to_int(found_multiplier_np)

                print(f"\n[+] COLLISION DITEMUKAN!")
                print(f"    Private Key: {hex(privkey_int)}")
                print(f"    Multiplier: {hex(multiplier_int)}")
                print(f"    Start scalar: {hex(current_start)}")
                print(f"    Total iterasi: {total_iterations_all:,}")
                print(f"    Start scalar ke: {start_scalars_tried}")
                print(f"    Waktu pencarian: {time.time() - start_time:.2f} detik")

                # Verifikasi: current_start * multiplier = private_key
                n = 0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEBAAEDCE6AF48A03BBFD25E8CD0364141
                expected = (current_start * multiplier_int) % n
                if expected == privkey_int:
                    print(f"    [VERIFIED] Perhitungan valid: {hex(current_start)} * {hex(multiplier_int)} = {hex(privkey_int)}")
                else:
                    print(f"    [WARNING] Perhitungan tidak valid: {hex(current_start)} * {hex(multiplier_int)} != {hex(privkey_int)}")

                with open("found_sequential.txt", "w") as f:
                    f.write(f"Private Key: {hex(privkey_int)}\n")
                    f.write(f"Multiplier: {hex(multiplier_int)}\n")
                    f.write(f"Start scalar: {hex(current_start)}\n")
                    f.write(f"Total iterations: {total_iterations_all}\n")
                    f.write(f"Start scalar index: {start_scalars_tried}\n")
                    f.write(f"Search time: {time.time() - start_time:.2f} seconds\n")
                    f.write(f"Range: {hex(args.range_min)} - {hex(args.range_max)}\n")

            else:
                # Tidak ditemukan di start scalar ini, lanjut ke berikutnya
                print(f"\n[*] Tidak ditemukan di start scalar {hex(current_start)}, melanjutkan ke {hex(current_start-1)}")
                print("[!] Jeda untuk mengurangi beban ")
                time.sleep(15)
                current_start -= 1

        # Handle final results
        if not found:
            print(f"\n\n[+] Pencarian selesai. Tidak ditemukan.")
            print(f"    Total start scalar dicoba: {start_scalars_tried}")
            print(f"    Total iterasi: {total_iterations_all:,}")
            print(f"    Start scalar terakhir: {hex(current_start)}")
            print(f"    Waktu pencarian: {time.time() - start_time:.2f} detik")

    except KeyboardInterrupt:
        print(f"\n\n[!] Dihentikan oleh pengguna.")
        print(f"    Start scalar saat ini: {hex(current_start)}")
        print(f"    Start scalar dicoba: {start_scalars_tried}")
        print(f"    Total iterasi: {total_iterations_all:,}")
        print(f"    Waktu pencarian: {time.time() - start_time:.2f} detik")
    except Exception as e:
        print(f"\n\n[!] Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    main()
