# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

#
#
# Compares the per-pair MP2 energy loop (test.py) against the batched
# strided-GEMM loop (test-strided.py) in the same process. Both loops consume
# the same RHF orbitals and the same B_ia tensor, so any difference between
# them comes only from the loop structure / floating-point summation order,
# not from run-to-run SCF noise.
#
#

import cuest.bindings as ce
import cuest_scf
import ctypes
import numpy as np

from cuest_mo_int_helper import CuestMOIntegralHelper
from kernel_helper import MP2KernelHelper
from cuest_scf.cuda_utility import CudaUtility

kernel_helper = MP2KernelHelper()

import os
filedir = os.path.dirname(os.path.realpath(__file__))
basisdir = os.path.join(filedir, '..', '..', 'data', 'gbs')

cuest_handle = cuest_scf.CuestHandle()

threshold_pq = 1.0E-18

molecule = cuest_scf.Molecule.parse_from_xyz_file('ethanol.xyz')
charge = 0

primary_name = 'def2-svp'
auxiliary_name = 'def2-universal-jkfit'
ri_auxiliary_name = 'def2-svp-rifit'
minao_name = 'minao-1'

primary_filename = os.path.join(basisdir, '%s.gbs' % (primary_name))
auxiliary_filename = os.path.join(basisdir, '%s.gbs' % (auxiliary_name))
ri_auxiliary_filename = os.path.join(basisdir, '%s.gbs' % (ri_auxiliary_name))
minao_filename = os.path.join(basisdir, '%s.gbs' % (minao_name))

primary = cuest_scf.AOBasis.parse_from_gbs_file(primary_filename, molecule=molecule)
auxiliary = cuest_scf.AOBasis.parse_from_gbs_file(auxiliary_filename, molecule=molecule)
ri_auxiliary = cuest_scf.AOBasis.parse_from_gbs_file(ri_auxiliary_filename, molecule=molecule)
minao = cuest_scf.AOBasis.parse_from_gbs_file(minao_filename, molecule=molecule)

rhf = cuest_scf.RHF(
    cuest_handle=cuest_handle,
    molecule=molecule,
    charge=charge,
    xc_functional_name='HF',
    primary=primary,
    auxiliary=auxiliary,
    minao=minao,
    primary_name=primary_name,
    auxiliary_name=auxiliary_name,
    minao_name=minao_name,
    threshold_pq=threshold_pq,
    )

rhf.solve()

print(f'RHF Energy:             {rhf.compute_energy():24.16E}')


# => Shared B_ia = (A | i a) <= #


def compute_B_ia(
    *,
    rhf : cuest_scf.RHF,
    ri_auxiliary : cuest_scf.AOBasis,
    ri_auxiliary_name : str,
    integral_direct : bool = False,
    df_fitting_eigenvalue_cutoff : float = 1.0e-12,
    ):

    C = rhf.tensors['C']
    nocc = rhf.sizes['nocc']
    nvir = rhf.sizes['nvir']
    naux = ri_auxiliary.nao

    mo_integral_helper = CuestMOIntegralHelper(
        cuest_handle=rhf.cuest_handle,
        molecule=rhf.molecule,
        primary=rhf.primary,
        ri_auxiliary=ri_auxiliary,
        primary_name=rhf.primary_name,
        ri_auxiliary_name=ri_auxiliary_name,
        threshold_pq=rhf.threshold_pq,
        integral_direct=integral_direct,
        df_fitting_eigenvalue_cutoff=df_fitting_eigenvalue_cutoff,
        )

    B_ia = cuest_scf.GPUMatrix(
        nrows=naux * nocc,
        ncols=nvir,
        dtype=np.double,
        )

    mo_integral_helper.compute_3c_integral(
        C=C,
        nmo_c1=nocc,
        c1_offset=0,
        nmo_c2=nvir,
        c2_offset=nocc,
        out_mo_integrals=B_ia,
        )

    return B_ia


# => Reference: per-pair loop from test.py <= #


def compute_mp2_energy_reference(
    *,
    rhf : cuest_scf.RHF,
    B_ia : cuest_scf.GPUMatrix,
    naux : int,
    ):

    eps = rhf.tensors['eps']
    nocc = rhf.sizes['nocc']
    nvir = rhf.sizes['nvir']

    pair_mo_ints = cuest_scf.GPUMatrix(
        nrows=nvir,
        ncols=nvir,
        dtype=np.double,
        )
    pair_amplitudes = cuest_scf.GPUMatrix(
        nrows=nvir,
        ncols=nvir,
        dtype=np.double,
        )

    mp2_os_energy = 0.0
    mp2_ss_energy = 0.0
    for i in range(nocc):
        for j in range(i, nocc):

            # => (ia|jb) = sum_A B[A,i,a] * B[A,j,b]

            cuest_scf.GPUMatrixUtility.c_dgemm(
                transa=True,
                transb=False,
                m=nvir,
                n=nvir,
                k=naux,
                alpha=1.0,
                a=B_ia,
                offa=i*nvir,
                lda=nocc*nvir,
                b=B_ia,
                offb=j*nvir,
                ldb=nocc*nvir,
                beta=0.0,
                c=pair_mo_ints,
                offc=0,
                ldc=nvir,
                )

            # => (ia|jb) -> (ib|ja) <= #

            transposed_pair_mo_ints = cuest_scf.GPUMatrixUtility.dgeam(
                transpose1=True,
                alpha=1.0,
                mat1=pair_mo_ints,
                transpose2=False,
                beta=0.0,
                mat2=pair_mo_ints,
                )

            # => T[i, j, a, b] = (i a | j b) / (eps[i] + eps[j] - eps[a] - eps[b]) <= #

            kernel_helper.make_pair_amplitudes(
                i=i,
                j=j,
                nocc=nocc,
                nvir=nvir,
                eps_ptr=eps.pointer,
                pair_mo_ints_ptr=pair_mo_ints.pointer,
                pair_amplitudes_ptr=pair_amplitudes.pointer,
                )

            iajb_term = cuest_scf.GPUMatrixUtility.ddot(
                n=nvir * nvir,
                x=pair_mo_ints,
                offx=0,
                incx=1,
                y=pair_amplitudes,
                offy=0,
                incy=1,
            )

            ibja_term = cuest_scf.GPUMatrixUtility.ddot(
                n=nvir * nvir,
                x=transposed_pair_mo_ints,
                offx=0,
                incx=1,
                y=pair_amplitudes,
                offy=0,
                incy=1,
            )

            pair_prefactor = 1.0 if i == j else 2.0

            mp2_os_energy += pair_prefactor * iajb_term
            mp2_ss_energy += pair_prefactor * (iajb_term - ibja_term)

    return mp2_os_energy, mp2_ss_energy


# => Batched: strided-GEMM loop from test-strided.py <= #


def compute_mp2_energy_batched(
    *,
    rhf : cuest_scf.RHF,
    B_ia : cuest_scf.GPUMatrix,
    naux : int,
    ):

    eps = rhf.tensors['eps']
    nocc = rhf.sizes['nocc']
    nvir = rhf.sizes['nvir']

    double_size = ctypes.sizeof(ctypes.c_double)

    # => per-pair scratch space <= #
    free, total = kernel_helper.get_mem_info()
    block_total = int((free * 0.8) // 3) // (double_size * nvir * nvir)
    batch_count = block_total if block_total < nocc else nocc

    print(f"free GPU memory: {free / 1024**3:.2f} GiB")
    print(f"GEMM batch count: {int(batch_count)}")

    size_in_bytes = double_size * nvir * nvir * batch_count

    pair_mo_ints_ptr = CudaUtility.cuda_malloc(size_in_bytes)
    transposed_pair_mo_ints_ptr = CudaUtility.cuda_malloc(size_in_bytes)
    pair_amplitudes_ptr = CudaUtility.cuda_malloc(size_in_bytes)

    # => loop sizing <= #
    stride = (nocc + batch_count - 1) // batch_count

    mp2_os_energy = 0.0
    mp2_ss_energy = 0.0
    for i in range(nocc):
        for j in range(stride):

            # => (ia|jb) = sum_A B[A,i,a] * B[A,j,b]

            a_ptr = B_ia.pointer + i * nvir * double_size
            b_ptr = B_ia.pointer + j * batch_count * nvir * double_size
            iter_batch_count = min(batch_count, nocc - j * batch_count)

            cuest_scf.GPUMatrixUtility.c_dgemm_strided_batched(
                transa=True,
                transb=False,
                m=nvir,
                n=nvir,
                k=naux,
                alpha=1.0,
                a=a_ptr,
                lda=nocc*nvir,
                stride_a=0,
                b=b_ptr,
                ldb=nocc*nvir,
                stride_b=nvir,
                beta=0.0,
                c=pair_mo_ints_ptr,
                ldc=nvir,
                stride_c=nvir*nvir,
                batch_count=iter_batch_count,
                raw_ptr=1,
                )

            # => (ia|jb) -> (ib|ja) <= #

            kernel_helper.general_transpose(
                input_ptr=pair_mo_ints_ptr,
                output_ptr=transposed_pair_mo_ints_ptr,
                nrows=nvir,
                ncols=nvir,
                batch_size=iter_batch_count,
                )

            # => T[i, j, a, b] = (i a | j b) / (eps[i] + eps[j] - eps[a] - eps[b]) <= #

            kernel_helper.pair_amplitudes_batched(
                i=i,
                j_start=j*batch_count,
                nocc=nocc,
                nvir=nvir,
                batch_size=iter_batch_count,
                pair_mo_ints_batched_ptr=pair_mo_ints_ptr,
                eps_ptr=eps.pointer,
                pair_amplitudes_batched_ptr=pair_amplitudes_ptr,
                )

            iajb_term = cuest_scf.GPUMatrixUtility.ddot(
                n=nvir*nvir*iter_batch_count,
                x=pair_mo_ints_ptr,
                offx=0,
                incx=1,
                y=pair_amplitudes_ptr,
                offy=0,
                incy=1,
                raw_ptr=1,
            )

            ibja_term = cuest_scf.GPUMatrixUtility.ddot(
                n=nvir*nvir*iter_batch_count,
                x=transposed_pair_mo_ints_ptr,
                offx=0,
                incx=1,
                y=pair_amplitudes_ptr,
                offy=0,
                incy=1,
                raw_ptr=1,
            )

            mp2_os_energy += iajb_term
            mp2_ss_energy += iajb_term - ibja_term

    CudaUtility.cuda_free(pointer=pair_mo_ints_ptr)
    CudaUtility.cuda_free(pointer=transposed_pair_mo_ints_ptr)
    CudaUtility.cuda_free(pointer=pair_amplitudes_ptr)

    return mp2_os_energy, mp2_ss_energy


# => One-shot: sliced GEMM loop from test-one-shot.py <= #


def compute_mp2_energy_one_shot(
    *,
    rhf : cuest_scf.RHF,
    B_ia : cuest_scf.GPUMatrix,
    naux : int,
    ):

    eps = rhf.tensors['eps']
    nocc = rhf.sizes['nocc']
    nvir = rhf.sizes['nvir']

    nvir2 = nvir * nvir

    free, total = kernel_helper.get_mem_info()

    slices_possible = int((free * 0.8) // 3) // (8 * nvir2 * nocc)
    slice_count = slices_possible if slices_possible < nocc else nocc
    if slice_count == 0:
        raise ValueError(f"Insufficient device memory for {3 * 8 * nvir2 * nocc / 1024**3:.2f} GiB scratch buffer")

    print(f"free GPU memory: {free / 1024**3:.2f} GiB")
    print(f"Performing (A|ia)(A|jb) in {slice_count} slice(s)")

    size_in_bytes = 8 * nvir2 * nocc * slice_count
    iajb_ptr = CudaUtility.cuda_malloc(size_in_bytes)
    ibja_ptr = CudaUtility.cuda_malloc(size_in_bytes)
    pair_amplitudes_ptr = CudaUtility.cuda_malloc(size_in_bytes)

    stride = (nocc + slice_count - 1) // slice_count

    mp2_os_energy = 0.0
    mp2_ss_energy = 0.0
    for i in range(stride):

        iter_slices = min(slice_count, nocc - i * slice_count)

        cuest_scf.GPUMatrixUtility.c_dgemm(
            transa=True,
            transb=False,
            m=iter_slices*nvir,
            n=nocc*nvir,
            k=naux,
            alpha=1.0,
            a=B_ia.pointer,
            offa=i*slice_count*nvir,
            lda=nocc*nvir,
            b=B_ia.pointer,
            offb=0,
            ldb=nocc*nvir,
            beta=0.0,
            c=iajb_ptr,
            offc=0,
            ldc=nocc*nvir,
            raw_ptr=1,
            )

        kernel_helper.pair_amplitudes_sliced(
            i_start=i*slice_count,
            nocc=nocc,
            nvir=nvir,
            batch_size=iter_slices*nocc,
            iajb_slice_ptr=iajb_ptr,
            eps_ptr=eps.pointer,
            pair_amplitudes_slice_ptr=pair_amplitudes_ptr,
            )

        kernel_helper.sliced_transpose(
            input_ptr=iajb_ptr,
            output_ptr=ibja_ptr,
            nocc=nocc,
            nvir=nvir,
            batch_size=iter_slices*nocc,
            )

        iajb_term = cuest_scf.GPUMatrixUtility.ddot(
            n=iter_slices*nvir*nocc*nvir,
            x=iajb_ptr,
            offx=0,
            incx=1,
            y=pair_amplitudes_ptr,
            offy=0,
            incy=1,
            raw_ptr=1,
        )

        ibja_term = cuest_scf.GPUMatrixUtility.ddot(
            n=iter_slices*nvir*nocc*nvir,
            x=ibja_ptr,
            offx=0,
            incx=1,
            y=pair_amplitudes_ptr,
            offy=0,
            incy=1,
            raw_ptr=1,
        )

        mp2_os_energy += iajb_term
        mp2_ss_energy += iajb_term - ibja_term

    CudaUtility.cuda_free(pointer=iajb_ptr)
    CudaUtility.cuda_free(pointer=ibja_ptr)
    CudaUtility.cuda_free(pointer=pair_amplitudes_ptr)

    return mp2_os_energy, mp2_ss_energy


# => Comparison <= #

B_ia = compute_B_ia(
    rhf=rhf,
    ri_auxiliary=ri_auxiliary,
    ri_auxiliary_name=ri_auxiliary_name,
    integral_direct=True,
    )

naux = ri_auxiliary.nao

ref_os_energy, ref_ss_energy = compute_mp2_energy_reference(rhf=rhf, B_ia=B_ia, naux=naux)
bat_os_energy, bat_ss_energy = compute_mp2_energy_batched(rhf=rhf, B_ia=B_ia, naux=naux)
one_os_energy, one_ss_energy = compute_mp2_energy_one_shot(rhf=rhf, B_ia=B_ia, naux=naux)

d_os = bat_os_energy - ref_os_energy
d_ss = bat_ss_energy - ref_ss_energy
d_corr = d_os + d_ss

d_one_os = one_os_energy - ref_os_energy
d_one_ss = one_ss_energy - ref_ss_energy
d_one_corr = d_one_os + d_one_ss

print()
print(f'                        {"reference":>24s} {"batched":>24s} {"diff":>11s}')
print(f'MP2 OS Energy:          {ref_os_energy:24.16E} {bat_os_energy:24.16E} {d_os:11.3E}')
print(f'MP2 SS Energy:          {ref_ss_energy:24.16E} {bat_ss_energy:24.16E} {d_ss:11.3E}')
print(f'MP2 Correlation Energy: {ref_os_energy + ref_ss_energy:24.16E} {bat_os_energy + bat_ss_energy:24.16E} {d_corr:11.3E}')

print()
print(f'                        {"reference":>24s} {"one-shot":>24s} {"diff":>11s}')
print(f'MP2 OS Energy:          {ref_os_energy:24.16E} {one_os_energy:24.16E} {d_one_os:11.3E}')
print(f'MP2 SS Energy:          {ref_ss_energy:24.16E} {one_ss_energy:24.16E} {d_one_ss:11.3E}')
print(f'MP2 Correlation Energy: {ref_os_energy + ref_ss_energy:24.16E} {one_os_energy + one_ss_energy:24.16E} {d_one_corr:11.3E}')

# Same orbitals and B_ia: only summation order differs
assert abs(d_os) < 1.0e-10
assert abs(d_ss) < 1.0e-10
assert abs(d_one_os) < 1.0e-10
assert abs(d_one_ss) < 1.0e-10

# Psi4 reference values
assert abs(bat_os_energy - -0.3739405675166669) < 1.0e-6
assert abs(bat_ss_energy - -0.1145863947025172) < 1.0e-6
