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
# This is an example demonstrating how to use the nonsymmetric exchange
# compute routines and multipole integral compute routines to solve the
# CPHF equations to obtain an RHF polarizability. This example is limited
# to gas-phase RHF polarizabilities.
#
# This implementation uses the cuest_scf.RHF object to access compute
# routines, pre-built data structures and the results of the RHF calculation.
#
#

import cuest.bindings as ce
import cuest_scf
import ctypes
import numpy as np
import nvmath.bindings.cublas as cublas

from cuest_mo_int_helper import CuestMOIntegralHelper
from kernel_helper import MP2KernelHelper

kernel_helper = MP2KernelHelper()

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--geometry', default='ethanol.xyz')
parser.add_argument('--part-timers', action=argparse.BooleanOptionalAction, default=True)
args = parser.parse_args()

enable_part_timers = args.part_timers

timer = cuest_scf.timer.Timer()

def start_part_timer(key, stream_handle=0):
    if enable_part_timers:
        timer.start(key=key, stream_handle=stream_handle)

def stop_part_timer(key, stream_handle=0):
    if enable_part_timers:
        timer.stop(key=key, stream_handle=stream_handle)

import os
filedir = os.path.dirname(os.path.realpath(__file__))
basisdir = os.path.join(filedir, '..', '..', 'data', 'gbs')

# Usually you want one CuestHandle for many computations - e.g., do not spin up
# separate handles for every RHF instance
cuest_handle = cuest_scf.CuestHandle()

threshold_pq = 1.0E-18

molecule = cuest_scf.Molecule.parse_from_xyz_file(args.geometry)
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

print(f'\n=> MP2 Energy Calculation <=\n')

# => MP2 Energy Calculation <= #


def compute_mp2_energy(
    *,
    rhf : cuest_scf.RHF,
    ri_auxiliary : cuest_scf.AOBasis,
    ri_auxiliary_name : str,
    integral_direct : bool = False,
    df_fitting_eigenvalue_cutoff : float = 1.0e-12,
    ):


    C = rhf.tensors['C']
    eps = rhf.tensors['eps']
    nocc = rhf.sizes['nocc']
    nvir = rhf.sizes['nvir']
    naux = ri_auxiliary.nao

    start_part_timer('mo_integrals', stream_handle=rhf.cuest_handle.stream_handle)

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

    # => B_ia = (A |i a); this is more efficient than forming (B | a i) <= #

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

    stop_part_timer('mo_integrals', stream_handle=rhf.cuest_handle.stream_handle)

    free, total = kernel_helper.get_mem_info()
    print(f'MEM after 3c integral transform: free {free / 1024**3:.2f} / {total / 1024**3:.2f} GiB', flush=True)

    # => per-pair scratch space <= #

    start_part_timer('scratch_alloc')

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
    transposed_pair_mo_ints = cuest_scf.GPUMatrix(
        nrows=nvir,
        ncols=nvir,
        dtype=np.double,
        )
    pair_terms = cuest_scf.GPUMatrix(
        nrows=nocc * (nocc + 1) // 2,
        ncols=2,
        dtype=np.double,
        )

    one = np.array(1.0, dtype=np.double)
    zero = np.array(0.0, dtype=np.double)

    device_handle = cublas.create()
    cublas.set_pointer_mode(device_handle, cublas.PointerMode.DEVICE)

    stop_part_timer('scratch_alloc')

    free, total = kernel_helper.get_mem_info()
    print(f'MEM after scratch allocation:    free {free / 1024**3:.2f} / {total / 1024**3:.2f} GiB', flush=True)

    pair = 0
    for i in range(nocc):
        for j in range(i, nocc):

            # => (ia|jb) = sum_A B[A,i,a] * B[A,j,b]

            start_part_timer('gemm')

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

            stop_part_timer('gemm')

            # => (ia|jb) -> (ib|ja) <= #

            start_part_timer('transpose')

            cublas.dgeam(
                handle=cuest_scf.gpu_matrix_utility.cublas_handle,
                transa=cublas.Operation.T,
                transb=cublas.Operation.N,
                m=nvir,
                n=nvir,
                alpha=one.ctypes.data,
                a=pair_mo_ints.pointer,
                lda=nvir,
                beta=zero.ctypes.data,
                b=pair_mo_ints.pointer,
                ldb=nvir,
                c=transposed_pair_mo_ints.pointer,
                ldc=nvir,
                )

            stop_part_timer('transpose')

            # => T[i, j, a, b] = (i a | j b) / (eps[i] + eps[j] - eps[a] - eps[b]) <= #

            start_part_timer('pair_amplitudes')

            kernel_helper.make_pair_amplitudes(
                i=i,
                j=j,
                nocc=nocc,
                nvir=nvir,
                eps_ptr=eps.pointer,
                pair_mo_ints_ptr=pair_mo_ints.pointer,
                pair_amplitudes_ptr=pair_amplitudes.pointer,
                )

            stop_part_timer('pair_amplitudes')

            start_part_timer('ddot')

            cublas.ddot(
                handle=device_handle,
                n=nvir * nvir,
                x=pair_mo_ints.pointer,
                incx=1,
                y=pair_amplitudes.pointer,
                incy=1,
                result=pair_terms.pointer + (2 * pair) * 8,
                )

            cublas.ddot(
                handle=device_handle,
                n=nvir * nvir,
                x=transposed_pair_mo_ints.pointer,
                incx=1,
                y=pair_amplitudes.pointer,
                incy=1,
                result=pair_terms.pointer + (2 * pair + 1) * 8,
                )

            stop_part_timer('ddot')

            pair += 1

    terms = pair_terms.to_numpy()
    cublas.destroy(device_handle)

    iu, ju = np.triu_indices(nocc)
    pair_prefactor = np.where(iu == ju, 1.0, 2.0)

    mp2_os_energy = float(np.sum(pair_prefactor * terms[:, 0]))
    mp2_ss_energy = float(np.sum(pair_prefactor * (terms[:, 0] - terms[:, 1])))

    free, total = kernel_helper.get_mem_info()
    print(f'MEM after MP2 energy loop:       free {free / 1024**3:.2f} / {total / 1024**3:.2f} GiB', flush=True)

    return mp2_os_energy, mp2_ss_energy


timer.start(key='mp2_total')

mp2_os_energy, mp2_ss_energy = compute_mp2_energy(
    rhf=rhf,
    ri_auxiliary=ri_auxiliary,
    ri_auxiliary_name=ri_auxiliary_name,
    integral_direct=True,
    )

timer.stop(key='mp2_total')

timer.print_out(header='MP2 Timing breakdown:')

print(f'MP2 OS Energy:          {mp2_os_energy:24.16E}')
print()
print(f'MP2 SS Energy:          {mp2_ss_energy:24.16E}')
print(f'MP2 Correlation Energy: {mp2_os_energy + mp2_ss_energy:24.16E}')
print()
print(f'MP2 Energy:             {rhf.compute_energy() + mp2_os_energy + mp2_ss_energy:24.16E}')

# Psi4 reference values
if os.path.basename(args.geometry) == 'ethanol.xyz':
    assert abs(mp2_os_energy - -0.3739405675166669) < 1.0e-6
    assert abs(mp2_ss_energy - -0.1145863947025172) < 1.0e-6
