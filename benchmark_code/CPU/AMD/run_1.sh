#!/bin/bash

script_dir="$(dirname "$(readlink -e "${BASH_SOURCE[0]}")")"
source "$script_dir"/config.sh
echo

if [[ "$(whoami)" == 'xexdgala' ]]; then
    path_other='/zhome/academic/HLRS/xex/xexdgala/Data/graphs/other'
    path_athena='/zhome/academic/HLRS/xex/xexdgala/Data/graphs/matrices_athena'
    path_selected='/zhome/academic/HLRS/xex/xexdgala/Data/graphs/selected_matrices'
    path_selected_sorted='/zhome/academic/HLRS/xex/xexdgala/Data/graphs/selected_matrices_sorted'
else
    path_other='/home/jim/Data/graphs/other'
    path_athena='/home/jim/Data/graphs/matrices_athena'
    path_selected='/home/jim/Data/graphs/selected_matrices'
    path_selected_sorted='/home/jim/Data/graphs/selected_matrices_sorted'
fi


# GOMP_CPU_AFFINITY pins the threads to specific cpus, even when assigning more cores than threads.
# e.g. with 'GOMP_CPU_AFFINITY=0,1,2,3' and 2 threads, the threads are pinned: t0->core0 and t1->core1.
# export GOMP_CPU_AFFINITY="$cpu_affinity"
# export XLSMPOPTS="PROCS=$cpu_affinity"

lscpu | grep -q -i amd
if (($? == 0)); then
    export MKL_DEBUG_CPU_TYPE=5
fi
# export MKL_ENABLE_INSTRUCTIONS=AVX512
export MKL_VERBOSE=1

# export LD_LIBRARY_PATH="${AOCL_PATH}/lib:${MKL_PATH}/lib/intel64:${LD_LIBRARY_PATH}"
# export LD_LIBRARY_PATH="${LD_LIBRARY_PATH}:${BOOST_LIB_PATH}:${LLVM_LIB_PATH}:${SPARSEX_LIB_PATH}"
# export LD_LIBRARY_PATH="${LD_LIBRARY_PATH}:/various/dgal/gcc/gcc-12.2.0/gcc_bin/lib64"
# export LD_LIBRARY_PATH="${LD_LIBRARY_PATH}:/various/dgal/epyc1/cuda/cuda_11_4_4/lib64"
# export LD_LIBRARY_PATH="${LD_LIBRARY_PATH}:${TACO_PATH}/build/lib"

# Encourages idle threads to spin rather than sleep.
# export OMP_WAIT_POLICY='active'
# Don't let the runtime deliver fewer threads than those we asked for.
export OMP_DYNAMIC='false'


matrices_openFoam=("$path_openFoam"/*.mtx)

# matrices_openFoam_own_neigh=( "$path_openFoam"/TestMatrices/*/*/* )

cd "$path_openFoam"
text="$(printf "%s\n" TestMatrices/*/*/*)"
cd - &>/dev/null
sorted_text="$(sort -t '/' -k2,2 -k3,3n -k4.10,4n <<<"${text}")"

IFS_buf="$IFS"
IFS=$'\n'
matrices_openFoam_own_neigh=( $(printf "${path_openFoam}/%s\n" ${sorted_text}) )
IFS="$IFS_buf"


matrices_validation=(
    scircuit.mtx
    mac_econ_fwd500.mtx
    raefsky3.mtx
    rgg_n_2_17_s0.mtx
    bbmat.mtx
    appu.mtx
    conf5_4-8x8-15.mtx
    mc2depi.mtx
    rma10.mtx
    cop20k_A.mtx
    thermomech_dK.mtx
    webbase-1M.mtx
    cant.mtx
    ASIC_680k.mtx
    roadNet-TX.mtx
    pdb1HYS.mtx
    TSOPF_RS_b300_c3.mtx
    Chebyshev4.mtx
    consph.mtx
    com-Youtube.mtx
    rajat30.mtx
    radiation.mtx
    Stanford_Berkeley.mtx
    shipsec1.mtx
    PR02R.mtx
    CurlCurl_2.mtx
    gupta3.mtx
    mip1.mtx
    rail4284.mtx
    pwtk.mtx
    crankseg_2.mtx
    Si41Ge41H72.mtx
    TSOPF_RS_b2383.mtx
    in-2004.mtx
    Ga41As41H72.mtx
    eu-2005.mtx
    wikipedia-20051105.mtx
    kron_g500-logn18.mtx
    rajat31.mtx
    human_gene1.mtx
    delaunay_n22.mtx
    GL7d20.mtx
    sx-stackoverflow.mtx
    dgreen.mtx
    mawi_201512012345.mtx
    ldoor.mtx
    dielFilterV2real.mtx
    circuit5M.mtx
    soc-LiveJournal1.mtx
    bone010.mtx
    audikw_1.mtx
    cage15.mtx
    kmer_V2a.mtx

)

validation_dirs=(
    "${path_validation}"
    "${path_validation}/new_matrices" 
)

matrices_validation=( $(
    for ((i=0;i<${#matrices_validation[@]};i++)); do
        m="${matrices_validation[i]}"
        for d in "${validation_dirs[@]}"; do
            if [[ -f "${d}/${m}" ]]; then
                echo "${d}/${m}"
                break
            fi
        done
    done
) )


matrices_paper_csr_rv=(

    ts-palko
    # neos
    # stat96v3
    # stormG2_1000
    # xenon2
    # s3dkq4m2
    # apache2
    # Si34H36
    # ecology2
    # LargeRegFile
    # largebasis
    # Goodwin_127
    # Hamrle3
    # boneS01
    # sls
    # cont1_l
    # CO
    # G3_circuit
    # degme
    # atmosmodl
    # SiO2
    # tp-6
    # af_shell3
    # circuit5M_dc
    # rajat31
    # CurlCurl_4
    # cage14
    # nlpkkt80
    # ss
    # boneS10

)
for ((i=0;i<${#matrices_paper_csr_rv[@]};i++)); do
    m="${matrices_paper_csr_rv[i]}"
    matrices_paper_csr_rv[i]="$path_tamu"/matrices/"$m"/"$m".mtx
done


matrices_compression_small=(

    # scircuit
    # mac_econ_fwd500
    # raefsky3
    # bbmat
    # appu
    # rma10
    # cop20k_A
    # thermomech_dK
    # webbase-1M
    # cant
    # ASIC_680k
    # pdb1HYS
    # TSOPF_RS_b300_c3
    # Chebyshev4
    # consph
    # rajat30
    # radiation
    # shipsec1
    # PR02R
    # CurlCurl_2
    # pwtk
    # crankseg_2
    # Si41Ge41H72
    # TSOPF_RS_b2383
    # Ga41As41H72
    # rajat31
    # human_gene1
    # dgreen

    cop20k_A
    ASIC_680k
    radiation
    PR02R
    crankseg_2
    human_gene1

)
for ((i=0;i<${#matrices_compression_small[@]};i++)); do
    m="${matrices_compression_small[i]}"
    matrices_compression_small[i]="$path_tamu"/matrices/"$m"/"$m".mtx
done


matrices_compression=(

    spal_004
    # ldoor
    # dielFilterV2real
    # nv2
    # af_shell10
    # boneS10
    # circuit5M
    # Hook_1498
    # Geo_1438
    # Serena
    # vas_stokes_2M
    # bone010
    # audikw_1
    # Long_Coup_dt0
    # Long_Coup_dt6
    # dielFilterV3real
    # nlpkkt120
    # cage15
    # ML_Geer
    # Flan_1565
    # Cube_Coup_dt0
    # Cube_Coup_dt6
    # Bump_2911
    # vas_stokes_4M
    # nlpkkt160
    # HV15R
    # Queen_4147
    # stokes
    # nlpkkt200

    # Transport
    # Freescale2
    # FullChip
    # cage14
    # ML_Laplace
    # vas_stokes_1M
    # ss
    # RM07R
    # dgreen
    # Hardesty3
    # nlpkkt240

    # MOLIERE_2016

    # Transport
    # Freescale2
    # ldoor
    # dielFilterV2real
    # af_shell10
    # FullChip
    # cage14
    # ML_Laplace
    # boneS10
    # Hook_1498
    # Geo_1438
    # Serena
    # vas_stokes_1M
    # ss
    # bone010
    # RM07R
    # dgreen
    # audikw_1
    # Hardesty3
    # Long_Coup_dt0
    # Long_Coup_dt6
    # dielFilterV3real
    # spal_004
    # nlpkkt120
    # nv2
    # Flan_1565
    # circuit5M
    # Cube_Coup_dt0
    # Cube_Coup_dt6
    # vas_stokes_2M
    # Bump_2911
    # cage15
    # ML_Geer
    # nlpkkt160
    # vas_stokes_4M
    # Queen_4147
    # nlpkkt200
    # HV15R
    # stokes
    # nlpkkt240
    # MOLIERE_2016

)
for ((i=0;i<${#matrices_compression[@]};i++)); do
    m="${matrices_compression[i]}"
    matrices_compression[i]="$path_tamu"/matrices/"$m"/"$m".mtx
done


matrices_M3E=(

    # StocF_1465

    Heel_1138
    Hook_1498
    Utemp20m
    guenda11m
    agg14m

)
for ((i=0;i<${#matrices_M3E[@]};i++)); do
    m="${matrices_M3E[i]}"
    matrices_M3E[i]="$path_M3E"/"$m"/"$m".mtx
done


matrices_validation_loop=()
for ((i=0;i<${#matrices_validation[@]};i++)); do
    path="${matrices_validation[i]}"
    dir="$(dirname "${path}")"
    filename="$(basename "${path}")"
    base="${filename%.*}"
    ext="${filename#${filename%.*}}"
    n=128
    for ((j=0;j<n;j++)); do
        matrices_validation_loop+=( "${matrices_validation_artificial_twins["$base"]}" )
    done
    matrices_validation_loop+=( "${matrices_validation[i]}" )
done


bench()
{
    declare args=("$@")
    declare prog="${args[0]}"
    declare prog_args=("${args[@]:1}")
    declare t

    for t in $cores
    do
        export OMP_NUM_THREADS="$t"

        while :; do
            # if [[ "$prog" == *"spmv_sparsex.exe"* ]]; then
                # # since affinity is set with the runtime variable, just reset it to "0" so no warnings are displayed, and reset it after execution of benchmark (for other benchmarks to run)
                # # mt_conf="${GOMP_CPU_AFFINITY}"
                # export GOMP_CPU_AFFINITY_backup="${GOMP_CPU_AFFINITY}"
                # export GOMP_CPU_AFFINITY="0"
                # mt_conf=$(seq -s ',' 0 1 "$(($t-1))")
                # if ((!USE_ARTIFICIAL_MATRICES)); then
                    # "$prog" "${prog_args[@]}" -t -o spx.rt.nr_threads=$t -o spx.rt.cpu_affinity=${mt_conf} -o spx.preproc.xform=all #-v  2>'tmp_1.err'
                # else
                    # prog_args2="${prog_args[@]}"
                    # "$prog" -p "${prog_args2[@]}" -t -o spx.rt.nr_threads=$t -o spx.rt.cpu_affinity=${mt_conf} -o spx.preproc.xform=all #-v  2>'tmp_1.err'
                # fi
                # export GOMP_CPU_AFFINITY="${GOMP_CPU_AFFINITY_backup}"
            # elif [[ "$prog" == *"spmv_sell-C-s.exe"* ]]; then
            if [[ "$prog" == *"spmv_sell-C-s.exe"* ]]; then
                if ((!USE_ARTIFICIAL_MATRICES)); then
                    "$prog" -c $OMP_NUM_THREADS -m "${prog_args[@]}" -f SELL-32-1  2>'tmp_1.err'
                    ret="$?"
                else
                    prog_args2="${prog_args[@]}"
                    "$prog" -c $OMP_NUM_THREADS --artif_args="${prog_args2[@]}" -f SELL-32-1  2>'tmp_1.err'
                    ret="$?"
                fi
            else
                # "$prog" 4690000 4 1.6 normal random 1 14  2>'tmp_1.err'

                # numactl -i all "$prog" "${prog_args[@]}"  2>'tmp_1.err'
                "$prog" "${prog_args[@]}"  2>'tmp_1.err'
                ret="$?"
            fi
            cat 'tmp_1.err'
            if ((!ret || !force_retry_on_error)); then      # If not retrying then print the error text to be able to notice it.
                cat 'tmp_1.err' >&2
                break
            fi
            echo "ERROR: Program exited with error [${ret}], retrying."
        done
    done

    rm 'tmp_1.err'
}


matrices=(
    # "${matrices_openFoam[@]}"
    "${matrices_validation[@]}"
    # "${matrices_paper_csr_rv[@]}"
    # "${matrices_compression_small[@]}"
    # "${matrices_compression[@]}"
    # "${matrices_M3E[@]}"

    # "$path_tamu"/matrices/ASIC_680k/ASIC_680k.mtx
    # '682862 682862 5.6699201303 659.8073579974 normal random 0.3746622132 69710.5639935502 0.6690077130 0.8254737741 14 ASIC_680k'

    # nr_rows nr_cols avg_nnz_per_row std_nnz_per_row distribution placement bw           skew          avg_num_neighbours cross_row_similarity seed
    # ' 16783   16783   555.5280343204  1233.5202594143 normal       random    1            0             0                  0                    14 gupta3-1'
    # ' 16783   16783   555.5280343204  1233.5202594143 normal       random    1            25.4109083495 0                  0                    14 gupta3-2'
    # ' 16783   16783   555.5280343204  1233.5202594143 normal       random    0.5718415058 25.4109083495 0                  0                    14 gupta3-3'
    # ' 16783   16783   555.5280343204  1233.5202594143 normal       random    0.5718415058 25.4109083495 1.9016586927       0                    14 gupta3-4'
    # ' 16783   16783   555.5280343204  1233.5202594143 normal       random    0.5718415058 25.4109083495 1.9016586927       0.9767488489         14 gupta3-5'
    # "${path_validation}"/gupta3.mtx

    # "${matrices_validation_artificial_twins[@]}"
    # "${matrices_validation_loop[@]}"

    # "$path_other"/simple.mtx
    # "$path_other"/simple_symmetric.mtx

    # /home/jim/Documents/Synced_Documents/other/ASIC_680k.mtx

    # "$path_openFoam"/100K.mtx
    # "$path_openFoam"/600K.mtx
    # "$path_openFoam"/TestMatrices/HEXmats/5krows/processor0
    # "${matrices_openFoam_own_neigh[@]}"

    # '/home/jim/Data/graphs/tamu/ML/thermomech_dK.mtx'

    # "$path_validation"/scircuit.mtx
    # "$path_validation"/mac_econ_fwd500.mtx
    # "$path_validation"/raefsky3.mtx
    # "$path_validation"/bbmat.mtx
    # "$path_validation"/conf5_4-8x8-15.mtx
    # "$path_validation"/mc2depi.mtx
    # "$path_validation"/rma10.mtx
    # "$path_validation"/cop20k_A.mtx
    # "$path_validation"/webbase-1M.mtx
    # "$path_validation"/cant.mtx
    # "$path_validation"/pdb1HYS.mtx
    # "$path_validation"/TSOPF_RS_b300_c3.mtx
    # "$path_validation"/Chebyshev4.mtx
    # "$path_validation"/consph.mtx
    # "$path_validation"/shipsec1.mtx
    # "$path_validation"/PR02R.mtx
    # "$path_validation"/mip1.mtx
    # "$path_validation"/rail4284.mtx
    # "$path_validation"/pwtk.mtx
    # "$path_validation"/crankseg_2.mtx
    # "$path_validation"/Si41Ge41H72.mtx
    # "$path_validation"/TSOPF_RS_b2383.mtx
    # "$path_validation"/in-2004.mtx
    # "$path_validation"/Ga41As41H72.mtx
    # "$path_validation"/eu-2005.mtx
    # "$path_validation"/wikipedia-20051105.mtx
    # "$path_validation"/ldoor.mtx
    # "$path_validation"/circuit5M.mtx
    # "$path_validation"/bone010.mtx
    # "$path_validation"/cage15.mtx

    # "$path_selected"/soc-LiveJournal1.mtx
    # "$path_selected"/soc-LiveJournal1_sorted_1.mtx
    # "$path_selected"/soc-LiveJournal1_sorted_2.mtx
    # "$path_selected"/soc-LiveJournal1_sorted_3.mtx
    # "$path_selected"/soc-LiveJournal1_sorted_4.mtx

    # "$path_selected"/dielFilterV3real.mtx
    # "$path_selected"/dielFilterV3real_sorted_1.mtx
    # "$path_selected"/dielFilterV3real_sorted_2.mtx
    # "$path_selected"/dielFilterV3real_sorted_3.mtx
    # "$path_selected"/dielFilterV3real_sorted_4.mtx

    # "$path_selected"/circuit5M.mtx
    # "$path_selected"/circuit5M_sorted_1.mtx
    # "$path_selected"/circuit5M_sorted_2.mtx
    # "$path_selected"/circuit5M_sorted_3.mtx
    # "$path_selected"/circuit5M_sorted_4.mtx

    # "$path_selected"/wikipedia-20051105.mtx
    # "$path_selected"/wikipedia-20051105_sorted_1.mtx
    # "$path_selected"/wikipedia-20051105_sorted_2.mtx
    # "$path_selected"/wikipedia-20051105_sorted_3.mtx
    # "$path_selected"/wikipedia-20051105_sorted_4.mtx

    # "$path_selected_sorted"/circuit5M.mtx
)

if ((USE_DLCM_MATRICES)); then
    prog_args_k=()
    prog_args_q=()
    prog_args_v=()
    prog_args=()
    tmp=()
    if ((PIPELINE)); then
        for f in "${dlmc_matrices_files_k[@]}"; do
            # IFS=$'\n' read -d '' -a tmp < "$f"
            mapfile -t tmp < "$f"
            for item in "${tmp[@]}"; do
                prog_args_k+=("${path_dlmc}/${item}")
                # echo $item
            done
            # prog_args+=("${path_dlmc}/${tmp[@]}")
            # item="${path_dlmc}/${tmp[@]}"
            # echo $item
        done
        for f in "${dlmc_matrices_files_q[@]}"; do
            # IFS=$'\n' read -d '' -a tmp < "$f"
            mapfile -t tmp < "$f"
            for item in "${tmp[@]}"; do
                prog_args_q+=("${path_dlmc}/${item}")
            done
            # prog_args+=("${path_dlmc}/${tmp[@]}")
            # item="${path_dlmc}/${tmp[@]}"
            # echo $item
        done
        for f in "${dlmc_matrices_files_v[@]}"; do
            # IFS=$'\n' read -d '' -a tmp < "$f"
            mapfile -t tmp < "$f"
            for item in "${tmp[@]}"; do
                prog_args_v+=("${path_dlmc}/${item}")
            done
            # prog_args+=("${path_dlmc}/${tmp[@]}")
            # item="${path_dlmc}/${tmp[@]}"
            # echo $item
        done

    else
        for f in "${dlmc_matrices_files[@]}"; do
            # IFS=$'\n' read -d '' -a tmp < "$f"
            mapfile -t tmp < "$f"
            for item in "${tmp[@]}"; do
                prog_args+=("${path_dlmc}/${item}")
            done
            # prog_args+=("${path_dlmc}/${tmp[@]}")
            # item="${path_dlmc}/${tmp[@]}"
            # echo $item
        done
    fi
elif ((!USE_ARTIFICIAL_MATRICES)); then
    prog_args=("${matrices[@]}")
else
    prog_args=()
    tmp=()
    for f in "${artificial_matrices_files[@]}"; do
        IFS=$'\n' read -d '' -a tmp < "$f"
        prog_args+=("${tmp[@]}")
    done
fi

# prog_args=(

    # '28508159 28508159 5 1.6667 normal random 0.05 0 0.05 0.05 14'              # This bugs at 128 threads with mkl and mkl_sparse_set_mv_hint() for some reason.

    # '5154859 5154859 19.24389 5.73672 normal random 0.21196 1.44233 0.19755 1.03234 14'
    # '952203 952203 48.85772782 11.94657153 normal random 0.2042067138 0.5760045224 1.79674 0.906047 14 ldoor'
# )

temp_labels=( $(printf "%s\n" /sys/class/hwmon/hwmon*/temp*_label | sort) )
temp_inputs=( ${temp_labels[@]/label/input} )

for format_name in "${!progs[@]}"; do
    prog="${progs["$format_name"]}"

    if ((output_to_files)); then
        > out/"${format_name}_${cores}.out"
        exec 1>>out/"${format_name}_${cores}.out"
        > out/"${format_name}_${cores}.csv"
        exec 2>>out/"${format_name}_${cores}.csv"
    fi

    echo "$config_str"

    echo "program: $prog"
    if ((PIPELINE)); then
        echo "number of matrices: ${#prog_args_k[@]}"
    else
        echo "number of matrices: ${#prog_args[@]}"
    fi

    "$prog"

    rep=1
    # rep=4
    # rep=5
    # rep=16
    # rep=1024


    LEVEL3_CACHE_SIZE="$(getconf LEVEL3_CACHE_SIZE)"
    csrcv_num_packet_vals=(
        # 128 
        $((2**6))
        # $((2**7))
        # $((2**10))
        # $((2**14))
        # $((2**17))
        # $((2**20))
        # $((2**12))
        # $((LEVEL3_CACHE_SIZE / 8 / 8 / 16))
    )

    # if [[ "$prog" == *'spmv_csr_cv'* ]]; then
        # csrcv_num_packet_vals=( $( declare -i i; for (( i=64;i<=2**24;i*=2 )); do echo "$i"; done ) )
    # fi
    # RANDOM ADD
# export OMP_NESTED=TRUE
# export OMP_DYNAMIC=FALSE
# export MKL_DYNAMIC=FALSE
# export MKL_NUM_THREADS=16
# export MKL_THREADING_LAYER=GNU
# export GOMP_CPU_AFFINITY="$cpu_affinity"
# export XLSMPOPTS="PROCS=$cpu_affinity"
export OMP_PROC_BIND=true
export OMP_PLACES="{0},{1},{2},{3},{4},{5},{6},{7},{8},{9},{10},{11},{12},{13},{14},{15},{16},{17},{18},{19},{20},{21},{22},{23},{24},{25},{26},{27},{28},{29},{30},{31},{32},{33},{34},{35},{36},{37},{38},{39},{40},{41},{42},{43},{44},{45},{46},{47},{48},{49},{50},{51},{52},{53},{54},{55},{56},{57},{58},{59},{60},{61},{62},{63}"
# export OMP_PLACES="{0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63}"
# export OMP_PLACES="{0},{1},{2},{3},{4},{5},{6},{7},{8},{9},{10},{11},{12},{13},{14},{15},{16},{17},{18},{19},{20},{21},{22},{23}"
# export OMP_PROC_BIND=spread,close
# export OMP_STACKSIZE=1000
# export OMP_DISPLAY_ENV=true
    for ((i=0;i<rep;i++)); do
        if ((PIPELINE)); then
            # for ((a=0; a<${#prog_args_k[@]}; a++));
            for ((a=0; a<1; a++));
            do

                rep_in=1
                # rep_in=10

                for packet_vals in "${csrcv_num_packet_vals[@]}"; do
                    export CSRCV_NUM_PACKET_VALS="$packet_vals"

                    printf "Temps: " >&1
                    for ((k=0;k<${#temp_labels[@]};k++)); do
                        printf "%s %s " $(cat ${temp_labels[k]}) $(cat ${temp_inputs[k]}) >&1
                    done
                    echo >&1
                    if [[ ${prog_args_k[a]} =~ /0\.([^/]+)/ ]]; then
                    result="${BASH_REMATCH[1]}"
                    export SPARSITY="0.${result}"
                    echo "sparsity: $result"
                    else
                        echo "No match found."
                    fi
                    echo "File: "${prog_args_k[a]}" "${prog_args_q[a]}" "${prog_args_v[a]}""
                    bench "$prog" "${prog_args_k[a]}" "${prog_args_q[a]}" "${prog_args_v[a]}"

                done
            done
        else
            for ((a=0; a<${#prog_args[@]}; a++));
            do

                rep_in=1
                # rep_in=10

                for packet_vals in "${csrcv_num_packet_vals[@]}"; do
                    export CSRCV_NUM_PACKET_VALS="$packet_vals"

                    printf "Temps: " >&1
                    for ((k=0;k<${#temp_labels[@]};k++)); do
                        printf "%s %s " $(cat ${temp_labels[k]}) $(cat ${temp_inputs[k]}) >&1
                    done
                    echo >&1
                    if [[ ${prog_args[a]} =~ /0\.([^/]+)/ ]]; then
                    result="${BASH_REMATCH[1]}"
                    export SPARSITY="0.${result}"
                    echo "sparsity: $result"
                    else
                        echo "No match found."
                    fi
                    echo "File: "${prog_args[a]}""

                    bench "$prog" "${prog_args[a]}"
                    

                done
            done
        fi
    done
done

