#! /bin/bash

epsilonarray=(0.005)
fractionarray=(0.005)
epsilonarray=(0.1 0.01 0.005 0.001)
fractionarray=(0.1 0.01 0.005 0.001)

python_name="run.py"

maxiter=50000

rhoarray=(1.00001)
# gammaarray=(8.0 5.0 3.0 1.00001)
kappaarray=(0.0)
# kappaarray=(-10.0 -5.0 -3.0 -2.0 -1.0 0.0)

zeta=0.5
boundcarray=(0 1 2 3)
hW1=0.06
hW2=0.015
for epsilon in ${epsilonarray[@]}; do
    for fraction in "${fractionarray[@]}"; do
        for rho in "${rhoarray[@]}"; do
            for gamma in "${gammaarray[@]}"; do
                for boundc in "${boundcarray[@]}"; do
                    for kappa in "${kappaarray[@]}"; do
                        count=0

                        action_name="TwoCapital_mul_bc_wo_hcons"

                        action_name="${action_name}"
                        output="${action_name}_bc_${boundc}"

                        mkdir -p ./job-outs/${output}/frac_${fraction}/eps_${epsilon}/

                        if [ -f ./bash/${output}/frac_${fraction}/eps_${epsilon}/rho_${rho}_gamma_${gamma}_kappa_${kappa}_zeta_${zeta}.sh ]; then
                            rm ./bash/${output}/frac_${fraction}/eps_${epsilon}/rho_${rho}_gamma_${gamma}_kappa_${kappa}_zeta_${zeta}.sh
                        fi

                        mkdir -p ./bash/${output}/frac_${fraction}/eps_${epsilon}/

                        touch ./bash/${output}/frac_${fraction}/eps_${epsilon}/rho_${rho}_gamma_${gamma}_kappa_${kappa}_zeta_${zeta}.sh

                        tee -a ./bash/${output}/frac_${fraction}/eps_${epsilon}/rho_${rho}_gamma_${gamma}_kappa_${kappa}_zeta_${zeta}.sh <<EOF
#! /bin/bash

######## login
#SBATCH --job-name=${boundc}_${fraction}_${epsilon}
#SBATCH --output=./job-outs/${output}/frac_${fraction}/eps_${epsilon}/rho_${rho}_gamma_${gamma}_kappa_${kappa}_zeta_${zeta}.out
#SBATCH --error=./job-outs/${output}/frac_${fraction}/eps_${epsilon}/rho_${rho}_gamma_${gamma}_kappa_${kappa}_zeta_${zeta}.err

#SBATCH --account=pi-lhansen
#SBATCH --partition=caslake
#SBATCH --cpus-per-task=1
#SBATCH --mem=1G
#SBATCH --time=12:00:00

####### load modules
module load python  gcc

echo "\$SLURM_JOB_NAME"

echo "Program starts \$(date)"
start_time=\$(date +%s)
# perform a task

python3 -u /project/lhansen/sluggish/$python_name  --rho ${rho} --gamma ${gamma} --kappa ${kappa} --zeta ${zeta} --epsilon ${epsilon}  --fraction ${fraction}  --maxiter ${maxiter} --output ${output} --action_name ${action_name} --boundc ${boundc} --hW1 ${hW1} --hW2 ${hW2}
echo "Program ends \$(date)"
end_time=\$(date +%s)

# elapsed time with second resolution
elapsed=\$((end_time - start_time))

eval "echo Elapsed time: \$(date -ud "@\$elapsed" +'\$((%s/3600/24)) days %H hr %M min %S sec')"

EOF
                        count=$(($count + 1))
                        sbatch ./bash/${output}/frac_${fraction}/eps_${epsilon}/rho_${rho}_gamma_${gamma}_kappa_${kappa}_zeta_${zeta}.sh
                    done
                done
            done
        done
    done
done
