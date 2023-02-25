#! /bin/bash

epsilonarray=(0.1 0.01 0.005 0.001)
fractionarray=(0.1 0.01 0.005 0.001)

actiontime=1

python_name="twocap3dplot.py"

maxiter=5000000

rhoarray=(1.00001)

gammaarray=(8.0)
Acaparray=(0.0)
# Acaparray=(0.2 0.3 0.35 0.4 0.45 0.5 0.6 0.7 0.8 0.9 1.0)
# Acaparray=(0.37 0.38 0.4)
A1caparray=(0.5 0.6 0.7)
A2caparray=(0.5 0.6 0.7)


for epsilon in ${epsilonarray[@]}; do
    for fraction in "${fractionarray[@]}"; do
        for rho in "${rhoarray[@]}"; do
            for gamma in "${gammaarray[@]}"; do
                for A1cap in "${Acaparray[@]}"; do
                    # for A2cap in "${Acaparray[@]}"; do
                        count=0

                        action_name="TwoCapital_natural49_hconstraint"
                        # action_name="newtestpe3"

                        dataname="${action_name}_${epsilon}_frac_${fraction}"

                        mkdir -p ./job-outs/${action_name}/eps_${epsilon}_frac_${fraction}/

                        if [ -f ./bash/${action_name}/eps_${epsilon}_frac_${fraction}/rho_${rho}_gamma_${gamma}_Acap_${A1cap}_plot.sh ]; then
                            rm ./bash/${action_name}/eps_${epsilon}_frac_${fraction}/rho_${rho}_gamma_${gamma}_Acap_${A1cap}_plot.sh
                        fi

                        mkdir -p ./bash/${action_name}/eps_${epsilon}_frac_${fraction}/

                        touch ./bash/${action_name}/eps_${epsilon}_frac_${fraction}/rho_${rho}_gamma_${gamma}_Acap_${A1cap}_plot.sh

                        tee -a ./bash/${action_name}/eps_${epsilon}_frac_${fraction}/rho_${rho}_gamma_${gamma}_Acap_${A1cap}_plot.sh <<EOF
#! /bin/bash

######## login
#SBATCH --job-name=${A1cap}_${epsilon}_${fraction}
#SBATCH --output=./job-outs/${action_name}/eps_${epsilon}_frac_${fraction}/rho_${rho}_gamma_${gamma}_Acap_${A1cap}_plot.out
#SBATCH --error=./job-outs/${action_name}/eps_${epsilon}_frac_${fraction}/rho_${rho}_gamma_${gamma}_Acap_${A1cap}_plot.err

#SBATCH --account=pi-lhansen
#SBATCH --partition=caslake
#SBATCH --cpus-per-task=1
#SBATCH --mem=1G
#SBATCH --time=1:00:00

####### load modules
module load python  gcc

echo "\$SLURM_JOB_NAME"

echo "Program starts \$(date)"
start_time=\$(date +%s)
# perform a task

python3 -u /project/lhansen/CTU/$python_name  --rho ${rho} --gamma ${gamma}  --A1cap ${A1cap} --epsilon ${epsilon}  --fraction ${fraction}   --maxiter ${maxiter} --dataname ${dataname} --figname ${action_name}
echo "Program ends \$(date)"
end_time=\$(date +%s)

# elapsed time with second resolution
elapsed=\$((end_time - start_time))

eval "echo Elapsed time: \$(date -ud "@\$elapsed" +'\$((%s/3600/24)) days %H hr %M min %S sec')"

EOF
                        count=$(($count + 1))
                        sbatch ./bash/${action_name}/eps_${epsilon}_frac_${fraction}/rho_${rho}_gamma_${gamma}_Acap_${A1cap}_plot.sh
                    # done
                done
            done
        done
    done
done
