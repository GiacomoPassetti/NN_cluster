for L in 6
do
for alpha in 4 
do
for seed in 1 2 
do
for steps in 2000
do
for samples in  50 80 100
do

echo "#!/usr/local_rwth/bin/zsh




#SBATCH --job-name=syk_nn$alpha$seed # The job name.
#SBATCH -c 8         # The number of cpu cores to use.
#SBATCH --time=47:59:00              # The time the job will take to run.
#SBATCH --mem-per-cpu=3200M        # The memory the job will use per cpu core.
### SBATCH --account=rwth0722
#SBATCH --mail-type=NONE
#SBATCH --mail-user=passetti@physik.rwth-aachen.de


#SBATCH --output=$HOME/neural_network_cl/syk_AI/NN_cluster/Bash/output$L
#

### Change to the work directory
cd $HOME/neural_network_cl/syk_AI/NN_cluster
### Execute your application



module load python/3.8.11



mpirun -np 4 python3 play.py $L $alpha $seed $steps $samples

echo \$?

date
" >$HOME/neural_network_cl/syk_AI/NN_cluster/Bash/addBash_GSs

sbatch <$HOME/neural_network_cl/syk_AI/NN_cluster/Bash/addBash_GSs

done
done
done
done
done

