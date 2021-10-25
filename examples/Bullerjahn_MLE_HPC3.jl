using DiffusionMLE, Plots, PyCall, CSV, DataFrames, HDF5

include("smeared_trajectory_integrator.jl")

const M = 1000 # Number of trajectories
const d = 2 # Dimension of trajectories

const N_sub = 10 # Number of substeps over which the trajectory is smeared out

N = [ rand(150:150) for i = 1 : M ] # Array of trajectory lengths

const a2_1 = 0.5
const a2_2 = 2.0
const a2_3 = 1.0
const σ2_1 = 0.1
const σ2_2 = 1.0
const σ2_3 = 10.0

B = [1/6 for m = 1 : M] # Array of blurring coefficients, where we have assumed a uniform illumination profile
data = vcat([make_2D_data(N[1:300],N_sub,a2_1,σ2_1), 
        make_2D_data(N[301:700],N_sub,a2_2,σ2_2), 
        make_2D_data(N[701:1000],N_sub,a2_3,σ2_3)]...); # Mock data set

function print_results(estimates,uncertainties)
    K = size(estimates,2)
    for k = 1 : K
        println(string("a2_", k, " = ", estimates[1,k], " ± ", uncertainties[1,k]))
    end
    for k = 1 : K
        println(string("σ2_", k, " = ", estimates[2,k], " ± ", uncertainties[2,k]))
    end
    for k = 1 : K
        println(string("P_", k, " = ", estimates[3,k]))
    end
end

const N_local = 5000 # Max number of expectation-maximization cycles
const N_global = 1000 # Number of iterations with different initial parameters

# Ranges from which the initial values for the parameters are drawn:
a2_range = [ 0.001, 1000. ];
σ2_range = [ 0.001, 1000. ];

using Base.Threads
println(string("Number of available cores for threading: ", nthreads()))

parameters = MLE_estimator(B,data)
parameter_matrix = reshape(vcat([parameters,[1.0]]...), 3, 1)
P1_estimates, P1_L, P1_T = local_EM_estimator!(d,M,1,N_local,parameter_matrix,B,data)
P1_uncertainties = MLE_errors(B,data,parameters)

println("Estimates:")
print_results(P1_estimates,P1_uncertainties)

Q_sub = subpopulation_analysis(P1_T,P1_estimates,B,data)
Q = vcat(Q_sub...)

println()

println("Kuiper statistic:")
println("κ = ",Kuiper_statistic!(Q))

# Load the 3 Cell Type data

# Loads the first 100 frames of a trajectory for processing
function load_100traj_csv(raw_csv_file_path)
    csv_data = CSV.read(raw_csv_file_path, DataFrame)
    csv_data = dropmissing(csv_data)
    # Get just the first 100 frames of each track
    # csv_data = csv_data[csv_data.Frame .< 101, :]
    # Get the first 100 trajectories
    csv_data = csv_data[csv_data.ID .< 200, :]
    trackIDs = unique(csv_data, :ID).ID
    trk_count = length(trackIDs)
    B_values = [1/6 for m = 1 : trk_count]
    sing_cond_tracks = Array{Array{Float64,2},1}(undef, trk_count)
    n = 0
    for i in trackIDs
        n += 1
        indiv_track = csv_data[in([i]).(csv_data.ID), :]
        sing_cond_tracks[n] = Array(indiv_track[:, [:X, :Y]])
    end
    return sing_cond_tracks, B_values
end

# Setup import paths for trajectories
# SVMv7_3CellTypes_37Degree = raw"C:\Users\User\OneDrive\Documents\Python Programs\Piezo1_MLE\Trajectory_JSONs\SVMv7_3CellTypes_Trajectory Data\tdTomato_37Degree\Mobile\tdTomato_37Degree_mobile_0-200frames_stride2.csv";
# SVMv7_3CellTypes_Endothelial = raw"C:\Users\User\OneDrive\Documents\Python Programs\Piezo1_MLE\Trajectory_JSONs\SVMv7_3CellTypes_Trajectory Data\tdTomato_Degree37_Endothelials\Mobile\tdTomato_Degree37_Endothelials_mobile_0-200frames_stride2.csv";
SVMv7_3CellTypes_mNSPC = raw"C:\Users\User\OneDrive\Documents\Python Programs\Piezo1_MLE\Trajectory_JSONs\SVMv7_3CellTypes_Trajectory Data\tdTomato_Degree37_mNSPCs\Mobile\tdTomato_Degree37_mNSPCs_mobile_0-200frames_stride2.csv";

# Prepare data for MLE
# mob_data_37Degree, B_values_37Degree = load_100traj_csv(SVMv7_3CellTypes_37Degree);
# mob_data_Endothelial, B_values_Endothelial = load_100traj_csv(SVMv7_3CellTypes_Endothelial);
mob_data_mNSPC, B_values_mNSPC = load_100traj_csv(SVMv7_3CellTypes_mNSPC);

mob_data_curr = mob_data_mNSPC
B_values_curr = B_values_mNSPC

K_vals = zeros(0)

for i in 2:10
    PX_estimates, PX_L, PX_T = global_EM_estimator(i,N_local,N_global,a2_range,σ2_range,B_values_curr,mob_data_curr);
    B_sub, X_sub = sort_trajectories(i,PX_T,B_values_curr,mob_data_curr)
    PX_uncertainties = hcat([ MLE_errors(B_sub[k], X_sub[k], PX_estimates[1:2,k]) for k = 1 : i ]...)
    Q_sub = subpopulation_analysis(PX_T,PX_estimates,B_values_curr,mob_data_curr)
    Q = vcat(Q_sub...)
    println(i, " Kuiper statistic:")
    println("K=",Kuiper_statistic!(Q))
    append!(K_vals, Kuiper_statistic!(Q))
    print(K_vals)
end

print(K_vals)