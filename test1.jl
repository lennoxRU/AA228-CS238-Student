using POMDPs, QMDP, POMDPModels, POMDPTools

# initialize problem and solver
pomdp = BabyPOMDP() # from POMDPModels
print("pomdp:", pomdp)
solver = QMDPSolver() # from QMDP

# compute a policy
policy = solve(solver, pomdp)

#evaluate the policy
belief_updater = updater(policy) # the default QMDP belief updater (discrete Bayesian filter)
init_dist = initialstate_distribution(pomdp) # from POMDPModels
hr = HistoryRecorder(max_steps=100) # from POMDPTools
hist = simulate(hr, pomdp, policy, belief_updater, init_dist) # run 100 step simulation
println("reward: $(discounted_reward(hist))")
println("policy: ", policy)

