using QMDP, FIB, POMDPs, POMDPTools, QuickPOMDPs, POMDPModelTools, BeliefUpdaters, Parameters, POMDPToolbox


pomdp = QuickPOMDP(
    states = collect(1:10),                   #𝒮
    actions = collect(1:10),               #𝒜
    observations = ["correct", "wrong"],          #𝒪
    initialstate = Uniform(collect(1:5)),    # Initial state of a student is random 1 to 5
    discount = 0.95,                          # γ

    transition = function (s, a)
        probs = zeros(10)
        if s == 10                      # can't go higher than 10  
            probs[s] == 1
        elseif a == s               # easier to grasp if material is at the level of a student
            probs[s] = 0.7
            probs[s+1] = 0.3
        elseif a == s + 1                # harder to grasp if material is higher than the level of a student 
            probs[s] = 0.85
            probs[s+1] = 0.15
        else 
            probs[s] = 1                # stays at the same level if given tasks lower or >= 2 from his current level
        return SparseCat(states, probs)
        end
    end,
        
    observation = function (s, a, s′)
        if s′ - s == 0                                       # hasn't mastered the level s yet
            SparseCat(observations, [0.8, 0.2])
        elseif s′ - s == 1                                   # mastered the level s and moved to the next one s+1
            SparseCat(observations, [0.3, 0.7])
        end 
    end,
        
    reward = (s,a)->(s == 10 ? 100 : -1)        # if reaches level 10 reward is 100 and -1 if each step if not
)


#belief = [0.2, 0.2, 0.2, 0.2, 0.2, 0, 0, 0, 0, 0]

#solver = FancyAlgorithmSolver()
#policy = solve(solver, pomdp) 

qmdp_iters = 50

#qmdp_solver = QMDPSolver(max_iterations=qmdp_iters);
#qmdp_policy = solve(qmdp_solver, pomdp)

fib_solver = FIBSolver()
fib_policy = solve(fib_solver, pomdp)

#pbvi_solver = PBVISolver()
#pbvi_policy = solve(pbvi_solver, pomdp)

#print("qmdp_policy: ", qmdp_policy)
print("fib_policy: ", fib_policy)


print("---------------------")
# initialize problem and solver

QMDP_solver = QMDPSolver() # from QMDP

# compute a policy
QMDP_policy = solve(QMDP_solver, pomdp)

#evaluate the policy
#belief_updater = updater(QMDP_policy) # the default QMDP belief updater (discrete Bayesian filter)
#init_dist = initialstate_distribution(pomdp) # from POMDPModels
#hr = HistoryRecorder(max_steps=100) # from POMDPTools
#hist = simulate(hr, pomdp, QMDP_policy, belief_updater, pomdp.initialstate) # run 100 step simulation
#println("reward: $(discounted_reward(hist))")
println("QMDP_policy: ", QMDP_policy)
