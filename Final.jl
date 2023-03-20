using POMDPs, QMDP, POMDPModels, POMDPTools, FIB, SARSOP, PlotlyJS
using Distributions, StatsBase, CSV, DataFrames, StatsPlots


mutable struct ITS_POMDP <: POMDP{Int64, Int64, Bool}
    r_step::Float64
    r_end::Float64
    p_next_when_current::Float64
    p_next_when_next::Float64
    p_correct_when_grasped::Float64
    p_correct_when_not_grasped::Float64
    discount::Float64
end

ITS_POMDP(p_next_when_current,p_next_when_next,p_correct_when_grasped,p_correct_when_not_grasped) = ITS_POMDP(-1, 0, p_next_when_current,p_next_when_next,p_correct_when_grasped,p_correct_when_not_grasped, 1)
ITS_POMDP() = ITS_POMDP(0.4, 0.2, 0.8, 0.3)

#updater(problem::ITS_POMDP) = DiscreteUpdater(problem)

POMDPs.actions(::ITS_POMDP) = collect(1:10)
POMDPs.actionindex(::ITS_POMDP, a::Int64) = a

POMDPs.states(::ITS_POMDP) = collect(1:10)
POMDPs.stateindex(::ITS_POMDP, s::Int64) = s

POMDPs.observations(::ITS_POMDP) = [true, false]
POMDPs.obsindex(::ITS_POMDP, o::Bool) = o + 1

POMDPs.discount(pomdp::ITS_POMDP) = pomdp.discount

# start knowing that at the start students' knowledge is uniformly distributed between 1 and 5


POMDPs.initialstate(::ITS_POMDP) = POMDPTools.Uniform(collect(1:5))       # could be any state from 1 to 5
#POMDPs.initial_belief(::ITS_POMDP) = DiscreteBelief(5)  

 

POMDPs.isterminal(::ITS_POMDP,s::Int64) = s==length(states(pomdp))

function POMDPs.transition(pomdp::ITS_POMDP, s::Int64, a::Int64)
    probs = zeros(10)
    if s == 10                      # can't go higher than 10  
        probs[s] == 1
    elseif a == s               # easier to grasp if material is at the level of a student
        probs[s] = 1 - pomdp.p_next_when_current
        probs[s+1] = pomdp.p_next_when_current
    elseif a == s + 1                # harder to grasp if material is higher than the level of a student 
        probs[s] = 1 - pomdp.p_next_when_next
        probs[s+1] = pomdp.p_next_when_next
    else
        probs[s] = 1                # stays at the same level if given tasks lower of >= 2 from his current level
    end
    return SparseCat(states(pomdp), probs)
end



function POMDPs.observation(pomdp::ITS_POMDP, s::Int64, a::Int64, sn::Int64)
    if (a > s) && (s == sn)                                # hasn't mastered the level s yet, 30% chance will succeed
        return BoolDistribution(pomdp.p_correct_when_not_grasped)
    elseif (a > s) && (sn == s+1)                          # mastered the level 80% will succeed
        return BoolDistribution(pomdp.p_correct_when_grasped)                     
    elseif a > s                                          # hasn't mastered the level s yet, 30% chance will succeed  
        return BoolDistribution(pomdp.p_correct_when_not_grasped)       
    else                                                  # a<=s, so mastered the level s, 80% chance will succed
        return BoolDistribution(pomdp.p_correct_when_grasped)
    end 
end



function POMDPs.reward(pomdp::ITS_POMDP, sn::Int64, a::Int64)
    r = pomdp.r_step
    if sn == length(actions(pomdp))
        r = pomdp.r_end
    end
    return r
end


# initialize problem and solver
pomdp = ITS_POMDP()
init_dist = initialstate(pomdp) # from POMDPModels
n_sim = 20   # number of simulations
max_steps = 40

### QMDP
qmdp_solver = QMDPSolver() # from QMDP
qmdp_policy = QMDP.solve(qmdp_solver, pomdp) # compute a policy
qmdp_belief_updater = updater(qmdp_policy) # the default QMDP belief updater (discrete Bayesian filter)
qmdp_rewards = zeros(1:n_sim)
qmdp_passed = []
qmdp_not_passed = []
qmdp_actions_all = [] # coll of coll of all actions
qmdp_hr = HistoryRecorder(max_steps=max_steps) # from POMDPTools


### FIB
fib_solver= FIBSolver()  # from FIB
fib_policy = FIB.solve(fib_solver, pomdp)
fib_belief_updater = updater(fib_policy) 
fib_rewards = zeros(1:n_sim)
fib_passed = []
fib_not_passed = []
fib_hr = HistoryRecorder(max_steps=max_steps) # from POMDPTools

"""
### SARSOP
sarsop_solver= SARSOPSolver()  # from FIB
sarsop_policy = SARSOP.solve(sarsop_solver, pomdp)
sarsop_belief_updater = updater(sarsop_policy) # the default QMDP belief updater (discrete Bayesian filter)
sarsop_rewards = zeros(1:n_sim)
sarsop_hr = HistoryRecorder(max_steps=max_steps) # from POMDPTools
"""

### Regular policy - assumes the teacher starts with a=1 and spends at each action exactly 3 steps until incrementing by 1.
mutable struct RegularPolicy <: Policy 
    n::Int64
    last_action::Int64
end

POMDPs.updater(::RegularPolicy) = NothingUpdater()

function POMDPs.action(rp::RegularPolicy, ::Any)
    if rp.n == 4
        rp.last_action += 1
        rp.n = 1
    else
        rp.n += 1
    end
    #println("a= ", rp.last_action)
    #println("n= ", rp.n)
    return rp.last_action
end
regpol_rewards = zeros(1:n_sim)
regpol_hr = HistoryRecorder(max_steps=max_steps) # from POMDPTools
regpol_passed = []
regpol_not_passed = []
regpol_actions = [] # coll of all actions

"""
### AEMS
TestSimulator = AEMS.RolloutSimulator()
aems_solver= AEMSSolver()  # from FIB
aems_policy = solve(aems_solver, pomdp)
aems_belief_updater = updater(aems_policy) # the default QMDP belief updater (discrete Bayesian filter)
aems_rewards = zeros(1:n_sim)
aems_hr = HistoryRecorder(max_steps=100) # from POMDPTools
"""


for i in 1:n_sim
    qmdp_hist = simulate(qmdp_hr, pomdp, qmdp_policy, qmdp_belief_updater, init_dist) # run 30 step simulation
    qmdp_rewards[i] = discounted_reward(qmdp_hist)
    qmdp_states = state_hist(qmdp_hist)
    qmdp_actions = collect(eachstep(qmdp_hist, "a"))
    append!(qmdp_actions_all, qmdp_actions)
    #qmdp_actions = action_hist(qmdp_hist)[a:] 
    #println("qmdp_S", qmdp_states)
    #println("qmdp_A", qmdp_actions)
    if (last(qmdp_states) == 10)
        append!(qmdp_passed, max_steps+1 - length(qmdp_states))   # time economized for those who mastered early
    else
        append!(qmdp_not_passed,last(qmdp_states))    # level of those who didn't master the course
    end

    fib_hist = simulate(fib_hr, pomdp, fib_policy, fib_belief_updater, init_dist) # run 30 step simulation
    fib_rewards[i] = discounted_reward(fib_hist)
    fib_states = state_hist(fib_hist)
    fib_actions = collect(eachstep(fib_hist, "a"))
    if (last(fib_states) == 10)
        append!(fib_passed, max_steps+1 - length(fib_states))   # time economized for those who mastered early
    else
        append!(fib_not_passed,last(fib_states))    # level of those who didn't master the course
    end


    regpol = RegularPolicy(0,1)
    regpol_belief_updater = updater(regpol)
    regpol_hist = simulate(regpol_hr, pomdp, regpol, regpol_belief_updater, init_dist) # run 30 step simulation
    regpol_rewards[i] = discounted_reward(regpol_hist)
    regpol_states = state_hist(regpol_hist)
    regpol_actions = collect(eachstep(regpol_hist, "a"))
    #regpol_actions = action_hist(regpol_hist)
    println("regpol_A: ", regpol_actions)
    #println("------------------------------")
    if (last(regpol_states) == 10)
        append!(regpol_passed, max_steps+1 - length(regpol_states))   # time economized for those who mastered early
    else
        append!(regpol_not_passed,last(regpol_states))    # level of those who didn't master the course
    end
    #println("regpol_hist: ",acts)
    #println("-----------------------------")
    """
    aems_hist = simulate(aems_hr, pomdp, aems_policy, aems_belief_updater, init_dist) # run 100 step simulation
    aems_rewards[i] = discounted_reward(aems_hist)
    """
end



println("QMDP: ", summarystats(qmdp_rewards))
println("FIB: ", summarystats(fib_rewards))
#println("SARSOP: ", summarystats(sarsop_rewards))
println("Regular: ", summarystats(regpol_rewards))
#println("QMDP_Passed", qmdp_passed)
#println("regpol_Passed", regpol_passed)

#println("AEMS: ", summarystats(aems_rewards))
#println("policy: ", qmdp_policy)


function boxplot_rewards()
    rewards1 = PlotlyJS.box(y=qmdp_rewards, marker_color="indianred", name="QMDP rewards")
    rewards2 = PlotlyJS.box(y=fib_rewards, marker_color="lightseagreen", name="FIB rewards")
    rewards3 = PlotlyJS.box(y=regpol_rewards, marker_color="gray", name="Regular rewards")
    PlotlyJS.plot([rewards1, rewards2, rewards3])
end
#boxplot_rewards()

function hist_steps_saved()
    trace1 = PlotlyJS.histogram(x=qmdp_passed, marker_color="indianred", opacity=0.5, name="QMDP steps saved")
    #trace2 = PlotlyJS.histogram(x=fib_passed, marker_color="lightseagreen", opacity=0.5, name="FIB steps saved")
    trace3 = PlotlyJS.histogram(x= regpol_passed, marker_color="gray", opacity=0.5, name="Regular steps saved")
    data = [trace1, trace3]
    layout = PlotlyJS.Layout(barmode="overlay")
    PlotlyJS.plot(data, layout)
end
# hist_steps_saved()

function hist_level_not_passed()
    trace1 = PlotlyJS.histogram(x=qmdp_not_passed, marker_color="indianred", opacity=0.5, name="QMDP")
    #trace2 = PlotlyJS.histogram(x=fib_passed, marker_color="lightseagreen", opacity=0.5, name="FIB steps saved")
    trace3 = PlotlyJS.histogram(x= regpol_not_passed, marker_color="gray", opacity=0.5, name="Regular")
    data = [trace1, trace3]
    layout = PlotlyJS.Layout(barmode="overlay")
    PlotlyJS.plot(data, layout)
end
#hist_level_not_passed()

function policy_viz()
   
    x_all = range(1, length=40)
    y_regpol = [1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5, 6, 6, 6, 6, 7, 7, 7, 7, 8, 8, 8, 8, 9, 9, 9, 9, 10, 10, 10, 10]
    data = []

    """
    for i in 1:n_sim
        y = qmdp_actions_all[i]
        trace = PlotlyJS.line(x=x_all, y=y)
        #trace = {type: "scatter", x: x_all, y: y, mode: "lines", name: "Blue", line: {color: "rgb(55, 128, 191)", width: 1}};
        append!(data,trace)
    end
    """

    qmdp_hist1 = simulate(qmdp_hr, pomdp, qmdp_policy, qmdp_belief_updater, init_dist) # run 30 step simulation
    qmdp_actions1 = collect(eachstep(qmdp_hist1, "a"))
    trace1 = PlotlyJS.scatter(x=x_all, y=qmdp_actions1, line=attr(color="indianred", width=2), name="QMDP_1")

    qmdp_hist2 = simulate(qmdp_hr, pomdp, qmdp_policy, qmdp_belief_updater, init_dist) # run 30 step simulation
    qmdp_actions2 = collect(eachstep(qmdp_hist2, "a"))
    trace2 = PlotlyJS.scatter(x=x_all, y=qmdp_actions2, line=attr(color="darkorange", width=2),name="QMDP_2")

    qmdp_hist3 = simulate(qmdp_hr, pomdp, qmdp_policy, qmdp_belief_updater, init_dist) # run 30 step simulation
    qmdp_actions3 = collect(eachstep(qmdp_hist3, "a"))
    trace3 = PlotlyJS.scatter(x=x_all, y=qmdp_actions3, line=attr(color="coral", width=2), name="QMDP_3")

    qmdp_hist4 = simulate(qmdp_hr, pomdp, qmdp_policy, qmdp_belief_updater, init_dist) # run 30 step simulation
    qmdp_actions4 = collect(eachstep(qmdp_hist4, "a"))
    trace4 = PlotlyJS.scatter(x=x_all, y=qmdp_actions4, line=attr(color="crimson", width=2),name="QMDP_4")

    qmdp_hist5 = simulate(qmdp_hr, pomdp, qmdp_policy, qmdp_belief_updater, init_dist) # run 30 step simulation
    qmdp_actions5 = collect(eachstep(qmdp_hist5, "a"))
    trace5 = PlotlyJS.scatter(x=x_all, y=qmdp_actions5, line=attr(color="mediumpurple", width=2), name="QMDP_5")

    qmdp_hist6 = simulate(qmdp_hr, pomdp, qmdp_policy, qmdp_belief_updater, init_dist) # run 30 step simulation
    qmdp_actions6 = collect(eachstep(qmdp_hist6, "a"))
    trace6 = PlotlyJS.scatter(x=x_all, y=qmdp_actions6, line=attr(color="indianred", width=1))

    qmdp_hist7 = simulate(qmdp_hr, pomdp, qmdp_policy, qmdp_belief_updater, init_dist) # run 30 step simulation
    qmdp_actions7 = collect(eachstep(qmdp_hist7, "a"))
    trace7 = PlotlyJS.scatter(x=x_all, y=qmdp_actions7, line=attr(color="indianred", width=1))

    qmdp_hist8 = simulate(qmdp_hr, pomdp, qmdp_policy, qmdp_belief_updater, init_dist) # run 30 step simulation
    qmdp_actions8 = collect(eachstep(qmdp_hist8, "a"))
    trace8 = PlotlyJS.scatter(x=x_all, y=qmdp_actions8, line=attr(color="indianred", width=1))

    qmdp_hist9 = simulate(qmdp_hr, pomdp, qmdp_policy, qmdp_belief_updater, init_dist) # run 30 step simulation
    qmdp_actions9 = collect(eachstep(qmdp_hist9, "a"))
    trace9 = PlotlyJS.scatter(x=x_all, y=qmdp_actions9, line=attr(color="indianred", width=1))

    qmdp_hist10 = simulate(qmdp_hr, pomdp, qmdp_policy, qmdp_belief_updater, init_dist) # run 30 step simulation
    qmdp_actions10 = collect(eachstep(qmdp_hist10, "a"))
    trace10 = PlotlyJS.scatter(x=x_all, y=qmdp_actions10, line=attr(color="indianred", width=1))

    trace11 = PlotlyJS.scatter(x=x_all, y=y_regpol, line=attr(color="gray", width=5),name="Regular")
    #trace2 = {type: "scatter", x: x_all, y: y_regpol, mode: "lines", name: "Red", line: {color: "rgb(219, 64, 82)", width: 3}};

    data = [trace1, trace2, trace3, trace4, trace5, trace11]

    layout = PlotlyJS.Layout(barmode="overlay")

    PlotlyJS.plot(data, layout)
end
policy_viz()