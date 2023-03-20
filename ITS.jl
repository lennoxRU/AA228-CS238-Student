# Stanford AA228/CS238, Winter 2023, Final project
# Alexey Tuzikov

# Linear Intelligent Tuturing System problem (LinearITS)
# State is 1 to 10 student knowledge level
# Action is true for up and false for stay
# Observation is true for Correct and false for Incorrect

mutable struct ITS_POMDP <: POMDP{Int64, Int64, Bool}
    r_step::Float64
    r_end::Float64
    p_next_when_current::Float64
    p_next_when_next::Float64
    p_correct_when_grasped::Float64
    p_correct_when_not_grasped::Float64
    discount::Float64
end

ITS_POMDP(p_next_when_current,p_next_when_next,p_correct_when_grasped,p_correct_when_not_grasped) = ITS_POMDP(-1, 100, p_next_when_current,p_next_when_next,p_correct_when_grasped,p_correct_when_not_grasped, 0.95)
ITS_POMDP() = ITS_POMDP(0.4, 0.2, 0.8, 0.3)

#updater(problem::ITS_POMDP) = DiscreteUpdater(problem)

POMDPs.actions(::ITS_POMDP) = [true, false]
POMDPs.actionindex(::ITS_POMDP, a::Bool) = a + 1

POMDPs.states(::ITS_POMDP) = collect(1:10)
POMDPs.stateindex(::ITS_POMDP, s::Int64) = s

POMDPs.observations(::ITS_POMDP) = [true, false]
POMDPs.obsindex(::ITS_POMDP, o::Bool) = o + 1

POMDPs.discount(pomdp::ITS_POMDP) = pomdp.discount

# start knowing that at the start students' knowledge is uniformly distributed between 1 and 5


POMDPs.initialstate(::ITS_POMDP) = Uniform(collect(1:5))       # could be any state from 1 to 5
#POMDPs.initial_belief(::ITS_POMDP) = DiscreteBelief(5)  

 

POMDPs.isterminal(::ITS_POMDP,s::Int64) = s==10

function POMDPs.transition(pomdp::ITS_POMDP, s::Int64, a::Int64)
    probs = zeros(10)
    if s == 10                      # can't go higher than 10  
        probs[s] == 1
    elseif a == false               # easier to grasp if material is at the level of a student
        probs[s] = 1 - pomdp.p_next_when_current
        probs[s+1] = pomdp.p_next_when_current
    elseif a == true                # harder to grasp if material is higher than the level of a student 
        probs[s] = 1 - pomdp.p_next_when_next
        probs[s+1] = pomdp.p_next_when_next
    else 
        probs[s] = 1                # stays at the same level if given tasks lower of >= 2 from his current level
    return SparseCat(states, probs)
    end
end


function POMDPs.observation(pomdp::ITS_POMDP, a::Int64, sn::Int64)
    if sn >= 5                             # hasn't mastered the level s yet, 30% chance will succeed
        return BoolDistribution(pomdp.p_correct_when_grasped)           # mastered the level 80% will succeed
    else                                                     # hasn't mastered the level s yet, 30% chance will succee                                                           # a<=s, so mastered the level s, 80% chance will succed
        return BoolDistribution(pomdp.p_correct_when_grasped)
    end 
end
"""
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

"""

function POMDPs.reward(pomdp::ITS_POMDP, s::Int64, a::Int64)
    r = -1
    if s == 10
        r = 100
    end
    return r
end
